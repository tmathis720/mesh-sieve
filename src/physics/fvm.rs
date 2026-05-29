//! Finite-volume oriented wrappers over mesh-sieve topology/data/geometry primitives.
//!
//! This module is the canonical entrypoint for finite-volume (FV) callers.
//! It intentionally exposes FV-ready mesh loops and interpolation helpers without
//! requiring users to couple directly to finite-element runtime APIs.
//!
//! # API tradeoffs: ergonomic maps vs packed arrays
//! - `cell_metrics(point)` / `face_metrics(point)` provide ergonomic point-addressed access and are
//!   convenient for setup/validation/debug flows.
//! - Hot loops should prefer pre-indexed contiguous arrays (`internal_owner_neighbor_indices`,
//!   `internal_face_geometry_indices`, `boundary_owner_indices`, `boundary_face_geometry_indices`)
//!   or `PackedFvmInputs` to avoid repeated map lookups and branch-heavy access.
//! - `FvmPackedCache` allows allocating packed buffers once and rebuilding content across
//!   timesteps/iterations, reducing allocator churn while preserving a stable high-level API.

use crate::data::section::Section;
use crate::data::storage::Storage;
use crate::discretization::runtime::{
    CellGeometry, FaceGeometry, FluxStencil, discretization_capability,
};
use crate::mesh_error::MeshSieveError;
use crate::topology::cell_type::CellType;
use crate::topology::coastal::{
    BOUNDARY_CLASS_LABEL, BOUNDARY_ROLE_LABEL, BoundaryClass, OpenBoundaryRole, WET_DRY_MASK_LABEL,
    WetDryMask,
};
use crate::topology::labels::LabelSet;
use crate::topology::point::PointId;
use crate::topology::sieve::Sieve;
use std::collections::{HashMap, HashSet};

/// Validate that a cell topology is part of the explicit PetscFV-style runtime matrix.
pub fn validate_fv_cell_type(cell_type: CellType) -> Result<(), MeshSieveError> {
    let supported = match cell_type {
        CellType::Simplex(dim) => (1..=3).contains(&dim),
        CellType::Polygon(_) | CellType::Polyhedron => true,
        other => discretization_capability(other)
            .map(|capability| capability.finite_volume)
            .unwrap_or(false),
    };
    if supported {
        Ok(())
    } else {
        Err(MeshSieveError::InvalidGeometry(format!(
            "unsupported finite-volume cell topology {cell_type:?}"
        )))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaceKind {
    Internal,
    Boundary,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ConvectiveScheme {
    Upwind,
    Central,
    BoundedLinear {
        blend: f64,
    },
    BlendUpwindCentral {
        blend: f64,
    },
    HighResolution {
        blend: f64,
        limiter: SlopeLimiterFamily,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReconstructionGradient {
    LeastSquares,
    GreenGauss,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReconstructionMode {
    GradientOnly(ReconstructionGradient),
    LeastSquaresWithGreenGaussFallback,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlopeLimiterFamily {
    None,
    Minmod,
    VanLeer,
    Superbee,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReconstructionSettings {
    pub mode: ReconstructionMode,
    pub limiter: LimiterOption,
}

impl Default for ReconstructionSettings {
    fn default() -> Self {
        Self {
            mode: ReconstructionMode::LeastSquaresWithGreenGaussFallback,
            limiter: LimiterOption::Family(SlopeLimiterFamily::None),
        }
    }
}

pub trait BoundedReconstructionLimiter {
    fn limiter_factor(&self, upwind: f64, downwind: f64) -> f64;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LimiterOption {
    None,
    Family(SlopeLimiterFamily),
}

impl BoundedReconstructionLimiter for LimiterOption {
    fn limiter_factor(&self, upwind: f64, downwind: f64) -> f64 {
        match self {
            LimiterOption::None => 1.0,
            LimiterOption::Family(f) => slope_limiter(*f, upwind, downwind),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BoundaryCondition {
    Dirichlet { value: f64 },
    Neumann { gradient: f64 },
    Robin { alpha: f64, beta: f64, gamma: f64 },
}

/// FV-oriented coastal boundary branch used to select flux closure behavior.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FvBoundaryBranch {
    Open,
    Inflow,
    Outflow,
    Tidal,
    Bed,
    FreeSurface,
}

/// Runtime errors when resolving boundary branch labels for FV assembly.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BoundaryBranchError {
    MissingBoundaryClass {
        face: PointId,
    },
    OpenBoundaryMissingRole {
        face: PointId,
    },
    OpenBoundaryConflictingRoles {
        face: PointId,
        roles: Vec<OpenBoundaryRole>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiffusionSettings {
    pub diffusivity: f64,
    pub non_orthogonal_mode: NonOrthogonalCorrectionMode,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NonOrthogonalCorrectionMode {
    OrthogonalOnly,
    Deferred,
    FullyCorrected,
}

impl Default for DiffusionSettings {
    fn default() -> Self {
        Self {
            diffusivity: 1.0,
            non_orthogonal_mode: NonOrthogonalCorrectionMode::FullyCorrected,
        }
    }
}

pub trait FvmPhysicsHooks {
    fn turbulence_flux(&self, _stencil: &FluxStencil, _inputs: &FvmInputs) -> f64 {
        0.0
    }
    fn free_surface_flux(&self, _stencil: &FluxStencil, _inputs: &FvmInputs) -> f64 {
        0.0
    }
    fn turbulence_source(&self, _cell: PointId, _inputs: &FvmInputs) -> f64 {
        0.0
    }
    fn free_surface_source(&self, _cell: PointId, _inputs: &FvmInputs) -> f64 {
        0.0
    }
}

pub trait FvmSourceFluxHook {
    fn additional_face_flux(&self, _stencil: &FluxStencil, _inputs: &FvmInputs) -> f64 {
        0.0
    }
    fn additional_cell_source(&self, _cell: PointId, _inputs: &FvmInputs) -> f64 {
        0.0
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct FluxAssembly {
    pub residual: HashMap<PointId, f64>,
    pub source: HashMap<PointId, f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FvmFieldSections {
    pub cell_scalar: HashMap<PointId, f64>,
    pub cell_gradient: HashMap<PointId, Vec<f64>>,
    pub face_mass_flux: HashMap<PointId, f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FvmSchemeSettings {
    pub convective: ConvectiveScheme,
    pub reconstruction: ReconstructionSettings,
    pub diffusion: DiffusionSettings,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FvmAssemblyOutput {
    pub convective: FluxAssembly,
    pub diffusive: FluxAssembly,
    pub total: FluxAssembly,
}

#[derive(Debug, PartialEq)]
pub enum FvmAssemblyError {
    MissingFaceMetric {
        face: PointId,
    },
    MissingCellMetric {
        cell: PointId,
    },
    MissingBoundaryFaceLabel {
        face: PointId,
    },
    LabelMappedToUnknownFace {
        face: PointId,
    },
    LabelMappedToInternalFace {
        face: PointId,
    },
    UnsupportedBoundaryBranch {
        face: PointId,
        branch: FvBoundaryBranch,
    },
    MissingBoundaryClosure {
        face: PointId,
        branch: FvBoundaryBranch,
        kind: FaceKind,
    },
    Kernel(MeshSieveError),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnsupportedBoundaryBehavior {
    Error,
    Ignore,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FvBoundaryPolicy {
    pub boundary_face_branches: HashMap<PointId, FvBoundaryBranch>,
    pub allowed_branches: HashSet<FvBoundaryBranch>,
    pub convective_branch_hooks: HashMap<FvBoundaryBranch, BoundaryCondition>,
    pub diffusive_branch_hooks: HashMap<FvBoundaryBranch, BoundaryCondition>,
    pub unsupported_behavior: UnsupportedBoundaryBehavior,
}

impl FvBoundaryPolicy {
    fn build_face_bcs(
        &self,
        kind: FaceKind,
    ) -> Result<HashMap<PointId, BoundaryCondition>, FvmAssemblyError> {
        let branch_hooks = match kind {
            FaceKind::Internal => return Ok(HashMap::new()),
            FaceKind::Boundary => &self.convective_branch_hooks,
        };
        let mut result = HashMap::with_capacity(self.boundary_face_branches.len());
        for (face, branch) in &self.boundary_face_branches {
            if let Some(bc) = branch_hooks.get(branch) {
                result.insert(*face, *bc);
            } else if self.unsupported_behavior == UnsupportedBoundaryBehavior::Error {
                return Err(FvmAssemblyError::MissingBoundaryClosure {
                    face: *face,
                    branch: *branch,
                    kind,
                });
            }
        }
        Ok(result)
    }

    fn build_diffusive_face_bcs(
        &self,
    ) -> Result<HashMap<PointId, BoundaryCondition>, FvmAssemblyError> {
        let mut result = HashMap::with_capacity(self.boundary_face_branches.len());
        for (face, branch) in &self.boundary_face_branches {
            if let Some(bc) = self.diffusive_branch_hooks.get(branch) {
                result.insert(*face, *bc);
            } else if self.unsupported_behavior == UnsupportedBoundaryBehavior::Error {
                return Err(FvmAssemblyError::MissingBoundaryClosure {
                    face: *face,
                    branch: *branch,
                    kind: FaceKind::Boundary,
                });
            }
        }
        Ok(result)
    }
}

impl From<MeshSieveError> for FvmAssemblyError {
    fn from(value: MeshSieveError) -> Self {
        Self::Kernel(value)
    }
}

pub fn assemble_fv_system(
    inputs: &FvmInputs,
    fields: &FvmFieldSections,
    boundary_policy: &FvBoundaryPolicy,
    schemes: FvmSchemeSettings,
) -> Result<FvmAssemblyOutput, FvmAssemblyError> {
    preflight_fv_assembly(inputs, boundary_policy)?;
    let convective_bcs = boundary_policy.build_face_bcs(FaceKind::Boundary)?;
    let diffusive_bcs = boundary_policy.build_diffusive_face_bcs()?;
    let convective = assemble_convective_fluxes_with_reconstruction(
        inputs,
        &fields.cell_scalar,
        &fields.face_mass_flux,
        &convective_bcs,
        schemes.convective,
        schemes.reconstruction,
    )?;
    let diffusive = assemble_diffusive_fluxes(
        inputs,
        &fields.cell_scalar,
        &fields.cell_gradient,
        &diffusive_bcs,
        schemes.diffusion,
    )?;
    let mut total = FluxAssembly::default();
    for (cell, value) in convective.residual.iter().chain(diffusive.residual.iter()) {
        total.add_residual(*cell, *value);
    }
    for (cell, value) in convective.source.iter().chain(diffusive.source.iter()) {
        total.add_source(*cell, *value);
    }
    Ok(FvmAssemblyOutput {
        convective,
        diffusive,
        total,
    })
}

fn preflight_fv_assembly(
    inputs: &FvmInputs,
    boundary_policy: &FvBoundaryPolicy,
) -> Result<(), FvmAssemblyError> {
    for stencil in inputs.internal_faces() {
        if inputs.face_metrics(stencil.face).is_none() {
            return Err(FvmAssemblyError::MissingFaceMetric { face: stencil.face });
        }
        if inputs.cell_metrics(stencil.left).is_none() {
            return Err(FvmAssemblyError::MissingCellMetric { cell: stencil.left });
        }
        let right = stencil.right.expect("internal");
        if inputs.cell_metrics(right).is_none() {
            return Err(FvmAssemblyError::MissingCellMetric { cell: right });
        }
    }
    let boundary_faces: HashSet<_> = inputs.boundary_faces().map(|s| s.face).collect();
    for face in boundary_policy.boundary_face_branches.keys() {
        if inputs.face_metrics(*face).is_none() {
            return Err(FvmAssemblyError::LabelMappedToUnknownFace { face: *face });
        }
        if !boundary_faces.contains(face) {
            return Err(FvmAssemblyError::LabelMappedToInternalFace { face: *face });
        }
    }
    for stencil in inputs.boundary_faces() {
        if inputs.face_metrics(stencil.face).is_none() {
            return Err(FvmAssemblyError::MissingFaceMetric { face: stencil.face });
        }
        if inputs.cell_metrics(stencil.left).is_none() {
            return Err(FvmAssemblyError::MissingCellMetric { cell: stencil.left });
        }
        let branch = boundary_policy
            .boundary_face_branches
            .get(&stencil.face)
            .copied()
            .ok_or(FvmAssemblyError::MissingBoundaryFaceLabel { face: stencil.face })?;
        if !boundary_policy.allowed_branches.contains(&branch) {
            if boundary_policy.unsupported_behavior == UnsupportedBoundaryBehavior::Error {
                return Err(FvmAssemblyError::UnsupportedBoundaryBranch {
                    face: stencil.face,
                    branch,
                });
            }
            continue;
        }
    }
    Ok(())
}

impl FluxAssembly {
    fn add_residual(&mut self, cell: PointId, value: f64) {
        *self.residual.entry(cell).or_insert(0.0) += value;
    }
    fn add_source(&mut self, cell: PointId, value: f64) {
        *self.source.entry(cell).or_insert(0.0) += value;
    }
}

#[derive(Clone, Debug, Default)]
pub struct FvmFaceLoops {
    pub internal: Vec<FluxStencil>,
    pub boundary: Vec<FluxStencil>,
}

#[derive(Clone, Debug, Default)]
pub struct FvmInputs {
    pub loops: FvmFaceLoops,
    pub cell_geometry: Vec<(PointId, CellGeometry)>,
    pub face_geometry: Vec<(PointId, FaceGeometry)>,
    cell_index: HashMap<PointId, usize>,
    face_index: HashMap<PointId, usize>,
    internal_owner_neighbor_idx: Vec<(usize, usize)>,
    boundary_owner_idx: Vec<usize>,
    internal_face_geom_idx: Vec<usize>,
    boundary_face_geom_idx: Vec<usize>,
    packed_cache: Option<PackedFvmInputs>,
}

#[derive(Clone, Debug, Default)]
pub struct PackedFvmInputs {
    pub internal_faces: Vec<PackedInternalFace>,
    pub boundary_faces: Vec<PackedBoundaryFace>,
}

#[derive(Clone, Debug, Default)]
pub struct FvmPackedCache {
    pub packed: PackedFvmInputs,
}

#[derive(Clone, Debug)]
pub struct PackedInternalFace {
    pub face: PointId,
    pub owner: PointId,
    pub neighbor: PointId,
    pub face_geom_idx: usize,
    pub owner_cell_geom_idx: usize,
    pub neighbor_cell_geom_idx: usize,
}

#[derive(Clone, Debug)]
pub struct PackedBoundaryFace {
    pub face: PointId,
    pub owner: PointId,
    pub face_geom_idx: usize,
    pub owner_cell_geom_idx: usize,
}

impl FvmInputs {
    pub fn new(
        stencils: impl IntoIterator<Item = FluxStencil>,
        cell_geometry: Vec<(PointId, CellGeometry)>,
        face_geometry: Vec<(PointId, FaceGeometry)>,
    ) -> Self {
        let mut loops = FvmFaceLoops::default();
        for stencil in stencils {
            if stencil.right.is_some() {
                loops.internal.push(stencil);
            } else {
                loops.boundary.push(stencil);
            }
        }
        let mut inputs = Self {
            loops,
            cell_geometry,
            face_geometry,
            cell_index: HashMap::new(),
            face_index: HashMap::new(),
            internal_owner_neighbor_idx: Vec::new(),
            boundary_owner_idx: Vec::new(),
            internal_face_geom_idx: Vec::new(),
            boundary_face_geom_idx: Vec::new(),
            packed_cache: None,
        };
        inputs.rebuild_indices();
        inputs
    }

    pub fn internal_faces(&self) -> impl Iterator<Item = &FluxStencil> {
        self.loops.internal.iter()
    }
    pub fn boundary_faces(&self) -> impl Iterator<Item = &FluxStencil> {
        self.loops.boundary.iter()
    }
    pub fn cell_metrics(&self, cell: PointId) -> Option<&CellGeometry> {
        self.cell_index
            .get(&cell)
            .and_then(|&idx| self.cell_geometry.get(idx).map(|(_, m)| m))
    }
    pub fn face_metrics(&self, face: PointId) -> Option<&FaceGeometry> {
        self.face_index
            .get(&face)
            .and_then(|&idx| self.face_geometry.get(idx).map(|(_, m)| m))
    }

    pub fn internal_owner_neighbor_indices(&self) -> &[(usize, usize)] {
        &self.internal_owner_neighbor_idx
    }

    pub fn boundary_owner_indices(&self) -> &[usize] {
        &self.boundary_owner_idx
    }

    pub fn internal_face_geometry_indices(&self) -> &[usize] {
        &self.internal_face_geom_idx
    }

    pub fn boundary_face_geometry_indices(&self) -> &[usize] {
        &self.boundary_face_geom_idx
    }

    pub fn packed(&self) -> Option<&PackedFvmInputs> {
        self.packed_cache.as_ref()
    }

    pub fn build_packed_cache(&mut self) {
        let mut internal_faces = Vec::with_capacity(self.loops.internal.len());
        let mut boundary_faces = Vec::with_capacity(self.loops.boundary.len());
        for stencil in &self.loops.internal {
            if let (Some(&fi), Some(&oi), Some(neighbor), Some(&ni)) = (
                self.face_index.get(&stencil.face),
                self.cell_index.get(&stencil.left),
                stencil.right,
                stencil.right.and_then(|r| self.cell_index.get(&r)),
            ) {
                internal_faces.push(PackedInternalFace {
                    face: stencil.face,
                    owner: stencil.left,
                    neighbor,
                    face_geom_idx: fi,
                    owner_cell_geom_idx: oi,
                    neighbor_cell_geom_idx: ni,
                });
            }
        }
        for stencil in &self.loops.boundary {
            if let (Some(&fi), Some(&oi)) = (
                self.face_index.get(&stencil.face),
                self.cell_index.get(&stencil.left),
            ) {
                boundary_faces.push(PackedBoundaryFace {
                    face: stencil.face,
                    owner: stencil.left,
                    face_geom_idx: fi,
                    owner_cell_geom_idx: oi,
                });
            }
        }
        self.packed_cache = Some(PackedFvmInputs {
            internal_faces,
            boundary_faces,
        });
    }

    pub fn build_packed_cache_into(&self, cache: &mut FvmPackedCache) {
        cache.packed.internal_faces.clear();
        cache.packed.boundary_faces.clear();
        cache
            .packed
            .internal_faces
            .reserve(self.loops.internal.len());
        cache
            .packed
            .boundary_faces
            .reserve(self.loops.boundary.len());
        for stencil in &self.loops.internal {
            if let (Some(&fi), Some(&oi), Some(neighbor), Some(&ni)) = (
                self.face_index.get(&stencil.face),
                self.cell_index.get(&stencil.left),
                stencil.right,
                stencil.right.and_then(|r| self.cell_index.get(&r)),
            ) {
                cache.packed.internal_faces.push(PackedInternalFace {
                    face: stencil.face,
                    owner: stencil.left,
                    neighbor,
                    face_geom_idx: fi,
                    owner_cell_geom_idx: oi,
                    neighbor_cell_geom_idx: ni,
                });
            }
        }
        for stencil in &self.loops.boundary {
            if let (Some(&fi), Some(&oi)) = (
                self.face_index.get(&stencil.face),
                self.cell_index.get(&stencil.left),
            ) {
                cache.packed.boundary_faces.push(PackedBoundaryFace {
                    face: stencil.face,
                    owner: stencil.left,
                    face_geom_idx: fi,
                    owner_cell_geom_idx: oi,
                });
            }
        }
    }

    fn rebuild_indices(&mut self) {
        self.cell_index.clear();
        self.face_index.clear();
        for (idx, (id, _)) in self.cell_geometry.iter().enumerate() {
            self.cell_index.insert(*id, idx);
        }
        for (idx, (id, _)) in self.face_geometry.iter().enumerate() {
            self.face_index.insert(*id, idx);
        }
        self.internal_owner_neighbor_idx.clear();
        self.internal_face_geom_idx.clear();
        self.internal_owner_neighbor_idx
            .reserve(self.loops.internal.len());
        self.internal_face_geom_idx
            .reserve(self.loops.internal.len());
        for stencil in &self.loops.internal {
            if let (Some(&owner), Some(neighbor), Some(&neigh)) = (
                self.cell_index.get(&stencil.left),
                stencil.right,
                stencil.right.and_then(|r| self.cell_index.get(&r)),
            ) {
                let _ = neighbor;
                self.internal_owner_neighbor_idx.push((owner, neigh));
            }
            if let Some(&fidx) = self.face_index.get(&stencil.face) {
                self.internal_face_geom_idx.push(fidx);
            }
        }
        self.boundary_owner_idx.clear();
        self.boundary_face_geom_idx.clear();
        self.boundary_owner_idx.reserve(self.loops.boundary.len());
        self.boundary_face_geom_idx
            .reserve(self.loops.boundary.len());
        for stencil in &self.loops.boundary {
            if let Some(&owner) = self.cell_index.get(&stencil.left) {
                self.boundary_owner_idx.push(owner);
            }
            if let Some(&fidx) = self.face_index.get(&stencil.face) {
                self.boundary_face_geom_idx.push(fidx);
            }
        }
    }
}

pub fn classify_face_loops<T>(
    topology: &T,
    faces: impl IntoIterator<Item = PointId>,
) -> Result<FvmFaceLoops, MeshSieveError>
where
    T: Sieve<Point = PointId>,
{
    let mut loops = FvmFaceLoops::default();
    for face in faces {
        let mut cells: Vec<_> = topology.support_points(face).collect();
        cells.sort_unstable();
        match cells.as_slice() {
            [left] => loops.boundary.push(FluxStencil {
                face,
                left: *left,
                right: None,
            }),
            [left, right] => loops.internal.push(FluxStencil {
                face,
                left: *left,
                right: Some(*right),
            }),
            [] => {
                return Err(MeshSieveError::InvalidGeometry(format!(
                    "face {face:?} has no supporting cells"
                )));
            }
            _ => {
                return Err(MeshSieveError::InvalidGeometry(format!(
                    "non-manifold face {face:?} has {} support cells",
                    cells.len()
                )));
            }
        }
    }
    Ok(loops)
}

pub fn interpolate_face_centered_scalar<S: Storage<f64>>(
    section: &Section<f64, S>,
    stencil: &FluxStencil,
) -> Result<f64, MeshSieveError> {
    let left = section.try_restrict(stencil.left)?;
    let lval = *left.first().ok_or_else(|| {
        MeshSieveError::InvalidGeometry(format!("missing scalar value at cell {}", stencil.left))
    })?;
    if let Some(right_cell) = stencil.right {
        let right = section.try_restrict(right_cell)?;
        let rval = *right.first().ok_or_else(|| {
            MeshSieveError::InvalidGeometry(format!("missing scalar value at cell {right_cell}"))
        })?;
        Ok(0.5 * (lval + rval))
    } else {
        Ok(lval)
    }
}

pub fn interpolate_cell_centered_scalar<S: Storage<f64>>(
    section: &Section<f64, S>,
    incident_faces: &[PointId],
) -> Result<f64, MeshSieveError> {
    if incident_faces.is_empty() {
        return Err(MeshSieveError::InvalidGeometry(
            "cannot interpolate to cell center from zero incident faces".to_string(),
        ));
    }
    let mut accum = 0.0;
    for face in incident_faces {
        let value = section.try_restrict(*face)?;
        let scalar = *value.first().ok_or_else(|| {
            MeshSieveError::InvalidGeometry(format!("missing scalar value at face {face}"))
        })?;
        accum += scalar;
    }
    Ok(accum / incident_faces.len() as f64)
}

pub fn assemble_convective_fluxes(
    inputs: &FvmInputs,
    cell_scalar: &HashMap<PointId, f64>,
    face_mass_flux: &HashMap<PointId, f64>,
    boundary_conditions: &HashMap<PointId, BoundaryCondition>,
    scheme: ConvectiveScheme,
) -> Result<FluxAssembly, MeshSieveError> {
    assemble_convective_fluxes_with_reconstruction(
        inputs,
        cell_scalar,
        face_mass_flux,
        boundary_conditions,
        scheme,
        ReconstructionSettings::default(),
    )
}

pub fn assemble_convective_fluxes_with_reconstruction(
    inputs: &FvmInputs,
    cell_scalar: &HashMap<PointId, f64>,
    face_mass_flux: &HashMap<PointId, f64>,
    boundary_conditions: &HashMap<PointId, BoundaryCondition>,
    scheme: ConvectiveScheme,
    reconstruction: ReconstructionSettings,
) -> Result<FluxAssembly, MeshSieveError> {
    assemble_convective_fluxes_masked(
        inputs,
        cell_scalar,
        face_mass_flux,
        boundary_conditions,
        scheme,
        reconstruction,
        None,
    )
}

/// Optional face-activity mask used to gate FV flux contributions.
#[derive(Clone, Debug, Default)]
pub struct FluxActivityMask {
    pub boundary_faces_active: HashMap<PointId, bool>,
    pub near_boundary_faces_active: HashMap<PointId, bool>,
}

fn face_is_active(mask: Option<&FluxActivityMask>, stencil: &FluxStencil, boundary: bool) -> bool {
    let Some(mask) = mask else { return true };
    if boundary {
        return mask
            .boundary_faces_active
            .get(&stencil.face)
            .copied()
            .unwrap_or(true);
    }
    if mask
        .near_boundary_faces_active
        .get(&stencil.face)
        .copied()
        .unwrap_or(true)
    {
        return true;
    }
    false
}

/// Build a flux-activity mask from coastal wet/dry labels on cells.
pub fn flux_activity_mask_from_wet_dry(inputs: &FvmInputs, labels: &LabelSet) -> FluxActivityMask {
    let mut mask = FluxActivityMask::default();
    let dry = WetDryMask::Dry.code();
    let cell_is_dry = |cell: PointId| labels.get_label(cell, WET_DRY_MASK_LABEL) == Some(dry);
    for stencil in inputs.boundary_faces() {
        mask.boundary_faces_active
            .insert(stencil.face, !cell_is_dry(stencil.left));
    }
    for stencil in inputs.internal_faces() {
        let near_boundary = labels
            .get_label(stencil.face, BOUNDARY_CLASS_LABEL)
            .is_some();
        if near_boundary {
            let left_dry = cell_is_dry(stencil.left);
            let right_dry = stencil.right.map(cell_is_dry).unwrap_or(true);
            mask.near_boundary_faces_active
                .insert(stencil.face, !(left_dry || right_dry));
        }
    }
    mask
}

/// Assemble convective fluxes with optional wet/dry gating.
pub fn assemble_convective_fluxes_masked(
    inputs: &FvmInputs,
    cell_scalar: &HashMap<PointId, f64>,
    face_mass_flux: &HashMap<PointId, f64>,
    boundary_conditions: &HashMap<PointId, BoundaryCondition>,
    scheme: ConvectiveScheme,
    reconstruction: ReconstructionSettings,
    activity_mask: Option<&FluxActivityMask>,
) -> Result<FluxAssembly, MeshSieveError> {
    let mut assembly = FluxAssembly::default();
    for stencil in inputs.internal_faces() {
        if !face_is_active(activity_mask, stencil, false) {
            continue;
        }
        let mdot = *face_mass_flux.get(&stencil.face).ok_or_else(|| {
            MeshSieveError::InvalidGeometry(format!("missing mass flux for face {}", stencil.face))
        })?;
        let l = *cell_scalar.get(&stencil.left).ok_or_else(|| {
            MeshSieveError::InvalidGeometry(format!("missing scalar for cell {}", stencil.left))
        })?;
        let rcell = stencil.right.expect("internal face must have neighbor");
        let r = *cell_scalar.get(&rcell).ok_or_else(|| {
            MeshSieveError::InvalidGeometry(format!("missing scalar for cell {rcell}"))
        })?;
        let phi_f = convective_face_value(l, r, mdot, scheme, reconstruction.limiter);
        let flux = mdot * phi_f;
        assembly.add_residual(stencil.left, flux);
        assembly.add_residual(rcell, -flux);
    }
    for stencil in inputs.boundary_faces() {
        if !face_is_active(activity_mask, stencil, true) {
            continue;
        }
        let mdot = *face_mass_flux.get(&stencil.face).ok_or_else(|| {
            MeshSieveError::InvalidGeometry(format!(
                "missing mass flux for boundary face {}",
                stencil.face
            ))
        })?;
        let phi_i = *cell_scalar.get(&stencil.left).ok_or_else(|| {
            MeshSieveError::InvalidGeometry(format!("missing scalar for cell {}", stencil.left))
        })?;
        let bc = boundary_conditions
            .get(&stencil.face)
            .copied()
            .ok_or_else(|| {
                MeshSieveError::InvalidGeometry(format!(
                    "missing boundary condition for face {}",
                    stencil.face
                ))
            })?;
        let phi_b = match bc {
            BoundaryCondition::Dirichlet { value } => value,
            BoundaryCondition::Neumann { .. } => phi_i,
            BoundaryCondition::Robin { alpha, beta, gamma } => {
                if alpha.abs() < 1e-14 {
                    phi_i
                } else {
                    (gamma - beta * phi_i) / alpha
                }
            }
        };
        let phi_f = convective_face_value(phi_i, phi_b, mdot, scheme, reconstruction.limiter);
        assembly.add_residual(stencil.left, mdot * phi_f);
    }
    Ok(assembly)
}

pub fn boundary_branch_for_face(labels: &LabelSet, face: PointId) -> Option<FvBoundaryBranch> {
    match labels.get_label(face, BOUNDARY_CLASS_LABEL) {
        Some(v) if v == BoundaryClass::FreeSurface.code() => Some(FvBoundaryBranch::FreeSurface),
        Some(v) if v == BoundaryClass::Bed.code() => Some(FvBoundaryBranch::Bed),
        Some(v) if v == BoundaryClass::Open.code() => {
            match labels.get_label(face, BOUNDARY_ROLE_LABEL) {
                Some(v) if v == OpenBoundaryRole::Inflow.code() => Some(FvBoundaryBranch::Inflow),
                Some(v) if v == OpenBoundaryRole::Outflow.code() => Some(FvBoundaryBranch::Outflow),
                Some(v) if v == OpenBoundaryRole::Tidal.code() => Some(FvBoundaryBranch::Tidal),
                _ => Some(FvBoundaryBranch::Open),
            }
        }
        _ => None,
    }
}

/// Resolve a boundary branch with explicit runtime validation for open-boundary roles.
pub fn boundary_branch_for_face_checked(
    labels: &LabelSet,
    face: PointId,
) -> Result<FvBoundaryBranch, BoundaryBranchError> {
    let Some(class) = labels.get_label(face, BOUNDARY_CLASS_LABEL) else {
        return Err(BoundaryBranchError::MissingBoundaryClass { face });
    };
    if class == BoundaryClass::FreeSurface.code() {
        return Ok(FvBoundaryBranch::FreeSurface);
    }
    if class == BoundaryClass::Bed.code() {
        return Ok(FvBoundaryBranch::Bed);
    }
    if class != BoundaryClass::Open.code() {
        return Err(BoundaryBranchError::MissingBoundaryClass { face });
    }
    let inflow =
        labels.get_label(face, BOUNDARY_ROLE_LABEL) == Some(OpenBoundaryRole::Inflow.code());
    let outflow =
        labels.get_label(face, BOUNDARY_ROLE_LABEL) == Some(OpenBoundaryRole::Outflow.code());
    let tidal = labels.get_label(face, BOUNDARY_ROLE_LABEL) == Some(OpenBoundaryRole::Tidal.code());
    let mut roles = Vec::new();
    if inflow {
        roles.push(OpenBoundaryRole::Inflow);
    }
    if outflow {
        roles.push(OpenBoundaryRole::Outflow);
    }
    if tidal {
        roles.push(OpenBoundaryRole::Tidal);
    }
    match roles.as_slice() {
        [OpenBoundaryRole::Inflow] => Ok(FvBoundaryBranch::Inflow),
        [OpenBoundaryRole::Outflow] => Ok(FvBoundaryBranch::Outflow),
        [OpenBoundaryRole::Tidal] => Ok(FvBoundaryBranch::Tidal),
        [] => Err(BoundaryBranchError::OpenBoundaryMissingRole { face }),
        _ => Err(BoundaryBranchError::OpenBoundaryConflictingRoles { face, roles }),
    }
}

pub fn assemble_diffusive_fluxes(
    inputs: &FvmInputs,
    cell_scalar: &HashMap<PointId, f64>,
    cell_gradient: &HashMap<PointId, Vec<f64>>,
    boundary_conditions: &HashMap<PointId, BoundaryCondition>,
    settings: DiffusionSettings,
) -> Result<FluxAssembly, MeshSieveError> {
    assemble_diffusive_fluxes_with_hooks(
        inputs,
        cell_scalar,
        cell_gradient,
        boundary_conditions,
        settings,
        None,
    )
}

pub fn assemble_diffusive_fluxes_with_hooks(
    inputs: &FvmInputs,
    cell_scalar: &HashMap<PointId, f64>,
    cell_gradient: &HashMap<PointId, Vec<f64>>,
    boundary_conditions: &HashMap<PointId, BoundaryCondition>,
    settings: DiffusionSettings,
    hooks: Option<&dyn FvmPhysicsHooks>,
) -> Result<FluxAssembly, MeshSieveError> {
    let mut assembly = FluxAssembly::default();
    for stencil in inputs.internal_faces() {
        let fg = inputs.face_metrics(stencil.face).ok_or_else(|| {
            MeshSieveError::InvalidGeometry(format!(
                "missing face geometry for face {}",
                stencil.face
            ))
        })?;
        let lc = inputs.cell_metrics(stencil.left).ok_or_else(|| {
            MeshSieveError::InvalidGeometry(format!(
                "missing cell geometry for cell {}",
                stencil.left
            ))
        })?;
        let rcell = stencil.right.expect("internal face");
        let rc = inputs.cell_metrics(rcell).ok_or_else(|| {
            MeshSieveError::InvalidGeometry(format!("missing cell geometry for cell {rcell}"))
        })?;
        let phi_l = *cell_scalar.get(&stencil.left).ok_or_else(|| {
            MeshSieveError::InvalidGeometry(format!("missing scalar for cell {}", stencil.left))
        })?;
        let phi_r = *cell_scalar.get(&rcell).ok_or_else(|| {
            MeshSieveError::InvalidGeometry(format!("missing scalar for cell {rcell}"))
        })?;
        let d = sub(&rc.centroid, &lc.centroid);
        let d2 = dot(&d, &d);
        let a = &fg.normal;
        let orth = if d2 > 0.0 {
            (phi_r - phi_l) * dot(a, &d) / d2
        } else {
            0.0
        };
        let mut flux = -settings.diffusivity * orth;
        let mut non_orth_flux = 0.0;
        if settings.non_orthogonal_mode != NonOrthogonalCorrectionMode::OrthogonalOnly {
            let gf = avg(
                cell_gradient.get(&stencil.left),
                cell_gradient.get(&rcell),
                a.len(),
            )?;
            let a_orth = if d2 > 0.0 {
                scale(&d, dot(a, &d) / d2)
            } else {
                vec![0.0; a.len()]
            };
            let a_non = sub(a, &a_orth);
            non_orth_flux = -settings.diffusivity * dot(&gf, &a_non);
        }
        match settings.non_orthogonal_mode {
            NonOrthogonalCorrectionMode::OrthogonalOnly => {}
            NonOrthogonalCorrectionMode::Deferred => {
                assembly.add_source(stencil.left, -non_orth_flux)
            }
            NonOrthogonalCorrectionMode::FullyCorrected => flux += non_orth_flux,
        }
        if let Some(h) = hooks {
            flux += h.turbulence_flux(stencil, inputs) + h.free_surface_flux(stencil, inputs);
        }
        assembly.add_residual(stencil.left, flux);
        assembly.add_residual(rcell, -flux);
    }
    for stencil in inputs.boundary_faces() {
        let fg = inputs.face_metrics(stencil.face).ok_or_else(|| {
            MeshSieveError::InvalidGeometry(format!(
                "missing face geometry for boundary face {}",
                stencil.face
            ))
        })?;
        let lc = inputs.cell_metrics(stencil.left).ok_or_else(|| {
            MeshSieveError::InvalidGeometry(format!(
                "missing cell geometry for cell {}",
                stencil.left
            ))
        })?;
        let phi_i = *cell_scalar.get(&stencil.left).ok_or_else(|| {
            MeshSieveError::InvalidGeometry(format!("missing scalar for cell {}", stencil.left))
        })?;
        let bc = boundary_conditions
            .get(&stencil.face)
            .copied()
            .ok_or_else(|| {
                MeshSieveError::InvalidGeometry(format!(
                    "missing boundary condition for face {}",
                    stencil.face
                ))
            })?;
        let n = normalize(&fg.normal);
        let d = norm(&sub(&fg.centroid, &lc.centroid)).max(1e-14);
        match bc {
            BoundaryCondition::Dirichlet { value } => {
                let flux = -settings.diffusivity * (value - phi_i) / d * fg.area;
                assembly.add_residual(stencil.left, flux);
            }
            BoundaryCondition::Neumann { gradient } => {
                let flux = -settings.diffusivity * gradient * fg.area;
                assembly.add_residual(stencil.left, flux);
            }
            BoundaryCondition::Robin { alpha, beta, gamma } => {
                let grad_n = (gamma - alpha * phi_i) / beta.max(1e-14);
                let flux = -settings.diffusivity * grad_n * fg.area;
                let _ = n;
                assembly.add_residual(stencil.left, flux);
                assembly.add_source(
                    stencil.left,
                    settings.diffusivity * gamma * fg.area / beta.max(1e-14),
                );
            }
        }
    }
    if let Some(h) = hooks {
        for (cell, _) in &inputs.cell_geometry {
            assembly.add_source(
                *cell,
                h.turbulence_source(*cell, inputs) + h.free_surface_source(*cell, inputs),
            );
        }
    }
    Ok(assembly)
}

fn convective_face_value(
    l: f64,
    r: f64,
    mdot: f64,
    scheme: ConvectiveScheme,
    limiter: LimiterOption,
) -> f64 {
    let up = if mdot >= 0.0 { l } else { r };
    let dn = if mdot >= 0.0 { r } else { l };
    let central = 0.5 * (l + r);
    let (mn, mx) = (l.min(r), l.max(r));
    match scheme {
        ConvectiveScheme::Upwind => up,
        ConvectiveScheme::Central => central,
        ConvectiveScheme::BoundedLinear { blend } => {
            clamp(up + clamp01(blend) * (central - up), mn, mx)
        }
        ConvectiveScheme::BlendUpwindCentral { blend } => {
            (1.0 - clamp01(blend)) * up + clamp01(blend) * central
        }
        ConvectiveScheme::HighResolution {
            blend,
            limiter: lim,
        } => {
            let psi = limiter.limiter_factor(up, dn) * slope_limiter(lim, up, dn);
            let hr = up + 0.5 * psi * (dn - up);
            let b = clamp01(blend);
            clamp((1.0 - b) * up + b * hr, mn, mx)
        }
    }
}
fn slope_limiter(limiter: SlopeLimiterFamily, up: f64, dn: f64) -> f64 {
    let r = if (dn - up).abs() < 1e-14 {
        1.0
    } else {
        ((up - dn) / (dn - up)).abs()
    };
    match limiter {
        SlopeLimiterFamily::None => 1.0,
        SlopeLimiterFamily::Minmod => r.clamp(0.0, 1.0),
        SlopeLimiterFamily::VanLeer => (r + r.abs()) / (1.0 + r.abs()),
        SlopeLimiterFamily::Superbee => (2.0 * r).min(1.0).max(r.min(2.0)).max(0.0),
    }
}
fn clamp(x: f64, lo: f64, hi: f64) -> f64 {
    x.max(lo).min(hi)
}
fn clamp01(x: f64) -> f64 {
    clamp(x, 0.0, 1.0)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}
fn sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}
fn scale(a: &[f64], s: f64) -> Vec<f64> {
    a.iter().map(|x| x * s).collect()
}
fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}
fn normalize(a: &[f64]) -> Vec<f64> {
    let n = norm(a).max(1e-14);
    scale(a, 1.0 / n)
}
fn avg(a: Option<&Vec<f64>>, b: Option<&Vec<f64>>, n: usize) -> Result<Vec<f64>, MeshSieveError> {
    let la = a.ok_or_else(|| MeshSieveError::InvalidGeometry("missing left gradient".into()))?;
    let lb = b.ok_or_else(|| MeshSieveError::InvalidGeometry("missing right gradient".into()))?;
    if la.len() != n || lb.len() != n {
        return Err(MeshSieveError::InvalidGeometry(
            "gradient dimension mismatch".into(),
        ));
    }
    Ok(la
        .iter()
        .zip(lb.iter())
        .map(|(x, y)| 0.5 * (x + y))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pid(id: u64) -> PointId {
        PointId::new(id).unwrap()
    }

    fn policy(
        branches: HashMap<PointId, FvBoundaryBranch>,
        allowed: HashSet<FvBoundaryBranch>,
    ) -> FvBoundaryPolicy {
        FvBoundaryPolicy {
            boundary_face_branches: branches,
            allowed_branches: allowed,
            convective_branch_hooks: HashMap::new(),
            diffusive_branch_hooks: HashMap::new(),
            unsupported_behavior: UnsupportedBoundaryBehavior::Error,
        }
    }

    #[test]
    fn convective_two_cell_conservation() {
        let c0 = pid(1);
        let c1 = pid(2);
        let f = pid(10);
        let inputs = FvmInputs::new(
            [FluxStencil {
                face: f,
                left: c0,
                right: Some(c1),
            }],
            vec![],
            vec![],
        );
        let phi = HashMap::from([(c0, 2.0), (c1, 6.0)]);
        let mdot = HashMap::from([(f, 3.0)]);
        let r = assemble_convective_fluxes(
            &inputs,
            &phi,
            &mdot,
            &HashMap::new(),
            ConvectiveScheme::Central,
        )
        .unwrap();
        assert!((r.residual[&c0] + r.residual[&c1]).abs() < 1e-12);
        assert_eq!(r.residual[&c0], 12.0);
    }

    #[test]
    fn diffusive_two_cell_symmetry() {
        let c0 = pid(1);
        let c1 = pid(2);
        let f = pid(9);
        let inputs = FvmInputs::new(
            [FluxStencil {
                face: f,
                left: c0,
                right: Some(c1),
            }],
            vec![
                (
                    c0,
                    CellGeometry {
                        centroid: vec![0.0, 0.0],
                        volume: 1.0,
                    },
                ),
                (
                    c1,
                    CellGeometry {
                        centroid: vec![1.0, 0.0],
                        volume: 1.0,
                    },
                ),
            ],
            vec![(
                f,
                FaceGeometry {
                    face: f,
                    centroid: vec![0.5, 0.0],
                    normal: vec![1.0, 0.0],
                    area: 1.0,
                    neighbors: vec![c0, c1],
                },
            )],
        );
        let phi = HashMap::from([(c0, 1.0), (c1, 3.0)]);
        let grad = HashMap::from([(c0, vec![2.0, 0.0]), (c1, vec![2.0, 0.0])]);
        let r = assemble_diffusive_fluxes(
            &inputs,
            &phi,
            &grad,
            &HashMap::new(),
            DiffusionSettings::default(),
        )
        .unwrap();
        assert!((r.residual[&c0] + r.residual[&c1]).abs() < 1e-12);
    }

    #[test]
    fn convective_bounded_linear_is_bounded_on_skewed_pair() {
        let c0 = pid(1);
        let c1 = pid(2);
        let f = pid(10);
        let inputs = FvmInputs::new(
            [FluxStencil {
                face: f,
                left: c0,
                right: Some(c1),
            }],
            vec![
                (
                    c0,
                    CellGeometry {
                        centroid: vec![0.0, 0.0],
                        volume: 1.0,
                    },
                ),
                (
                    c1,
                    CellGeometry {
                        centroid: vec![1.2, 0.3],
                        volume: 1.0,
                    },
                ),
            ],
            vec![(
                f,
                FaceGeometry {
                    face: f,
                    centroid: vec![0.55, 0.2],
                    normal: vec![0.9, 0.4],
                    area: 0.98,
                    neighbors: vec![c0, c1],
                },
            )],
        );
        let phi = HashMap::from([(c0, 0.0), (c1, 1.0)]);
        let mdot = HashMap::from([(f, 4.0)]);
        let r = assemble_convective_fluxes_with_reconstruction(
            &inputs,
            &phi,
            &mdot,
            &HashMap::new(),
            ConvectiveScheme::BoundedLinear { blend: 1.0 },
            ReconstructionSettings {
                mode: ReconstructionMode::GradientOnly(ReconstructionGradient::GreenGauss),
                limiter: LimiterOption::Family(SlopeLimiterFamily::VanLeer),
            },
        )
        .unwrap();
        let face_value = r.residual[&c0] / mdot[&f];
        assert!((0.0..=1.0).contains(&face_value));
        assert!((r.residual[&c0] + r.residual[&c1]).abs() < 1e-12);
    }

    #[test]
    fn diffusive_non_orthogonal_modes_preserve_pair_conservation() {
        let c0 = pid(1);
        let c1 = pid(2);
        let f = pid(9);
        let inputs = FvmInputs::new(
            [FluxStencil {
                face: f,
                left: c0,
                right: Some(c1),
            }],
            vec![
                (
                    c0,
                    CellGeometry {
                        centroid: vec![0.0, 0.0],
                        volume: 1.0,
                    },
                ),
                (
                    c1,
                    CellGeometry {
                        centroid: vec![1.0, 0.25],
                        volume: 1.0,
                    },
                ),
            ],
            vec![(
                f,
                FaceGeometry {
                    face: f,
                    centroid: vec![0.45, 0.15],
                    normal: vec![0.8, 0.6],
                    area: 1.0,
                    neighbors: vec![c0, c1],
                },
            )],
        );
        let phi = HashMap::from([(c0, 1.0), (c1, 3.0)]);
        let grad = HashMap::from([(c0, vec![2.0, 0.2]), (c1, vec![2.0, 0.2])]);
        for mode in [
            NonOrthogonalCorrectionMode::OrthogonalOnly,
            NonOrthogonalCorrectionMode::Deferred,
            NonOrthogonalCorrectionMode::FullyCorrected,
        ] {
            let r = assemble_diffusive_fluxes(
                &inputs,
                &phi,
                &grad,
                &HashMap::new(),
                DiffusionSettings {
                    diffusivity: 1.0,
                    non_orthogonal_mode: mode,
                },
            )
            .unwrap();
            assert!((r.residual[&c0] + r.residual[&c1]).abs() < 1e-12);
        }
    }

    #[test]
    fn limiter_options_bias_toward_bounded_face_values() {
        let c0 = pid(1);
        let c1 = pid(2);
        let f = pid(44);
        let inputs = FvmInputs::new(
            [FluxStencil {
                face: f,
                left: c0,
                right: Some(c1),
            }],
            vec![],
            vec![],
        );
        let phi = HashMap::from([(c0, -1.0), (c1, 3.0)]);
        let mdot = HashMap::from([(f, 2.0)]);
        let unlimited = convective_face_value(
            phi[&c0],
            phi[&c1],
            mdot[&f],
            ConvectiveScheme::HighResolution {
                blend: 1.0,
                limiter: SlopeLimiterFamily::Superbee,
            },
            LimiterOption::None,
        );
        let bounded = convective_face_value(
            phi[&c0],
            phi[&c1],
            mdot[&f],
            ConvectiveScheme::HighResolution {
                blend: 1.0,
                limiter: SlopeLimiterFamily::Superbee,
            },
            LimiterOption::Family(SlopeLimiterFamily::Minmod),
        );
        assert!(bounded <= unlimited + 1e-12);
        let _ = inputs;
    }

    #[test]
    fn diffusive_non_orthogonality_mode_sensitivity_on_skew_patch() {
        let c0 = pid(1);
        let c1 = pid(2);
        let c2 = pid(3);
        let f01 = pid(101);
        let f12 = pid(102);
        let inputs = FvmInputs::new(
            [
                FluxStencil {
                    face: f01,
                    left: c0,
                    right: Some(c1),
                },
                FluxStencil {
                    face: f12,
                    left: c1,
                    right: Some(c2),
                },
            ],
            vec![
                (
                    c0,
                    CellGeometry {
                        centroid: vec![0.0, 0.0],
                        volume: 1.0,
                    },
                ),
                (
                    c1,
                    CellGeometry {
                        centroid: vec![1.0, 0.35],
                        volume: 1.0,
                    },
                ),
                (
                    c2,
                    CellGeometry {
                        centroid: vec![2.1, 0.05],
                        volume: 1.0,
                    },
                ),
            ],
            vec![
                (
                    f01,
                    FaceGeometry {
                        face: f01,
                        centroid: vec![0.45, 0.21],
                        normal: vec![0.75, 0.62],
                        area: 1.0,
                        neighbors: vec![c0, c1],
                    },
                ),
                (
                    f12,
                    FaceGeometry {
                        face: f12,
                        centroid: vec![1.55, 0.16],
                        normal: vec![0.68, 0.71],
                        area: 1.0,
                        neighbors: vec![c1, c2],
                    },
                ),
            ],
        );
        let phi = HashMap::from([(c0, 0.0), (c1, 1.0), (c2, 2.0)]);
        let grad = HashMap::from([
            (c0, vec![1.0, 0.2]),
            (c1, vec![1.0, -0.15]),
            (c2, vec![1.0, 0.1]),
        ]);
        let ortho = assemble_diffusive_fluxes(
            &inputs,
            &phi,
            &grad,
            &HashMap::new(),
            DiffusionSettings {
                diffusivity: 1.0,
                non_orthogonal_mode: NonOrthogonalCorrectionMode::OrthogonalOnly,
            },
        )
        .unwrap();
        let full = assemble_diffusive_fluxes(
            &inputs,
            &phi,
            &grad,
            &HashMap::new(),
            DiffusionSettings {
                diffusivity: 1.0,
                non_orthogonal_mode: NonOrthogonalCorrectionMode::FullyCorrected,
            },
        )
        .unwrap();
        let sum_ortho: f64 = ortho.residual.values().sum();
        let sum_full: f64 = full.residual.values().sum();
        assert!(sum_ortho.abs() < 1e-12);
        assert!(sum_full.abs() < 1e-12);
        assert_ne!(ortho.residual, full.residual);
    }

    #[test]
    fn assemble_driver_internal_only_mesh() {
        let c0 = pid(1);
        let c1 = pid(2);
        let f = pid(11);
        let inputs = FvmInputs::new(
            [FluxStencil {
                face: f,
                left: c0,
                right: Some(c1),
            }],
            vec![
                (
                    c0,
                    CellGeometry {
                        centroid: vec![0.0, 0.0],
                        volume: 1.0,
                    },
                ),
                (
                    c1,
                    CellGeometry {
                        centroid: vec![1.0, 0.0],
                        volume: 1.0,
                    },
                ),
            ],
            vec![(
                f,
                FaceGeometry {
                    face: f,
                    centroid: vec![0.5, 0.0],
                    normal: vec![1.0, 0.0],
                    area: 1.0,
                    neighbors: vec![c0, c1],
                },
            )],
        );
        let out = assemble_fv_system(
            &inputs,
            &FvmFieldSections {
                cell_scalar: HashMap::from([(c0, 1.0), (c1, 3.0)]),
                cell_gradient: HashMap::from([(c0, vec![2.0, 0.0]), (c1, vec![2.0, 0.0])]),
                face_mass_flux: HashMap::from([(f, 1.0)]),
            },
            &policy(HashMap::new(), HashSet::new()),
            FvmSchemeSettings {
                convective: ConvectiveScheme::Central,
                reconstruction: ReconstructionSettings::default(),
                diffusion: DiffusionSettings::default(),
            },
        )
        .unwrap();
        assert!((out.total.residual[&c0] + out.total.residual[&c1]).abs() < 1e-12);
    }

    #[test]
    fn assemble_driver_mixed_internal_boundary_mesh() {
        let c0 = pid(1);
        let c1 = pid(2);
        let fi = pid(20);
        let fb = pid(21);
        let inputs = FvmInputs::new(
            [
                FluxStencil {
                    face: fi,
                    left: c0,
                    right: Some(c1),
                },
                FluxStencil {
                    face: fb,
                    left: c0,
                    right: None,
                },
            ],
            vec![
                (
                    c0,
                    CellGeometry {
                        centroid: vec![0.0, 0.0],
                        volume: 1.0,
                    },
                ),
                (
                    c1,
                    CellGeometry {
                        centroid: vec![1.0, 0.0],
                        volume: 1.0,
                    },
                ),
            ],
            vec![
                (
                    fi,
                    FaceGeometry {
                        face: fi,
                        centroid: vec![0.5, 0.0],
                        normal: vec![1.0, 0.0],
                        area: 1.0,
                        neighbors: vec![c0, c1],
                    },
                ),
                (
                    fb,
                    FaceGeometry {
                        face: fb,
                        centroid: vec![-0.5, 0.0],
                        normal: vec![-1.0, 0.0],
                        area: 1.0,
                        neighbors: vec![c0],
                    },
                ),
            ],
        );
        let branches = HashMap::from([(fb, FvBoundaryBranch::Inflow)]);
        let mut boundary_policy = policy(branches, HashSet::from([FvBoundaryBranch::Inflow]));
        boundary_policy.convective_branch_hooks.insert(
            FvBoundaryBranch::Inflow,
            BoundaryCondition::Dirichlet { value: 5.0 },
        );
        boundary_policy.diffusive_branch_hooks.insert(
            FvBoundaryBranch::Inflow,
            BoundaryCondition::Neumann { gradient: 0.0 },
        );
        let err = assemble_fv_system(
            &inputs,
            &FvmFieldSections {
                cell_scalar: HashMap::from([(c0, 1.0), (c1, 2.0)]),
                cell_gradient: HashMap::from([(c0, vec![1.0, 0.0]), (c1, vec![1.0, 0.0])]),
                face_mass_flux: HashMap::from([(fi, 1.0), (fb, 1.0)]),
            },
            &boundary_policy,
            FvmSchemeSettings {
                convective: ConvectiveScheme::Upwind,
                reconstruction: ReconstructionSettings::default(),
                diffusion: DiffusionSettings::default(),
            },
        )
        .unwrap();
        assert!(err.total.residual.contains_key(&c0));
    }

    #[test]
    fn assemble_driver_boundary_only_rejects_invalid_label_mapping() {
        let c0 = pid(1);
        let fb = pid(31);
        let inputs = FvmInputs::new(
            [FluxStencil {
                face: fb,
                left: c0,
                right: None,
            }],
            vec![(
                c0,
                CellGeometry {
                    centroid: vec![0.0, 0.0],
                    volume: 1.0,
                },
            )],
            vec![(
                fb,
                FaceGeometry {
                    face: fb,
                    centroid: vec![0.0, 1.0],
                    normal: vec![0.0, 1.0],
                    area: 1.0,
                    neighbors: vec![c0],
                },
            )],
        );
        let bad_map = HashMap::from([(pid(99), FvBoundaryBranch::Open)]);
        let err = assemble_fv_system(
            &inputs,
            &FvmFieldSections {
                cell_scalar: HashMap::from([(c0, 1.0)]),
                cell_gradient: HashMap::from([(c0, vec![0.0, 1.0])]),
                face_mass_flux: HashMap::from([(fb, 1.0)]),
            },
            &policy(bad_map, HashSet::from([FvBoundaryBranch::Open])),
            FvmSchemeSettings {
                convective: ConvectiveScheme::Central,
                reconstruction: ReconstructionSettings::default(),
                diffusion: DiffusionSettings::default(),
            },
        )
        .unwrap_err();
        assert!(matches!(
            err,
            FvmAssemblyError::LabelMappedToUnknownFace { .. }
        ));
    }

    #[test]
    fn preflight_reports_missing_metrics_and_internal_label_mapping() {
        let c0 = pid(1);
        let c1 = pid(2);
        let fi = pid(41);
        let fb = pid(42);

        let inputs_missing_face_metric = FvmInputs::new(
            [FluxStencil {
                face: fb,
                left: c0,
                right: None,
            }],
            vec![(
                c0,
                CellGeometry {
                    centroid: vec![0.0, 0.0],
                    volume: 1.0,
                },
            )],
            vec![],
        );
        let err = preflight_fv_assembly(
            &inputs_missing_face_metric,
            &policy(HashMap::new(), HashSet::new()),
        )
        .unwrap_err();
        assert_eq!(err, FvmAssemblyError::MissingFaceMetric { face: fb });

        let inputs_missing_cell_metric = FvmInputs::new(
            [FluxStencil {
                face: fb,
                left: c0,
                right: None,
            }],
            vec![],
            vec![(
                fb,
                FaceGeometry {
                    face: fb,
                    centroid: vec![0.0, 0.0],
                    normal: vec![1.0, 0.0],
                    area: 1.0,
                    neighbors: vec![c0],
                },
            )],
        );
        let err = preflight_fv_assembly(
            &inputs_missing_cell_metric,
            &policy(HashMap::new(), HashSet::new()),
        )
        .unwrap_err();
        assert_eq!(err, FvmAssemblyError::MissingCellMetric { cell: c0 });

        let inputs_internal_face_label = FvmInputs::new(
            [FluxStencil {
                face: fi,
                left: c0,
                right: Some(c1),
            }],
            vec![
                (
                    c0,
                    CellGeometry {
                        centroid: vec![0.0, 0.0],
                        volume: 1.0,
                    },
                ),
                (
                    c1,
                    CellGeometry {
                        centroid: vec![1.0, 0.0],
                        volume: 1.0,
                    },
                ),
            ],
            vec![(
                fi,
                FaceGeometry {
                    face: fi,
                    centroid: vec![0.5, 0.0],
                    normal: vec![1.0, 0.0],
                    area: 1.0,
                    neighbors: vec![c0, c1],
                },
            )],
        );
        let err = preflight_fv_assembly(
            &inputs_internal_face_label,
            &policy(
                HashMap::from([(fi, FvBoundaryBranch::Open)]),
                HashSet::new(),
            ),
        )
        .unwrap_err();
        assert_eq!(
            err,
            FvmAssemblyError::LabelMappedToInternalFace { face: fi }
        );
    }
}
