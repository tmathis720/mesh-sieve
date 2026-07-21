//! Device-ready finite-volume plans and CPU reference execution.

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};

use crate::discretization::runtime::FiniteVolumeMetadata;
use crate::physics::fvm::{
    BoundaryCondition, ConvectiveScheme, FvBoundaryBranch, FvBoundaryPolicy, FvmInputs,
    FvmSchemeSettings, LimiterOption, NonOrthogonalCorrectionMode, ReconstructionGradient,
    ReconstructionMode, SlopeLimiterFamily, UnsupportedBoundaryBehavior,
};
use crate::topology::coastal::{BOUNDARY_CLASS_LABEL, WET_DRY_MASK_LABEL, WetDryMask};
use crate::topology::labels::LabelSet;
use crate::topology::point::PointId;

use super::backend::CpuBackend;
use super::plan::{checked_u32, upload};
use super::{AcceleratorBackend, AcceleratorError, DeviceBuffer, DeviceValue, PlanEpochs};

/// Stable device ABI for an internal face.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct DeviceInternalFace {
    /// Owner cell dense index.
    pub owner: u32,
    /// Neighbor cell dense index.
    pub neighbor: u32,
    /// Geometry dense index.
    pub geometry: u32,
    /// Reserved flags for model-specific use.
    pub flags: u32,
}

// SAFETY: repr(C), Copy, and every field is a CUDA-representable u32.
#[cfg(feature = "cuda")]
unsafe impl cudarc::driver::DeviceRepr for DeviceInternalFace {}
// SAFETY: the all-zero bit pattern is valid for every u32 field.
#[cfg(feature = "cuda")]
unsafe impl cudarc::driver::ValidAsZeroBits for DeviceInternalFace {}

/// Stable device ABI for a boundary face.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct DeviceBoundaryFace {
    /// Owner cell dense index.
    pub owner: u32,
    /// Geometry dense index.
    pub geometry: u32,
    /// Encoded [`FvBoundaryBranch`], or `u32::MAX` if unclassified.
    pub kind: u32,
    /// Reserved flags for model-specific use.
    pub flags: u32,
}

// SAFETY: repr(C), Copy, and every field is a CUDA-representable u32.
#[cfg(feature = "cuda")]
unsafe impl cudarc::driver::DeviceRepr for DeviceBoundaryFace {}
// SAFETY: the all-zero bit pattern is valid for every u32 field.
#[cfg(feature = "cuda")]
unsafe impl cudarc::driver::ValidAsZeroBits for DeviceBoundaryFace {}

/// Built-in device physics parameters. Solver-specific state stays outside
/// mesh-sieve and may be supplied through custom PTX entry points.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct DevicePhysicsParams {
    /// Fluid density.
    pub density: f64,
    /// Kinematic/dynamic viscosity selected by the caller's model.
    pub viscosity: f64,
    /// Gravitational acceleration.
    pub gravity: f64,
    /// Scalar diffusivity.
    pub diffusivity: f64,
}

// SAFETY: repr(C), Copy, and every field is a CUDA-representable f64.
#[cfg(feature = "cuda")]
unsafe impl cudarc::driver::DeviceRepr for DevicePhysicsParams {}
// SAFETY: the all-zero bit pattern is valid for every f64 field.
#[cfg(feature = "cuda")]
unsafe impl cudarc::driver::ValidAsZeroBits for DevicePhysicsParams {}

impl Default for DevicePhysicsParams {
    fn default() -> Self {
        Self {
            density: 1.0,
            viscosity: 0.0,
            gravity: 9.80665,
            diffusivity: 0.0,
        }
    }
}

/// Scalar convective interpolation available in the first device executor.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalarFluxScheme {
    /// Select the upwind cell value using the sign of mass flux.
    Upwind = 0,
    /// Average owner and neighbor/boundary values.
    Central = 1,
}

/// Floating-point types supported by the packed FVM executor.
pub trait FvmScalar: DeviceValue + Copy + Default + std::fmt::Debug + PartialEq {
    /// Convert from the reference executor's accumulation type.
    fn from_f64(value: f64) -> Self;
    /// Convert to the reference executor's accumulation type.
    fn to_f64(self) -> f64;
}

impl FvmScalar for f32 {
    fn from_f64(value: f64) -> Self {
        value as f32
    }
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl FvmScalar for f64 {
    fn from_f64(value: f64) -> Self {
        value
    }
    fn to_f64(self) -> f64 {
        self
    }
}

/// Structure-of-arrays device plan for conservative face/cell execution.
pub struct DeviceFvmPlan<B: AcceleratorBackend> {
    /// Epochs captured when this plan was compiled.
    pub epochs: PlanEpochs,
    /// Stable cell IDs in dense field order (host-only diagnostics/output map).
    pub cell_ids: Vec<PointId>,
    /// Stable face IDs in flux-array order: internal, then boundary.
    pub face_ids: Vec<PointId>,
    /// Internal face owner cell indices.
    pub internal_owner: B::Buffer<u32>,
    /// Internal face neighbor cell indices.
    pub internal_neighbor: B::Buffer<u32>,
    /// Internal face geometry indices.
    pub internal_geometry: B::Buffer<u32>,
    /// Boundary face owner cell indices.
    pub boundary_owner: B::Buffer<u32>,
    /// Boundary face geometry indices.
    pub boundary_geometry: B::Buffer<u32>,
    /// Encoded boundary branch.
    pub boundary_kind: B::Buffer<u32>,
    /// Face area in input geometry order.
    pub face_area: B::Buffer<f64>,
    /// X component of the owner-oriented face normal.
    pub face_normal_x: B::Buffer<f64>,
    /// Y component of the owner-oriented face normal.
    pub face_normal_y: B::Buffer<f64>,
    /// Z component of the owner-oriented face normal.
    pub face_normal_z: B::Buffer<f64>,
    /// X component of face center.
    pub face_center_x: B::Buffer<f64>,
    /// Y component of face center.
    pub face_center_y: B::Buffer<f64>,
    /// Z component of face center.
    pub face_center_z: B::Buffer<f64>,
    /// Geometry index for each face in combined flux order.
    pub face_geometry_indices: B::Buffer<u32>,
    /// Cell volume in dense field order.
    pub cell_volume: B::Buffer<f64>,
    /// X component of cell center.
    pub cell_center_x: B::Buffer<f64>,
    /// Y component of cell center.
    pub cell_center_y: B::Buffer<f64>,
    /// Z component of cell center.
    pub cell_center_z: B::Buffer<f64>,
    /// Cell-to-face CSR offsets.
    pub cell_face_offsets: B::Buffer<u32>,
    /// Indices into the combined face flux array.
    pub cell_face_indices: B::Buffer<u32>,
    /// `+1` for owner and `-1` for neighbor contribution.
    pub cell_face_signs: B::Buffer<i8>,
    /// One byte per face; zero suppresses its flux.
    pub face_active: B::Buffer<u8>,
    /// One byte per cell; zero marks a dry/inactive cell.
    pub cell_active: B::Buffer<u8>,
    /// Number of spatial dimensions represented by the geometry.
    pub dimension: usize,
    /// Number of internal faces.
    pub internal_face_count: usize,
    /// Number of boundary faces.
    pub boundary_face_count: usize,
}

impl<B: AcceleratorBackend> DeviceFvmPlan<B> {
    /// Compile FVM stencils, geometry, and optional wet/dry/boundary labels.
    pub fn compile(
        backend: &B,
        inputs: &FvmInputs,
        labels: Option<&LabelSet>,
        epochs: PlanEpochs,
    ) -> Result<Self, AcceleratorError> {
        checked_u32(inputs.cell_geometry.len(), "cell count")?;
        checked_u32(inputs.face_geometry.len(), "face geometry count")?;
        let total_faces = inputs
            .loops
            .internal
            .len()
            .checked_add(inputs.loops.boundary.len())
            .ok_or(AcceleratorError::IndexOverflow {
                what: "face count",
                value: usize::MAX,
            })?;
        checked_u32(total_faces, "face count")?;

        let dimension = inputs.cell_geometry.first().map_or_else(
            || {
                inputs
                    .face_geometry
                    .first()
                    .map_or(0, |(_, g)| g.normal.len())
            },
            |(_, g)| g.centroid.len(),
        );
        if dimension > 3 {
            return Err(AcceleratorError::InvalidPlan(format!(
                "CUDA FVM supports at most 3 dimensions, found {dimension}"
            )));
        }

        let mut cell_index = HashMap::with_capacity(inputs.cell_geometry.len());
        let mut cell_ids = Vec::with_capacity(inputs.cell_geometry.len());
        let mut cell_volume = Vec::with_capacity(inputs.cell_geometry.len());
        let mut cell_center_x = Vec::with_capacity(inputs.cell_geometry.len());
        let mut cell_center_y = Vec::with_capacity(inputs.cell_geometry.len());
        let mut cell_center_z = Vec::with_capacity(inputs.cell_geometry.len());
        for (idx, (id, geometry)) in inputs.cell_geometry.iter().enumerate() {
            if geometry.centroid.len() != dimension {
                return Err(AcceleratorError::InvalidPlan(format!(
                    "cell {id} has dimension {}, expected {dimension}",
                    geometry.centroid.len()
                )));
            }
            if !geometry.volume.is_finite() || geometry.volume <= 0.0 {
                return Err(AcceleratorError::InvalidPlan(format!(
                    "cell {id} has non-positive or non-finite volume {}",
                    geometry.volume
                )));
            }
            if cell_index
                .insert(*id, checked_u32(idx, "cell index")?)
                .is_some()
            {
                return Err(AcceleratorError::InvalidPlan(format!(
                    "duplicate cell geometry for {id}"
                )));
            }
            cell_ids.push(*id);
            cell_volume.push(geometry.volume);
            cell_center_x.push(component(&geometry.centroid, 0));
            cell_center_y.push(component(&geometry.centroid, 1));
            cell_center_z.push(component(&geometry.centroid, 2));
        }

        let mut face_index = HashMap::with_capacity(inputs.face_geometry.len());
        let mut face_area = Vec::with_capacity(inputs.face_geometry.len());
        let mut face_normal_x = Vec::with_capacity(inputs.face_geometry.len());
        let mut face_normal_y = Vec::with_capacity(inputs.face_geometry.len());
        let mut face_normal_z = Vec::with_capacity(inputs.face_geometry.len());
        let mut face_center_x = Vec::with_capacity(inputs.face_geometry.len());
        let mut face_center_y = Vec::with_capacity(inputs.face_geometry.len());
        let mut face_center_z = Vec::with_capacity(inputs.face_geometry.len());
        for (idx, (id, geometry)) in inputs.face_geometry.iter().enumerate() {
            if geometry.normal.len() != dimension || geometry.centroid.len() != dimension {
                return Err(AcceleratorError::InvalidPlan(format!(
                    "face {id} has normal/centroid dimensions {}/{}, expected {dimension}",
                    geometry.normal.len(),
                    geometry.centroid.len()
                )));
            }
            if geometry.face != *id {
                return Err(AcceleratorError::InvalidPlan(format!(
                    "face geometry key {id} disagrees with embedded face {}",
                    geometry.face
                )));
            }
            if !geometry.area.is_finite() || geometry.area < 0.0 {
                return Err(AcceleratorError::InvalidPlan(format!(
                    "face {id} has negative or non-finite area {}",
                    geometry.area
                )));
            }
            if face_index
                .insert(*id, checked_u32(idx, "face geometry index")?)
                .is_some()
            {
                return Err(AcceleratorError::InvalidPlan(format!(
                    "duplicate face geometry for {id}"
                )));
            }
            face_area.push(geometry.area);
            face_normal_x.push(component(&geometry.normal, 0));
            face_normal_y.push(component(&geometry.normal, 1));
            face_normal_z.push(component(&geometry.normal, 2));
            face_center_x.push(component(&geometry.centroid, 0));
            face_center_y.push(component(&geometry.centroid, 1));
            face_center_z.push(component(&geometry.centroid, 2));
        }

        let mut internal_owner = Vec::with_capacity(inputs.loops.internal.len());
        let mut internal_neighbor = Vec::with_capacity(inputs.loops.internal.len());
        let mut internal_geometry = Vec::with_capacity(inputs.loops.internal.len());
        let mut boundary_owner = Vec::with_capacity(inputs.loops.boundary.len());
        let mut boundary_geometry = Vec::with_capacity(inputs.loops.boundary.len());
        let mut boundary_kind = Vec::with_capacity(inputs.loops.boundary.len());
        let mut face_ids = Vec::with_capacity(total_faces);
        let mut face_geometry_indices = Vec::with_capacity(total_faces);
        let mut incidence: Vec<Vec<(u32, i8)>> = vec![Vec::new(); cell_ids.len()];
        let mut face_active = Vec::with_capacity(total_faces);
        let cell_active: Vec<u8> = cell_ids
            .iter()
            .map(|id| {
                u8::from(
                    labels.and_then(|set| set.get_label(*id, WET_DRY_MASK_LABEL))
                        != Some(WetDryMask::Dry.code()),
                )
            })
            .collect();

        for (flux_idx, stencil) in inputs.loops.internal.iter().enumerate() {
            let owner = lookup(&cell_index, stencil.left, "owner cell")?;
            let neighbor_id = stencil.right.ok_or_else(|| {
                AcceleratorError::InvalidPlan(format!(
                    "internal face {} has no neighbor",
                    stencil.face
                ))
            })?;
            let neighbor = lookup(&cell_index, neighbor_id, "neighbor cell")?;
            let geometry = lookup(&face_index, stencil.face, "face geometry")?;
            internal_owner.push(owner);
            internal_neighbor.push(neighbor);
            internal_geometry.push(geometry);
            face_ids.push(stencil.face);
            face_geometry_indices.push(geometry);
            let fi = checked_u32(flux_idx, "flux index")?;
            incidence[owner as usize].push((fi, 1));
            incidence[neighbor as usize].push((fi, -1));
            face_active.push(u8::from(
                cell_active[owner as usize] != 0 && cell_active[neighbor as usize] != 0,
            ));
        }
        let internal_face_count = internal_owner.len();
        for (boundary_idx, stencil) in inputs.loops.boundary.iter().enumerate() {
            let owner = lookup(&cell_index, stencil.left, "boundary owner cell")?;
            let geometry = lookup(&face_index, stencil.face, "boundary face geometry")?;
            boundary_owner.push(owner);
            boundary_geometry.push(geometry);
            boundary_kind.push(
                labels
                    .and_then(|set| {
                        crate::physics::fvm::boundary_branch_for_face(set, stencil.face)
                    })
                    .map_or(u32::MAX, encode_boundary_kind),
            );
            face_ids.push(stencil.face);
            face_geometry_indices.push(geometry);
            let flux_idx = checked_u32(internal_face_count + boundary_idx, "flux index")?;
            incidence[owner as usize].push((flux_idx, 1));
            face_active.push(cell_active[owner as usize]);
        }

        let mut cell_face_offsets = Vec::with_capacity(cell_ids.len() + 1);
        let mut cell_face_indices = Vec::new();
        let mut cell_face_signs = Vec::new();
        cell_face_offsets.push(0);
        for entries in &incidence {
            for &(face, sign) in entries {
                cell_face_indices.push(face);
                cell_face_signs.push(sign);
            }
            cell_face_offsets.push(checked_u32(cell_face_indices.len(), "cell-face offset")?);
        }

        Ok(Self {
            epochs,
            cell_ids,
            face_ids,
            internal_owner: upload(backend, &internal_owner)?,
            internal_neighbor: upload(backend, &internal_neighbor)?,
            internal_geometry: upload(backend, &internal_geometry)?,
            boundary_owner: upload(backend, &boundary_owner)?,
            boundary_geometry: upload(backend, &boundary_geometry)?,
            boundary_kind: upload(backend, &boundary_kind)?,
            face_area: upload(backend, &face_area)?,
            face_normal_x: upload(backend, &face_normal_x)?,
            face_normal_y: upload(backend, &face_normal_y)?,
            face_normal_z: upload(backend, &face_normal_z)?,
            face_center_x: upload(backend, &face_center_x)?,
            face_center_y: upload(backend, &face_center_y)?,
            face_center_z: upload(backend, &face_center_z)?,
            face_geometry_indices: upload(backend, &face_geometry_indices)?,
            cell_volume: upload(backend, &cell_volume)?,
            cell_center_x: upload(backend, &cell_center_x)?,
            cell_center_y: upload(backend, &cell_center_y)?,
            cell_center_z: upload(backend, &cell_center_z)?,
            cell_face_offsets: upload(backend, &cell_face_offsets)?,
            cell_face_indices: upload(backend, &cell_face_indices)?,
            cell_face_signs: upload(backend, &cell_face_signs)?,
            face_active: upload(backend, &face_active)?,
            cell_active: upload(backend, &cell_active)?,
            dimension,
            internal_face_count,
            boundary_face_count: boundary_owner.len(),
        })
    }

    /// Reject execution with stale topology, atlas, or geometry epochs.
    pub fn validate(&self, current: PlanEpochs) -> Result<(), AcceleratorError> {
        self.epochs.validate(current)
    }

    /// Total number of faces in the combined flux array.
    pub fn face_count(&self) -> usize {
        self.internal_face_count + self.boundary_face_count
    }
}

/// Persistent scalar fields used by the two-kernel conservative FVM path.
pub struct DeviceFvmState<T: FvmScalar, B: AcceleratorBackend> {
    /// Number of component-major fields in each cell/face workspace.
    pub components: usize,
    /// Cell-centered scalar values.
    pub cell_values: B::Buffer<T>,
    /// Mass flux values in the plan's combined face order.
    pub face_mass_flux: B::Buffer<T>,
    /// Boundary exterior values in boundary-face order.
    pub boundary_values: B::Buffer<T>,
    /// Oriented flux values in combined face order.
    pub face_flux: B::Buffer<T>,
    /// Explicit caller-provided face flux additions.
    pub face_source: B::Buffer<T>,
    /// Deferred non-orthogonal source contribution per face/component.
    pub face_deferred_source: B::Buffer<T>,
    /// Interpolated face values used by Green--Gauss gradients.
    pub face_values: B::Buffer<T>,
    /// Gathered cell residuals.
    pub residual: B::Buffer<T>,
    /// Explicit caller-provided cell source additions.
    pub cell_source: B::Buffer<T>,
    /// X component of the cell gradient.
    pub gradient_x: B::Buffer<T>,
    /// Y component of the cell gradient.
    pub gradient_y: B::Buffer<T>,
    /// Z component of the cell gradient.
    pub gradient_z: B::Buffer<T>,
}

impl<T: FvmScalar, B: AcceleratorBackend> DeviceFvmState<T, B> {
    /// Upload persistent field data and allocate flux/residual workspaces.
    pub fn upload(
        backend: &B,
        plan: &DeviceFvmPlan<B>,
        cell_values: &[T],
        face_mass_flux: &[T],
        boundary_values: &[T],
    ) -> Result<Self, AcceleratorError> {
        Self::upload_components(
            backend,
            plan,
            1,
            cell_values,
            face_mass_flux,
            boundary_values,
        )
    }

    /// Upload component-major state and allocate persistent workspaces.
    pub fn upload_components(
        backend: &B,
        plan: &DeviceFvmPlan<B>,
        components: usize,
        cell_values: &[T],
        face_mass_flux: &[T],
        boundary_values: &[T],
    ) -> Result<Self, AcceleratorError> {
        if components == 0 {
            return Err(AcceleratorError::InvalidPlan(
                "FVM state requires at least one component".into(),
            ));
        }
        let cell_values_len = checked_product(plan.cell_ids.len(), components, "cell state")?;
        let face_values_len = checked_product(plan.face_count(), components, "face state")?;
        let boundary_values_len =
            checked_product(plan.boundary_face_count, components, "boundary state")?;
        check_len(cell_values_len, cell_values.len())?;
        check_len(plan.face_count(), face_mass_flux.len())?;
        check_len(boundary_values_len, boundary_values.len())?;
        Ok(Self {
            components,
            cell_values: upload(backend, cell_values)?,
            face_mass_flux: upload(backend, face_mass_flux)?,
            boundary_values: upload(backend, boundary_values)?,
            face_flux: allocate(backend, face_values_len)?,
            face_source: allocate(backend, face_values_len)?,
            face_deferred_source: allocate(backend, face_values_len)?,
            face_values: allocate(backend, face_values_len)?,
            residual: allocate(backend, cell_values_len)?,
            cell_source: allocate(backend, cell_values_len)?,
            gradient_x: allocate(backend, cell_values_len)?,
            gradient_y: allocate(backend, cell_values_len)?,
            gradient_z: allocate(backend, cell_values_len)?,
        })
    }

    /// Refresh explicit component-major face and cell source arrays.
    pub fn upload_sources(
        &mut self,
        backend: &B,
        face_source: &[T],
        cell_source: &[T],
    ) -> Result<(), AcceleratorError> {
        check_len(self.face_source.len(), face_source.len())?;
        check_len(self.cell_source.len(), cell_source.len())?;
        backend
            .upload_into(face_source, &mut self.face_source)
            .map_err(|e| AcceleratorError::DeviceTransferFailed(e.to_string()))?;
        backend
            .upload_into(cell_source, &mut self.cell_source)
            .map_err(|e| AcceleratorError::DeviceTransferFailed(e.to_string()))
    }

    /// Download the gathered residual vector.
    pub fn download_residual(&self, backend: &B) -> Result<Vec<T>, AcceleratorError> {
        let mut host = vec![T::zeroed(); self.residual.len()];
        backend
            .download(&self.residual, &mut host)
            .map_err(|e| AcceleratorError::DeviceTransferFailed(e.to_string()))?;
        Ok(host)
    }
}

impl DeviceFvmPlan<CpuBackend> {
    /// Reference Green--Gauss gradient using the plan's deterministic CSR.
    pub fn compute_gradients<T: FvmScalar>(
        &self,
        state: &mut DeviceFvmState<T, CpuBackend>,
    ) -> Result<(), AcceleratorError> {
        validate_state(self, state)?;
        for face in 0..self.internal_face_count {
            let owner = self.internal_owner.0[face] as usize;
            let neighbor = self.internal_neighbor.0[face] as usize;
            state.face_values.0[face] = T::from_f64(
                0.5 * (state.cell_values.0[owner].to_f64()
                    + state.cell_values.0[neighbor].to_f64()),
            );
        }
        for boundary in 0..self.boundary_face_count {
            let face = self.internal_face_count + boundary;
            state.face_values.0[face] = state.boundary_values.0[boundary];
        }
        for cell in 0..self.cell_ids.len() {
            if self.cell_active.0[cell] == 0 {
                state.gradient_x.0[cell] = T::zeroed();
                state.gradient_y.0[cell] = T::zeroed();
                state.gradient_z.0[cell] = T::zeroed();
                continue;
            }
            let begin = self.cell_face_offsets.0[cell] as usize;
            let end = self.cell_face_offsets.0[cell + 1] as usize;
            let mut gradient = [0.0; 3];
            for incidence in begin..end {
                let face = self.cell_face_indices.0[incidence] as usize;
                if self.face_active.0[face] == 0 {
                    continue;
                }
                let geometry = self.face_geometry_indices.0[face] as usize;
                let scale =
                    self.cell_face_signs.0[incidence] as f64 * state.face_values.0[face].to_f64();
                gradient[0] += scale * self.face_normal_x.0[geometry];
                gradient[1] += scale * self.face_normal_y.0[geometry];
                gradient[2] += scale * self.face_normal_z.0[geometry];
            }
            let volume = self.cell_volume.0[cell];
            if volume == 0.0 {
                return Err(AcceleratorError::InvalidPlan(format!(
                    "cell {} has zero volume",
                    self.cell_ids[cell]
                )));
            }
            state.gradient_x.0[cell] = T::from_f64(gradient[0] / volume);
            state.gradient_y.0[cell] = T::from_f64(gradient[1] / volume);
            state.gradient_z.0[cell] = T::from_f64(gradient[2] / volume);
        }
        Ok(())
    }

    /// Reference orthogonal internal/boundary diffusive face fluxes.
    pub fn compute_diffusive_face_fluxes<T: FvmScalar>(
        &self,
        state: &mut DeviceFvmState<T, CpuBackend>,
        diffusivity: f64,
    ) -> Result<(), AcceleratorError> {
        validate_state(self, state)?;
        for face in 0..self.internal_face_count {
            if self.face_active.0[face] == 0 {
                state.face_flux.0[face] = T::zeroed();
                continue;
            }
            let owner = self.internal_owner.0[face] as usize;
            let neighbor = self.internal_neighbor.0[face] as usize;
            let geometry = self.internal_geometry.0[face] as usize;
            let d = [
                self.cell_center_x.0[neighbor] - self.cell_center_x.0[owner],
                self.cell_center_y.0[neighbor] - self.cell_center_y.0[owner],
                self.cell_center_z.0[neighbor] - self.cell_center_z.0[owner],
            ];
            let normal = [
                self.face_normal_x.0[geometry],
                self.face_normal_y.0[geometry],
                self.face_normal_z.0[geometry],
            ];
            let d2 = dot3(d, d);
            let flux = if d2 == 0.0 {
                0.0
            } else {
                -diffusivity
                    * (state.cell_values.0[neighbor].to_f64() - state.cell_values.0[owner].to_f64())
                    * dot3(normal, d)
                    / d2
            };
            state.face_flux.0[face] = T::from_f64(flux);
        }
        for boundary in 0..self.boundary_face_count {
            let face = self.internal_face_count + boundary;
            if self.face_active.0[face] == 0 {
                state.face_flux.0[face] = T::zeroed();
                continue;
            }
            let owner = self.boundary_owner.0[boundary] as usize;
            let geometry = self.boundary_geometry.0[boundary] as usize;
            let d = [
                self.face_center_x.0[geometry] - self.cell_center_x.0[owner],
                self.face_center_y.0[geometry] - self.cell_center_y.0[owner],
                self.face_center_z.0[geometry] - self.cell_center_z.0[owner],
            ];
            let normal = [
                self.face_normal_x.0[geometry],
                self.face_normal_y.0[geometry],
                self.face_normal_z.0[geometry],
            ];
            let d2 = dot3(d, d);
            let flux = if d2 == 0.0 {
                0.0
            } else {
                -diffusivity
                    * (state.boundary_values.0[boundary].to_f64()
                        - state.cell_values.0[owner].to_f64())
                    * dot3(normal, d)
                    / d2
            };
            state.face_flux.0[face] = T::from_f64(flux);
        }
        Ok(())
    }

    /// Reference implementation of internal/boundary face flux kernels.
    pub fn compute_face_fluxes<T: FvmScalar>(
        &self,
        state: &mut DeviceFvmState<T, CpuBackend>,
        scheme: ScalarFluxScheme,
    ) -> Result<(), AcceleratorError> {
        validate_state(self, state)?;
        for face in 0..self.internal_face_count {
            if self.face_active.0[face] == 0 {
                state.face_flux.0[face] = T::zeroed();
                continue;
            }
            let owner = self.internal_owner.0[face] as usize;
            let neighbor = self.internal_neighbor.0[face] as usize;
            let mass = state.face_mass_flux.0[face].to_f64();
            let left = state.cell_values.0[owner].to_f64();
            let right = state.cell_values.0[neighbor].to_f64();
            let face_value = match scheme {
                ScalarFluxScheme::Upwind => {
                    if mass >= 0.0 {
                        left
                    } else {
                        right
                    }
                }
                ScalarFluxScheme::Central => 0.5 * (left + right),
            };
            state.face_flux.0[face] = T::from_f64(mass * face_value);
        }
        for boundary in 0..self.boundary_face_count {
            let face = self.internal_face_count + boundary;
            if self.face_active.0[face] == 0 {
                state.face_flux.0[face] = T::zeroed();
                continue;
            }
            let owner = self.boundary_owner.0[boundary] as usize;
            let mass = state.face_mass_flux.0[face].to_f64();
            let inside = state.cell_values.0[owner].to_f64();
            let outside = state.boundary_values.0[boundary].to_f64();
            let face_value = match scheme {
                ScalarFluxScheme::Upwind => {
                    if mass >= 0.0 {
                        inside
                    } else {
                        outside
                    }
                }
                ScalarFluxScheme::Central => 0.5 * (inside + outside),
            };
            state.face_flux.0[face] = T::from_f64(mass * face_value);
        }
        Ok(())
    }

    /// Reference implementation of deterministic cell residual gathering.
    pub fn assemble_cell_residuals<T: FvmScalar>(
        &self,
        state: &mut DeviceFvmState<T, CpuBackend>,
    ) -> Result<(), AcceleratorError> {
        validate_state(self, state)?;
        for cell in 0..self.cell_ids.len() {
            if self.cell_active.0[cell] == 0 {
                state.residual.0[cell] = T::zeroed();
                continue;
            }
            let begin = self.cell_face_offsets.0[cell] as usize;
            let end = self.cell_face_offsets.0[cell + 1] as usize;
            let mut sum = 0.0;
            for incidence in begin..end {
                let face = self.cell_face_indices.0[incidence] as usize;
                let sign = self.cell_face_signs.0[incidence] as f64;
                sum += sign * state.face_flux.0[face].to_f64();
            }
            state.residual.0[cell] = T::from_f64(sum);
        }
        Ok(())
    }
}

fn validate_state<T: FvmScalar>(
    plan: &DeviceFvmPlan<CpuBackend>,
    state: &DeviceFvmState<T, CpuBackend>,
) -> Result<(), AcceleratorError> {
    if state.components != 1 {
        return Err(AcceleratorError::InvalidPlan(
            "scalar DeviceFvmPlan methods require a one-component state; use DeviceFvmOperator"
                .into(),
        ));
    }
    check_len(plan.cell_ids.len(), state.cell_values.0.len())?;
    check_len(plan.cell_ids.len(), state.residual.0.len())?;
    check_len(plan.face_count(), state.face_mass_flux.0.len())?;
    check_len(plan.face_count(), state.face_flux.0.len())?;
    check_len(plan.face_count(), state.face_values.0.len())?;
    check_len(plan.cell_ids.len(), state.gradient_x.0.len())?;
    check_len(plan.cell_ids.len(), state.gradient_y.0.len())?;
    check_len(plan.cell_ids.len(), state.gradient_z.0.len())?;
    check_len(plan.boundary_face_count, state.boundary_values.0.len())
}

/// Packed standard boundary conditions for convective and diffusive kernels.
pub struct DeviceFvmBoundaryConditions<B: AcceleratorBackend> {
    /// Boundary-condition tag: 0 Dirichlet, 1 Neumann, 2 Robin.
    pub convective_kind: B::Buffer<u8>,
    /// Convective Robin/Dirichlet alpha coefficient.
    pub convective_alpha: B::Buffer<f64>,
    /// Convective beta coefficient.
    pub convective_beta: B::Buffer<f64>,
    /// Convective prescribed value/gradient/gamma coefficient.
    pub convective_gamma: B::Buffer<f64>,
    /// Boundary-condition tag: 0 Dirichlet, 1 Neumann, 2 Robin.
    pub diffusive_kind: B::Buffer<u8>,
    /// Diffusive alpha coefficient.
    pub diffusive_alpha: B::Buffer<f64>,
    /// Diffusive beta coefficient.
    pub diffusive_beta: B::Buffer<f64>,
    /// Diffusive prescribed value/gradient/gamma coefficient.
    pub diffusive_gamma: B::Buffer<f64>,
}

/// Precomputed least-squares coefficients aligned with cell-face CSR entries.
pub struct DeviceLeastSquaresPlan<B: AcceleratorBackend> {
    /// Neighbor cell, or `u32::MAX` for a boundary sample.
    pub neighbor: B::Buffer<u32>,
    /// X coefficient multiplying the neighbor/boundary value difference.
    pub weight_x: B::Buffer<f64>,
    /// Y coefficient multiplying the neighbor/boundary value difference.
    pub weight_y: B::Buffer<f64>,
    /// Z coefficient multiplying the neighbor/boundary value difference.
    pub weight_z: B::Buffer<f64>,
    /// One for cells whose least-squares normal matrix was singular.
    pub fallback: B::Buffer<u8>,
    fallback_host: Vec<u8>,
}

/// Complete, persistent finite-volume numerical operator.
pub struct DeviceFvmOperator<B: AcceleratorBackend> {
    /// Connectivity, geometry, masks, and deterministic gather structure.
    pub plan: DeviceFvmPlan<B>,
    /// Number of transported components and reconstruction order.
    pub metadata: FiniteVolumeMetadata,
    /// Numerical schemes captured by the operator.
    pub schemes: FvmSchemeSettings,
    /// Packed standard boundary conditions.
    pub boundary: DeviceFvmBoundaryConditions<B>,
    /// Precomputed least-squares reconstruction data.
    pub least_squares: DeviceLeastSquaresPlan<B>,
}

impl<B: AcceleratorBackend> DeviceFvmOperator<B> {
    /// Compile a complete numerical operator without uploading solution state.
    pub fn compile(
        backend: &B,
        inputs: &FvmInputs,
        metadata: FiniteVolumeMetadata,
        labels: Option<&LabelSet>,
        boundary_policy: &FvBoundaryPolicy,
        schemes: FvmSchemeSettings,
        epochs: PlanEpochs,
    ) -> Result<Self, AcceleratorError> {
        if metadata.components == 0 {
            return Err(AcceleratorError::InvalidPlan(
                "finite-volume metadata requires at least one component".into(),
            ));
        }
        if metadata.reconstruction_order == 0 {
            return Err(AcceleratorError::InvalidPlan(
                "finite-volume reconstruction order must be at least one".into(),
            ));
        }
        if !schemes.diffusion.diffusivity.is_finite() {
            return Err(AcceleratorError::InvalidPlan(
                "finite-volume diffusivity must be finite".into(),
            ));
        }
        let blend = match schemes.convective {
            ConvectiveScheme::BoundedLinear { blend }
            | ConvectiveScheme::BlendUpwindCentral { blend }
            | ConvectiveScheme::HighResolution { blend, .. } => Some(blend),
            ConvectiveScheme::Upwind | ConvectiveScheme::Central => None,
        };
        if blend.is_some_and(|value| !value.is_finite()) {
            return Err(AcceleratorError::InvalidPlan(
                "finite-volume convection blend must be finite".into(),
            ));
        }
        if let Some(labels) = labels {
            for stencil in &inputs.loops.boundary {
                if labels
                    .get_label(stencil.face, BOUNDARY_CLASS_LABEL)
                    .is_none()
                {
                    continue;
                }
                let labeled =
                    crate::physics::fvm::boundary_branch_for_face_checked(labels, stencil.face)
                        .map_err(|error| {
                            AcceleratorError::InvalidPlan(format!(
                                "invalid boundary labels for face {}: {error:?}",
                                stencil.face
                            ))
                        })?;
                if boundary_policy.boundary_face_branches.get(&stencil.face) != Some(&labeled) {
                    return Err(AcceleratorError::InvalidPlan(format!(
                        "boundary face {} is labeled {labeled:?} but its policy branch is {:?}",
                        stencil.face,
                        boundary_policy.boundary_face_branches.get(&stencil.face)
                    )));
                }
            }
        }
        let plan = DeviceFvmPlan::compile(backend, inputs, labels, epochs)?;
        let boundary_host = pack_boundary_conditions(
            inputs.loops.boundary.iter().map(|face| face.face),
            boundary_policy,
        )?;
        let least_squares_host = build_least_squares(inputs, &plan.cell_ids)?;
        if matches!(
            schemes.reconstruction.mode,
            ReconstructionMode::GradientOnly(ReconstructionGradient::LeastSquares)
        ) && least_squares_host.fallback.iter().any(|&value| value != 0)
        {
            return Err(AcceleratorError::InvalidPlan(
                "least-squares reconstruction is singular for at least one cell; select the Green--Gauss fallback mode"
                    .into(),
            ));
        }
        Ok(Self {
            boundary: upload_boundary(backend, boundary_host)?,
            least_squares: DeviceLeastSquaresPlan {
                neighbor: upload(backend, &least_squares_host.neighbor)?,
                weight_x: upload(backend, &least_squares_host.weight_x)?,
                weight_y: upload(backend, &least_squares_host.weight_y)?,
                weight_z: upload(backend, &least_squares_host.weight_z)?,
                fallback: upload(backend, &least_squares_host.fallback)?,
                fallback_host: least_squares_host.fallback,
            },
            plan,
            metadata,
            schemes,
        })
    }

    /// Refresh boundary coefficients while preserving topology and geometry.
    pub fn refresh_boundary_conditions(
        &mut self,
        backend: &B,
        boundary_policy: &FvBoundaryPolicy,
    ) -> Result<(), AcceleratorError> {
        let host = pack_boundary_conditions(
            self.plan.face_ids[self.plan.internal_face_count..]
                .iter()
                .copied(),
            boundary_policy,
        )?;
        upload_boundary_into(backend, &mut self.boundary, &host)
    }

    /// Validate epochs and component-major state lengths.
    pub fn validate_state<T: FvmScalar>(
        &self,
        state: &DeviceFvmState<T, B>,
        current_epochs: PlanEpochs,
    ) -> Result<(), AcceleratorError> {
        self.plan.validate(current_epochs)?;
        if state.components != self.metadata.components {
            return Err(AcceleratorError::InvalidPlan(format!(
                "operator has {} components but state has {}",
                self.metadata.components, state.components
            )));
        }
        let cells = checked_product(self.plan.cell_ids.len(), state.components, "cell state")?;
        let faces = checked_product(self.plan.face_count(), state.components, "face state")?;
        let boundaries = checked_product(
            self.plan.boundary_face_count,
            state.components,
            "boundary state",
        )?;
        check_len(cells, state.cell_values.len())?;
        check_len(cells, state.residual.len())?;
        check_len(cells, state.cell_source.len())?;
        check_len(cells, state.gradient_x.len())?;
        check_len(cells, state.gradient_y.len())?;
        check_len(cells, state.gradient_z.len())?;
        check_len(faces, state.face_flux.len())?;
        check_len(faces, state.face_source.len())?;
        check_len(faces, state.face_deferred_source.len())?;
        check_len(faces, state.face_values.len())?;
        check_len(self.plan.face_count(), state.face_mass_flux.len())?;
        check_len(boundaries, state.boundary_values.len())
    }
}

#[derive(Default)]
struct BoundaryHost {
    convective_kind: Vec<u8>,
    convective_alpha: Vec<f64>,
    convective_beta: Vec<f64>,
    convective_gamma: Vec<f64>,
    diffusive_kind: Vec<u8>,
    diffusive_alpha: Vec<f64>,
    diffusive_beta: Vec<f64>,
    diffusive_gamma: Vec<f64>,
}

fn pack_boundary_conditions(
    faces: impl IntoIterator<Item = PointId>,
    policy: &FvBoundaryPolicy,
) -> Result<BoundaryHost, AcceleratorError> {
    let mut host = BoundaryHost::default();
    for face in faces {
        let branch = policy.boundary_face_branches.get(&face).copied();
        let branch = match branch {
            Some(branch) if policy.allowed_branches.contains(&branch) => Some(branch),
            Some(branch) if policy.unsupported_behavior == UnsupportedBoundaryBehavior::Error => {
                return Err(AcceleratorError::InvalidPlan(format!(
                    "boundary face {face} uses unsupported branch {branch:?}"
                )));
            }
            None if policy.unsupported_behavior == UnsupportedBoundaryBehavior::Error => {
                return Err(AcceleratorError::InvalidPlan(format!(
                    "boundary face {face} has no boundary branch"
                )));
            }
            _ => None,
        };
        let resolve = |hooks: &HashMap<FvBoundaryBranch, BoundaryCondition>, kind: &str| {
            if let Some(branch) = branch {
                if let Some(condition) = hooks.get(&branch) {
                    return Ok(*condition);
                }
                if policy.unsupported_behavior == UnsupportedBoundaryBehavior::Error {
                    return Err(AcceleratorError::InvalidPlan(format!(
                        "boundary face {face} has no {kind} closure for branch {branch:?}"
                    )));
                }
            }
            Ok(BoundaryCondition::Neumann { gradient: 0.0 })
        };
        let convective = resolve(&policy.convective_branch_hooks, "convective")?;
        let diffusive = resolve(&policy.diffusive_branch_hooks, "diffusive")?;
        if !boundary_condition_is_finite(convective) || !boundary_condition_is_finite(diffusive) {
            return Err(AcceleratorError::InvalidPlan(format!(
                "boundary face {face} has non-finite coefficients"
            )));
        }
        push_boundary(
            convective,
            &mut host.convective_kind,
            &mut host.convective_alpha,
            &mut host.convective_beta,
            &mut host.convective_gamma,
        );
        push_boundary(
            diffusive,
            &mut host.diffusive_kind,
            &mut host.diffusive_alpha,
            &mut host.diffusive_beta,
            &mut host.diffusive_gamma,
        );
    }
    Ok(host)
}

fn boundary_condition_is_finite(condition: BoundaryCondition) -> bool {
    match condition {
        BoundaryCondition::Dirichlet { value } => value.is_finite(),
        BoundaryCondition::Neumann { gradient } => gradient.is_finite(),
        BoundaryCondition::Robin { alpha, beta, gamma } => {
            alpha.is_finite() && beta.is_finite() && gamma.is_finite()
        }
    }
}

fn push_boundary(
    condition: BoundaryCondition,
    kind: &mut Vec<u8>,
    alpha: &mut Vec<f64>,
    beta: &mut Vec<f64>,
    gamma: &mut Vec<f64>,
) {
    let (tag, a, b, g) = match condition {
        BoundaryCondition::Dirichlet { value } => (0, 1.0, 0.0, value),
        BoundaryCondition::Neumann { gradient } => (1, 0.0, 1.0, gradient),
        BoundaryCondition::Robin { alpha, beta, gamma } => (2, alpha, beta, gamma),
    };
    kind.push(tag);
    alpha.push(a);
    beta.push(b);
    gamma.push(g);
}

fn upload_boundary<B: AcceleratorBackend>(
    backend: &B,
    host: BoundaryHost,
) -> Result<DeviceFvmBoundaryConditions<B>, AcceleratorError> {
    Ok(DeviceFvmBoundaryConditions {
        convective_kind: upload(backend, &host.convective_kind)?,
        convective_alpha: upload(backend, &host.convective_alpha)?,
        convective_beta: upload(backend, &host.convective_beta)?,
        convective_gamma: upload(backend, &host.convective_gamma)?,
        diffusive_kind: upload(backend, &host.diffusive_kind)?,
        diffusive_alpha: upload(backend, &host.diffusive_alpha)?,
        diffusive_beta: upload(backend, &host.diffusive_beta)?,
        diffusive_gamma: upload(backend, &host.diffusive_gamma)?,
    })
}

fn upload_boundary_into<B: AcceleratorBackend>(
    backend: &B,
    device: &mut DeviceFvmBoundaryConditions<B>,
    host: &BoundaryHost,
) -> Result<(), AcceleratorError> {
    macro_rules! refresh {
        ($field:ident) => {
            backend
                .upload_into(&host.$field, &mut device.$field)
                .map_err(|e| AcceleratorError::DeviceTransferFailed(e.to_string()))?;
        };
    }
    refresh!(convective_kind);
    refresh!(convective_alpha);
    refresh!(convective_beta);
    refresh!(convective_gamma);
    refresh!(diffusive_kind);
    refresh!(diffusive_alpha);
    refresh!(diffusive_beta);
    refresh!(diffusive_gamma);
    Ok(())
}

#[derive(Default)]
struct LeastSquaresHost {
    neighbor: Vec<u32>,
    weight_x: Vec<f64>,
    weight_y: Vec<f64>,
    weight_z: Vec<f64>,
    fallback: Vec<u8>,
}

fn build_least_squares(
    inputs: &FvmInputs,
    cell_ids: &[PointId],
) -> Result<LeastSquaresHost, AcceleratorError> {
    let dimension = inputs
        .cell_geometry
        .first()
        .map_or(0, |(_, geometry)| geometry.centroid.len());
    let cell_index: HashMap<_, _> = cell_ids
        .iter()
        .copied()
        .enumerate()
        .map(|(index, point)| (point, index))
        .collect();
    let cell_centers: HashMap<_, _> = inputs
        .cell_geometry
        .iter()
        .map(|(point, geometry)| (*point, geometry.centroid.as_slice()))
        .collect();
    let face_centers: HashMap<_, _> = inputs
        .face_geometry
        .iter()
        .map(|(point, geometry)| (*point, geometry.centroid.as_slice()))
        .collect();
    let mut entries: Vec<Vec<(u32, [f64; 3])>> = vec![Vec::new(); cell_ids.len()];
    for stencil in &inputs.loops.internal {
        let owner = *cell_index.get(&stencil.left).ok_or_else(|| {
            AcceleratorError::InvalidPlan(format!("missing owner cell {}", stencil.left))
        })?;
        let neighbor_id = stencil.right.ok_or_else(|| {
            AcceleratorError::InvalidPlan(format!("internal face {} has no neighbor", stencil.face))
        })?;
        let neighbor = *cell_index.get(&neighbor_id).ok_or_else(|| {
            AcceleratorError::InvalidPlan(format!("missing neighbor cell {neighbor_id}"))
        })?;
        let owner_center = cell_centers[&stencil.left];
        let neighbor_center = cell_centers[&neighbor_id];
        entries[owner].push((
            checked_u32(neighbor, "least-squares neighbor")?,
            delta3(neighbor_center, owner_center),
        ));
        entries[neighbor].push((
            checked_u32(owner, "least-squares neighbor")?,
            delta3(owner_center, neighbor_center),
        ));
    }
    for stencil in &inputs.loops.boundary {
        let owner = *cell_index.get(&stencil.left).ok_or_else(|| {
            AcceleratorError::InvalidPlan(format!("missing boundary owner {}", stencil.left))
        })?;
        let center = face_centers.get(&stencil.face).ok_or_else(|| {
            AcceleratorError::InvalidPlan(format!(
                "missing boundary face geometry {}",
                stencil.face
            ))
        })?;
        entries[owner].push((u32::MAX, delta3(center, cell_centers[&stencil.left])));
    }

    let mut host = LeastSquaresHost::default();
    host.fallback.reserve(cell_ids.len());
    for cell_entries in entries {
        let mut normal = [[0.0; 3]; 3];
        for (_, delta) in &cell_entries {
            let d2 = dot3(*delta, *delta);
            let scale = if d2 > 1.0e-28 { 1.0 / d2 } else { 0.0 };
            for row in 0..dimension {
                for col in 0..dimension {
                    normal[row][col] += scale * delta[row] * delta[col];
                }
            }
        }
        let inverse = invert_normal(normal, dimension);
        host.fallback.push(u8::from(inverse.is_none()));
        for (neighbor, delta) in cell_entries {
            host.neighbor.push(neighbor);
            if let Some(inverse) = inverse {
                let d2 = dot3(delta, delta);
                let scale = if d2 > 1.0e-28 { 1.0 / d2 } else { 0.0 };
                let mut weight = [0.0; 3];
                for row in 0..dimension {
                    for col in 0..dimension {
                        weight[row] += inverse[row][col] * scale * delta[col];
                    }
                }
                host.weight_x.push(weight[0]);
                host.weight_y.push(weight[1]);
                host.weight_z.push(weight[2]);
            } else {
                host.weight_x.push(0.0);
                host.weight_y.push(0.0);
                host.weight_z.push(0.0);
            }
        }
    }
    Ok(host)
}

fn delta3(to: &[f64], from: &[f64]) -> [f64; 3] {
    [
        component(to, 0) - component(from, 0),
        component(to, 1) - component(from, 1),
        component(to, 2) - component(from, 2),
    ]
}

fn invert_normal(matrix: [[f64; 3]; 3], dimension: usize) -> Option<[[f64; 3]; 3]> {
    let mut inverse = [[0.0; 3]; 3];
    match dimension {
        0 => Some(inverse),
        1 => {
            if matrix[0][0].abs() <= 1.0e-12 {
                None
            } else {
                inverse[0][0] = 1.0 / matrix[0][0];
                Some(inverse)
            }
        }
        2 => {
            let determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
            if determinant.abs() <= 1.0e-12 {
                return None;
            }
            inverse[0][0] = matrix[1][1] / determinant;
            inverse[0][1] = -matrix[0][1] / determinant;
            inverse[1][0] = -matrix[1][0] / determinant;
            inverse[1][1] = matrix[0][0] / determinant;
            Some(inverse)
        }
        3 => {
            let determinant = matrix[0][0]
                * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
                - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
                + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);
            if determinant.abs() <= 1.0e-12 {
                return None;
            }
            inverse[0][0] =
                (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) / determinant;
            inverse[0][1] =
                (matrix[0][2] * matrix[2][1] - matrix[0][1] * matrix[2][2]) / determinant;
            inverse[0][2] =
                (matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1]) / determinant;
            inverse[1][0] =
                (matrix[1][2] * matrix[2][0] - matrix[1][0] * matrix[2][2]) / determinant;
            inverse[1][1] =
                (matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0]) / determinant;
            inverse[1][2] =
                (matrix[0][2] * matrix[1][0] - matrix[0][0] * matrix[1][2]) / determinant;
            inverse[2][0] =
                (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]) / determinant;
            inverse[2][1] =
                (matrix[0][1] * matrix[2][0] - matrix[0][0] * matrix[2][1]) / determinant;
            inverse[2][2] =
                (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]) / determinant;
            Some(inverse)
        }
        _ => None,
    }
}

impl DeviceFvmOperator<CpuBackend> {
    /// Evaluate all configured FVM terms into the resident residual buffer.
    pub fn evaluate_residual<T: FvmScalar>(
        &self,
        state: &mut DeviceFvmState<T, CpuBackend>,
        current_epochs: PlanEpochs,
    ) -> Result<(), AcceleratorError> {
        self.validate_state(state, current_epochs)?;
        self.compute_operator_gradients(state)?;
        self.compute_operator_face_fluxes(state)?;
        self.gather_operator_residual(state);
        Ok(())
    }

    fn compute_operator_gradients<T: FvmScalar>(
        &self,
        state: &mut DeviceFvmState<T, CpuBackend>,
    ) -> Result<(), AcceleratorError> {
        match self.schemes.reconstruction.mode {
            ReconstructionMode::GradientOnly(ReconstructionGradient::GreenGauss) => {
                self.green_gauss(state)
            }
            ReconstructionMode::GradientOnly(ReconstructionGradient::LeastSquares) => {
                self.least_squares(state, false)
            }
            ReconstructionMode::LeastSquaresWithGreenGaussFallback => {
                self.green_gauss(state)?;
                self.least_squares(state, true)
            }
        }
    }

    fn green_gauss<T: FvmScalar>(
        &self,
        state: &mut DeviceFvmState<T, CpuBackend>,
    ) -> Result<(), AcceleratorError> {
        let cells = self.plan.cell_ids.len();
        let faces = self.plan.face_count();
        let boundaries = self.plan.boundary_face_count;
        for component_index in 0..state.components {
            for face in 0..self.plan.internal_face_count {
                let owner = self.plan.internal_owner.0[face] as usize;
                let neighbor = self.plan.internal_neighbor.0[face] as usize;
                state.face_values.0[component_index * faces + face] = T::from_f64(
                    0.5 * (state.cell_values.0[component_index * cells + owner].to_f64()
                        + state.cell_values.0[component_index * cells + neighbor].to_f64()),
                );
            }
            for boundary in 0..boundaries {
                let face = self.plan.internal_face_count + boundary;
                let owner = self.plan.boundary_owner.0[boundary] as usize;
                let inside = state.cell_values.0[component_index * cells + owner].to_f64();
                let exterior = boundary_exterior(
                    self.boundary.convective_kind.0[boundary],
                    self.boundary.convective_alpha.0[boundary],
                    self.boundary.convective_beta.0[boundary],
                    self.boundary.convective_gamma.0[boundary],
                    inside,
                    state.boundary_values.0[component_index * boundaries + boundary].to_f64(),
                );
                state.face_values.0[component_index * faces + face] = T::from_f64(exterior);
            }
            for cell in 0..cells {
                let output = component_index * cells + cell;
                if self.plan.cell_active.0[cell] == 0 {
                    state.gradient_x.0[output] = T::zeroed();
                    state.gradient_y.0[output] = T::zeroed();
                    state.gradient_z.0[output] = T::zeroed();
                    continue;
                }
                let begin = self.plan.cell_face_offsets.0[cell] as usize;
                let end = self.plan.cell_face_offsets.0[cell + 1] as usize;
                let mut gradient = [0.0; 3];
                for incidence in begin..end {
                    let face = self.plan.cell_face_indices.0[incidence] as usize;
                    if self.plan.face_active.0[face] == 0 {
                        continue;
                    }
                    let geometry = self.plan.face_geometry_indices.0[face] as usize;
                    let value = state.face_values.0[component_index * faces + face].to_f64();
                    let scale = self.plan.cell_face_signs.0[incidence] as f64 * value;
                    gradient[0] += scale * self.plan.face_normal_x.0[geometry];
                    gradient[1] += scale * self.plan.face_normal_y.0[geometry];
                    gradient[2] += scale * self.plan.face_normal_z.0[geometry];
                }
                let volume = self.plan.cell_volume.0[cell];
                state.gradient_x.0[output] = T::from_f64(gradient[0] / volume);
                state.gradient_y.0[output] = T::from_f64(gradient[1] / volume);
                state.gradient_z.0[output] = T::from_f64(gradient[2] / volume);
            }
        }
        Ok(())
    }

    fn least_squares<T: FvmScalar>(
        &self,
        state: &mut DeviceFvmState<T, CpuBackend>,
        preserve_fallback: bool,
    ) -> Result<(), AcceleratorError> {
        let cells = self.plan.cell_ids.len();
        for component_index in 0..state.components {
            for cell in 0..cells {
                if preserve_fallback && self.least_squares.fallback_host[cell] != 0 {
                    continue;
                }
                let output = component_index * cells + cell;
                if self.plan.cell_active.0[cell] == 0 {
                    state.gradient_x.0[output] = T::zeroed();
                    state.gradient_y.0[output] = T::zeroed();
                    state.gradient_z.0[output] = T::zeroed();
                    continue;
                }
                let center_value = state.cell_values.0[output].to_f64();
                let begin = self.plan.cell_face_offsets.0[cell] as usize;
                let end = self.plan.cell_face_offsets.0[cell + 1] as usize;
                let mut gradient = [0.0; 3];
                for incidence in begin..end {
                    let face = self.plan.cell_face_indices.0[incidence] as usize;
                    if self.plan.face_active.0[face] == 0 {
                        continue;
                    }
                    let neighbor = self.least_squares.neighbor.0[incidence];
                    let sample = if neighbor == u32::MAX {
                        let boundary =
                            face.checked_sub(self.plan.internal_face_count)
                                .ok_or_else(|| {
                                    AcceleratorError::InvalidPlan(
                                        "boundary LS sample is internal".into(),
                                    )
                                })?;
                        boundary_exterior(
                            self.boundary.convective_kind.0[boundary],
                            self.boundary.convective_alpha.0[boundary],
                            self.boundary.convective_beta.0[boundary],
                            self.boundary.convective_gamma.0[boundary],
                            center_value,
                            state.boundary_values.0
                                [component_index * self.plan.boundary_face_count + boundary]
                                .to_f64(),
                        )
                    } else {
                        state.cell_values.0[component_index * cells + neighbor as usize].to_f64()
                    };
                    let difference = sample - center_value;
                    gradient[0] += self.least_squares.weight_x.0[incidence] * difference;
                    gradient[1] += self.least_squares.weight_y.0[incidence] * difference;
                    gradient[2] += self.least_squares.weight_z.0[incidence] * difference;
                }
                state.gradient_x.0[output] = T::from_f64(gradient[0]);
                state.gradient_y.0[output] = T::from_f64(gradient[1]);
                state.gradient_z.0[output] = T::from_f64(gradient[2]);
            }
        }
        Ok(())
    }

    fn compute_operator_face_fluxes<T: FvmScalar>(
        &self,
        state: &mut DeviceFvmState<T, CpuBackend>,
    ) -> Result<(), AcceleratorError> {
        let cells = self.plan.cell_ids.len();
        let faces = self.plan.face_count();
        let reconstruct = self.metadata.reconstruction_order > 1;
        for component_index in 0..state.components {
            for face in 0..self.plan.internal_face_count {
                let output = component_index * faces + face;
                if self.plan.face_active.0[face] == 0 {
                    state.face_flux.0[output] = T::zeroed();
                    state.face_deferred_source.0[output] = T::zeroed();
                    continue;
                }
                let owner = self.plan.internal_owner.0[face] as usize;
                let neighbor = self.plan.internal_neighbor.0[face] as usize;
                let geometry = self.plan.internal_geometry.0[face] as usize;
                let left = reconstructed_value(
                    state,
                    component_index,
                    cells,
                    owner,
                    geometry,
                    &self.plan,
                    reconstruct,
                );
                let right = reconstructed_value(
                    state,
                    component_index,
                    cells,
                    neighbor,
                    geometry,
                    &self.plan,
                    reconstruct,
                );
                let mass = state.face_mass_flux.0[face].to_f64();
                let convective = mass
                    * operator_face_value(
                        left,
                        right,
                        mass,
                        self.schemes.convective,
                        self.schemes.reconstruction.limiter,
                    );
                let delta = [
                    self.plan.cell_center_x.0[neighbor] - self.plan.cell_center_x.0[owner],
                    self.plan.cell_center_y.0[neighbor] - self.plan.cell_center_y.0[owner],
                    self.plan.cell_center_z.0[neighbor] - self.plan.cell_center_z.0[owner],
                ];
                let normal = [
                    self.plan.face_normal_x.0[geometry],
                    self.plan.face_normal_y.0[geometry],
                    self.plan.face_normal_z.0[geometry],
                ];
                let d2 = dot3(delta, delta);
                let inside_left = state.cell_values.0[component_index * cells + owner].to_f64();
                let inside_right = state.cell_values.0[component_index * cells + neighbor].to_f64();
                let orthogonal = if d2 > 0.0 {
                    -self.schemes.diffusion.diffusivity
                        * (inside_right - inside_left)
                        * dot3(normal, delta)
                        / d2
                } else {
                    0.0
                };
                let average_gradient = [
                    0.5 * (state.gradient_x.0[component_index * cells + owner].to_f64()
                        + state.gradient_x.0[component_index * cells + neighbor].to_f64()),
                    0.5 * (state.gradient_y.0[component_index * cells + owner].to_f64()
                        + state.gradient_y.0[component_index * cells + neighbor].to_f64()),
                    0.5 * (state.gradient_z.0[component_index * cells + owner].to_f64()
                        + state.gradient_z.0[component_index * cells + neighbor].to_f64()),
                ];
                let orthogonal_normal = if d2 > 0.0 {
                    let scale = dot3(normal, delta) / d2;
                    [delta[0] * scale, delta[1] * scale, delta[2] * scale]
                } else {
                    [0.0; 3]
                };
                let nonorthogonal = -self.schemes.diffusion.diffusivity
                    * dot3(
                        average_gradient,
                        [
                            normal[0] - orthogonal_normal[0],
                            normal[1] - orthogonal_normal[1],
                            normal[2] - orthogonal_normal[2],
                        ],
                    );
                let (diffusive, deferred) = match self.schemes.diffusion.non_orthogonal_mode {
                    NonOrthogonalCorrectionMode::OrthogonalOnly => (orthogonal, 0.0),
                    NonOrthogonalCorrectionMode::Deferred => (orthogonal, -nonorthogonal),
                    NonOrthogonalCorrectionMode::FullyCorrected => {
                        (orthogonal + nonorthogonal, 0.0)
                    }
                };
                state.face_deferred_source.0[output] = T::from_f64(deferred);
                state.face_flux.0[output] =
                    T::from_f64(convective + diffusive + state.face_source.0[output].to_f64());
            }
            for boundary in 0..self.plan.boundary_face_count {
                let face = self.plan.internal_face_count + boundary;
                let output = component_index * faces + face;
                if self.plan.face_active.0[face] == 0 {
                    state.face_flux.0[output] = T::zeroed();
                    state.face_deferred_source.0[output] = T::zeroed();
                    continue;
                }
                let owner = self.plan.boundary_owner.0[boundary] as usize;
                let geometry = self.plan.boundary_geometry.0[boundary] as usize;
                let inside = reconstructed_value(
                    state,
                    component_index,
                    cells,
                    owner,
                    geometry,
                    &self.plan,
                    reconstruct,
                );
                let exterior = boundary_exterior(
                    self.boundary.convective_kind.0[boundary],
                    self.boundary.convective_alpha.0[boundary],
                    self.boundary.convective_beta.0[boundary],
                    self.boundary.convective_gamma.0[boundary],
                    inside,
                    state.boundary_values.0
                        [component_index * self.plan.boundary_face_count + boundary]
                        .to_f64(),
                );
                let mass = state.face_mass_flux.0[face].to_f64();
                let convective = mass
                    * operator_face_value(
                        inside,
                        exterior,
                        mass,
                        self.schemes.convective,
                        self.schemes.reconstruction.limiter,
                    );
                let cell_value = state.cell_values.0[component_index * cells + owner].to_f64();
                let area = self.plan.face_area.0[geometry];
                let (diffusive, boundary_source) = match self.boundary.diffusive_kind.0[boundary] {
                    0 => {
                        let delta = [
                            self.plan.face_center_x.0[geometry] - self.plan.cell_center_x.0[owner],
                            self.plan.face_center_y.0[geometry] - self.plan.cell_center_y.0[owner],
                            self.plan.face_center_z.0[geometry] - self.plan.cell_center_z.0[owner],
                        ];
                        let distance = dot3(delta, delta).sqrt().max(1.0e-14);
                        (
                            -self.schemes.diffusion.diffusivity
                                * (state.boundary_values.0
                                    [component_index * self.plan.boundary_face_count + boundary]
                                    .to_f64()
                                    - cell_value)
                                / distance
                                * area,
                            0.0,
                        )
                    }
                    1 => (
                        -self.schemes.diffusion.diffusivity
                            * self.boundary.diffusive_gamma.0[boundary]
                            * area,
                        0.0,
                    ),
                    _ => {
                        let beta = self.boundary.diffusive_beta.0[boundary].max(1.0e-14);
                        let gradient = (self.boundary.diffusive_gamma.0[boundary]
                            - self.boundary.diffusive_alpha.0[boundary] * cell_value)
                            / beta;
                        (
                            -self.schemes.diffusion.diffusivity * gradient * area,
                            self.schemes.diffusion.diffusivity
                                * self.boundary.diffusive_gamma.0[boundary]
                                * area
                                / beta,
                        )
                    }
                };
                state.face_deferred_source.0[output] = T::from_f64(boundary_source);
                state.face_flux.0[output] =
                    T::from_f64(convective + diffusive + state.face_source.0[output].to_f64());
            }
        }
        Ok(())
    }

    fn gather_operator_residual<T: FvmScalar>(&self, state: &mut DeviceFvmState<T, CpuBackend>) {
        let cells = self.plan.cell_ids.len();
        let faces = self.plan.face_count();
        for component_index in 0..state.components {
            for cell in 0..cells {
                let output = component_index * cells + cell;
                if self.plan.cell_active.0[cell] == 0 {
                    state.residual.0[output] = T::zeroed();
                    continue;
                }
                let begin = self.plan.cell_face_offsets.0[cell] as usize;
                let end = self.plan.cell_face_offsets.0[cell + 1] as usize;
                let mut residual = state.cell_source.0[output].to_f64();
                for incidence in begin..end {
                    let face = self.plan.cell_face_indices.0[incidence] as usize;
                    let sign = self.plan.cell_face_signs.0[incidence] as f64;
                    residual += sign * state.face_flux.0[component_index * faces + face].to_f64();
                    if sign > 0.0 {
                        residual +=
                            state.face_deferred_source.0[component_index * faces + face].to_f64();
                    }
                }
                state.residual.0[output] = T::from_f64(residual);
            }
        }
    }
}

fn reconstructed_value<T: FvmScalar>(
    state: &DeviceFvmState<T, CpuBackend>,
    component_index: usize,
    cell_count: usize,
    cell: usize,
    geometry: usize,
    plan: &DeviceFvmPlan<CpuBackend>,
    reconstruct: bool,
) -> f64 {
    let index = component_index * cell_count + cell;
    let value = state.cell_values.0[index].to_f64();
    if !reconstruct {
        return value;
    }
    value
        + state.gradient_x.0[index].to_f64()
            * (plan.face_center_x.0[geometry] - plan.cell_center_x.0[cell])
        + state.gradient_y.0[index].to_f64()
            * (plan.face_center_y.0[geometry] - plan.cell_center_y.0[cell])
        + state.gradient_z.0[index].to_f64()
            * (plan.face_center_z.0[geometry] - plan.cell_center_z.0[cell])
}

fn boundary_exterior(
    kind: u8,
    alpha: f64,
    beta: f64,
    gamma: f64,
    inside: f64,
    dirichlet_value: f64,
) -> f64 {
    match kind {
        0 => dirichlet_value,
        1 => inside,
        _ if alpha.abs() < 1.0e-14 => inside,
        _ => (gamma - beta * inside) / alpha,
    }
}

fn operator_face_value(
    left: f64,
    right: f64,
    mass: f64,
    scheme: ConvectiveScheme,
    limiter: LimiterOption,
) -> f64 {
    let upwind = if mass >= 0.0 { left } else { right };
    let downwind = if mass >= 0.0 { right } else { left };
    let central = 0.5 * (left + right);
    let minimum = left.min(right);
    let maximum = left.max(right);
    match scheme {
        ConvectiveScheme::Upwind => upwind,
        ConvectiveScheme::Central => central,
        ConvectiveScheme::BoundedLinear { blend } => {
            (upwind + blend.clamp(0.0, 1.0) * (central - upwind)).clamp(minimum, maximum)
        }
        ConvectiveScheme::BlendUpwindCentral { blend } => {
            let blend = blend.clamp(0.0, 1.0);
            (1.0 - blend) * upwind + blend * central
        }
        ConvectiveScheme::HighResolution {
            blend,
            limiter: scheme_limiter,
        } => {
            let psi = limiter_factor(limiter, upwind, downwind)
                * slope_limiter_factor(scheme_limiter, upwind, downwind);
            let high_resolution = upwind + 0.5 * psi * (downwind - upwind);
            let blend = blend.clamp(0.0, 1.0);
            ((1.0 - blend) * upwind + blend * high_resolution).clamp(minimum, maximum)
        }
    }
}

fn limiter_factor(limiter: LimiterOption, upwind: f64, downwind: f64) -> f64 {
    match limiter {
        LimiterOption::None => 1.0,
        LimiterOption::Family(family) => slope_limiter_factor(family, upwind, downwind),
    }
}

fn slope_limiter_factor(limiter: SlopeLimiterFamily, upwind: f64, downwind: f64) -> f64 {
    let ratio = if (downwind - upwind).abs() < 1.0e-14 {
        1.0
    } else {
        ((upwind - downwind) / (downwind - upwind)).abs()
    };
    match limiter {
        SlopeLimiterFamily::None => 1.0,
        SlopeLimiterFamily::Minmod => ratio.clamp(0.0, 1.0),
        SlopeLimiterFamily::VanLeer => (ratio + ratio.abs()) / (1.0 + ratio.abs()),
        SlopeLimiterFamily::Superbee => (2.0 * ratio).min(1.0).max(ratio.min(2.0)).max(0.0),
    }
}

fn checked_product(
    count: usize,
    components: usize,
    what: &'static str,
) -> Result<usize, AcceleratorError> {
    count
        .checked_mul(components)
        .ok_or(AcceleratorError::IndexOverflow {
            what,
            value: usize::MAX,
        })
}

fn allocate<T: FvmScalar, B: AcceleratorBackend>(
    backend: &B,
    len: usize,
) -> Result<B::Buffer<T>, AcceleratorError> {
    backend
        .allocate(len)
        .map_err(|e| AcceleratorError::AllocationFailed {
            bytes: len.saturating_mul(std::mem::size_of::<T>()),
            reason: e.to_string(),
        })
}

fn component(values: &[f64], index: usize) -> f64 {
    values.get(index).copied().unwrap_or(0.0)
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn lookup(
    map: &HashMap<PointId, u32>,
    point: PointId,
    kind: &'static str,
) -> Result<u32, AcceleratorError> {
    map.get(&point).copied().ok_or_else(|| {
        AcceleratorError::InvalidPlan(format!("{kind} {point} is absent from packed geometry"))
    })
}

fn check_len(expected: usize, found: usize) -> Result<(), AcceleratorError> {
    if expected == found {
        Ok(())
    } else {
        Err(AcceleratorError::LengthMismatch { expected, found })
    }
}

fn encode_boundary_kind(branch: FvBoundaryBranch) -> u32 {
    match branch {
        FvBoundaryBranch::Open => 0,
        FvBoundaryBranch::Inflow => 1,
        FvBoundaryBranch::Outflow => 2,
        FvBoundaryBranch::Tidal => 3,
        FvBoundaryBranch::Bed => 4,
        FvBoundaryBranch::FreeSurface => 5,
    }
}
