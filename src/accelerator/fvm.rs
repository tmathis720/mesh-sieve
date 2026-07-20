//! Device-ready finite-volume plans and CPU reference execution.

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};

use crate::physics::fvm::{FvBoundaryBranch, FvmInputs};
use crate::topology::coastal::{WET_DRY_MASK_LABEL, WetDryMask};
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
    /// Cell-centered scalar values.
    pub cell_values: B::Buffer<T>,
    /// Mass flux values in the plan's combined face order.
    pub face_mass_flux: B::Buffer<T>,
    /// Boundary exterior values in boundary-face order.
    pub boundary_values: B::Buffer<T>,
    /// Oriented flux values in combined face order.
    pub face_flux: B::Buffer<T>,
    /// Interpolated face values used by Green--Gauss gradients.
    pub face_values: B::Buffer<T>,
    /// Gathered cell residuals.
    pub residual: B::Buffer<T>,
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
        check_len(plan.cell_ids.len(), cell_values.len())?;
        check_len(plan.face_count(), face_mass_flux.len())?;
        check_len(plan.boundary_face_count, boundary_values.len())?;
        Ok(Self {
            cell_values: upload(backend, cell_values)?,
            face_mass_flux: upload(backend, face_mass_flux)?,
            boundary_values: upload(backend, boundary_values)?,
            face_flux: backend.allocate(plan.face_count()).map_err(|e| {
                AcceleratorError::AllocationFailed {
                    bytes: plan.face_count() * std::mem::size_of::<T>(),
                    reason: e.to_string(),
                }
            })?,
            face_values: backend.allocate(plan.face_count()).map_err(|e| {
                AcceleratorError::AllocationFailed {
                    bytes: plan.face_count() * std::mem::size_of::<T>(),
                    reason: e.to_string(),
                }
            })?,
            residual: backend.allocate(plan.cell_ids.len()).map_err(|e| {
                AcceleratorError::AllocationFailed {
                    bytes: plan.cell_ids.len() * std::mem::size_of::<T>(),
                    reason: e.to_string(),
                }
            })?,
            gradient_x: backend.allocate(plan.cell_ids.len()).map_err(|e| {
                AcceleratorError::AllocationFailed {
                    bytes: plan.cell_ids.len() * std::mem::size_of::<T>(),
                    reason: e.to_string(),
                }
            })?,
            gradient_y: backend.allocate(plan.cell_ids.len()).map_err(|e| {
                AcceleratorError::AllocationFailed {
                    bytes: plan.cell_ids.len() * std::mem::size_of::<T>(),
                    reason: e.to_string(),
                }
            })?,
            gradient_z: backend.allocate(plan.cell_ids.len()).map_err(|e| {
                AcceleratorError::AllocationFailed {
                    bytes: plan.cell_ids.len() * std::mem::size_of::<T>(),
                    reason: e.to_string(),
                }
            })?,
        })
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
