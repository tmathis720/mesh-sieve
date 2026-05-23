//! Finite-volume oriented wrappers over mesh-sieve topology/data/geometry primitives.
//!
//! This module is the canonical entrypoint for finite-volume (FV) callers.
//! It intentionally exposes FV-ready mesh loops and interpolation helpers without
//! requiring users to couple directly to finite-element runtime APIs.

use crate::data::section::Section;
use crate::data::storage::Storage;
use crate::discretization::runtime::{CellGeometry, FaceGeometry, FluxStencil};
use crate::mesh_error::MeshSieveError;
use crate::topology::point::PointId;
use crate::topology::sieve::Sieve;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaceKind {
    Internal,
    Boundary,
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
        Self {
            loops,
            cell_geometry,
            face_geometry,
        }
    }

    pub fn internal_faces(&self) -> impl Iterator<Item = &FluxStencil> {
        self.loops.internal.iter()
    }

    pub fn boundary_faces(&self) -> impl Iterator<Item = &FluxStencil> {
        self.loops.boundary.iter()
    }

    pub fn cell_metrics(&self, cell: PointId) -> Option<&CellGeometry> {
        self.cell_geometry
            .iter()
            .find_map(|(id, metrics)| (*id == cell).then_some(metrics))
    }

    pub fn face_metrics(&self, face: PointId) -> Option<&FaceGeometry> {
        self.face_geometry
            .iter()
            .find_map(|(id, metrics)| (*id == face).then_some(metrics))
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
            MeshSieveError::InvalidGeometry(format!("missing scalar value at cell {}", right_cell))
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
            MeshSieveError::InvalidGeometry(format!("missing scalar value at face {}", face))
        })?;
        accum += scalar;
    }
    Ok(accum / incident_faces.len() as f64)
}

pub fn assemble_convective_fluxes(_inputs: &FvmInputs) -> Result<(), MeshSieveError> {
    Ok(())
}

pub fn assemble_diffusive_fluxes(_inputs: &FvmInputs) -> Result<(), MeshSieveError> {
    Ok(())
}
