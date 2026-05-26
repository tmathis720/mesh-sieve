//! Boundary condition helpers using label queries.

use crate::data::constrained_section::ConstrainedSection;
use crate::data::section::Section;
use crate::data::storage::Storage;
use crate::mesh_error::MeshSieveError;
use crate::physics::fvm::{
    BoundaryBranchError, BoundaryCondition, FvBoundaryBranch, boundary_branch_for_face_checked,
};
use crate::topology::cache::InvalidateCache;
use crate::topology::coastal::CoastalLabelQueries;
use crate::topology::labels::LabelSet;
use crate::topology::point::PointId;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CoastalBoundaryAssemblyError {
    InvalidBoundaryFaceRole(BoundaryBranchError),
}

/// Label query selector for boundary condition application.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LabelQuery {
    name: String,
    value: i32,
}

impl LabelQuery {
    /// Create a new label query.
    pub fn new(name: impl Into<String>, value: i32) -> Self {
        Self {
            name: name.into(),
            value,
        }
    }

    /// Label name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Label value.
    pub fn value(&self) -> i32 {
        self.value
    }
}

/// Per-field DOF indices for packed multi-field sections.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldDofIndices {
    /// Field index in the packed layout.
    pub field: usize,
    /// DOF indices within the field slice.
    pub dof_indices: Vec<usize>,
}

impl FieldDofIndices {
    /// Create a new field DOF index set.
    pub fn new(field: usize, dof_indices: impl Into<Vec<usize>>) -> Self {
        Self {
            field,
            dof_indices: dof_indices.into(),
        }
    }
}

fn label_points(labels: &LabelSet, query: &LabelQuery) -> Vec<PointId> {
    labels.stratum_points(query.name(), query.value())
}

/// Boundary face sets derived from canonical coastal labels.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CoastalBoundaryFaceSets {
    pub free_surface: Vec<PointId>,
    pub bed: Vec<PointId>,
    pub open: Vec<PointId>,
    pub inflow: Vec<PointId>,
    pub outflow: Vec<PointId>,
    pub tidal: Vec<PointId>,
}

/// Build boundary-face sets by intersecting supplied boundary faces with coastal labels.
pub fn coastal_boundary_face_sets(
    labels: &LabelSet,
    boundary_faces: impl IntoIterator<Item = PointId>,
) -> CoastalBoundaryFaceSets {
    let faces: HashSet<_> = boundary_faces.into_iter().collect();
    let filter = |pts: Vec<PointId>| -> Vec<PointId> {
        let mut v: Vec<_> = pts.into_iter().filter(|p| faces.contains(p)).collect();
        v.sort_unstable();
        v
    };
    CoastalBoundaryFaceSets {
        free_surface: filter(labels.free_surface_points()),
        bed: filter(labels.bed_points()),
        open: filter(labels.open_boundary_points()),
        inflow: filter(labels.inflow_points()),
        outflow: filter(labels.outflow_points()),
        tidal: filter(labels.tidal_points()),
    }
}

/// Assign per-face boundary conditions using coastal boundary class/role labels.
pub fn map_coastal_boundary_conditions(
    labels: &LabelSet,
    boundary_faces: impl IntoIterator<Item = PointId>,
    branch_closure: impl Fn(FvBoundaryBranch, PointId) -> BoundaryCondition,
) -> HashMap<PointId, BoundaryCondition> {
    let mut out = HashMap::new();
    for face in boundary_faces {
        if let Ok(branch) = boundary_branch_for_face_checked(labels, face) {
            out.insert(face, branch_closure(branch, face));
        }
    }
    out
}

/// Resolved coastal boundary data ready for FV flux assembly.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CoastalBoundaryAssembly {
    pub face_sets: CoastalBoundaryFaceSets,
    pub boundary_conditions: HashMap<PointId, BoundaryCondition>,
    pub branches: HashMap<PointId, FvBoundaryBranch>,
}

/// High-level coastal API: resolve boundary face groups and BC closures in one pass.
pub fn resolve_coastal_boundary_assembly(
    labels: &LabelSet,
    boundary_faces: impl IntoIterator<Item = PointId>,
    branch_closure: impl Fn(FvBoundaryBranch, PointId) -> BoundaryCondition,
) -> Result<CoastalBoundaryAssembly, CoastalBoundaryAssemblyError> {
    let faces: Vec<_> = boundary_faces.into_iter().collect();
    let face_sets = coastal_boundary_face_sets(labels, faces.iter().copied());
    let mut boundary_conditions = HashMap::new();
    let mut branches = HashMap::new();
    for face in faces {
        let branch = boundary_branch_for_face_checked(labels, face)
            .map_err(CoastalBoundaryAssemblyError::InvalidBoundaryFaceRole)?;
        boundary_conditions.insert(face, branch_closure(branch, face));
        branches.insert(face, branch);
    }
    Ok(CoastalBoundaryAssembly {
        face_sets,
        boundary_conditions,
        branches,
    })
}

fn field_offsets(field_dofs: &[usize]) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(field_dofs.len());
    let mut total = 0usize;
    for dof in field_dofs {
        offsets.push(total);
        total += *dof;
    }
    offsets
}

/// Apply Dirichlet constraints directly to a section for all points matching a label query.
pub fn apply_dirichlet_to_section<V, S, F>(
    section: &mut Section<V, S>,
    labels: &LabelSet,
    query: &LabelQuery,
    dof_indices: &[usize],
    mut value: F,
) -> Result<(), MeshSieveError>
where
    V: Clone,
    S: Storage<V>,
    F: FnMut(PointId, usize) -> V,
{
    for point in label_points(labels, query) {
        let slice = section.try_restrict_mut(point)?;
        let len = slice.len();
        for &index in dof_indices {
            if index >= len {
                return Err(MeshSieveError::ConstraintIndexOutOfBounds { point, index, len });
            }
            slice[index] = value(point, index);
        }
    }
    section.invalidate_cache();
    Ok(())
}

/// Apply Dirichlet constraints directly to a section using packed per-field DOF indices.
pub fn apply_dirichlet_to_section_fields<V, S, F>(
    section: &mut Section<V, S>,
    labels: &LabelSet,
    query: &LabelQuery,
    field_dofs: &[usize],
    field_indices: &[FieldDofIndices],
    mut value: F,
) -> Result<(), MeshSieveError>
where
    V: Clone,
    S: Storage<V>,
    F: FnMut(PointId, usize, usize) -> V,
{
    let offsets = field_offsets(field_dofs);
    for point in label_points(labels, query) {
        let slice = section.try_restrict_mut(point)?;
        let len = slice.len();
        for field_spec in field_indices {
            let field = field_spec.field;
            let field_len =
                *field_dofs
                    .get(field)
                    .ok_or_else(|| MeshSieveError::SectionAccess {
                        point,
                        source: Box::new(std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            "field index out of bounds",
                        )),
                    })?;
            let base = offsets[field];
            for &dof in &field_spec.dof_indices {
                if dof >= field_len {
                    return Err(MeshSieveError::ConstraintIndexOutOfBounds {
                        point,
                        index: dof,
                        len: field_len,
                    });
                }
                let index = base + dof;
                if index >= len {
                    return Err(MeshSieveError::ConstraintIndexOutOfBounds { point, index, len });
                }
                slice[index] = value(point, field, dof);
            }
        }
    }
    section.invalidate_cache();
    Ok(())
}

/// Apply Dirichlet constraints to a constrained section for all points matching a label query.
pub fn apply_dirichlet_to_constrained_section<V, S, F>(
    section: &mut ConstrainedSection<V, S>,
    labels: &LabelSet,
    query: &LabelQuery,
    dof_indices: &[usize],
    mut value: F,
) -> Result<(), MeshSieveError>
where
    V: Clone,
    S: Storage<V>,
    F: FnMut(PointId, usize) -> V,
{
    for point in label_points(labels, query) {
        for &index in dof_indices {
            let val = value(point, index);
            section.insert_constraint(point, index, val)?;
        }
    }
    section.apply_constraints()
}

/// Apply Dirichlet constraints to a constrained section using packed per-field DOF indices.
pub fn apply_dirichlet_to_constrained_section_fields<V, S, F>(
    section: &mut ConstrainedSection<V, S>,
    labels: &LabelSet,
    query: &LabelQuery,
    field_dofs: &[usize],
    field_indices: &[FieldDofIndices],
    mut value: F,
) -> Result<(), MeshSieveError>
where
    V: Clone,
    S: Storage<V>,
    F: FnMut(PointId, usize, usize) -> V,
{
    let offsets = field_offsets(field_dofs);
    for point in label_points(labels, query) {
        for field_spec in field_indices {
            let field = field_spec.field;
            let field_len =
                *field_dofs
                    .get(field)
                    .ok_or_else(|| MeshSieveError::SectionAccess {
                        point,
                        source: Box::new(std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            "field index out of bounds",
                        )),
                    })?;
            let base = offsets[field];
            for &dof in &field_spec.dof_indices {
                if dof >= field_len {
                    return Err(MeshSieveError::ConstraintIndexOutOfBounds {
                        point,
                        index: dof,
                        len: field_len,
                    });
                }
                let index = base + dof;
                let val = value(point, field, dof);
                section.insert_constraint(point, index, val)?;
            }
        }
    }
    section.apply_constraints()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discretization::runtime::FluxStencil;
    use crate::physics::fvm::{FvmInputs, flux_activity_mask_from_wet_dry};
    use crate::topology::coastal::{
        BOUNDARY_CLASS_LABEL, BOUNDARY_ROLE_LABEL, BoundaryClass, OpenBoundaryRole,
        WET_DRY_MASK_LABEL, WetDryMask,
    };

    fn p(id: u64) -> PointId {
        PointId::new(id).unwrap()
    }

    #[test]
    fn resolves_all_boundary_branches_to_bc_closures() {
        let mut labels = LabelSet::new();
        labels.set_label(
            p(10),
            BOUNDARY_CLASS_LABEL,
            BoundaryClass::FreeSurface.code(),
        );
        labels.set_label(p(11), BOUNDARY_CLASS_LABEL, BoundaryClass::Bed.code());
        labels.set_label(p(12), BOUNDARY_CLASS_LABEL, BoundaryClass::Open.code());
        labels.set_label(p(12), BOUNDARY_ROLE_LABEL, OpenBoundaryRole::Inflow.code());
        labels.set_label(p(13), BOUNDARY_CLASS_LABEL, BoundaryClass::Open.code());
        labels.set_label(p(13), BOUNDARY_ROLE_LABEL, OpenBoundaryRole::Outflow.code());
        labels.set_label(p(14), BOUNDARY_CLASS_LABEL, BoundaryClass::Open.code());
        labels.set_label(p(14), BOUNDARY_ROLE_LABEL, OpenBoundaryRole::Tidal.code());

        let resolved = resolve_coastal_boundary_assembly(
            &labels,
            [p(10), p(11), p(12), p(13), p(14)],
            |branch, _| match branch {
                FvBoundaryBranch::FreeSurface => BoundaryCondition::Neumann { gradient: 1.0 },
                FvBoundaryBranch::Bed => BoundaryCondition::Neumann { gradient: -1.0 },
                FvBoundaryBranch::Inflow => BoundaryCondition::Dirichlet { value: 10.0 },
                FvBoundaryBranch::Outflow => BoundaryCondition::Robin {
                    alpha: 1.0,
                    beta: 0.5,
                    gamma: 0.2,
                },
                FvBoundaryBranch::Tidal => BoundaryCondition::Dirichlet { value: 7.0 },
                FvBoundaryBranch::Open => unreachable!(),
            },
        )
        .unwrap();
        assert_eq!(resolved.face_sets.free_surface, vec![p(10)]);
        assert_eq!(resolved.face_sets.bed, vec![p(11)]);
        assert_eq!(resolved.face_sets.inflow, vec![p(12)]);
        assert_eq!(resolved.face_sets.outflow, vec![p(13)]);
        assert_eq!(resolved.face_sets.tidal, vec![p(14)]);
        assert_eq!(resolved.boundary_conditions.len(), 5);
    }

    #[test]
    fn errors_on_missing_open_role_in_assembly_faces() {
        let mut labels = LabelSet::new();
        labels.set_label(p(12), BOUNDARY_CLASS_LABEL, BoundaryClass::Open.code());
        let err = resolve_coastal_boundary_assembly(&labels, [p(12)], |_, _| {
            BoundaryCondition::Dirichlet { value: 0.0 }
        })
        .unwrap_err();
        assert!(matches!(
            err,
            CoastalBoundaryAssemblyError::InvalidBoundaryFaceRole(BoundaryBranchError::OpenBoundaryMissingRole { face }) if face == p(12)
        ));
    }

    #[test]
    fn wet_dry_mask_permutations_and_mixed_boundary_sets() {
        let c0 = p(1);
        let c1 = p(2);
        let c2 = p(3);
        let b0 = p(10);
        let b1 = p(11);
        let i0 = p(20);
        let inputs = FvmInputs::new(
            [
                FluxStencil {
                    face: b0,
                    left: c0,
                    right: None,
                },
                FluxStencil {
                    face: b1,
                    left: c1,
                    right: None,
                },
                FluxStencil {
                    face: i0,
                    left: c1,
                    right: Some(c2),
                },
            ],
            vec![],
            vec![],
        );
        let mut labels = LabelSet::new();
        labels.set_label(b0, BOUNDARY_CLASS_LABEL, BoundaryClass::Open.code());
        labels.set_label(b0, BOUNDARY_ROLE_LABEL, OpenBoundaryRole::Inflow.code());
        labels.set_label(b1, BOUNDARY_CLASS_LABEL, BoundaryClass::Bed.code());
        labels.set_label(i0, BOUNDARY_CLASS_LABEL, BoundaryClass::Open.code());
        labels.set_label(c1, WET_DRY_MASK_LABEL, WetDryMask::Dry.code());
        labels.set_label(c2, WET_DRY_MASK_LABEL, WetDryMask::Wet.code());
        let mask = flux_activity_mask_from_wet_dry(&inputs, &labels);
        assert_eq!(mask.boundary_faces_active.get(&b0), Some(&true));
        assert_eq!(mask.boundary_faces_active.get(&b1), Some(&false));
        assert_eq!(mask.near_boundary_faces_active.get(&i0), Some(&false));
    }
}
