//! Boundary condition helpers using label queries.

use crate::data::constrained_section::ConstrainedSection;
use crate::data::section::Section;
use crate::data::storage::Storage;
use crate::mesh_error::MeshSieveError;
use crate::topology::cache::InvalidateCache;
use crate::topology::labels::LabelSet;
use crate::topology::point::PointId;

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
