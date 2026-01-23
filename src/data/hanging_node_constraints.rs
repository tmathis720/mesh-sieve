//! Hanging node constraints: linear relationships between DOFs.

use crate::data::section::Section;
use crate::data::storage::Storage;
use crate::mesh_error::MeshSieveError;
use crate::topology::cache::InvalidateCache;
use crate::topology::point::PointId;
use std::collections::BTreeMap;

/// A linear term referencing a parent point/DOF with a weight.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearConstraintTerm<V> {
    /// Parent point containing the source DOF.
    pub point: PointId,
    /// Index into the parent point slice.
    pub index: usize,
    /// Weight applied to the parent DOF value.
    pub weight: V,
}

impl<V> LinearConstraintTerm<V> {
    /// Create a new linear term.
    pub fn new(point: PointId, index: usize, weight: V) -> Self {
        Self {
            point,
            index,
            weight,
        }
    }
}

/// A linear constraint for a single DOF in a child point.
#[derive(Clone, Debug, PartialEq)]
pub struct HangingDofConstraint<V> {
    /// Index into the constrained (child) point slice.
    pub index: usize,
    /// Linear terms defining the constraint.
    pub terms: Vec<LinearConstraintTerm<V>>,
}

impl<V> HangingDofConstraint<V> {
    /// Create a new hanging DOF constraint.
    pub fn new(index: usize, terms: Vec<LinearConstraintTerm<V>>) -> Self {
        Self { index, terms }
    }
}

/// Stores hanging node constraints for a collection of points.
#[derive(Clone, Debug)]
pub struct HangingNodeConstraints<V> {
    constraints: BTreeMap<PointId, Vec<HangingDofConstraint<V>>>,
}

impl<V> Default for HangingNodeConstraints<V> {
    fn default() -> Self {
        Self {
            constraints: BTreeMap::new(),
        }
    }
}

impl<V> HangingNodeConstraints<V> {
    /// Borrow the constraint map.
    pub fn constraints(&self) -> &BTreeMap<PointId, Vec<HangingDofConstraint<V>>> {
        &self.constraints
    }

    /// Mutable access to the constraint map.
    pub fn constraints_mut(&mut self) -> &mut BTreeMap<PointId, Vec<HangingDofConstraint<V>>> {
        &mut self.constraints
    }

    /// Insert or update a linear constraint for a point DOF.
    pub fn insert_constraint(
        &mut self,
        point: PointId,
        index: usize,
        terms: Vec<LinearConstraintTerm<V>>,
    ) {
        let entry = self.constraints.entry(point).or_default();
        if let Some(existing) = entry.iter_mut().find(|c| c.index == index) {
            existing.terms = terms;
        } else {
            entry.push(HangingDofConstraint { index, terms });
        }
    }

    /// Remove all constraints for a point.
    pub fn clear_constraints_for_point(&mut self, point: PointId) {
        self.constraints.remove(&point);
    }

    /// Remove all constraints.
    pub fn clear_constraints(&mut self) {
        self.constraints.clear();
    }
}

/// Apply hanging node constraints to a mutable section.
pub fn apply_hanging_constraints_to_section<V, S>(
    section: &mut Section<V, S>,
    constraints: &HangingNodeConstraints<V>,
) -> Result<(), MeshSieveError>
where
    V: Clone + Default + core::ops::AddAssign + core::ops::Mul<Output = V>,
    S: Storage<V>,
{
    let mut updates: Vec<(PointId, usize, V)> = Vec::new();
    for (point, list) in constraints.constraints.iter() {
        let len = section.try_restrict(*point)?.len();
        for constraint in list {
            if constraint.index >= len {
                return Err(MeshSieveError::ConstraintIndexOutOfBounds {
                    point: *point,
                    index: constraint.index,
                    len,
                });
            }
            let mut value = V::default();
            for term in &constraint.terms {
                let source = section.try_restrict(term.point)?;
                let src_len = source.len();
                if term.index >= src_len {
                    return Err(MeshSieveError::ConstraintIndexOutOfBounds {
                        point: term.point,
                        index: term.index,
                        len: src_len,
                    });
                }
                value += source[term.index].clone() * term.weight.clone();
            }
            updates.push((*point, constraint.index, value));
        }
    }

    for (point, index, value) in updates {
        let slice = section.try_restrict_mut(point)?;
        slice[index] = value;
    }
    section.invalidate_cache();
    Ok(())
}
