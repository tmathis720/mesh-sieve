//! ConstrainedSection: field data with per-point constrained DOF values.

use crate::data::section::{FallibleMap, Section};
use crate::data::storage::Storage;
use crate::mesh_error::MeshSieveError;
use crate::topology::cache::InvalidateCache;
use crate::topology::point::PointId;
use std::collections::BTreeMap;

/// A single constrained degree of freedom within a point slice.
#[derive(Clone, Debug, PartialEq)]
pub struct DofConstraint<V> {
    /// Index into the point slice.
    pub index: usize,
    /// Value to enforce at that index.
    pub value: V,
}

impl<V> DofConstraint<V> {
    /// Create a new DOF constraint at `index` with `value`.
    pub fn new(index: usize, value: V) -> Self {
        Self { index, value }
    }
}

/// Read-only access to a per-point constraint store.
pub trait ConstraintSet<V> {
    /// Returns an iterator over points with constraints.
    fn points(&self) -> Box<dyn Iterator<Item = PointId> + '_>;

    /// Returns the constraints for a point, if any.
    fn constraints_for(&self, point: PointId) -> Option<&[DofConstraint<V>]>;
}

/// Section wrapper that carries per-point constraints on DOF indices.
#[derive(Clone, Debug)]
pub struct ConstrainedSection<V, S: Storage<V>> {
    section: Section<V, S>,
    constraints: BTreeMap<PointId, Vec<DofConstraint<V>>>,
}

impl<V, S> ConstrainedSection<V, S>
where
    S: Storage<V>,
{
    /// Create a constrained section from an existing [`Section`].
    pub fn new(section: Section<V, S>) -> Self {
        Self {
            section,
            constraints: BTreeMap::new(),
        }
    }

    /// Access the underlying section.
    pub fn section(&self) -> &Section<V, S> {
        &self.section
    }

    /// Mutable access to the underlying section.
    pub fn section_mut(&mut self) -> &mut Section<V, S> {
        &mut self.section
    }

    /// Take ownership of the underlying section.
    pub fn into_section(self) -> Section<V, S> {
        self.section
    }

    /// Borrow the constraint map.
    pub fn constraints(&self) -> &BTreeMap<PointId, Vec<DofConstraint<V>>> {
        &self.constraints
    }

    /// Mutable access to the constraint map.
    pub fn constraints_mut(&mut self) -> &mut BTreeMap<PointId, Vec<DofConstraint<V>>> {
        &mut self.constraints
    }

    /// Insert or update a single constraint for a point.
    pub fn insert_constraint(
        &mut self,
        point: PointId,
        index: usize,
        value: V,
    ) -> Result<(), MeshSieveError> {
        let len = self.section.try_restrict(point)?.len();
        if index >= len {
            return Err(MeshSieveError::ConstraintIndexOutOfBounds { point, index, len });
        }
        let entry = self.constraints.entry(point).or_default();
        if let Some(existing) = entry.iter_mut().find(|c| c.index == index) {
            existing.value = value;
        } else {
            entry.push(DofConstraint { index, value });
        }
        Ok(())
    }

    /// Remove all constraints for a point.
    pub fn clear_constraints_for_point(&mut self, point: PointId) {
        self.constraints.remove(&point);
    }

    /// Remove all constraints.
    pub fn clear_constraints(&mut self) {
        self.constraints.clear();
    }

    /// Remove a single constraint by index, if present.
    pub fn remove_constraint(&mut self, point: PointId, index: usize) -> Option<DofConstraint<V>> {
        let entry = self.constraints.get_mut(&point)?;
        let pos = entry.iter().position(|c| c.index == index)?;
        Some(entry.remove(pos))
    }

    /// Apply constraints for a single point to the underlying section.
    pub fn apply_constraints_for_point(&mut self, point: PointId) -> Result<(), MeshSieveError>
    where
        V: Clone,
    {
        if let Some(constraints) = self.constraints.get(&point) {
            apply_constraints_to_slice(&mut self.section, point, constraints)?;
            self.section.invalidate_cache();
        }
        Ok(())
    }

    /// Apply all stored constraints to the underlying section.
    pub fn apply_constraints(&mut self) -> Result<(), MeshSieveError>
    where
        V: Clone,
    {
        apply_constraints_to_section(&mut self.section, &self.constraints)
    }
}

impl<V, S> ConstraintSet<V> for ConstrainedSection<V, S>
where
    S: Storage<V>,
{
    fn points(&self) -> Box<dyn Iterator<Item = PointId> + '_> {
        Box::new(self.constraints.keys().copied())
    }

    fn constraints_for(&self, point: PointId) -> Option<&[DofConstraint<V>]> {
        self.constraints
            .get(&point)
            .map(|constraints| constraints.as_slice())
    }
}

impl<V> ConstraintSet<V> for BTreeMap<PointId, Vec<DofConstraint<V>>> {
    fn points(&self) -> Box<dyn Iterator<Item = PointId> + '_> {
        Box::new(self.keys().copied())
    }

    fn constraints_for(&self, point: PointId) -> Option<&[DofConstraint<V>]> {
        self.get(&point).map(|constraints| constraints.as_slice())
    }
}

impl<V, S> FallibleMap<V> for ConstrainedSection<V, S>
where
    S: Storage<V>,
{
    #[inline]
    fn try_get(&self, p: PointId) -> Result<&[V], MeshSieveError> {
        self.section.try_restrict(p)
    }

    #[inline]
    fn try_get_mut(&mut self, p: PointId) -> Result<&mut [V], MeshSieveError> {
        self.section.try_restrict_mut(p)
    }
}

impl<V, S> InvalidateCache for ConstrainedSection<V, S>
where
    S: Storage<V>,
{
    fn invalidate_cache(&mut self) {
        self.section.invalidate_cache();
    }
}

/// Apply constraints from a [`ConstraintSet`] to a mutable section.
pub fn apply_constraints_to_section<V, S, C>(
    section: &mut Section<V, S>,
    constraints: &C,
) -> Result<(), MeshSieveError>
where
    V: Clone,
    S: Storage<V>,
    C: ConstraintSet<V>,
{
    for point in constraints.points() {
        if let Some(list) = constraints.constraints_for(point) {
            apply_constraints_to_slice(section, point, list)?;
        }
    }
    section.invalidate_cache();
    Ok(())
}

fn apply_constraints_to_slice<V, S>(
    section: &mut Section<V, S>,
    point: PointId,
    constraints: &[DofConstraint<V>],
) -> Result<(), MeshSieveError>
where
    V: Clone,
    S: Storage<V>,
{
    let slice = section.try_restrict_mut(point)?;
    let len = slice.len();
    for constraint in constraints {
        if constraint.index >= len {
            return Err(MeshSieveError::ConstraintIndexOutOfBounds {
                point,
                index: constraint.index,
                len,
            });
        }
        slice[constraint.index] = constraint.value.clone();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::atlas::Atlas;
    use crate::data::storage::VecStorage;

    #[test]
    fn apply_constraints_sets_values() {
        let mut atlas = Atlas::default();
        let p = PointId::new(1).unwrap();
        atlas.try_insert(p, 3).unwrap();
        let mut section = Section::<i32, VecStorage<i32>>::new(atlas);
        section.try_set(p, &[1, 2, 3]).unwrap();
        let mut constrained = ConstrainedSection::new(section);
        constrained.insert_constraint(p, 0, 10).unwrap();
        constrained.insert_constraint(p, 2, 30).unwrap();
        constrained.apply_constraints().unwrap();
        assert_eq!(constrained.section().try_restrict(p).unwrap(), &[10, 2, 30]);
    }
}
