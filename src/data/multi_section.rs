//! MultiSection: group multiple field sections with shared point offsets.

use crate::data::atlas::Atlas;
use crate::data::constrained_section::{DofConstraint, apply_constraints_to_section};
use crate::data::section::Section;
use crate::data::storage::Storage;
use crate::mesh_error::MeshSieveError;
use crate::topology::point::PointId;
use std::collections::{BTreeMap, HashSet};

/// A named field section with field-specific constraints.
#[derive(Clone, Debug)]
pub struct FieldSection<V, S: Storage<V>> {
    name: String,
    section: Section<V, S>,
    constraints: BTreeMap<PointId, Vec<DofConstraint<V>>>,
}

impl<V, S> FieldSection<V, S>
where
    S: Storage<V>,
{
    /// Create a new field section with a name and data section.
    pub fn new(name: impl Into<String>, section: Section<V, S>) -> Self {
        Self {
            name: name.into(),
            section,
            constraints: BTreeMap::new(),
        }
    }

    /// Field name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Access the underlying section.
    pub fn section(&self) -> &Section<V, S> {
        &self.section
    }

    /// Mutable access to the underlying section.
    pub fn section_mut(&mut self) -> &mut Section<V, S> {
        &mut self.section
    }

    /// Access the constraint map for this field.
    pub fn constraints(&self) -> &BTreeMap<PointId, Vec<DofConstraint<V>>> {
        &self.constraints
    }

    /// Mutable access to the constraint map for this field.
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

    /// Apply all stored constraints to the underlying section.
    pub fn apply_constraints(&mut self) -> Result<(), MeshSieveError>
    where
        V: Clone,
    {
        apply_constraints_to_section(&mut self.section, &self.constraints)
    }
}

/// Group multiple field sections with consistent per-point offsets.
#[derive(Clone, Debug)]
pub struct MultiSection<V, S: Storage<V>> {
    atlas: Atlas,
    fields: Vec<FieldSection<V, S>>,
}

impl<V, S> MultiSection<V, S>
where
    S: Storage<V>,
{
    /// Construct a multi-section from field sections.
    ///
    /// All fields must share the same point set. Per-point total DOFs are the
    /// sum of field DOFs, in field order.
    pub fn new(fields: Vec<FieldSection<V, S>>) -> Result<Self, MeshSieveError> {
        let atlas = Self::build_atlas(&fields)?;
        Ok(Self { atlas, fields })
    }

    /// Borrow the combined atlas.
    pub fn atlas(&self) -> &Atlas {
        &self.atlas
    }

    /// Number of fields.
    pub fn field_count(&self) -> usize {
        self.fields.len()
    }

    /// Borrow all fields.
    pub fn fields(&self) -> &[FieldSection<V, S>] {
        &self.fields
    }

    /// Mutable access to all fields.
    pub fn fields_mut(&mut self) -> &mut [FieldSection<V, S>] {
        &mut self.fields
    }

    /// Borrow a field by index.
    pub fn field(&self, index: usize) -> Option<&FieldSection<V, S>> {
        self.fields.get(index)
    }

    /// Mutable access to a field by index.
    pub fn field_mut(&mut self, index: usize) -> Option<&mut FieldSection<V, S>> {
        self.fields.get_mut(index)
    }

    /// Borrow a field by name.
    pub fn field_by_name(&self, name: &str) -> Option<&FieldSection<V, S>> {
        self.fields.iter().find(|field| field.name == name)
    }

    /// Mutable access to a field by name.
    pub fn field_by_name_mut(&mut self, name: &str) -> Option<&mut FieldSection<V, S>> {
        self.fields.iter_mut().find(|field| field.name == name)
    }

    /// Total DOF count at point `p` across all fields.
    pub fn dof(&self, p: PointId) -> Result<usize, MeshSieveError> {
        let (_, len) = self
            .atlas
            .get(p)
            .ok_or(MeshSieveError::PointNotInAtlas(p))?;
        Ok(len)
    }

    /// Offset for point `p` in the combined layout.
    pub fn offset(&self, p: PointId) -> Result<usize, MeshSieveError> {
        let (offset, _) = self
            .atlas
            .get(p)
            .ok_or(MeshSieveError::PointNotInAtlas(p))?;
        Ok(offset)
    }

    /// DOF count for a specific field at point `p`.
    pub fn field_dof(&self, p: PointId, field: usize) -> Result<usize, MeshSieveError> {
        let section = self
            .fields
            .get(field)
            .ok_or(MeshSieveError::SectionAccess {
                point: p,
                source: Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "field index out of bounds",
                )),
            })?
            .section();
        if self.atlas.get(p).is_none() {
            return Err(MeshSieveError::PointNotInAtlas(p));
        }
        Ok(section.atlas().get(p).map(|(_, len)| len).unwrap_or(0))
    }

    /// Field offset for point `p` in the combined layout.
    ///
    /// This mirrors PetscSection field offsets: `offset(p) + sum_{i < field} dof_i(p)`.
    pub fn field_offset(&self, p: PointId, field: usize) -> Result<usize, MeshSieveError> {
        let base = self.offset(p)?;
        let mut offset = base;
        for idx in 0..field {
            offset += self.field_dof(p, idx)?;
        }
        Ok(offset)
    }

    /// (offset, dof) pair for a field at point `p`.
    pub fn field_span(&self, p: PointId, field: usize) -> Result<(usize, usize), MeshSieveError> {
        Ok((self.field_offset(p, field)?, self.field_dof(p, field)?))
    }

    /// DOF count for a field at point `p`, by field name.
    pub fn field_dof_by_name(&self, p: PointId, name: &str) -> Result<usize, MeshSieveError> {
        let field = self
            .fields
            .iter()
            .position(|field| field.name == name)
            .ok_or(MeshSieveError::SectionAccess {
                point: p,
                source: Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "field name not found",
                )),
            })?;
        self.field_dof(p, field)
    }

    /// Field offset for point `p`, by field name.
    pub fn field_offset_by_name(&self, p: PointId, name: &str) -> Result<usize, MeshSieveError> {
        let field = self
            .fields
            .iter()
            .position(|field| field.name == name)
            .ok_or(MeshSieveError::SectionAccess {
                point: p,
                source: Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "field name not found",
                )),
            })?;
        self.field_offset(p, field)
    }

    /// Apply constraints for all fields.
    pub fn apply_constraints(&mut self) -> Result<(), MeshSieveError>
    where
        V: Clone,
    {
        for field in &mut self.fields {
            field.apply_constraints()?;
        }
        Ok(())
    }

    fn build_atlas(fields: &[FieldSection<V, S>]) -> Result<Atlas, MeshSieveError> {
        if fields.is_empty() {
            return Ok(Atlas::default());
        }
        let mut atlas = Atlas::default();
        let mut points = Vec::new();
        let mut seen = HashSet::new();
        for field in fields {
            for point in field.section().atlas().points() {
                if seen.insert(point) {
                    points.push(point);
                }
            }
        }

        for point in points {
            let mut total = 0usize;
            for field in fields {
                if let Some((_, len)) = field.section().atlas().get(point) {
                    total = total.saturating_add(len);
                }
            }
            atlas.try_insert(point, total)?;
        }
        Ok(atlas)
    }
}

impl<V, S> MultiSection<V, S>
where
    V: Clone,
    S: Storage<V>,
{
    /// Gather all fields into a flat buffer in combined atlas order.
    pub fn gather_in_order(&self) -> Result<Vec<V>, MeshSieveError> {
        let mut out = Vec::with_capacity(self.atlas.total_len());
        for point in self.atlas.points() {
            for field in &self.fields {
                if field.section().atlas().get(point).is_some() {
                    let slice = field.section().try_restrict(point)?;
                    out.extend_from_slice(slice);
                }
            }
        }
        if out.len() != self.atlas.total_len() {
            return Err(MeshSieveError::ScatterLengthMismatch {
                expected: self.atlas.total_len(),
                found: out.len(),
            });
        }
        Ok(out)
    }

    /// Scatter values from a flat buffer in combined atlas order.
    pub fn try_scatter_in_order(&mut self, buf: &[V]) -> Result<(), MeshSieveError> {
        let total = self.atlas.total_len();
        if buf.len() != total {
            return Err(MeshSieveError::ScatterLengthMismatch {
                expected: total,
                found: buf.len(),
            });
        }
        let points: Vec<PointId> = self.atlas.points().collect();
        let mut cursor = 0usize;
        for point in points {
            for field in &mut self.fields {
                if field.section().atlas().get(point).is_some() {
                    let len = {
                        let slice = field.section().try_restrict(point)?;
                        slice.len()
                    };
                    let end = cursor
                        .checked_add(len)
                        .ok_or(MeshSieveError::ScatterChunkMismatch {
                            offset: cursor,
                            len,
                        })?;
                    let chunk = buf.get(cursor..end).ok_or(MeshSieveError::ScatterChunkMismatch {
                        offset: cursor,
                        len,
                    })?;
                    field.section_mut().try_set(point, chunk)?;
                    cursor = end;
                }
            }
        }
        if cursor != total {
            return Err(MeshSieveError::ScatterLengthMismatch {
                expected: total,
                found: cursor,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{FieldSection, MultiSection};
    use crate::data::atlas::Atlas;
    use crate::data::section::Section;
    use crate::data::storage::VecStorage;
    use crate::topology::point::PointId;

    #[test]
    fn field_offsets_match_stacked_layout() {
        let mut atlas_a = Atlas::default();
        let p1 = PointId::new(1).unwrap();
        let p2 = PointId::new(2).unwrap();
        atlas_a.try_insert(p1, 2).unwrap();
        atlas_a.try_insert(p2, 1).unwrap();
        let section_a = Section::<f64, VecStorage<f64>>::new(atlas_a);

        let mut atlas_b = Atlas::default();
        atlas_b.try_insert(p1, 1).unwrap();
        atlas_b.try_insert(p2, 3).unwrap();
        let section_b = Section::<f64, VecStorage<f64>>::new(atlas_b);

        let fields = vec![
            FieldSection::new("a", section_a),
            FieldSection::new("b", section_b),
        ];
        let multi = MultiSection::new(fields).unwrap();

        assert_eq!(multi.offset(p1).unwrap(), 0);
        assert_eq!(multi.offset(p2).unwrap(), 3);
        assert_eq!(multi.field_offset(p1, 0).unwrap(), 0);
        assert_eq!(multi.field_offset(p1, 1).unwrap(), 2);
        assert_eq!(multi.field_offset(p2, 0).unwrap(), 3);
        assert_eq!(multi.field_offset(p2, 1).unwrap(), 4);
        assert_eq!(multi.field_dof(p2, 1).unwrap(), 3);
    }

    #[test]
    fn mixed_fields_have_independent_dofs() {
        let p1 = PointId::new(1).unwrap();
        let p2 = PointId::new(2).unwrap();
        let p3 = PointId::new(3).unwrap();

        let mut velocity_atlas = Atlas::default();
        velocity_atlas.try_insert(p1, 3).unwrap();
        velocity_atlas.try_insert(p2, 2).unwrap();
        let velocity = Section::<f64, VecStorage<f64>>::new(velocity_atlas);

        let mut pressure_atlas = Atlas::default();
        pressure_atlas.try_insert(p2, 1).unwrap();
        pressure_atlas.try_insert(p3, 1).unwrap();
        let pressure = Section::<f64, VecStorage<f64>>::new(pressure_atlas);

        let fields = vec![
            FieldSection::new("velocity", velocity),
            FieldSection::new("pressure", pressure),
        ];
        let multi = MultiSection::new(fields).unwrap();

        assert_eq!(multi.offset(p1).unwrap(), 0);
        assert_eq!(multi.offset(p2).unwrap(), 3);
        assert_eq!(multi.offset(p3).unwrap(), 6);

        assert_eq!(multi.field_dof(p1, 0).unwrap(), 3);
        assert_eq!(multi.field_dof(p1, 1).unwrap(), 0);
        assert_eq!(multi.field_offset(p1, 1).unwrap(), 3);

        assert_eq!(multi.field_dof(p2, 0).unwrap(), 2);
        assert_eq!(multi.field_dof(p2, 1).unwrap(), 1);
        assert_eq!(multi.field_offset(p2, 1).unwrap(), 5);

        assert_eq!(multi.field_dof(p3, 0).unwrap(), 0);
        assert_eq!(multi.field_dof_by_name(p3, "pressure").unwrap(), 1);
        assert_eq!(multi.field_offset_by_name(p3, "pressure").unwrap(), 6);
    }
}
