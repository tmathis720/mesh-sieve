//! Closure-driven DOF layout helpers for sections and multi-sections.

use crate::data::constrained_section::{ConstraintSet, DofConstraint};
use crate::data::multi_section::MultiSection;
use crate::data::section::Section;
use crate::data::storage::Storage;
use crate::mesh_error::MeshSieveError;
use crate::topology::periodic::PeriodicMap;
use crate::topology::point::PointId;
use std::collections::{BTreeSet, HashMap};

/// Contiguous DOF layout for a point set.
#[derive(Clone, Debug, Default)]
pub struct DofLayout {
    offsets: Vec<Option<u64>>,
    dof_lengths: Vec<Option<usize>>,
    total_dofs: u64,
}

impl DofLayout {
    /// Return the offset for a point in the layout.
    pub fn offset(&self, point: PointId) -> Result<u64, MeshSieveError> {
        let idx = point_index(point)?;
        self.offsets
            .get(idx)
            .and_then(|val| *val)
            .ok_or(MeshSieveError::PointNotInAtlas(point))
    }

    /// Return the DOF count for a point in the layout.
    pub fn dof_len(&self, point: PointId) -> Result<usize, MeshSieveError> {
        let idx = point_index(point)?;
        self.dof_lengths
            .get(idx)
            .and_then(|val| *val)
            .ok_or(MeshSieveError::PointNotInAtlas(point))
    }

    /// Return the global index for a local point/DOF pair.
    pub fn global_index(&self, point: PointId, dof: usize) -> Result<u64, MeshSieveError> {
        let len = self.dof_len(point)?;
        if dof >= len {
            return Err(MeshSieveError::ConstraintIndexOutOfBounds {
                point,
                index: dof,
                len,
            });
        }
        Ok(self.offset(point)? + dof as u64)
    }

    /// Total number of DOFs in the layout.
    pub fn total_dofs(&self) -> u64 {
        self.total_dofs
    }
}

/// Compute a constraint-aware DOF count for a point slice.
pub fn constrained_dof_len<V>(
    point: PointId,
    base_len: usize,
    constraints: Option<&[DofConstraint<V>]>,
) -> Result<usize, MeshSieveError> {
    let Some(constraints) = constraints else {
        return Ok(base_len);
    };
    let mut seen = BTreeSet::new();
    for constraint in constraints {
        if constraint.index >= base_len {
            return Err(MeshSieveError::ConstraintIndexOutOfBounds {
                point,
                index: constraint.index,
                len: base_len,
            });
        }
        seen.insert(constraint.index);
    }
    Ok(base_len.saturating_sub(seen.len()))
}

/// Build a DOF layout using closures for DOF length and representative points.
pub fn build_layout_with<I, F, R>(
    points: I,
    dof_len: F,
    representative: R,
) -> Result<DofLayout, MeshSieveError>
where
    I: IntoIterator<Item = PointId>,
    F: Fn(PointId) -> Result<usize, MeshSieveError>,
    R: Fn(PointId) -> PointId,
{
    let points_vec: Vec<PointId> = points.into_iter().collect();
    let points_set: BTreeSet<PointId> = points_vec.iter().copied().collect();
    let max_id = points_vec.iter().map(|p| p.get()).max().unwrap_or(0) as usize;
    let mut layout = DofLayout {
        offsets: vec![None; max_id],
        dof_lengths: vec![None; max_id],
        total_dofs: 0,
    };

    let mut rep_lengths: HashMap<PointId, usize> = HashMap::new();
    for point in &points_vec {
        let rep = representative(*point);
        if !points_set.contains(&rep) {
            return Err(MeshSieveError::InvalidGeometry(format!(
                "representative {rep:?} missing from layout points",
            )));
        }
        let len = dof_len(*point)?;
        if let Some(existing) = rep_lengths.insert(rep, len) {
            if existing != len {
                return Err(MeshSieveError::InvalidGeometry(format!(
                    "periodic layout mismatch for {point:?}: expected {existing}, got {len}",
                )));
            }
        }
    }

    let mut rep_offsets: HashMap<PointId, u64> = HashMap::new();
    let mut total = 0u64;
    for point in &points_vec {
        let rep = representative(*point);
        if rep_offsets.contains_key(&rep) {
            continue;
        }
        let len = *rep_lengths
            .get(&rep)
            .ok_or_else(|| MeshSieveError::PointNotInAtlas(rep))? as u64;
        rep_offsets.insert(rep, total);
        total = total.saturating_add(len);
    }

    layout.total_dofs = total;
    for point in &points_vec {
        let idx = point_index(*point)?;
        if idx >= layout.offsets.len() {
            layout.offsets.resize(idx + 1, None);
            layout.dof_lengths.resize(idx + 1, None);
        }
        let rep = representative(*point);
        let offset = *rep_offsets
            .get(&rep)
            .ok_or_else(|| MeshSieveError::PointNotInAtlas(rep))?;
        let len = *rep_lengths
            .get(&rep)
            .ok_or_else(|| MeshSieveError::PointNotInAtlas(rep))?;
        layout.offsets[idx] = Some(offset);
        layout.dof_lengths[idx] = Some(len);
    }

    Ok(layout)
}

/// Build a layout for a section using optional constraints and periodic identification.
pub fn layout_for_section_with_constraints_and_periodic<V, S, C>(
    section: &Section<V, S>,
    constraints: &C,
    periodic: Option<&PeriodicMap>,
) -> Result<DofLayout, MeshSieveError>
where
    S: Storage<V>,
    C: ConstraintSet<V>,
{
    let dof_len = |point: PointId| {
        let (_, len) = section
            .atlas()
            .get(point)
            .ok_or(MeshSieveError::PointNotInAtlas(point))?;
        constrained_dof_len(point, len, constraints.constraints_for(point))
    };
    let rep = |point: PointId| {
        periodic
            .and_then(|map| map.master_of(point))
            .unwrap_or(point)
    };
    build_layout_with(section.atlas().points(), dof_len, rep)
}

/// Build a layout for a multi-section using field constraints and optional periodic mapping.
pub fn layout_for_multi_section_with_periodic<V, S>(
    section: &MultiSection<V, S>,
    periodic: Option<&PeriodicMap>,
) -> Result<DofLayout, MeshSieveError>
where
    S: Storage<V>,
{
    let dof_len = |point: PointId| multi_section_dof_len_with_constraints(section, point);
    let rep = |point: PointId| {
        periodic
            .and_then(|map| map.master_of(point))
            .unwrap_or(point)
    };
    build_layout_with(section.atlas().points(), dof_len, rep)
}

/// Compute the constraint-aware DOF length for a multi-section point.
pub fn multi_section_dof_len_with_constraints<V, S>(
    section: &MultiSection<V, S>,
    point: PointId,
) -> Result<usize, MeshSieveError>
where
    S: Storage<V>,
{
    if section.atlas().get(point).is_none() {
        return Err(MeshSieveError::PointNotInAtlas(point));
    }
    let mut total = 0usize;
    for field in section.fields() {
        let len = field
            .section()
            .atlas()
            .get(point)
            .map(|(_, len)| len)
            .unwrap_or(0);
        let constraints = field.constraints().get(&point).map(|c| c.as_slice());
        total = total.saturating_add(constrained_dof_len(point, len, constraints)?);
    }
    Ok(total)
}

/// Allocate a zero-initialized local vector for a section.
pub fn local_vector_for_section<V, S>(section: &Section<V, S>) -> Vec<V>
where
    V: Clone + Default,
    S: Storage<V>,
{
    vec![V::default(); section.atlas().total_len()]
}

/// Allocate a zero-initialized local vector for a layout.
pub fn local_vector_for_layout<V>(layout: &DofLayout) -> Vec<V>
where
    V: Clone + Default,
{
    vec![V::default(); layout.total_dofs as usize]
}

fn point_index(point: PointId) -> Result<usize, MeshSieveError> {
    point
        .get()
        .checked_sub(1)
        .ok_or(MeshSieveError::InvalidPointId)
        .map(|idx| idx as usize)
}
