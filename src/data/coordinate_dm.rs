//! Coordinate data management container.

use crate::data::atlas::Atlas;
use crate::data::coordinates::{Coordinates, HighOrderCoordinates};
use crate::data::discretization::Discretization;
use crate::data::section::Section;
use crate::data::storage::Storage;
use crate::mesh_error::MeshSieveError;
use crate::topology::labels::LabelSet;
use crate::topology::point::PointId;
use std::collections::{HashMap, HashSet};

/// Coordinate-specific numbering for coordinate points.
#[derive(Clone, Debug, Default)]
pub struct CoordinateNumbering {
    order: Vec<PointId>,
    index: HashMap<PointId, usize>,
}

impl CoordinateNumbering {
    /// Build a numbering from an iterator of points.
    pub fn from_points<I>(points: I) -> Self
    where
        I: IntoIterator<Item = PointId>,
    {
        let mut order = Vec::new();
        let mut index = HashMap::new();
        for (idx, point) in points.into_iter().enumerate() {
            index.insert(point, idx);
            order.push(point);
        }
        Self { order, index }
    }

    /// Build a numbering while validating uniqueness.
    pub fn try_from_points<I>(points: I) -> Result<Self, MeshSieveError>
    where
        I: IntoIterator<Item = PointId>,
    {
        let mut order = Vec::new();
        let mut index = HashMap::new();
        for (idx, point) in points.into_iter().enumerate() {
            if index.insert(point, idx).is_some() {
                return Err(MeshSieveError::InvalidPermutation(format!(
                    "duplicate coordinate point {point:?}"
                )));
            }
            order.push(point);
        }
        Ok(Self { order, index })
    }

    /// Ordered list of points tracked by the numbering.
    pub fn points(&self) -> &[PointId] {
        &self.order
    }

    /// Lookup the numbering index for a point.
    pub fn index(&self, point: PointId) -> Option<usize> {
        self.index.get(&point).copied()
    }
}

/// Coordinate data management wrapper decoupled from the main mesh.
#[derive(Clone, Debug)]
pub struct CoordinateDM<V, St>
where
    St: Storage<V>,
{
    /// Coordinate storage for mesh points.
    pub coordinates: Coordinates<V, St>,
    /// Optional labels associated with coordinate points.
    pub labels: Option<LabelSet>,
    /// Optional discretization metadata for coordinate fields.
    pub discretization: Option<Discretization>,
    /// Coordinate-specific numbering for points.
    pub numbering: CoordinateNumbering,
}

impl<V, St> CoordinateDM<V, St>
where
    St: Storage<V>,
{
    /// Construct a coordinate DM from coordinates only.
    pub fn new(coordinates: Coordinates<V, St>) -> Self {
        let numbering = CoordinateNumbering::from_points(coordinates.section().atlas().points());
        Self {
            coordinates,
            labels: None,
            discretization: None,
            numbering,
        }
    }
}

impl<V, St> CoordinateDM<V, St>
where
    V: Clone + Default,
    St: Storage<V> + Clone,
{
    /// Update a single point's coordinate values.
    pub fn try_update_point(&mut self, point: PointId, values: &[V]) -> Result<(), MeshSieveError> {
        self.coordinates.section_mut().try_set(point, values)
    }

    /// Update multiple points' coordinate values.
    pub fn try_update_points<I>(&mut self, updates: I) -> Result<(), MeshSieveError>
    where
        I: IntoIterator<Item = (PointId, Vec<V>)>,
    {
        for (point, values) in updates {
            self.coordinates
                .section_mut()
                .try_set(point, values.as_slice())?;
        }
        Ok(())
    }

    /// Reorder coordinate points using a permutation of existing points.
    pub fn reorder_points(&self, permutation: &[PointId]) -> Result<Self, MeshSieveError> {
        let points: Vec<PointId> = self.coordinates.section().atlas().points().collect();
        let (old_to_new, new_to_old) = build_renumber_map(&points, permutation)?;
        let coordinates = remap_coordinates(&self.coordinates, &old_to_new, &new_to_old)?;
        let labels = match &self.labels {
            Some(labels) => {
                let out = remap_labels(labels, &old_to_new)?;
                (!out.is_empty()).then_some(out)
            }
            None => None,
        };
        let numbering =
            CoordinateNumbering::try_from_points(coordinates.section().atlas().points())?;
        Ok(Self {
            coordinates,
            labels,
            discretization: self.discretization.clone(),
            numbering,
        })
    }

    /// Rename coordinate points using an explicit mapping.
    pub fn rename_points(
        &self,
        mapping: &HashMap<PointId, PointId>,
    ) -> Result<Self, MeshSieveError> {
        let points: Vec<PointId> = self.coordinates.section().atlas().points().collect();
        validate_rename_mapping(&points, mapping)?;
        let mut old_to_new = HashMap::with_capacity(points.len());
        for &old in &points {
            let new = mapping.get(&old).copied().unwrap_or(old);
            old_to_new.insert(old, new);
        }
        let coordinates = remap_coordinates(&self.coordinates, &old_to_new, &points)?;
        let labels = match &self.labels {
            Some(labels) => {
                let out = remap_labels(labels, &old_to_new)?;
                (!out.is_empty()).then_some(out)
            }
            None => None,
        };
        let numbering =
            CoordinateNumbering::try_from_points(coordinates.section().atlas().points())?;
        Ok(Self {
            coordinates,
            labels,
            discretization: self.discretization.clone(),
            numbering,
        })
    }
}

fn validate_rename_mapping(
    points: &[PointId],
    mapping: &HashMap<PointId, PointId>,
) -> Result<(), MeshSieveError> {
    let expected: HashSet<PointId> = points.iter().copied().collect();
    for key in mapping.keys() {
        if !expected.contains(key) {
            return Err(MeshSieveError::InvalidPermutation(format!(
                "unknown coordinate point {key:?}"
            )));
        }
    }
    let mut seen = HashSet::with_capacity(points.len());
    for &old in points {
        let new = mapping.get(&old).copied().unwrap_or(old);
        if !seen.insert(new) {
            return Err(MeshSieveError::InvalidPermutation(format!(
                "duplicate renamed point {new:?}"
            )));
        }
    }
    Ok(())
}

fn build_renumber_map(
    points: &[PointId],
    permutation: &[PointId],
) -> Result<(HashMap<PointId, PointId>, Vec<PointId>), MeshSieveError> {
    let total = points.len();
    if permutation.len() != total {
        return Err(MeshSieveError::InvalidPermutation(format!(
            "expected {total} coordinate points, got {}",
            permutation.len()
        )));
    }
    let mut expected: HashSet<PointId> = points.iter().copied().collect();
    let mut old_to_new = HashMap::with_capacity(permutation.len());
    let mut new_to_old = Vec::with_capacity(permutation.len());
    for (idx, &old) in permutation.iter().enumerate() {
        if !expected.remove(&old) {
            return Err(MeshSieveError::InvalidPermutation(format!(
                "duplicate or unknown point {old:?}"
            )));
        }
        let new = PointId::new((idx + 1) as u64)?;
        if old_to_new.insert(old, new).is_some() {
            return Err(MeshSieveError::InvalidPermutation(format!(
                "duplicate point {old:?}"
            )));
        }
        new_to_old.push(old);
    }
    if !expected.is_empty() {
        return Err(MeshSieveError::InvalidPermutation(format!(
            "missing points: {expected:?}"
        )));
    }
    Ok((old_to_new, new_to_old))
}

fn remap_section<V, S>(
    section: &Section<V, S>,
    old_to_new: &HashMap<PointId, PointId>,
    new_to_old: &[PointId],
) -> Result<Section<V, S>, MeshSieveError>
where
    V: Clone + Default,
    S: Storage<V> + Clone,
{
    let mut atlas = Atlas::default();
    for &old in new_to_old {
        if let Some((_, len)) = section.atlas().get(old) {
            let new = old_to_new[&old];
            atlas.try_insert(new, len)?;
        }
    }
    let mut out = Section::new(atlas);
    for &old in new_to_old {
        if section.atlas().contains(old) {
            let new = old_to_new[&old];
            let data = section.try_restrict(old)?;
            out.try_set(new, data)?;
        }
    }
    Ok(out)
}

fn remap_coordinates<V, S>(
    coords: &Coordinates<V, S>,
    old_to_new: &HashMap<PointId, PointId>,
    new_to_old: &[PointId],
) -> Result<Coordinates<V, S>, MeshSieveError>
where
    V: Clone + Default,
    S: Storage<V> + Clone,
{
    let section = remap_section(coords.section(), old_to_new, new_to_old)?;
    let mut out = Coordinates::from_section(
        coords.topological_dimension(),
        coords.embedding_dimension(),
        section,
    )?;
    if let Some(high_order) = coords.high_order() {
        let ho_section = remap_section(high_order.section(), old_to_new, new_to_old)?;
        let ho = HighOrderCoordinates::from_section(high_order.dimension(), ho_section)?;
        out.set_high_order(ho)?;
    }
    Ok(out)
}

fn remap_labels(
    labels: &LabelSet,
    old_to_new: &HashMap<PointId, PointId>,
) -> Result<LabelSet, MeshSieveError> {
    let mut out = LabelSet::new();
    for (name, point, value) in labels.iter() {
        let new_point = old_to_new.get(&point).copied().ok_or_else(|| {
            MeshSieveError::InvalidPermutation(format!("missing label point {point:?}"))
        })?;
        out.set_label(new_point, name, value);
    }
    Ok(out)
}
