//! Point label storage for topology metadata.
//!
//! Labels map `PointId` → integer tags, grouped by label name.
//! This is useful for boundary conditions, material IDs, or other
//! integer annotations on mesh points.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::ops::RangeBounds;

use crate::algs::submesh::{SubmeshMaps, SubmeshSelection, extract_by_label};
use crate::data::storage::Storage;
use crate::io::MeshData;
use crate::mesh_error::MeshSieveError;
use crate::topology::cell_type::CellType;
use crate::topology::point::PointId;
use crate::topology::sieve::{InMemorySieve, Sieve};

/// Named integer labels for mesh points.
#[derive(Clone, Debug, Default)]
pub struct LabelSet {
    labels: HashMap<String, HashMap<PointId, i32>>,
}

impl LabelSet {
    /// Creates an empty label set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Assigns `value` for `point` under label `name`.
    ///
    /// Returns the previous value, if any.
    pub fn set_label(&mut self, point: PointId, name: &str, value: i32) -> Option<i32> {
        self.labels
            .entry(name.to_string())
            .or_default()
            .insert(point, value)
    }

    /// Returns the label value for `point` under `name`.
    pub fn get_label(&self, point: PointId, name: &str) -> Option<i32> {
        self.labels
            .get(name)
            .and_then(|map| map.get(&point).copied())
    }

    /// Returns all points with label `name == value`.
    pub fn points_with_label<'a>(
        &'a self,
        name: &'a str,
        value: i32,
    ) -> impl Iterator<Item = PointId> + 'a {
        self.labels.get(name).into_iter().flat_map(move |map| {
            map.iter()
                .filter_map(move |(&point, &label_value)| (label_value == value).then_some(point))
        })
    }

    /// Returns the number of points with label `name == value`.
    pub fn stratum_size(&self, name: &str, value: i32) -> usize {
        self.labels.get(name).map_or(0, |map| {
            map.values()
                .filter(|&&label_value| label_value == value)
                .count()
        })
    }

    /// Returns all points with label `name == value` in deterministic order.
    pub fn stratum_points(&self, name: &str, value: i32) -> Vec<PointId> {
        let mut points: Vec<_> = self.points_with_label(name, value).collect();
        points.sort_unstable();
        points
    }

    /// Returns all distinct values stored for label `name`, sorted ascending.
    pub fn stratum_values(&self, name: &str) -> Vec<i32> {
        let mut values: Vec<i32> = self
            .labels
            .get(name)
            .map_or_else(Vec::new, |map| map.values().copied().collect());
        values.sort_unstable();
        values.dedup();
        values
    }

    /// Backwards-compatible alias for [`LabelSet::stratum_values`].
    pub fn values(&self, name: &str) -> Vec<i32> {
        self.stratum_values(name)
    }

    /// Builds a value-indexed view of the label stratum for efficient range queries.
    pub fn strata_by_value(&self, name: &str) -> BTreeMap<i32, Vec<PointId>> {
        let Some(map) = self.labels.get(name) else {
            return BTreeMap::new();
        };
        let mut strata: BTreeMap<i32, Vec<PointId>> = BTreeMap::new();
        for (&point, &value) in map {
            strata.entry(value).or_default().push(point);
        }
        for points in strata.values_mut() {
            points.sort_unstable();
        }
        strata
    }

    /// Returns points with `name` values inside the provided range.
    ///
    /// Points are ordered by value, then by point ID.
    pub fn stratum_points_in_range<R>(&self, name: &str, range: R) -> Vec<PointId>
    where
        R: RangeBounds<i32>,
    {
        let strata = self.strata_by_value(name);
        let mut points = Vec::new();
        for (_, stratum_points) in strata.range(range) {
            points.extend(stratum_points.iter().copied());
        }
        points
    }

    /// Returns the union of two label strata, sorted deterministically.
    pub fn stratum_union(&self, other: &LabelSet, name: &str, value: i32) -> Vec<PointId> {
        let mut union: HashSet<PointId> = self.points_with_label(name, value).collect();
        union.extend(other.points_with_label(name, value));
        let mut points: Vec<_> = union.into_iter().collect();
        points.sort_unstable();
        points
    }

    /// Returns the intersection of two label strata, sorted deterministically.
    pub fn stratum_intersection(&self, other: &LabelSet, name: &str, value: i32) -> Vec<PointId> {
        let other_points: HashSet<PointId> = other.points_with_label(name, value).collect();
        let mut points: Vec<_> = self
            .points_with_label(name, value)
            .filter(|point| other_points.contains(point))
            .collect();
        points.sort_unstable();
        points
    }

    /// Returns the difference between two label strata, sorted deterministically.
    ///
    /// Points returned are those in `self` that are not in `other`.
    pub fn stratum_difference(&self, other: &LabelSet, name: &str, value: i32) -> Vec<PointId> {
        let other_points: HashSet<PointId> = other.points_with_label(name, value).collect();
        let mut points: Vec<_> = self
            .points_with_label(name, value)
            .filter(|point| !other_points.contains(point))
            .collect();
        points.sort_unstable();
        points
    }

    /// Removes all points with label `name == value`.
    ///
    /// Returns the number of removed points.
    pub fn clear_label_value(&mut self, name: &str, value: i32) -> usize {
        let Some(map) = self.labels.get_mut(name) else {
            return 0;
        };

        let before = map.len();
        map.retain(|_, label_value| *label_value != value);
        let removed = before - map.len();
        if map.is_empty() {
            self.labels.remove(name);
        }
        removed
    }

    /// Returns a new label set containing only labels on the provided points.
    pub fn filtered_to_points<I>(&self, points: I) -> Self
    where
        I: IntoIterator<Item = PointId>,
    {
        let keep: HashSet<PointId> = points.into_iter().collect();
        if keep.is_empty() {
            return Self::default();
        }

        let mut out = LabelSet::new();
        for (name, values) in &self.labels {
            for (&point, &value) in values {
                if keep.contains(&point) {
                    out.set_label(point, name, value);
                }
            }
        }
        out
    }

    /// Returns true when the label set has no entries.
    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    /// Remove all label entries for the provided points.
    pub fn clear_points<I>(&mut self, points: I)
    where
        I: IntoIterator<Item = PointId>,
    {
        let targets: HashSet<PointId> = points.into_iter().collect();
        if targets.is_empty() {
            return;
        }
        let mut empty_labels = Vec::new();
        for (name, map) in &mut self.labels {
            map.retain(|point, _| !targets.contains(point));
            if map.is_empty() {
                empty_labels.push(name.clone());
            }
        }
        for name in empty_labels {
            self.labels.remove(&name);
        }
    }

    /// Iterate over all labels as `(name, point, value)`.
    pub fn iter(&self) -> impl Iterator<Item = (&str, PointId, i32)> + '_ {
        self.labels.iter().flat_map(|(name, map)| {
            map.iter()
                .map(move |(&point, &value)| (name.as_str(), point, value))
        })
    }
}

/// Expand a label stratum to include the closure of its points.
///
/// This mirrors the behavior of DMPlexLabelComplete for a single label value by applying
/// the label to every point in the closure of each labeled point.
pub fn complete_label_value<S>(sieve: &S, labels: &LabelSet, name: &str, value: i32) -> LabelSet
where
    S: Sieve<Point = PointId>,
{
    propagate_label_value_closure(sieve, labels, name, value)
}

/// Expand all labels to include the closure of their points.
pub fn complete_label_set<S>(sieve: &S, labels: &LabelSet) -> LabelSet
where
    S: Sieve<Point = PointId>,
{
    propagate_label_set_closure(sieve, labels)
}

/// Propagate a label stratum through the closure of its points.
pub fn propagate_label_value_closure<S>(
    sieve: &S,
    labels: &LabelSet,
    name: &str,
    value: i32,
) -> LabelSet
where
    S: Sieve<Point = PointId>,
{
    let mut out = labels.clone();
    let seeds: Vec<PointId> = labels.points_with_label(name, value).collect();
    if seeds.is_empty() {
        return out;
    }
    for point in sieve.closure_iter(seeds) {
        out.set_label(point, name, value);
    }
    out
}

/// Propagate a label stratum through the star of its points.
pub fn propagate_label_value_star<S>(
    sieve: &S,
    labels: &LabelSet,
    name: &str,
    value: i32,
) -> LabelSet
where
    S: Sieve<Point = PointId>,
{
    let mut out = labels.clone();
    let seeds: Vec<PointId> = labels.points_with_label(name, value).collect();
    if seeds.is_empty() {
        return out;
    }
    for point in sieve.star_iter(seeds) {
        out.set_label(point, name, value);
    }
    out
}

/// Propagate all labels through the closure of their points.
pub fn propagate_label_set_closure<S>(sieve: &S, labels: &LabelSet) -> LabelSet
where
    S: Sieve<Point = PointId>,
{
    let mut out = labels.clone();
    let mut names: HashSet<String> = HashSet::new();
    for (name, _, _) in labels.iter() {
        names.insert(name.to_string());
    }
    for name in names {
        for value in labels.stratum_values(&name) {
            out = propagate_label_value_closure(sieve, &out, &name, value);
        }
    }
    out
}

/// Propagate all labels through the star of their points.
pub fn propagate_label_set_star<S>(sieve: &S, labels: &LabelSet) -> LabelSet
where
    S: Sieve<Point = PointId>,
{
    let mut out = labels.clone();
    let mut names: HashSet<String> = HashSet::new();
    for (name, _, _) in labels.iter() {
        names.insert(name.to_string());
    }
    for name in names {
        for value in labels.stratum_values(&name) {
            out = propagate_label_value_star(sieve, &out, &name, value);
        }
    }
    out
}

/// Extract a submesh from points tagged by a label, including the full closure.
pub fn extract_submesh_from_label<S, V, St, CtSt>(
    mesh: &MeshData<S, V, St, CtSt>,
    labels: &LabelSet,
    label_name: &str,
    label_value: i32,
) -> Result<
    (
        MeshData<InMemorySieve<PointId, S::Payload>, V, St, CtSt>,
        SubmeshMaps,
    ),
    MeshSieveError,
>
where
    S: Sieve<Point = PointId>,
    S::Payload: Clone,
    V: Clone + Default,
    St: Storage<V> + Clone,
    CtSt: Storage<CellType> + Clone,
{
    let completed = complete_label_value(&mesh.sieve, labels, label_name, label_value);
    extract_by_label(
        mesh,
        &completed,
        label_name,
        label_value,
        SubmeshSelection::FullClosure,
    )
}
