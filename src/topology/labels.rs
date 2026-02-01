//! Point label storage for topology metadata.
//!
//! Labels map `PointId` → integer tags, grouped by label name.
//! This is useful for boundary conditions, material IDs, or other
//! integer annotations on mesh points.

use std::collections::{HashMap, HashSet};

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
    pub fn values(&self, name: &str) -> Vec<i32> {
        let mut values: Vec<i32> = self
            .labels
            .get(name)
            .map_or_else(Vec::new, |map| map.values().copied().collect());
        values.sort_unstable();
        values.dedup();
        values
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

/// Expand all labels to include the closure of their points.
pub fn complete_label_set<S>(sieve: &S, labels: &LabelSet) -> LabelSet
where
    S: Sieve<Point = PointId>,
{
    let mut out = labels.clone();
    let mut names: HashSet<String> = HashSet::new();
    for (name, _, _) in labels.iter() {
        names.insert(name.to_string());
    }
    for name in names {
        for value in labels.values(&name) {
            out = complete_label_value(sieve, &out, &name, value);
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
