//! Point label storage for topology metadata.
//!
//! Labels map `PointId` → integer tags, grouped by label name.
//! This is useful for boundary conditions, material IDs, or other
//! integer annotations on mesh points.

use std::collections::{HashMap, HashSet};

use crate::topology::point::PointId;

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
}
