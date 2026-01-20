//! Discretization metadata for field layouts keyed by regions.

use crate::topology::cell_type::CellType;
use std::collections::{BTreeMap, HashMap};

/// Region selector for discretization metadata.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum RegionKey {
    /// Match points tagged by a label name/value pair.
    Label { name: String, value: i32 },
    /// Match points with a specific cell type.
    CellType(CellType),
}

impl RegionKey {
    /// Construct a label-based region selector.
    pub fn label(name: impl Into<String>, value: i32) -> Self {
        Self::Label {
            name: name.into(),
            value,
        }
    }

    /// Construct a cell-type-based region selector.
    pub fn cell_type(cell_type: CellType) -> Self {
        Self::CellType(cell_type)
    }
}

/// DOF layout metadata for a region.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DofLayout {
    /// Number of degrees of freedom per point in the region.
    pub dofs_per_point: usize,
}

impl DofLayout {
    /// Create a new layout with `dofs_per_point` entries per point.
    pub fn new(dofs_per_point: usize) -> Self {
        Self { dofs_per_point }
    }
}

/// Per-field discretization metadata keyed by region selectors.
#[derive(Clone, Debug, Default)]
pub struct FieldDiscretization {
    layouts: HashMap<RegionKey, DofLayout>,
}

impl FieldDiscretization {
    /// Create an empty field discretization.
    pub fn new() -> Self {
        Self::default()
    }

    /// Associate a DOF layout with a region selector.
    pub fn set_layout(&mut self, region: RegionKey, layout: DofLayout) -> Option<DofLayout> {
        self.layouts.insert(region, layout)
    }

    /// Convenience wrapper for label-based layouts.
    pub fn set_label_layout(
        &mut self,
        name: impl Into<String>,
        value: i32,
        layout: DofLayout,
    ) -> Option<DofLayout> {
        self.set_layout(RegionKey::label(name, value), layout)
    }

    /// Convenience wrapper for cell-type-based layouts.
    pub fn set_cell_type_layout(
        &mut self,
        cell_type: CellType,
        layout: DofLayout,
    ) -> Option<DofLayout> {
        self.set_layout(RegionKey::cell_type(cell_type), layout)
    }

    /// Retrieve the layout for a region selector, if any.
    pub fn layout_for(&self, region: &RegionKey) -> Option<&DofLayout> {
        self.layouts.get(region)
    }

    /// Iterate over region layouts.
    pub fn iter(&self) -> impl Iterator<Item = (&RegionKey, &DofLayout)> {
        self.layouts.iter()
    }
}

/// Discretization metadata for multiple named fields.
#[derive(Clone, Debug, Default)]
pub struct Discretization {
    fields: BTreeMap<String, FieldDiscretization>,
}

impl Discretization {
    /// Create an empty discretization metadata container.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a field discretization by name.
    pub fn insert_field(
        &mut self,
        name: impl Into<String>,
        field: FieldDiscretization,
    ) -> Option<FieldDiscretization> {
        self.fields.insert(name.into(), field)
    }

    /// Retrieve a field discretization by name.
    pub fn field(&self, name: &str) -> Option<&FieldDiscretization> {
        self.fields.get(name)
    }

    /// Retrieve or create a field discretization by name.
    pub fn field_mut(&mut self, name: &str) -> &mut FieldDiscretization {
        self.fields.entry(name.to_string()).or_default()
    }

    /// Iterate over all fields.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &FieldDiscretization)> {
        self.fields
            .iter()
            .map(|(name, field)| (name.as_str(), field))
    }
}
