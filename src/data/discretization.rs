//! Discretization metadata for basis and quadrature keyed by regions.

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

/// Basis and quadrature metadata for a region.
#[derive(Clone, Debug, PartialEq)]
pub struct DiscretizationMetadata {
    /// Basis function identifier for the region.
    pub basis: String,
    /// Optional polynomial order for the basis functions.
    pub basis_order: Option<usize>,
    /// Optional labels describing the basis shape functions.
    pub shape_functions: Vec<String>,
    /// Quadrature rule identifier for the region.
    pub quadrature: String,
    /// Optional quadrature order.
    pub quadrature_order: Option<usize>,
    /// Optional quadrature points in reference coordinates.
    pub quadrature_points: Vec<Vec<f64>>,
    /// Optional quadrature weights matching `quadrature_points`.
    pub quadrature_weights: Vec<f64>,
}

impl DiscretizationMetadata {
    /// Create a new metadata record with the provided basis and quadrature labels.
    pub fn new(basis: impl Into<String>, quadrature: impl Into<String>) -> Self {
        Self {
            basis: basis.into(),
            basis_order: None,
            shape_functions: Vec::new(),
            quadrature: quadrature.into(),
            quadrature_order: None,
            quadrature_points: Vec::new(),
            quadrature_weights: Vec::new(),
        }
    }

    /// Add basis metadata describing order and shape-function labels.
    pub fn with_basis_metadata(
        mut self,
        order: usize,
        shape_functions: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.basis_order = Some(order);
        self.shape_functions = shape_functions.into_iter().map(Into::into).collect();
        self
    }

    /// Add quadrature metadata describing order, points, and weights.
    pub fn with_quadrature_metadata(
        mut self,
        order: usize,
        points: Vec<Vec<f64>>,
        weights: Vec<f64>,
    ) -> Self {
        self.quadrature_order = Some(order);
        self.quadrature_points = points;
        self.quadrature_weights = weights;
        self
    }

    /// Returns true when explicit quadrature points and weights are available.
    pub fn has_quadrature_data(&self) -> bool {
        !self.quadrature_points.is_empty()
            && self.quadrature_points.len() == self.quadrature_weights.len()
    }
}

/// Per-field discretization metadata keyed by region selectors.
#[derive(Clone, Debug, Default)]
pub struct FieldDiscretization {
    metadata: HashMap<RegionKey, DiscretizationMetadata>,
}

impl FieldDiscretization {
    /// Create an empty field discretization.
    pub fn new() -> Self {
        Self::default()
    }

    /// Associate discretization metadata with a region selector.
    pub fn set_metadata(
        &mut self,
        region: RegionKey,
        metadata: DiscretizationMetadata,
    ) -> Option<DiscretizationMetadata> {
        self.metadata.insert(region, metadata)
    }

    /// Convenience wrapper for label-based metadata.
    pub fn set_label_metadata(
        &mut self,
        name: impl Into<String>,
        value: i32,
        metadata: DiscretizationMetadata,
    ) -> Option<DiscretizationMetadata> {
        self.set_metadata(RegionKey::label(name, value), metadata)
    }

    /// Convenience wrapper for cell-type-based metadata.
    pub fn set_cell_type_metadata(
        &mut self,
        cell_type: CellType,
        metadata: DiscretizationMetadata,
    ) -> Option<DiscretizationMetadata> {
        self.set_metadata(RegionKey::cell_type(cell_type), metadata)
    }

    /// Retrieve the metadata for a region selector, if any.
    pub fn metadata_for(&self, region: &RegionKey) -> Option<&DiscretizationMetadata> {
        self.metadata.get(region)
    }

    /// Iterate over region metadata.
    pub fn iter(&self) -> impl Iterator<Item = (&RegionKey, &DiscretizationMetadata)> {
        self.metadata.iter()
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
