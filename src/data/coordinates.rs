//! Geometry/coordinates storage for mesh points.
//!
//! Coordinates are stored in a `Section` with a fixed dimension per point. Optional
//! higher-order geometry data can be stored per entity (typically per-cell).

use crate::data::atlas::Atlas;
use crate::data::section::Section;
use crate::data::storage::Storage;
use crate::mesh_error::MeshSieveError;
use crate::topology::point::PointId;

/// Higher-order coordinate storage (e.g., per-cell geometry DOFs).
#[derive(Clone, Debug)]
pub struct HighOrderCoordinates<V, S: Storage<V>> {
    dimension: usize,
    section: Section<V, S>,
}

impl<V, S> HighOrderCoordinates<V, S>
where
    S: Storage<V>,
{
    /// Returns the spatial dimension per coordinate tuple.
    #[inline]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns a read-only reference to the underlying section.
    #[inline]
    pub fn section(&self) -> &Section<V, S> {
        &self.section
    }

    /// Returns a mutable reference to the underlying section.
    #[inline]
    pub fn section_mut(&mut self) -> &mut Section<V, S> {
        &mut self.section
    }
}

impl<V, S> HighOrderCoordinates<V, S>
where
    V: Clone + Default,
    S: Storage<V> + Clone,
{
    /// Construct a new higher-order coordinate section with a fixed dimension.
    ///
    /// Each entry length must be a non-zero multiple of `dimension`.
    pub fn try_new(dimension: usize, atlas: Atlas) -> Result<Self, MeshSieveError> {
        validate_high_order_dimension(dimension, &atlas)?;
        Ok(Self {
            dimension,
            section: Section::new(atlas),
        })
    }

    /// Wrap an existing section as higher-order coordinates, validating slice lengths.
    pub fn from_section(dimension: usize, section: Section<V, S>) -> Result<Self, MeshSieveError> {
        validate_high_order_dimension(dimension, section.atlas())?;
        Ok(Self { dimension, section })
    }
}

/// Coordinate storage with an attached dimension.
#[derive(Clone, Debug)]
pub struct Coordinates<V, S: Storage<V>> {
    dimension: usize,
    section: Section<V, S>,
    high_order: Option<HighOrderCoordinates<V, S>>,
}

impl<V, S> Coordinates<V, S>
where
    S: Storage<V>,
{
    /// Returns the spatial dimension per point.
    #[inline]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns a read-only reference to the underlying section.
    #[inline]
    pub fn section(&self) -> &Section<V, S> {
        &self.section
    }

    /// Returns a mutable reference to the underlying section.
    #[inline]
    pub fn section_mut(&mut self) -> &mut Section<V, S> {
        &mut self.section
    }

    /// Optional higher-order coordinate data (e.g., per-cell geometry DOFs).
    #[inline]
    pub fn high_order(&self) -> Option<&HighOrderCoordinates<V, S>> {
        self.high_order.as_ref()
    }

    /// Mutable view of optional higher-order coordinate data.
    #[inline]
    pub fn high_order_mut(&mut self) -> Option<&mut HighOrderCoordinates<V, S>> {
        self.high_order.as_mut()
    }

    /// Attach higher-order coordinate data, validating the dimension.
    pub fn set_high_order(
        &mut self,
        high_order: HighOrderCoordinates<V, S>,
    ) -> Result<(), MeshSieveError> {
        if high_order.dimension != self.dimension {
            return Err(MeshSieveError::InvalidGeometry(format!(
                "higher-order coordinate dimension {} does not match base dimension {}",
                high_order.dimension, self.dimension
            )));
        }
        self.high_order = Some(high_order);
        Ok(())
    }

    /// Consumes the wrapper and returns the underlying section.
    #[inline]
    pub fn into_section(self) -> Section<V, S> {
        self.section
    }

    /// Read-only view of the coordinate slice for a point `p`.
    #[inline]
    pub fn try_restrict(&self, p: PointId) -> Result<&[V], MeshSieveError> {
        self.section.try_restrict(p)
    }

    /// Mutable view of the coordinate slice for a point `p`.
    #[inline]
    pub fn try_restrict_mut(&mut self, p: PointId) -> Result<&mut [V], MeshSieveError> {
        self.section.try_restrict_mut(p)
    }
}

impl<V, S> Coordinates<V, S>
where
    V: Clone + Default,
    S: Storage<V> + Clone,
{
    /// Construct a new coordinates section with a fixed dimension.
    ///
    /// The provided `atlas` must store slices of length `dimension` for all points.
    pub fn try_new(dimension: usize, atlas: Atlas) -> Result<Self, MeshSieveError> {
        validate_dimension(dimension, &atlas)?;
        Ok(Self {
            dimension,
            section: Section::new(atlas),
            high_order: None,
        })
    }

    /// Wrap an existing section as coordinates, validating slice lengths.
    pub fn from_section(dimension: usize, section: Section<V, S>) -> Result<Self, MeshSieveError> {
        validate_dimension(dimension, section.atlas())?;
        Ok(Self {
            dimension,
            section,
            high_order: None,
        })
    }

    /// Adds a new point with the configured coordinate dimension.
    pub fn try_add_point(&mut self, p: PointId) -> Result<(), MeshSieveError> {
        self.section.try_add_point(p, self.dimension)
    }
}

fn validate_dimension(dimension: usize, atlas: &Atlas) -> Result<(), MeshSieveError> {
    if dimension == 0 {
        return Err(MeshSieveError::ZeroLengthSlice);
    }
    for (point, (_offset, len)) in atlas.iter_entries() {
        if len != dimension {
            return Err(MeshSieveError::SliceLengthMismatch {
                point,
                expected: dimension,
                found: len,
            });
        }
    }
    Ok(())
}

fn validate_high_order_dimension(dimension: usize, atlas: &Atlas) -> Result<(), MeshSieveError> {
    if dimension == 0 {
        return Err(MeshSieveError::ZeroLengthSlice);
    }
    for (point, (_offset, len)) in atlas.iter_entries() {
        if len == 0 || len % dimension != 0 {
            return Err(MeshSieveError::SliceLengthMismatch {
                point,
                expected: dimension,
                found: len,
            });
        }
    }
    Ok(())
}
