//! Geometry/coordinates storage for mesh points.
//!
//! Coordinates are stored in a `Section` with a fixed dimension per point.

use crate::data::atlas::Atlas;
use crate::data::section::Section;
use crate::data::storage::Storage;
use crate::mesh_error::MeshSieveError;
use crate::topology::point::PointId;

/// Coordinate storage with an attached dimension.
#[derive(Clone, Debug)]
pub struct Coordinates<V, S: Storage<V>> {
    dimension: usize,
    section: Section<V, S>,
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
        })
    }

    /// Wrap an existing section as coordinates, validating slice lengths.
    pub fn from_section(dimension: usize, section: Section<V, S>) -> Result<Self, MeshSieveError> {
        validate_dimension(dimension, section.atlas())?;
        Ok(Self { dimension, section })
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
