//! Geometry/coordinates storage for mesh points.
//!
//! Coordinates are stored in a `Section` with fixed topological and embedding
//! dimensions per point. Optional
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

/// Velocity storage aligned with coordinate dimensions.
#[derive(Clone, Debug)]
pub struct MeshVelocity<V, S: Storage<V>> {
    dimension: usize,
    section: Section<V, S>,
}

impl<V, S> MeshVelocity<V, S>
where
    S: Storage<V>,
{
    /// Returns the spatial dimension per velocity tuple.
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

    /// Read-only view of the velocity slice for a point `p`.
    #[inline]
    pub fn try_restrict(&self, p: PointId) -> Result<&[V], MeshSieveError> {
        self.section.try_restrict(p)
    }

    /// Mutable view of the velocity slice for a point `p`.
    #[inline]
    pub fn try_restrict_mut(&mut self, p: PointId) -> Result<&mut [V], MeshSieveError> {
        self.section.try_restrict_mut(p)
    }
}

impl<V, S> MeshVelocity<V, S>
where
    V: Clone + Default,
    S: Storage<V> + Clone,
{
    /// Construct a new velocity section with a fixed dimension.
    ///
    /// The provided `atlas` must store slices of length `dimension` for all points.
    pub fn try_new(dimension: usize, atlas: Atlas) -> Result<Self, MeshSieveError> {
        validate_dimension(dimension, &atlas)?;
        Ok(Self {
            dimension,
            section: Section::new(atlas),
        })
    }

    /// Wrap an existing section as velocity data, validating slice lengths.
    pub fn from_section(dimension: usize, section: Section<V, S>) -> Result<Self, MeshSieveError> {
        validate_dimension(dimension, section.atlas())?;
        Ok(Self { dimension, section })
    }

    /// Adds a new point with the configured velocity dimension.
    pub fn try_add_point(&mut self, p: PointId) -> Result<(), MeshSieveError> {
        self.section.try_add_point(p, self.dimension)
    }
}

/// Coordinate storage with an attached topological and embedding dimension.
#[derive(Clone, Debug)]
pub struct Coordinates<V, S: Storage<V>> {
    topological_dimension: usize,
    embedding_dimension: usize,
    section: Section<V, S>,
    high_order: Option<HighOrderCoordinates<V, S>>,
}

impl<V, S> Coordinates<V, S>
where
    S: Storage<V>,
{
    /// Returns the embedding dimension per point.
    #[inline]
    pub fn dimension(&self) -> usize {
        self.embedding_dimension
    }

    /// Returns the topological dimension of the mesh.
    #[inline]
    pub fn topological_dimension(&self) -> usize {
        self.topological_dimension
    }

    /// Returns the embedding dimension per point.
    #[inline]
    pub fn embedding_dimension(&self) -> usize {
        self.embedding_dimension
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
        if high_order.dimension != self.embedding_dimension {
            return Err(MeshSieveError::InvalidGeometry(format!(
                "higher-order coordinate dimension {} does not match base dimension {}",
                high_order.dimension, self.embedding_dimension
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

impl<S> Coordinates<f64, S>
where
    S: Storage<f64>,
{
    /// Advance coordinates using a velocity field and timestep.
    pub fn advance_with_velocity<St>(
        &mut self,
        velocity: &MeshVelocity<f64, St>,
        dt: f64,
    ) -> Result<(), MeshSieveError>
    where
        St: Storage<f64>,
    {
        let dim = self.embedding_dimension;
        let points: Vec<PointId> = self.section.atlas().points().collect();
        for point in points {
            let vel = velocity.try_restrict(point)?;
            if vel.len() != dim {
                return Err(MeshSieveError::SliceLengthMismatch {
                    point,
                    expected: dim,
                    found: vel.len(),
                });
            }
            let coord = self.try_restrict_mut(point)?;
            if coord.len() != dim {
                return Err(MeshSieveError::SliceLengthMismatch {
                    point,
                    expected: dim,
                    found: coord.len(),
                });
            }
            for (coord_value, vel_value) in coord.iter_mut().zip(vel.iter()) {
                *coord_value += dt * vel_value;
            }
        }
        Ok(())
    }
}

impl<V, S> Coordinates<V, S>
where
    V: Clone + Default,
    S: Storage<V> + Clone,
{
    /// Construct a new coordinates section with fixed topological and embedding dimensions.
    ///
    /// The provided `atlas` must store slices of length `embedding_dimension`
    /// for all points.
    pub fn try_new(
        topological_dimension: usize,
        embedding_dimension: usize,
        atlas: Atlas,
    ) -> Result<Self, MeshSieveError> {
        validate_coordinate_dimensions(topological_dimension, embedding_dimension, &atlas)?;
        Ok(Self {
            topological_dimension,
            embedding_dimension,
            section: Section::new(atlas),
            high_order: None,
        })
    }

    /// Wrap an existing section as coordinates, validating slice lengths.
    pub fn from_section(
        topological_dimension: usize,
        embedding_dimension: usize,
        section: Section<V, S>,
    ) -> Result<Self, MeshSieveError> {
        validate_coordinate_dimensions(
            topological_dimension,
            embedding_dimension,
            section.atlas(),
        )?;
        Ok(Self {
            topological_dimension,
            embedding_dimension,
            section,
            high_order: None,
        })
    }

    /// Adds a new point with the configured coordinate dimension.
    pub fn try_add_point(&mut self, p: PointId) -> Result<(), MeshSieveError> {
        self.section.try_add_point(p, self.embedding_dimension)
    }
}

fn validate_coordinate_dimensions(
    topological_dimension: usize,
    embedding_dimension: usize,
    atlas: &Atlas,
) -> Result<(), MeshSieveError> {
    if embedding_dimension == 0 {
        return Err(MeshSieveError::ZeroLengthSlice);
    }
    if topological_dimension > embedding_dimension {
        return Err(MeshSieveError::InvalidGeometry(format!(
            "topological dimension {topological_dimension} exceeds embedding dimension {embedding_dimension}"
        )));
    }
    for (point, (_offset, len)) in atlas.iter_entries() {
        if len != embedding_dimension {
            return Err(MeshSieveError::SliceLengthMismatch {
                point,
                expected: embedding_dimension,
                found: len,
            });
        }
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::{Coordinates, MeshVelocity};
    use crate::data::atlas::Atlas;
    use crate::data::storage::VecStorage;
    use crate::topology::point::PointId;

    #[test]
    fn advance_coordinates_over_multiple_steps() {
        let mut atlas = Atlas::default();
        let p1 = PointId::new(1).unwrap();
        let p2 = PointId::new(2).unwrap();
        atlas.try_insert(p1, 3).unwrap();
        atlas.try_insert(p2, 3).unwrap();

        let mut coords = Coordinates::<f64, VecStorage<f64>>::try_new(3, 3, atlas.clone()).unwrap();
        let mut velocity = MeshVelocity::<f64, VecStorage<f64>>::try_new(3, atlas).unwrap();

        coords.section_mut().try_set(p1, &[0.0, 0.0, 0.0]).unwrap();
        coords.section_mut().try_set(p2, &[1.0, 1.0, 1.0]).unwrap();
        velocity
            .section_mut()
            .try_set(p1, &[1.0, 0.0, -1.0])
            .unwrap();
        velocity
            .section_mut()
            .try_set(p2, &[0.5, -0.5, 1.0])
            .unwrap();

        let dt = 0.25;
        for _ in 0..4 {
            coords.advance_with_velocity(&velocity, dt).unwrap();
        }

        assert_eq!(coords.try_restrict(p1).unwrap(), &[1.0, 0.0, -1.0]);
        assert_eq!(coords.try_restrict(p2).unwrap(), &[1.5, 0.5, 2.0]);
    }
}
