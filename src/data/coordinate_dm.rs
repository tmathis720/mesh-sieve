//! Coordinate data management container.

use crate::data::coordinates::Coordinates;
use crate::data::discretization::Discretization;
use crate::data::storage::Storage;
use crate::topology::labels::LabelSet;

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
}

impl<V, St> CoordinateDM<V, St>
where
    St: Storage<V>,
{
    /// Construct a coordinate DM from coordinates only.
    pub fn new(coordinates: Coordinates<V, St>) -> Self {
        Self {
            coordinates,
            labels: None,
            discretization: None,
        }
    }
}
