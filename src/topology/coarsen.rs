//! Coarsening helpers for merging mesh entities with transfer maps.
//!
//! Coarsening operates by replacing a set of fine entities with a coarse entity
//! and emitting a transfer map suitable for assembling data from fine to coarse.

use crate::mesh_error::MeshSieveError;
use crate::topology::arrow::Polarity;
use crate::topology::point::PointId;
use crate::topology::sieve::{InMemorySieve, Sieve};
use std::collections::HashSet;

/// Mapping from coarse entities to fine entities with orientation hints.
pub type CoarseningMap = Vec<(PointId, Vec<(PointId, Polarity)>)>;

/// Description of a coarse entity and the fine entities it replaces.
#[derive(Clone, Debug)]
pub struct CoarsenEntity {
    /// The coarse entity (cell/edge/etc.) created by coarsening.
    pub coarse_point: PointId,
    /// Fine entities that are merged into the coarse entity.
    pub fine_points: Vec<(PointId, Polarity)>,
    /// The cone (connectivity) for the coarse entity.
    pub cone: Vec<PointId>,
}

/// Output of [`coarsen_topology`], including the coarsened topology and transfer map.
#[derive(Clone, Debug)]
pub struct CoarsenedTopology {
    /// Coarsened topology (entities → cone points).
    pub sieve: InMemorySieve<PointId, ()>,
    /// Mapping from coarse entity to the fine entities it replaces.
    pub transfer_map: CoarseningMap,
}

/// Coarsen a topology by merging fine entities into coarse entities.
///
/// The provided [`CoarsenEntity`] entries define the coarse entity cones
/// (e.g., cells → vertices or edges → vertices) and the fine entities they
/// replace. The returned transfer map can be used to assemble data from fine
/// entities into coarse entities.
pub fn coarsen_topology(
    fine: &impl Sieve<Point = PointId>,
    entities: &[CoarsenEntity],
) -> Result<CoarsenedTopology, MeshSieveError> {
    let fine_points: HashSet<_> = fine.points().collect();
    let mut coarse = InMemorySieve::<PointId, ()>::default();
    let mut transfer_map = Vec::with_capacity(entities.len());

    for entity in entities {
        for (fine_point, _) in &entity.fine_points {
            if !fine_points.contains(fine_point) {
                return Err(MeshSieveError::UnknownPoint(format!(
                    "{fine_point:?}"
                )));
            }
        }
        for cone_point in &entity.cone {
            if !fine_points.contains(cone_point) {
                return Err(MeshSieveError::UnknownPoint(format!(
                    "{cone_point:?}"
                )));
            }
            coarse.add_arrow(entity.coarse_point, *cone_point, ());
        }
        transfer_map.push((entity.coarse_point, entity.fine_points.clone()));
    }

    Ok(CoarsenedTopology {
        sieve: coarse,
        transfer_map,
    })
}
