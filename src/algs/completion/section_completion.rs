//! High-level `complete_section` orchestration: neighbour_links → exchange_sizes → exchange_data.
//!
//! The public API now accepts explicit [`CommTag`]s for deterministic, collision-free
//! communication epochs.

use std::collections::{BTreeSet, HashSet};

use crate::algs::communicator::{CommTag, Communicator, SectionCommTags};
use crate::algs::completion::{
    data_exchange, neighbour_links::neighbour_links, size_exchange::exchange_sizes_symmetric,
};
use crate::data::section::Section;
use crate::data::storage::Storage;
use crate::mesh_error::MeshSieveError;
use crate::overlap::delta::ValueDelta;
use crate::overlap::overlap::Overlap;

/// Complete a section using explicit communication tags.
pub fn complete_section_with_tags<V, S, D, C>(
    section: &mut Section<V, S>,
    overlap: &Overlap,
    comm: &C,
    my_rank: usize,
    tags: SectionCommTags,
) -> Result<(), MeshSieveError>
where
    V: Clone + Default + Send + PartialEq + 'static,
    S: Storage<V>,
    D: ValueDelta<V> + Send + Sync + 'static,
    D::Part: bytemuck::Pod + Default,
    C: Communicator + Sync,
{
    #[cfg(any(
        debug_assertions,
        feature = "strict-invariants",
        feature = "check-invariants"
    ))]
    overlap.validate_invariants()?;

    // Fast-path: if there are no neighbors (excluding self), nothing to exchange.
    // Mirrors the behavior in sieve completion: treat as a successful no-op.
    {
        use std::collections::BTreeSet;
        let mut nb: BTreeSet<usize> = overlap.neighbor_ranks().collect();
        nb.remove(&my_rank);
        if nb.is_empty() {
            return Ok(());
        }
    }

    // 1) discover which points each neighbor needs
    let links = neighbour_links::<V, S>(section, overlap, my_rank).map_err(|e| {
        MeshSieveError::CommError {
            neighbor: my_rank,
            source: format!("neighbour_links failed: {e}").into(),
        }
    })?;

    // 2) Build true neighbor set (both outgoing and incoming), deterministically ordered
    let mut all: BTreeSet<usize> = overlap.neighbor_ranks().collect();
    all.extend(links.keys().copied());
    all.remove(&my_rank);
    let all_neighbors: HashSet<usize> = all.into_iter().collect();

    // 3) exchange the item counts
    let counts =
        exchange_sizes_symmetric(&links, comm, tags.sizes, &all_neighbors).map_err(|e| {
            MeshSieveError::CommError {
                neighbor: my_rank,
                source: format!("exchange_sizes_symmetric failed: {e}").into(),
            }
        })?;

    // 4) exchange the actual data parts & fuse into our section
    data_exchange::exchange_data_symmetric::<V, S, D, C>(
        &links,
        &counts,
        comm,
        tags.data.as_u16(),
        section,
        &all_neighbors,
    )?;

    #[cfg(debug_assertions)]
    comm.barrier_result()?;

    Ok(())
}

/// Convenience wrapper using a legacy default tag (0xBEEF).
pub fn complete_section<V, S, D, C>(
    section: &mut Section<V, S>,
    overlap: &Overlap,
    comm: &C,
    my_rank: usize,
) -> Result<(), MeshSieveError>
where
    V: Clone + Default + Send + PartialEq + 'static,
    S: Storage<V>,
    D: ValueDelta<V> + Send + Sync + 'static,
    D::Part: bytemuck::Pod + Default,
    C: Communicator + Sync,
{
    // Legacy default tags keep ranks in sync for thread-local comms; use
    // complete_section_with_tags for concurrent or coordinated epochs.
    let tags = SectionCommTags::from_base(CommTag::new(0xBEEF));
    complete_section_with_tags::<V, S, D, C>(section, overlap, comm, my_rank, tags)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algs::communicator::NoComm;
    use crate::data::atlas::Atlas;
    use crate::data::storage::VecStorage;
    use crate::topology::point::PointId;

    use crate::overlap::delta::CopyDelta;

    // Helper to build a section with points set to their ID as value
    fn make_section(points: &[u64]) -> Section<i32, VecStorage<i32>> {
        let mut atlas = Atlas::default();
        for &p in points {
            atlas.try_insert(PointId::new(p).unwrap(), 1).unwrap();
        }
        let mut s = Section::<i32, VecStorage<i32>>::new(atlas);
        for &p in points {
            s.try_set(PointId::new(p).unwrap(), &[p as i32]).unwrap();
        }
        s
    }

    #[test]
    fn unresolved_mapping_errors() {
        let mut section = make_section(&[1]);
        let mut ovlp = Overlap::new();
        // structural link without resolving the remote point
        ovlp.add_link_structural_one(PointId::new(1).unwrap(), 1);
        let comm = NoComm;
        let tags = SectionCommTags::from_base(CommTag::new(0x4100));
        let res = complete_section_with_tags::<i32, VecStorage<i32>, CopyDelta, _>(
            &mut section,
            &ovlp,
            &comm,
            0,
            tags,
        );
        assert!(matches!(
            res,
            Err(MeshSieveError::CommError { neighbor: 0, .. })
        ));
    }

    #[test]
    fn no_neighbors_is_noop() {
        let mut section = make_section(&[]);
        let ovlp = Overlap::new();
        let comm = NoComm;
        let tags = SectionCommTags::from_base(CommTag::new(0x4200));
        let res = complete_section_with_tags::<i32, VecStorage<i32>, CopyDelta, _>(
            &mut section,
            &ovlp,
            &comm,
            0,
            tags,
        );
        // With no neighbors, completion is a no-op and succeeds.
        assert!(res.is_ok());
    }
}
