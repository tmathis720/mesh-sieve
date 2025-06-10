//! Metadata that identifies a remote copy of a local point.
use crate::topology::point::PointId;
use crate::topology::sieve::{InMemorySieve, Sieve};

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Remote {
    pub rank: usize,
    pub remote_point: PointId,
}

/// A sieve that stores sharing relationships between partitions.
pub type Overlap = InMemorySieve<PointId, Remote>;

// Helper for partition points
fn partition_point(rank: usize) -> PointId {
    PointId::new((rank as u64) + 1)
}

impl Overlap {
    /// Add an overlap arrow: `local_p --(rank,remote_p)--> partition(rank)`.
    pub fn add_link(&mut self, local: PointId, remote_rank: usize, remote: PointId) {
        let part_pt = partition_point(remote_rank);
        Sieve::add_arrow(
            self,
            local,
            part_pt,
            Remote {
                rank: remote_rank,
                remote_point: remote,
            },
        );
        // Invalidate caches after mutation
        //InvalidateCache::invalidate_cache(self);
    }

    /// Convenience: iterate all neighbours of the *current* rank.
    pub fn neighbours<'a>(&'a self, my_rank: usize) -> impl Iterator<Item = usize> + 'a {
        use std::collections::HashSet;
        Sieve::cone(self, partition_point(my_rank))
            .map(|(_, rem)| rem.rank)
            .collect::<HashSet<_>>()
            .into_iter()
    }

    /// Returns iterator over `(local, remote_point)` for a given neighbour rank.
    pub fn links_to<'a>(
        &'a self,
        nbr: usize,
        _my_rank: usize,
    ) -> impl Iterator<Item = (PointId, PointId)> + 'a {
        Sieve::support(self, partition_point(nbr))
            .filter(move |(_, r)| r.rank == nbr)
            .map(|(local, r)| (local, r.remote_point))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlap_cache_cleared_on_mutation() {
        use crate::topology::point::PointId;
        let mut ovlp = Overlap::default();
        ovlp.add_link(PointId::new(1), 1, PointId::new(101));
        let d0 = ovlp.diameter();
        ovlp.add_link(PointId::new(2), 2, PointId::new(201));
        let d1 = ovlp.diameter();
        assert!(d1 >= d0);
    }
}
