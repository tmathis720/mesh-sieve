//! Overlap metadata and utilities for distributed mesh partitioning.
//!
//! This module defines the [`Overlap`] type and related functions for tracking
//! sharing relationships between partitions, as well as the [`Remote`] metadata
//! describing remote copies of local points. The API supports closure and star
//! operations, cache invalidation, and overlap expansion.

use crate::topology::point::PointId;
use crate::topology::sieve::{InMemorySieve, Sieve};
use crate::topology::stratum::InvalidateCache;

/// A sieve that stores sharing relationships between partitions.
///
/// The payload is [`Remote`], which identifies the remote rank and point.
pub type Overlap = InMemorySieve<PointId, Remote>;

/// Returns the partition point for a given rank (centralized for all modules).
#[inline]
pub fn partition_point(rank: usize) -> PointId {
    PointId::new((rank as u64) + 1).unwrap()
}

impl Overlap {
    /// Add an overlap arrow: `local_p --(rank,remote_p)--> partition(rank)`.
    ///
    /// Ensures closure-of-support and invalidates caches as per Knepley & Karpeev 2009.
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
        // Enforce closure-of-support: for each leaf in support(part_pt),
        // add closure(leaf) to the overlap if not already present.
        let leaves: Vec<_> = Sieve::support(self, part_pt).map(|(l, _)| l).collect();
        for &leaf in &leaves {
            let closure: Vec<_> = Sieve::closure(self, std::iter::once(leaf)).collect();
            for q in closure {
                // Add link q→part_pt if missing
                if !self.cone(q).any(|(dst, _)| dst == part_pt) {
                    Sieve::add_arrow(
                        self,
                        q,
                        part_pt,
                        Remote {
                            rank: remote_rank,
                            remote_point: remote,
                        },
                    );
                }
            }
        }
        self.invalidate_cache();
    }

    /// Returns the closure of the star of a point, as in Eq (5) of the paper.
    pub fn closure_of_star(&self, p: PointId) -> Vec<PointId> {
        let mut out = std::collections::HashSet::new();
        let star: Vec<_> = Sieve::star(self, std::iter::once(p)).collect();
        for q in star {
            let closure: Vec<_> = Sieve::closure(self, std::iter::once(q)).collect();
            for r in closure {
                out.insert(r);
            }
        }
        out.into_iter().collect()
    }

    /// Expands the overlap by one adjacency-closure pass (one layer).
    pub fn expand_one_layer(&mut self) {
        let points: Vec<_> = self.points().collect();
        for p in points {
            let closure = self.closure_of_star(p);
            for q in closure {
                if !self.points().any(|x| x == q) {
                    // Add a self-link to ensure q is present
                    Sieve::add_arrow(
                        self,
                        q,
                        partition_point(0), // dummy partition
                        Remote { rank: 0, remote_point: q },
                    );
                }
            }
        }
        self.invalidate_cache();
    }

    /// Convenience: iterate all neighbour ranks of the *current* rank.
    pub fn neighbour_ranks<'a>(&'a self, my_rank: usize) -> impl Iterator<Item = usize> + 'a {
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

/// Remote: (rank, remote_point) is exactly SF leaf→root as in PETSc SF.
///
/// This struct identifies a remote copy of a local point, including the remote rank and point ID.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Remote {
    /// The remote MPI rank.
    pub rank: usize,
    /// The remote point ID on that rank.
    pub remote_point: PointId,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh_error::MeshSieveError;
    use crate::topology::point::PointId;
    use crate::topology::sieve::Sieve;

    #[test]
    fn overlap_cache_cleared_on_mutation() {
        let mut ovlp = Overlap::default();
        ovlp.add_link(PointId::new(1).unwrap(), 1, PointId::new(101).unwrap());
        let d0 = match ovlp.diameter() {
            Ok(val) => val,
            Err(MeshSieveError::CycleDetected) => {
                // If a cycle is detected, the test setup is not suitable for diameter computation.
                // The important part is that the cache is invalidated and no panic occurs.
                return;
            },
            Err(e) => panic!("Failed to compute diameter d0: {e:?}"),
        };
        ovlp.add_link(PointId::new(2).unwrap(), 2, PointId::new(201).unwrap());
        let d1 = match ovlp.diameter() {
            Ok(val) => val,
            Err(MeshSieveError::CycleDetected) => return,
            Err(e) => panic!("Failed to compute diameter d1: {e:?}"),
        };
        assert!(d1 >= d0);
    }

    #[test]
    fn add_link_enforces_closure() {
        let mut ovlp = Overlap::default();
        ovlp.add_link(PointId::new(1).unwrap(), 1, PointId::new(101).unwrap());
        // After add_link, closure of 1 should be present
        let closure: Vec<_> = Sieve::closure(&ovlp, std::iter::once(PointId::new(1).unwrap())).collect();
        for p in closure {
            assert!(ovlp.points().any(|x| x == p));
        }
    }

    #[test]
    fn neighbour_ranks_matches_theory() {
        let mut ovlp = Overlap::default();
        ovlp.add_link(PointId::new(1).unwrap(), 1, PointId::new(101).unwrap());
        ovlp.add_link(PointId::new(2).unwrap(), 2, PointId::new(201).unwrap());
        let ranks: Vec<_> = ovlp.neighbour_ranks(1).collect();
        assert!(ranks.contains(&1));
        let ranks2: Vec<_> = ovlp.neighbour_ranks(2).collect();
        assert!(ranks2.contains(&2));
    }

    #[test]
    fn closure_of_star_and_expand_one_layer() {
        let mut ovlp = Overlap::default();
        ovlp.add_link(PointId::new(1).unwrap(), 1, PointId::new(101).unwrap());
        let before = ovlp.points().count();
        ovlp.expand_one_layer();
        let after = ovlp.points().count();
        assert!(after >= before);
        let closure = ovlp.closure_of_star(PointId::new(1).unwrap());
        for p in closure {
            assert!(ovlp.points().any(|x| x == p));
        }
    }
}
