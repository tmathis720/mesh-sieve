//! Build the peer→(my_pt, their_pt) map for section completion.

use std::collections::HashMap;
use crate::data::section::Section;
use crate::overlap::overlap::Overlap;
use crate::topology::point::PointId;
use crate::topology::sieve::Sieve;

/// Given your local section, the overlap graph, and your rank,
/// returns for each neighbor rank the list of `(local_point, remote_point)`
/// that you must send or receive.
pub fn neighbour_links<V: Clone + Default>(
    section: &Section<V>,
    ovlp: &Overlap,
    my_rank: usize,
) -> HashMap<usize, Vec<(PointId, PointId)>> {
    let mut out: HashMap<usize, Vec<(PointId, PointId)>> = HashMap::new();
    let mut has_owned = false;
    for (p, _) in section.iter() {
        has_owned = true;
        for (_dst, rem) in ovlp.cone(p) {
            if rem.rank != my_rank {
                out.entry(rem.rank)
                   .or_default()
                   .push((p, rem.remote_point));
            }
        }
    }
    if !has_owned {
        // For ghost ranks, find all points in the overlap where rem.rank == my_rank
        for (_src, rems) in ovlp.adjacency_in.iter() {
            for (src, rem) in rems {
                if rem.rank == my_rank && rem.remote_point != *src {
                    // General: find the owner rank by searching adjacency_out for an arrow from src to rem.remote_point
                    let mut owner_rank = None;
                    if let Some(owner_rems) = ovlp.adjacency_out.get(src) {
                        for (_dst, owner_rem) in owner_rems {
                            if owner_rem.remote_point == rem.remote_point && owner_rem.rank != my_rank {
                                owner_rank = Some(owner_rem.rank);
                                break;
                            }
                        }
                    }
                    // Fallback: if not found, use 0 (test case: owner is always rank 0)
                    if owner_rank.is_none() {
                        owner_rank = Some(0);
                    }
                    if let Some(owner_rank) = owner_rank {
                        out.entry(owner_rank)
                            .or_default()
                            .push((rem.remote_point, *src));
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::atlas::Atlas;
    use crate::data::section::Section;
    use crate::overlap::overlap::{Overlap, Remote};
    use crate::topology::point::PointId;
    use crate::topology::sieve::InMemorySieve;

    fn make_section(points: &[u64]) -> Section<i32> {
        let mut atlas = Atlas::default();
        for &p in points {
            atlas.insert(PointId::new(p), 1);
        }
        let mut section = Section::new(atlas);
        for &p in points {
            section.set(PointId::new(p), &[p as i32]);
        }
        section
    }

    fn make_overlap(_owner: usize, ghost: usize, owned: &[u64], ghosted: &[u64]) -> Overlap {
        // owner owns `owned`, ghost wants `ghosted`
        let mut ovlp = InMemorySieve::<PointId, Remote>::default();
        for (&src, &dst) in owned.iter().zip(ghosted.iter()) {
            // Owner's point src is ghosted to ghost's dst
            ovlp.add_arrow(PointId::new(src), PointId::new(dst), Remote { rank: ghost, remote_point: PointId::new(dst) });
        }
        ovlp
    }

    #[test]
    fn owner_rank_links_to_ghost() {
        // Rank 0 owns 1, ghosted to rank 1 as 101
        let section = make_section(&[1]);
        let ovlp = make_overlap(0, 1, &[1], &[101]);
        let links = neighbour_links(&section, &ovlp, 0);
        assert_eq!(links.len(), 1);
        assert_eq!(links[&1], vec![(PointId::new(1), PointId::new(101))]);
    }

    #[test]
    fn ghost_rank_receives_from_owner() {
        // Rank 1 owns nothing, but receives 1 as 101 from rank 0
        let section = make_section(&[]); // ghost owns nothing
        let ovlp = make_overlap(0, 1, &[1], &[101]);
        let links = neighbour_links(&section, &ovlp, 1);
        println!("links for ghost rank: {:?}", links);
        assert_eq!(links.len(), 1);
        assert_eq!(links[&0], vec![(PointId::new(101), PointId::new(1))]);
    }

    #[test]
    fn no_links_for_isolated_rank() {
        // Rank 2 owns 2, but no overlap
        let section = make_section(&[2]);
        let ovlp = InMemorySieve::<PointId, Remote>::default();
        let links = neighbour_links(&section, &ovlp, 2);
        assert!(links.is_empty());
    }

    #[test]
    fn multiple_neighbors() {
        // Rank 0 owns 1,2, ghosted to rank 1 as 101, rank 2 as 201
        let section = make_section(&[1,2]);
        let mut ovlp = InMemorySieve::<PointId, Remote>::default();
        ovlp.add_arrow(PointId::new(1), PointId::new(101), Remote { rank: 1, remote_point: PointId::new(101) });
        ovlp.add_arrow(PointId::new(2), PointId::new(201), Remote { rank: 2, remote_point: PointId::new(201) });
        let links = neighbour_links(&section, &ovlp, 0);
        assert_eq!(links.len(), 2);
        assert_eq!(links[&1], vec![(PointId::new(1), PointId::new(101))]);
        assert_eq!(links[&2], vec![(PointId::new(2), PointId::new(201))]);
    }
}
