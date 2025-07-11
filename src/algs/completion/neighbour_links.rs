//! Build the peer→(my_pt, their_pt) map for section completion.

use crate::data::section::Section;
use crate::overlap::overlap::Overlap;
use crate::topology::point::PointId;
use crate::topology::sieve::Sieve;
use std::collections::HashMap;

/// Given your local section, the overlap graph, and your rank,
/// returns for each neighbor rank the list of `(local_point, remote_point)`
/// that you must send or receive.
pub fn neighbour_links<V: Clone + Default + PartialEq>(
    section: &Section<V>,
    ovlp: &Overlap,
    my_rank: usize,
) -> HashMap<usize, Vec<(PointId, PointId)>> {
    let default_val = V::default();
    let mut out: HashMap<usize, Vec<(PointId, PointId)>> = HashMap::new();
    let mut has_owned = false;
    for (p, vals) in section.iter() {
        if vals.iter().all(|v| *v == default_val) {
            continue;
        }
        has_owned = true;
        for (_dst, rem) in ovlp.cone(p) {
            if rem.rank != my_rank {
                out.entry(rem.rank).or_default().push((p, rem.remote_point));
            }
        }
    }
    if !has_owned {
        for (&_src, rems) in &ovlp.adjacency_in {
            for (orig_src, rem) in rems {
                if rem.rank == my_rank && rem.remote_point != *orig_src {
                    let owner = ovlp.adjacency_out
                        .get(orig_src)
                        .and_then(|outs| {
                            outs.iter()
                                .find(|(_, orem)| {
                                    orem.remote_point == rem.remote_point && orem.rank != my_rank
                                })
                                .map(|(_, orem)| orem.rank)
                        })
                        .unwrap_or(0);
                    out.entry(owner)
                        .or_default()
                        .push((rem.remote_point, *orig_src));
                }
            }
        }
    }
    // Optional: sort for deterministic order
    // for vec in out.values_mut() {
    //     vec.sort_unstable_by_key(|&(l, r)| (l.get(), r.get()));
    // }
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
            atlas.try_insert(PointId::new(p).unwrap(), 1).expect("Failed to insert point into atlas");
        }
        let mut section = Section::new(atlas);
        for &p in points {
            section.try_set(PointId::new(p).unwrap(), &[p as i32]).expect("Failed to set section value");
        }
        section
    }

    fn make_overlap(_owner: usize, ghost: usize, owned: &[u64], ghosted: &[u64]) -> Overlap {
        // owner owns `owned`, ghost wants `ghosted`
        let mut ovlp = InMemorySieve::<PointId, Remote>::default();
        for (&src, &dst) in owned.iter().zip(ghosted.iter()) {
            // Owner's point src is ghosted to ghost's dst
            ovlp.add_arrow(
                PointId::new(src).unwrap(),
                PointId::new(dst).unwrap(),
                Remote {
                    rank: ghost,
                    remote_point: PointId::new(dst).unwrap(),
                },
            );
        }
        ovlp
    }

    #[test]
    fn owner_rank_links_to_ghost() {
        // Rank 0 owns 1, ghosted to rank 1 as 101
        let section = make_section(&[1]);
        let mut ovlp = make_overlap(0, 1, &[1], &[101]);
        let links = neighbour_links(&section, &mut ovlp, 0);
        assert_eq!(links.len(), 1);
        assert_eq!(links[&1], vec![(PointId::new(1).unwrap(), PointId::new(101).unwrap())]);
    }

    #[test]
    fn ghost_rank_receives_from_owner() {
        // Rank 1 owns nothing, but receives 1 as 101 from rank 0
        let section = make_section(&[]); // ghost owns nothing
        let mut ovlp = make_overlap(0, 1, &[1], &[101]);
        let links = neighbour_links(&section, &mut ovlp, 1);
        println!("links for ghost rank: {:?}", links);
        assert_eq!(links.len(), 1);
        assert_eq!(links[&0], vec![(PointId::new(101).unwrap(), PointId::new(1).unwrap())]);
    }

    #[test]
    fn no_links_for_isolated_rank() {
        // Rank 2 owns 2, but no overlap
        let section = make_section(&[2]);
        let mut ovlp = InMemorySieve::<PointId, Remote>::default();
        let links = neighbour_links(&section, &mut ovlp, 2);
        assert!(links.is_empty());
    }

    #[test]
    fn multiple_neighbors() {
        // Rank 0 owns 1,2, ghosted to rank 1 as 101, rank 2 as 201
        let section = make_section(&[1, 2]);
        let mut ovlp = InMemorySieve::<PointId, Remote>::default();
        ovlp.add_arrow(
            PointId::new(1).unwrap(),
            PointId::new(101).unwrap(),
            Remote {
                rank: 1,
                remote_point: PointId::new(101).unwrap(),
            },
        );
        ovlp.add_arrow(
            PointId::new(2).unwrap(),
            PointId::new(201).unwrap(),
            Remote {
                rank: 2,
                remote_point: PointId::new(201).unwrap(),
            },
        );
        let links = neighbour_links(&section, &mut ovlp, 0);
        assert_eq!(links.len(), 2);
        assert_eq!(links[&1], vec![(PointId::new(1).unwrap(), PointId::new(101).unwrap())]);
        assert_eq!(links[&2], vec![(PointId::new(2).unwrap(), PointId::new(201).unwrap())]);
    }
}
