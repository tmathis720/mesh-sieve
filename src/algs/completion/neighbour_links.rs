//! Build the peer→(send_loc, recv_loc) map for section completion.
//!
//! This returns, for each neighbor rank, the list of
//! `(local_point, remote_point)` pairs you must exchange:
//!  - If you own values at `local_point`, you will send them to your neighbor’s `remote_point`.
//!  - If you own *no* values, you instead receive from each neighbor’s `remote_point` into your `local_point`.

use crate::data::section::Section;
use crate::mesh_error::MeshSieveError;
use crate::overlap::overlap::Overlap;
use crate::topology::point::PointId;
use crate::topology::sieve::sieve_trait::Sieve;
use std::collections::HashMap;

/// Compute the neighbor‐links for section completion.
///
/// On success, returns a map `neighbor_rank -> Vec<(send_loc, recv_loc)>`.
/// You then use these pairs in the symmetric size‐ and data‐exchange phases.
///
/// # Errors
/// Returns `MeshSieveError::MissingOverlap` if the overlap graph doesn’t
/// contain the information needed to build the send/receive pairs.
pub fn neighbour_links<V>(
    section: &Section<V>,
    ovlp: &Overlap,
    my_rank: usize,
) -> Result<HashMap<usize, Vec<(PointId, PointId)>>, MeshSieveError>
where
    V: Clone + Default + PartialEq,
{
    let default_val = V::default();
    let mut out: HashMap<usize, Vec<(PointId, PointId)>> = HashMap::new();
    let mut has_owned = false;

    // 1) Outbound: for every point where we have a non‐default value,
    //    send to all overlapping remote points.
    for (p, vals) in section.iter() {
        if vals.iter().all(|v| *v == default_val) {
            continue; // nothing to send for this point
        }
        has_owned = true;
        for (_unused, rem) in ovlp.cone(p) {
            if rem.rank != my_rank {
                out.entry(rem.rank)
                    .or_default()
                    .push((p, rem.remote_point));
            }
        }
    }

    // 2) Inbound: if we had *no* owned values, we must pull from every neighbor
    //    that appears in adjacency_in.  rems holds (local_mesh_pt, Remote),
    //    so fetch `rem.remote_point` into `local_mesh_pt`.
    if !has_owned {
        for (_part_pt, rems) in &ovlp.adjacency_in {
            for &(local_pt, ref rem) in rems {
                if rem.rank == my_rank {
                    continue;
                }
                // send from remote_point, receive into the real local_pt
                out.entry(rem.rank)
                    .or_default()
                    .push((rem.remote_point, local_pt));
            }
        }
    }

    // If we still have no links at all, something’s wrong with the Overlap
    if out.is_empty() {
        return Err(MeshSieveError::MissingOverlap {
            source: format!("rank {} has no neighbour links", my_rank).into(),
        });
    }

    // Optional: sort each neighbor’s list deterministically
    // for vec in out.values_mut() {
    //     vec.sort_unstable_by_key(|&(l, r)| (l.get(), r.get()));
    // }

    Ok(out)
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
        let links = neighbour_links(&section, &mut ovlp, 0).unwrap();
        assert_eq!(links.len(), 1);
        assert_eq!(links[&1], vec![(PointId::new(1).unwrap(), PointId::new(101).unwrap())]);
    }

    #[test]
    fn ghost_rank_receives_from_owner() {
        // Rank 1 owns nothing, but receives 1 as 101 from rank 0
        let section = make_section(&[]); // ghost owns nothing
        let mut ovlp = make_overlap(0, 1, &[1], &[101]);
        let links = neighbour_links(&section, &mut ovlp, 1);
        assert!(matches!(links, Err(MeshSieveError::MissingOverlap { .. })), "Expected MissingOverlap error for ghost rank with no inbound links");
    }

    #[test]
    fn no_links_for_isolated_rank() {
        // Rank 2 owns 2, but no overlap
        let section = make_section(&[2]);
        let mut ovlp = InMemorySieve::<PointId, Remote>::default();
        let links = neighbour_links(&section, &mut ovlp, 2);
        assert!(matches!(links, Err(MeshSieveError::MissingOverlap { .. })));
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
        let links = neighbour_links(&section, &mut ovlp, 0).unwrap();
        assert_eq!(links.len(), 2);
        assert_eq!(links[&1], vec![(PointId::new(1).unwrap(), PointId::new(101).unwrap())]);
        assert_eq!(links[&2], vec![(PointId::new(2).unwrap(), PointId::new(201).unwrap())]);
    }
}
