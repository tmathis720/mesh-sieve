//! Build the peer→(my_pt, their_pt) map for section completion.

use std::collections::HashMap;
use crate::data::section::Section;
use crate::overlap::overlap::Overlap;
use crate::topology::point::PointId;
use crate::topology::sieve::Sieve;
use crate::algs::completion::partition_point;

/// Given your local section, the overlap graph, and your rank,
/// returns for each neighbor rank the list of `(local_point, remote_point)`
/// that you must send or receive.
pub fn neighbour_links<V: Clone + Default>(
    section: &Section<V>,
    ovlp: &Overlap,
    my_rank: usize,
) -> HashMap<usize, Vec<(PointId, PointId)>> {
    let mut out: HashMap<usize, Vec<(PointId, PointId)>> = HashMap::new();
    let me_pt = crate::algs::completion::partition_point(my_rank);
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
        for (src, rem) in ovlp.support(me_pt) {
            if rem.rank != my_rank {
                out.entry(rem.rank)
                   .or_default()
                   .push((rem.remote_point, src));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    // TODO: Add tests for owner/ghost behaviour
}
