//! Set-lattice helpers: meet, join, adjacency and helpers.
//! All output vectors are **sorted & deduplicated** for deterministic behaviour.
//!
//! This module provides utilities for set-lattice operations on mesh topologies,
//! including adjacency queries and helpers for use with [`Sieve`] structures.

use crate::algs::traversal::star;
use crate::topology::point::PointId;
use crate::topology::sieve::Sieve;

type P = PointId;

/// Cells adjacent to `p` that are **not** `p` itself.
///
/// Adjacent = share a face/edge (= “support” of cone items).
/// Returns a sorted, deduplicated vector of adjacent cells.
pub fn adjacent<S>(sieve: &mut S, p: P) -> Vec<P>
where
    S: Sieve<Point = P>,
{
    use std::collections::HashSet;
    let st = star(sieve, [p]);
    let mut neigh = HashSet::new();
    for q in &st {
        for (cell, _) in sieve.support(*q) {
            if cell != p {
                neigh.insert(cell);
            }
        }
    }
    let mut out: Vec<P> = neigh.into_iter().collect();
    out.sort_unstable();
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::sieve::InMemorySieve;

    fn simple_pair() -> (InMemorySieve<P, ()>, P, P) {
        // two triangles sharing an edge
        let v = |i| PointId::new(i).unwrap();
        let t1 = v(10);
        let t2 = v(11);
        let mut s = InMemorySieve::<P, ()>::default();
        // triangle 1 cone
        for x in [v(1), v(2), v(3)] {
            s.add_arrow(t1, x, ());
            s.add_arrow(x, t1, ());
        }
        // triangle 2 cone
        for x in [v(2), v(3), v(4)] {
            s.add_arrow(t2, x, ());
            s.add_arrow(x, t2, ());
        }
        (s, t1, t2)
    }

    #[test]
    fn adjacent_symmetry() {
        let (mut s, a, b) = simple_pair();
        let adj_a = adjacent(&mut s, a);
        let adj_b = adjacent(&mut s, b);
        assert!(adj_a.contains(&b));
        assert!(adj_b.contains(&a));
    }
    // Property-based tests can be added here in the future.
}
