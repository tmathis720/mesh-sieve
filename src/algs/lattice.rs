//! Set-lattice helpers: meet, join, adjacency and helpers.
//! All output vectors are **sorted & deduplicated** for deterministic behaviour.

use std::cmp::Ordering;

use crate::topology::point::PointId;
use crate::topology::sieve::Sieve;
use crate::algs::traversal::{closure, star};

type P = PointId;

fn set_union(a: &[P], b: &[P], out: &mut Vec<P>) {
    let mut i = 0;
    let mut j = 0;
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            Ordering::Less    => { out.push(a[i]); i += 1; }
            Ordering::Greater => { out.push(b[j]); j += 1; }
            Ordering::Equal   => { out.push(a[i]); i += 1; j += 1; }
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
}

fn set_intersection(a: &[P], b: &[P], out: &mut Vec<P>) {
    let mut i = 0;
    let mut j = 0;
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            Ordering::Less    => i += 1,
            Ordering::Greater => j += 1,
            Ordering::Equal   => { out.push(a[i]); i += 1; j += 1; }
        }
    }
}

/// meet(a,b) = closure(a) ∩ closure(b)
pub fn meet<S>(sieve: &S, a: P, b: P) -> Vec<P>
where
    S: Sieve<Point = P>,
{
    let mut ca = closure(sieve, [a]);
    let mut cb = closure(sieve, [b]);
    ca.sort_unstable();
    cb.sort_unstable();
    let mut out = Vec::with_capacity(ca.len().min(cb.len()));
    set_intersection(&ca, &cb, &mut out);
    out
}

/// join(a,b) = star(a) ∪ star(b)
pub fn join<S>(sieve: &S, a: P, b: P) -> Vec<P>
where
    S: Sieve<Point = P>,
{
    let mut sa = star(sieve, [a]);
    let mut sb = star(sieve, [b]);
    sa.sort_unstable();
    sb.sort_unstable();
    let mut out = Vec::with_capacity(sa.len() + sb.len());
    set_union(&sa, &sb, &mut out);
    out
}

/// Cells adjacent to `p` that are **not** `p` itself.
/// Adjacent = share a face/edge (= “support” of cone items).
pub fn adjacent<S>(sieve: &S, p: P) -> Vec<P>
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
        let v  = |i| PointId::new(i);
        let t1 = v(10); let t2 = v(11);
        let mut s = InMemorySieve::<P, ()>::default();
        // triangle 1 cone
        for x in [v(1), v(2), v(3)] { s.add_arrow(t1, x, ()); s.add_arrow(x, t1, ()); }
        // triangle 2 cone
        for x in [v(2), v(3), v(4)] { s.add_arrow(t2, x, ()); s.add_arrow(x, t2, ()); }
        (s, t1, t2)
    }

    #[test]
    fn meet_contains_shared_verts() {
        let (s, a, b) = simple_pair();
        let m = meet(&s, a, b);
        // The intersection of closures includes both triangles and all shared cone points
        let expected = vec![PointId::new(1), PointId::new(2), PointId::new(3), PointId::new(4), PointId::new(10), PointId::new(11)];
        assert_eq!(m, expected);
    }

    #[test]
    fn join_contains_both_cells() {
        let (s, a, b) = simple_pair();
        let j = join(&s, a, b);
        assert!(j.contains(&a) && j.contains(&b));
    }

    #[test]
    fn adjacent_symmetry() {
        let (s, a, b) = simple_pair();
        let adj_a = adjacent(&s, a);
        let adj_b = adjacent(&s, b);
        assert!(adj_a.contains(&b));
        assert!(adj_b.contains(&a));
    }
    // Property-based tests can be added here in the future.
}
