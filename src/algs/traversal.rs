//! DFS/BFS traversal helpers for Sieve topologies.

use std::collections::{HashSet, VecDeque};
use crate::topology::sieve::Sieve;
use crate::topology::point::PointId;

/// Shorthand so callers don't have to spell the full bound.
pub type Point = PointId;

fn dfs<F, I, S>(sieve: &S, seeds: I, mut neighbour_fn: F) -> Vec<Point>
where
    S: Sieve<Point = Point>,
    F: FnMut(&S, Point) -> Vec<Point>,
    I: IntoIterator<Item = Point>,
{
    let mut stack: Vec<Point> = seeds.into_iter().collect();
    let mut seen: HashSet<Point> = stack.iter().copied().collect();

    while let Some(p) = stack.pop() {
        for q in neighbour_fn(sieve, p) {
            if seen.insert(q) {
                stack.push(q);
            }
        }
    }
    let mut out: Vec<Point> = seen.into_iter().collect();
    out.sort_unstable();
    out
}

/// Complete transitive closure following `cone` arrows.
pub fn closure<I, S>(sieve: &S, seeds: I) -> Vec<Point>
where
    S: Sieve<Point = Point>,
    I: IntoIterator<Item = Point>,
{
    dfs(sieve, seeds, |s, p| s.cone(p).map(|(q, _)| q).collect())
}

/// Complete transitive star following `support` arrows.
pub fn star<I, S>(sieve: &S, seeds: I) -> Vec<Point>
where
    S: Sieve<Point = Point>,
    I: IntoIterator<Item = Point>,
{
    dfs(sieve, seeds, |s, p| s.support(p).map(|(q, _)| q).collect())
}

/// link(p) = star(p) ∩ closure(p)
pub fn link<S: Sieve<Point = Point>>(sieve: &S, p: Point) -> Vec<Point> {
    let mut cl = closure(sieve, [p]);
    let mut st = star(sieve, [p]);
    cl.sort_unstable();
    st.sort_unstable();
    // Remove p, cone(p), and support(p) from the intersection
    let cone: HashSet<_> = sieve.cone(p).map(|(q, _)| q).collect();
    let sup: HashSet<_> = sieve.support(p).map(|(q, _)| q).collect();
    cl.retain(|x| st.binary_search(x).is_ok() && *x != p && !cone.contains(x) && !sup.contains(x));
    cl
}

/// Optional BFS distance map – used by coarsening / agglomeration.
pub fn depth_map<S: Sieve<Point = Point>>(sieve: &S, seed: Point) -> Vec<(Point, u32)> {
    let mut depths = Vec::new();
    let mut seen   = HashSet::new();
    let mut q      = VecDeque::from([(seed, 0)]);

    while let Some((p, d)) = q.pop_front() {
        if seen.insert(p) {
            depths.push((p, d));
            for (q_pt, _) in sieve.cone(p) {
                q.push_back((q_pt, d + 1));
            }
        }
    }
    depths.sort_by_key(|&(p, _)| p);
    depths
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::sieve::InMemorySieve;

    /// helper that builds a toy 2-triangle mesh used in many tests
    fn tiny_mesh() -> InMemorySieve<Point, ()> {
        let v = |i| PointId::new(i);
        let mut s = InMemorySieve::<Point, ()>::default();
        // triangle 10 (verts 1,2,3)
        s.add_arrow(v(10), v(1), ());
        s.add_arrow(v(10), v(2), ());
        s.add_arrow(v(10), v(3), ());
        // triangle 11 (verts 2,3,4)
        s.add_arrow(v(11), v(2), ());
        s.add_arrow(v(11), v(3), ());
        s.add_arrow(v(11), v(4), ());
        // reverse incidences
        for (src, dsts) in [ (v(10), [v(1),v(2),v(3)]),
                             (v(11), [v(2),v(3),v(4)]) ] {
            for d in dsts {
                s.add_arrow(d, src, ()); // support edge
            }
        }
        s
    }

    #[test]
    fn closure_contains_cone() {
        let s = tiny_mesh();
        let cl = closure(&s, [PointId::new(10)]);
        assert!(cl.contains(&PointId::new(1)));
        assert!(cl.contains(&PointId::new(2)));
    }

    #[test]
    fn link_disjoint_from_cone_and_support() {
        let s = tiny_mesh();
        let p = PointId::new(10);
        let lk = link(&s, p);
        let cone: Vec<_> = s.cone(p).map(|(q, _)| q).collect();
        let sup : Vec<_> = s.support(p).map(|(q, _)| q).collect();
        for x in lk {
            assert!(!cone.contains(&x));
            assert!(!sup.contains(&x));
        }
    }
    // Property-based tests can be added here in the future.
}
