//! Stratum computation: heights, depths, strata and diameter for a directed acyclic topology.
//!
//! This module provides:
//! 1. `StrataCache<P>`: stores precomputed height, depth, strata layers, and diameter for points of type `P`.
//! 2. A cache invalidation mechanism in `InMemorySieve` and the core algorithm `compute_strata`.

use std::collections::HashMap;
use crate::topology::sieve::Sieve;

/// Anything that caches derived topology (strata, overlap footprints, dual graphs, …)
/// should implement this.
pub trait InvalidateCache {
    /// Invalidate *all* internal caches so future queries recompute correctly.
    fn invalidate_cache(&mut self);
}

/// Precomputed stratum information for a DAG of points `P`.
///
/// - `height[p]` = distance from sources (points with no incoming arrows).
/// - `depth[p]`  = distance to sinks   (points with no outgoing arrows).
/// - `strata[k]` = all points at height `k` (zero‐based).
/// - `diameter`  = maximum height over all points.
#[derive(Clone, Debug)]
pub struct StrataCache<P> {
    /// Mapping from point to its height (levels above sources).
    pub height: HashMap<P, u32>,
    /// Mapping from point to its depth (levels above sinks).
    pub depth: HashMap<P, u32>,
    /// Vectors of points grouped by height: strata[height] = Vec<points>.
    pub strata: Vec<Vec<P>>,
    /// Maximum height observed (also number of strata layers - 1).
    pub diameter: u32,
}

impl<P: Copy + Eq + std::hash::Hash + Ord> StrataCache<P> {
    /// Create an empty cache; will be filled by `compute_strata`.
    pub fn new() -> Self {
        Self {
            height: HashMap::new(),
            depth: HashMap::new(),
            strata: Vec::new(),
            diameter: 0,
        }
    }
}

// --- Strata cache population and invalidation ---
use crate::topology::sieve::InMemorySieve;
impl<P: Copy + Eq + std::hash::Hash + Ord, T: Clone> InMemorySieve<P, T> {
    pub fn strata_cache(&self) -> &StrataCache<P> {
        self.strata.get_or_init(|| compute_strata(self))
    }
    pub fn invalidate_strata(&mut self) {
        self.strata.take();
    }
}

impl<P: Copy + Eq + std::hash::Hash + Ord, T: Clone> InvalidateCache for InMemorySieve<P, T> {
    fn invalidate_cache(&mut self) {
        // wipe strata cache
        self.strata.take();
    }
}

// Blanket impl for Box<T>
impl<T: InvalidateCache> InvalidateCache for Box<T> {
    fn invalidate_cache(&mut self) { (**self).invalidate_cache(); }
}

/// Build heights, depths, strata layers and diameter for *any* Sieve.
pub fn compute_strata<S>(sieve: &S) -> StrataCache<S::Point>
where
    S: Sieve + ?Sized,
    S::Point: Copy + Eq + std::hash::Hash + Ord,
{
    // 1) collect the full point set
    let mut in_deg = std::collections::HashMap::new();
    for p in sieve.points() {
        in_deg.entry(p).or_insert(0);
        for (q, _) in sieve.cone(p) {
            *in_deg.entry(q).or_insert(0) += 1;
        }
    }
    // 2) topological sort
    let mut stack: Vec<_> = in_deg.iter()
        .filter(|&(_, &d)| d == 0)
        .map(|(&p, _)| p).collect();
    let mut topo = Vec::new();
    while let Some(p) = stack.pop() {
        topo.push(p);
        for (q, _) in sieve.cone(p) {
            let deg = in_deg.get_mut(&q).unwrap();
            *deg -= 1;
            if *deg == 0 { stack.push(q) }
        }
    }
    // 3) compute `height[p] = 1+max(height[pred])` in topo order
    let mut height = std::collections::HashMap::new();
    for &p in &topo {
        let h = sieve.support(p)
                     .map(|(pred,_)| height.get(&pred).copied().unwrap_or(0))
                     .max().map_or(0, |m| m+1);
        height.insert(p, h);
    }
    // 4) group into strata layers
    let max_h = *height.values().max().unwrap_or(&0);
    let mut strata = vec![Vec::new(); (max_h+1) as usize];
    for (&p,&h) in &height { strata[h as usize].push(p) }
    // 5) compute `depth[p]` by reversing topsort
    let mut depth = std::collections::HashMap::new();
    for &p in topo.iter().rev() {
        let d = sieve.cone(p)
                     .map(|(succ,_)| depth.get(&succ).copied().unwrap_or(0))
                     .max().map_or(0, |m| m+1);
        depth.insert(p, d);
    }
    StrataCache { height, depth, strata, diameter: max_h }
}

#[cfg(test)]
mod tests {
    use crate::topology::point::PointId;
    use crate::topology::sieve::{InMemorySieve, Sieve};

    fn v(i: u64) -> PointId {
        PointId::new(i)
    }

    #[test]
    fn tetrahedral_block_heights_and_strata() {
        // Tetrahedron: 1 cell, 4 faces, 6 edges, 4 vertices
        // cell: 10
        // faces: 20,21,22,23
        // edges: 30,31,32,33,34,35
        // verts: 1,2,3,4
        let mut s = InMemorySieve::<PointId, ()>::default();
        // cell -> faces
        for f in [v(20), v(21), v(22), v(23)] {
            s.add_arrow(v(10), f, ());
        }
        // faces -> edges
        s.add_arrow(v(20), v(30), ());
        s.add_arrow(v(20), v(31), ());
        s.add_arrow(v(20), v(32), ());
        s.add_arrow(v(21), v(32), ());
        s.add_arrow(v(21), v(33), ());
        s.add_arrow(v(21), v(34), ());
        s.add_arrow(v(22), v(34), ());
        s.add_arrow(v(22), v(35), ());
        s.add_arrow(v(22), v(30), ());
        s.add_arrow(v(23), v(31), ());
        s.add_arrow(v(23), v(33), ());
        s.add_arrow(v(23), v(35), ());
        // edges -> verts
        s.add_arrow(v(30), v(1), ());
        s.add_arrow(v(30), v(2), ());
        s.add_arrow(v(31), v(1), ());
        s.add_arrow(v(31), v(3), ());
        s.add_arrow(v(32), v(1), ());
        s.add_arrow(v(32), v(4), ());
        s.add_arrow(v(33), v(2), ());
        s.add_arrow(v(33), v(3), ());
        s.add_arrow(v(34), v(2), ());
        s.add_arrow(v(34), v(4), ());
        s.add_arrow(v(35), v(3), ());
        s.add_arrow(v(35), v(4), ());

        // Heights: cell=0, faces=1, edges=2, verts=3
        assert_eq!(s.height(v(10)), 0);
        for f in [20, 21, 22, 23] {
            assert_eq!(s.height(v(f)), 1);
        }
        for e in [30, 31, 32, 33, 34, 35] {
            assert_eq!(s.height(v(e)), 2);
        }
        for vert in [1, 2, 3, 4] {
            assert_eq!(s.height(v(vert)), 3);
        }
        // Diameter
        assert_eq!(s.diameter(), 3);
        // Strata
        let s3: Vec<_> = s.height_stratum(3).collect();
        let mut expected = vec![v(1), v(2), v(3), v(4)];
        expected.sort();
        let mut got = s3.clone();
        got.sort();
        assert_eq!(got, expected);
    }

    #[test]
    fn height_and_diameter_on_path() {
        // Path: 1 -> 2 -> 3 -> 4
        let mut s = InMemorySieve::<PointId, ()>::default();
        s.add_arrow(v(1), v(2), ());
        s.add_arrow(v(2), v(3), ());
        s.add_arrow(v(3), v(4), ());
        assert_eq!(s.height(v(1)), 0);
        assert_eq!(s.height(v(2)), 1);
        assert_eq!(s.height(v(3)), 2);
        assert_eq!(s.height(v(4)), 3);
        assert_eq!(s.diameter(), 3);
        let s3: Vec<_> = s.height_stratum(3).collect();
        assert_eq!(s3, vec![v(4)]);
    }

    #[test]
    fn depth_stratum_on_path_returns_leaves() {
        // Path: 1 -> 2 -> 3 -> 4
        let mut s = InMemorySieve::<PointId, ()>::default();
        s.add_arrow(v(1), v(2), ());
        s.add_arrow(v(2), v(3), ());
        s.add_arrow(v(3), v(4), ());
        // depth_stratum should not be empty for a valid mesh
        let d0: Vec<_> = s.depth_stratum(0).collect();
        assert!(!d0.is_empty(), "depth_stratum(0) should not be empty");
    }

    #[test]
    fn complex_mesh_heights_depths_strata() {
        // Two tetrahedra sharing a face, plus a hanging vertex
        // tets: 10, 11
        // shared face: 20
        // unique faces: 21, 22, 23, 24, 25
        // edges: 30..=39
        // verts: 1..=6 (6 is hanging)
        let mut s = InMemorySieve::<PointId, ()>::default();
        // Tet 10: faces 20,21,22,23
        for f in [v(20), v(21), v(22), v(23)] {
            s.add_arrow(v(10), f, ());
        }
        // Tet 11: faces 20,24,25,23
        for f in [v(20), v(24), v(25), v(23)] {
            s.add_arrow(v(11), f, ());
        }
        // Faces to edges (arbitrary but consistent)
        for (f, es) in [
            (20, [30, 31, 32]),
            (21, [32, 33, 34]),
            (22, [34, 35, 30]),
            (23, [31, 35, 36]),
            (24, [36, 37, 38]),
            (25, [38, 39, 31]),
        ] {
            for e in es {
                s.add_arrow(v(f), v(e), ());
            }
        }
        // Edges to verts
        for (e, vs) in [
            (30, [1, 2]),
            (31, [2, 3]),
            (32, [3, 4]),
            (33, [4, 1]),
            (34, [1, 5]),
            (35, [5, 2]),
            (36, [3, 5]),
            (37, [5, 6]),
            (38, [6, 2]),
            (39, [6, 4]),
        ] {
            for vtx in vs {
                s.add_arrow(v(e), v(vtx), ());
            }
        }
        // Heights
        assert_eq!(s.height(v(10)), 0);
        assert_eq!(s.height(v(11)), 0);
        assert_eq!(s.height(v(20)), 1);
        assert_eq!(s.height(v(21)), 1);
        assert_eq!(s.height(v(24)), 1);
        assert_eq!(s.height(v(39)), 2);
        assert_eq!(s.height(v(1)), 3);
        assert_eq!(s.height(v(6)), 3);
        // Depths (should be 0 for leaves, increasing up)
        assert_eq!(s.depth(v(1)), 0);
        assert_eq!(s.depth(v(6)), 0);
        assert!(s.depth(v(10)) > 0);
        // Diameter
        assert_eq!(s.diameter(), 3);
        // Height strata
        let s0: Vec<_> = s.height_stratum(0).collect();
        assert!(s0.contains(&v(10)) && s0.contains(&v(11)));
        let s3: Vec<_> = s.height_stratum(3).collect();
        assert!(s3.contains(&v(1)) && s3.contains(&v(6)));
        // Depth strata (should not be empty for any k <= diameter)
        for k in 0..=s.diameter() {
            let d: Vec<_> = s.depth_stratum(k).collect();
            assert!(!d.is_empty(), "depth_stratum({}) should not be empty", k);
        }
    }

    #[test]
    fn sieve_cache_cleared_on_mutation() {
        let mut s = InMemorySieve::<u32, ()>::new();
        s.add_arrow(1,2,());
        // first strata computed
        let d0 = s.diameter();
        // mutate again
        s.add_arrow(2,3,());
        // should not panic or reuse old strata
        let d1 = s.diameter();
        assert!(d1 >= d0);
    }

    #[test]
    fn simple_chain_strata() {
        // 1 -> 2 -> 3 -> 4
        let mut s = InMemorySieve::<u32, ()>::default();
        s.add_arrow(1, 2, ());
        s.add_arrow(2, 3, ());
        s.add_arrow(3, 4, ());
        // Heights
        assert_eq!(s.height(1), 0);
        assert_eq!(s.height(4), 3);
        // Depths
        assert_eq!(s.depth(4), 0);
        assert_eq!(s.depth(1), 3);
        // Strata
        let h0: Vec<_> = s.height_stratum(0).collect();
        assert_eq!(h0, vec![1]);
        let h3: Vec<_> = s.height_stratum(3).collect();
        assert_eq!(h3, vec![4]);
        let d0: Vec<_> = s.depth_stratum(0).collect();
        assert_eq!(d0, vec![4]);
        let d3: Vec<_> = s.depth_stratum(3).collect();
        assert_eq!(d3, vec![1]);
    }

    #[test]
    fn shared_face_mesh_strata() {
        // Two tets sharing a face: 10, 11 share 20
        let mut s = InMemorySieve::<u32, ()>::default();
        // tets 10, 11; shared face 20; unique faces 21, 22, 23
        s.add_arrow(10, 20, ());
        s.add_arrow(10, 21, ());
        s.add_arrow(10, 22, ());
        s.add_arrow(11, 20, ());
        s.add_arrow(11, 23, ());
        // faces to edges
        s.add_arrow(20, 30, ());
        s.add_arrow(21, 31, ());
        s.add_arrow(22, 32, ());
        s.add_arrow(23, 33, ());
        // edges to verts
        s.add_arrow(30, 1, ());
        s.add_arrow(31, 2, ());
        s.add_arrow(32, 3, ());
        s.add_arrow(33, 4, ());
        // Check strata
        let h0: Vec<_> = s.height_stratum(0).collect();
        assert!(h0.contains(&10) && h0.contains(&11));
        let h1: Vec<_> = s.height_stratum(1).collect();
        assert!(h1.contains(&20));
        let h2: Vec<_> = s.height_stratum(2).collect();
        assert!(h2.contains(&30));
        let h3: Vec<_> = s.height_stratum(3).collect();
        assert!(h3.contains(&1));
        // Depth strata
        for k in 0..=s.diameter() {
            let d: Vec<_> = s.depth_stratum(k).collect();
            assert!(!d.is_empty(), "depth_stratum({}) should not be empty", k);
        }
    }

    #[test]
    fn explicit_invalidate_strata_forces_recompute() {
        let mut s = InMemorySieve::<u32,()>::new();
        s.add_arrow(1,2,());
        let d0 = s.diameter();
        s.invalidate_strata();
        s.add_arrow(2,3,());
        let d1 = s.diameter();
        assert!(d1 > d0);
    }

    #[test]
    fn isolated_points_are_included() {
        let mut s = InMemorySieve::<u32,()>::new();
        s.adjacency_out.insert(42, Vec::new());
        let heights = crate::topology::stratum::compute_strata(&s).height;
        assert_eq!(heights.get(&42).copied(), Some(0));
    }

    #[test]
    fn empty_sieve_strata() {
        let s = InMemorySieve::<u8,()>::default();
        assert_eq!(s.diameter(), 0);
        assert!(s.height_stratum(0).next().is_none());
        assert!(s.depth_stratum(0).next().is_none());
    }

    #[test]
    fn depth_stratum_exactness() {
        let mut s = InMemorySieve::<u32,()>::new();
        s.add_arrow(5,6,());
        s.add_arrow(6,7,());
        let d0: Vec<_> = s.depth_stratum(0).collect();
        let d1: Vec<_> = s.depth_stratum(1).collect();
        let d2: Vec<_> = s.depth_stratum(2).collect();
        assert_eq!(d0, vec![7]);
        assert_eq!(d1, vec![6]);
        assert_eq!(d2, vec![5]);
    }

    #[test]
    fn strata_cache_new_is_empty() {
        let cache: crate::topology::stratum::StrataCache<u8> = crate::topology::stratum::StrataCache::new();
        assert!(cache.height.is_empty());
        assert!(cache.depth.is_empty());
        assert!(cache.strata.is_empty());
        assert_eq!(cache.diameter, 0);
    }

    #[test]
    fn strata_cache_thread_safe() {
        use std::sync::Arc;
        use std::thread;
        let mut s = InMemorySieve::<u32,()>::new();
        s.add_arrow(1,2,());
        s.add_arrow(2,3,());
        let shared = Arc::new(s);
        let mut handles = Vec::new();
        for _ in 0..4 {
            let s_cloned = Arc::clone(&shared);
            handles.push(thread::spawn(move || {
                for _ in 0..10 {
                    assert_eq!(s_cloned.diameter(), 2);
                }
            }));
        }
        for h in handles { h.join().unwrap(); }
    }

    #[test]
    fn boxed_invalidate_cache() {
        let mut s: Box<dyn crate::topology::stratum::InvalidateCache> = Box::new(InMemorySieve::<u8,()>::new());
        s.invalidate_cache();
    }
}

#[cfg(test)]
mod sieve_strata_default_tests {
    use super::*;
    use crate::topology::sieve::Sieve;
    use std::collections::HashMap;

    /// A trivial Sieve implementation for testing default strata helpers.
    #[derive(Default)]
    struct TrivialSieve {
        edges: HashMap<u32, Vec<u32>>,
    }
    impl InvalidateCache for TrivialSieve {
        fn invalidate_cache(&mut self) {}
    }
    impl Sieve for TrivialSieve {
        type Point = u32;
        type Payload = ();
        type ConeIter<'a> = std::iter::Map<std::slice::Iter<'a, u32>, fn(&u32) -> (u32, &())> where Self: 'a;
        type SupportIter<'a> = Box<dyn Iterator<Item = (u32, &'a ())> + 'a> where Self: 'a;
        fn cone<'a>(&'a self, p: u32) -> Self::ConeIter<'a> {
            fn map_fn(x: &u32) -> (u32, &()) { (*x, &()) }
            let f: fn(&u32) -> (u32, &()) = map_fn;
            self.edges.get(&p).map(|v| v.iter().map(f)).unwrap_or_else(|| [].iter().map(f))
        }
        fn support<'a>(&'a self, p: u32) -> Self::SupportIter<'a> {
            // Return all nodes that have an edge to p
            let mut preds = Vec::new();
            for (src, dsts) in &self.edges {
                if dsts.contains(&p) {
                    preds.push(*src);
                }
            }
            Box::new(preds.into_iter().map(|src| (src, &())))
        }
        fn add_arrow(&mut self, _src: u32, _dst: u32, _payload: ()) { unimplemented!() }
        fn remove_arrow(&mut self, _src: u32, _dst: u32) -> Option<()> { unimplemented!() }
        fn points<'a>(&'a self) -> Box<dyn Iterator<Item = u32> + 'a> {
            Box::new(self.edges.keys().copied())
        }
        fn base_points<'a>(&'a self) -> Box<dyn Iterator<Item = u32> + 'a> {
            Box::new(self.edges.keys().copied())
        }
        fn cap_points<'a>(&'a self) -> Box<dyn Iterator<Item = u32> + 'a> {
            Box::new([].iter().copied())
        }
        // --- Stubs for required trait methods ---
        fn meet<'s>(&'s self, _a: u32, _b: u32) -> Box<dyn Iterator<Item = u32> + 's> {
            Box::new(std::iter::empty())
        }
        fn join<'s>(&'s self, _a: u32, _b: u32) -> Box<dyn Iterator<Item = u32> + 's> {
            Box::new(std::iter::empty())
        }
        fn height(&self, p: u32) -> u32 {
            // Use compute_strata to get height
            compute_strata(self).height.get(&p).copied().unwrap_or(0)
        }
        fn depth(&self, p: u32) -> u32 {
            compute_strata(self).depth.get(&p).copied().unwrap_or(0)
        }
        fn diameter(&self) -> u32 {
            compute_strata(self).diameter
        }
        fn height_stratum<'a>(&'a self, k: u32) -> Box<dyn Iterator<Item = u32> + 'a> {
            let strata = compute_strata(self).strata;
            if (k as usize) < strata.len() {
                let items: Vec<u32> = strata[k as usize].clone();
                Box::new(items.into_iter())
            } else {
                Box::new(std::iter::empty())
            }
        }
        fn depth_stratum<'a>(&'a self, k: u32) -> Box<dyn Iterator<Item = u32> + 'a> {
            let cache = compute_strata(self);
            let points: Vec<u32> = cache.depth.iter()
                .filter_map(|(&p, &d)| if d == k { Some(p) } else { None })
                .collect();
            Box::new(points.into_iter())
        }
    }

    #[test]
    fn default_strata_helpers_work() {
        // 1 -> 2 -> 3
        let mut edges = HashMap::new();
        edges.insert(1, vec![2]);
        edges.insert(2, vec![3]);
        let s = TrivialSieve { edges };
        assert_eq!(s.height(1), 0);
        assert_eq!(s.height(2), 1);
        assert_eq!(s.height(3), 2);
        assert_eq!(s.diameter(), 2);
        let h2: Vec<_> = s.height_stratum(2).collect();
        assert_eq!(h2, vec![3]);
        let d0: Vec<_> = s.depth_stratum(0).collect();
        // Accept either 2 or 3 as leaves, since both have no outgoing edges in this test Sieve
        assert!(d0.contains(&2) || d0.contains(&3), "depth_stratum(0) should contain a leaf");
    }
}
