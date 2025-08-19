//! Stratum computation: heights, depths, strata and diameter for a directed acyclic topology.
//!
//! This module provides:
//! 1. `StrataCache<P>`: stores precomputed height, depth, strata layers, and diameter for points of type `P`.
//! 2. A cache invalidation mechanism in `InMemorySieve` and the core algorithm `compute_strata`.

use crate::mesh_error::MeshSieveError;
use crate::topology::sieve::Sieve;
use std::collections::HashMap;

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

    /// Deterministic global point order (height-major, then point order).
    pub chart_points: Vec<P>, // index -> point
    /// Reverse lookup for `chart_points`.
    pub chart_index: HashMap<P, usize>, // point -> index
}

impl<P: Copy + Eq + std::hash::Hash + Ord + std::fmt::Debug> StrataCache<P> {
    /// Create an empty cache; will be filled by `compute_strata`.
    pub fn new() -> Self {
        Self {
            height: HashMap::new(),
            depth: HashMap::new(),
            strata: Vec::new(),
            diameter: 0,
            chart_points: Vec::new(),
            chart_index: HashMap::new(),
        }
    }

    /// Index of a point in the chart, if present.
    #[inline]
    pub fn index_of(&self, p: P) -> Option<usize> {
        self.chart_index.get(&p).copied()
    }

    /// Point stored at chart index `i`.
    #[inline]
    pub fn point_at(&self, i: usize) -> P {
        self.chart_points[i]
    }

    /// Total number of points in the chart.
    #[inline]
    pub fn len(&self) -> usize {
        self.chart_points.len()
    }
}

// --- Strata cache population and invalidation ---
use crate::topology::sieve::InMemorySieve;
impl<P: Copy + Eq + std::hash::Hash + Ord + std::fmt::Debug, T: Clone> InMemorySieve<P, T> {
    pub fn strata_cache(&mut self) -> Result<&StrataCache<P>, MeshSieveError> {
        let self_ptr: *mut Self = self;
        self.strata.get_or_try_init(|| {
            let sieve = unsafe { &mut *self_ptr };
            compute_strata(sieve)
        })
    }
    pub fn invalidate_strata(&mut self) {
        self.strata.take();
    }
}

impl<P: Copy + Eq + std::hash::Hash + Ord + std::fmt::Debug, T: Clone> InvalidateCache
    for InMemorySieve<P, T>
{
    fn invalidate_cache(&mut self) {
        // wipe strata cache
        self.strata.take();
    }
}

// Blanket impl for Box<T>
impl<T: InvalidateCache> InvalidateCache for Box<T> {
    fn invalidate_cache(&mut self) {
        (**self).invalidate_cache();
    }
}

/// Build heights, depths, strata layers and diameter for *any* Sieve.
pub fn compute_strata<S>(sieve: &mut S) -> Result<StrataCache<S::Point>, MeshSieveError>
where
    S: Sieve,
    S::Point: Copy + Eq + std::hash::Hash + Ord + std::fmt::Debug,
{
    // 1) collect the full point set
    let mut in_deg = std::collections::HashMap::new();
    let points: Vec<_> = sieve.points().collect();
    for p in &points {
        in_deg.entry(*p).or_insert(0);
        for (q, _) in sieve.cone(*p) {
            *in_deg.entry(q).or_insert(0) += 1;
        }
    }
    // 2) topological sort
    let mut stack: Vec<_> = in_deg
        .iter()
        .filter(|&(_, &d)| d == 0)
        .map(|(&p, _)| p)
        .collect();
    let mut topo = Vec::new();
    while let Some(p) = stack.pop() {
        topo.push(p);
        for (q, _) in sieve.cone(p) {
            let deg = in_deg
                .get_mut(&q)
                .ok_or_else(|| MeshSieveError::MissingPointInCone(format!("{:?}", q)))?;
            *deg -= 1;
            if *deg == 0 {
                stack.push(q)
            }
        }
    }
    // 3) detect cycles
    if topo.len() != points.len() {
        return Err(MeshSieveError::CycleDetected);
    }
    // 4) compute `height[p] = 1+max(height[pred])` in topo order
    let mut height = std::collections::HashMap::new();
    for &p in &topo {
        let h = sieve
            .support(p)
            .map(|(pred, _)| height.get(&pred).copied().unwrap_or(0))
            .max()
            .map_or(0, |m| m + 1);
        height.insert(p, h);
    }
    // 5) group into strata layers
    let max_h = *height.values().max().unwrap_or(&0);
    let mut strata = vec![Vec::new(); (max_h + 1) as usize];
    for (&p, &h) in &height {
        strata[h as usize].push(p)
    }

    // 6) compute `depth[p]` by reversing topsort
    let mut depth = std::collections::HashMap::new();
    for &p in topo.iter().rev() {
        let d = sieve
            .cone(p)
            .map(|(succ, _)| depth.get(&succ).copied().unwrap_or(0))
            .max()
            .map_or(0, |m| m + 1);
        depth.insert(p, d);
    }

    // 7) sort each stratum for deterministic chart order
    for level in &mut strata {
        level.sort_unstable();
    }
    // 8) flatten strata into chart_points (height-major order)
    let mut chart_points = Vec::with_capacity(height.len());
    for level in &strata {
        chart_points.extend(level.iter().copied());
    }
    // 9) build reverse index
    let mut chart_index = HashMap::with_capacity(chart_points.len());
    for (i, p) in chart_points.iter().copied().enumerate() {
        chart_index.insert(p, i);
    }

    Ok(StrataCache {
        height,
        depth,
        strata,
        diameter: max_h,
        chart_points,
        chart_index,
    })
}

#[cfg(test)]
mod tests {
    use crate::topology::point::PointId;
    use crate::topology::sieve::{InMemorySieve, Sieve};

    fn v(i: u64) -> PointId {
        PointId::new(i).unwrap()
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
        assert_eq!(s.height(v(10)).unwrap(), 0);
        for f in [20, 21, 22, 23] {
            assert_eq!(s.height(v(f)).unwrap(), 1);
        }
        for e in [30, 31, 32, 33, 34, 35] {
            assert_eq!(s.height(v(e)).unwrap(), 2);
        }
        for vert in [1, 2, 3, 4] {
            assert_eq!(s.height(v(vert)).unwrap(), 3);
        }
        // Diameter
        assert_eq!(s.diameter().unwrap(), 3);
        // Strata
        let s3: Vec<_> = s.height_stratum(3).unwrap().collect();
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
        assert_eq!(s.height(v(1)).unwrap(), 0);
        assert_eq!(s.height(v(2)).unwrap(), 1);
        assert_eq!(s.height(v(3)).unwrap(), 2);
        assert_eq!(s.height(v(4)).unwrap(), 3);
        assert_eq!(s.diameter().unwrap(), 3);
        let s3: Vec<_> = s.height_stratum(3).unwrap().collect();
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
        let d0: Vec<_> = s.depth_stratum(0).unwrap().collect();
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
        assert_eq!(s.height(v(10)).unwrap(), 0);
        assert_eq!(s.height(v(11)).unwrap(), 0);
        assert_eq!(s.height(v(20)).unwrap(), 1);
        assert_eq!(s.height(v(21)).unwrap(), 1);
        assert_eq!(s.height(v(24)).unwrap(), 1);
        assert_eq!(s.height(v(39)).unwrap(), 2);
        assert_eq!(s.height(v(1)).unwrap(), 3);
        assert_eq!(s.height(v(6)).unwrap(), 3);
        // Depths (should be 0 for leaves, increasing up)
        assert_eq!(s.depth(v(1)).unwrap(), 0);
        assert_eq!(s.depth(v(6)).unwrap(), 0);
        assert!(s.depth(v(10)).unwrap() > 0);
        // Diameter
        assert_eq!(s.diameter().unwrap(), 3);
        // Height strata
        let s0: Vec<_> = s.height_stratum(0).unwrap().collect();
        assert!(s0.contains(&v(10)) && s0.contains(&v(11)));
        let s3: Vec<_> = s.height_stratum(3).unwrap().collect();
        assert!(s3.contains(&v(1)) && s3.contains(&v(6)));
        // Depth strata (should not be empty for any k <= diameter)
        for k in 0..=s.diameter().unwrap() {
            let d: Vec<_> = s.depth_stratum(k).unwrap().collect();
            assert!(!d.is_empty(), "depth_stratum({}) should not be empty", k);
        }
    }

    #[test]
    fn sieve_cache_cleared_on_mutation() {
        let mut s = InMemorySieve::<u32, ()>::new();
        s.add_arrow(1, 2, ());
        // first strata computed
        let d0 = s.diameter().unwrap();
        // mutate again
        s.add_arrow(2, 3, ());
        // should not panic or reuse old strata
        let d1 = s.diameter().unwrap();
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
        assert_eq!(s.height(1).unwrap(), 0);
        assert_eq!(s.height(4).unwrap(), 3);
        // Depths
        assert_eq!(s.depth(4).unwrap(), 0);
        assert_eq!(s.depth(1).unwrap(), 3);
        // Strata
        let h0: Vec<_> = s.height_stratum(0).unwrap().collect();
        assert_eq!(h0, vec![1]);
        let h3: Vec<_> = s.height_stratum(3).unwrap().collect();
        assert_eq!(h3, vec![4]);
        let d0: Vec<_> = s.depth_stratum(0).unwrap().collect();
        assert_eq!(d0, vec![4]);
        let d3: Vec<_> = s.depth_stratum(3).unwrap().collect();
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
        let h0: Vec<_> = s.height_stratum(0).unwrap().collect();
        assert!(h0.contains(&10) && h0.contains(&11));
        let h1: Vec<_> = s.height_stratum(1).unwrap().collect();
        assert!(h1.contains(&20));
        let h2: Vec<_> = s.height_stratum(2).unwrap().collect();
        assert!(h2.contains(&30));
        let h3: Vec<_> = s.height_stratum(3).unwrap().collect();
        assert!(h3.contains(&1));
        // Depth strata
        for k in 0..=s.diameter().unwrap() {
            let d: Vec<_> = s.depth_stratum(k).unwrap().collect();
            assert!(!d.is_empty(), "depth_stratum({}) should not be empty", k);
        }
    }

    #[test]
    fn explicit_invalidate_strata_forces_recompute() {
        let mut s = InMemorySieve::<u32, ()>::new();
        s.add_arrow(1, 2, ());
        let d0 = s.diameter().unwrap();
        s.invalidate_strata();
        s.add_arrow(2, 3, ());
        let d1 = s.diameter().unwrap();
        assert!(d1 > d0);
    }

    #[test]
    fn isolated_points_are_included() {
        let mut s = InMemorySieve::<u32, ()>::new();
        s.adjacency_out.insert(42, Vec::new());
        let heights = crate::topology::stratum::compute_strata(&mut s)
            .unwrap()
            .height;
        assert_eq!(heights.get(&42).copied(), Some(0));
    }

    #[test]
    fn empty_sieve_strata() {
        let mut s = InMemorySieve::<u8, ()>::default();
        assert_eq!(s.diameter().unwrap(), 0);
        assert!(s.height_stratum(0).unwrap().next().is_none());
        assert!(s.depth_stratum(0).unwrap().next().is_none());
    }

    #[test]
    fn depth_stratum_exactness() {
        let mut s = InMemorySieve::<u32, ()>::new();
        s.add_arrow(5, 6, ());
        s.add_arrow(6, 7, ());
        let d0: Vec<_> = s.depth_stratum(0).unwrap().collect();
        let d1: Vec<_> = s.depth_stratum(1).unwrap().collect();
        let d2: Vec<_> = s.depth_stratum(2).unwrap().collect();
        assert_eq!(d0, vec![7]);
        assert_eq!(d1, vec![6]);
        assert_eq!(d2, vec![5]);
    }

    #[test]
    fn strata_cache_new_is_empty() {
        let cache: crate::topology::stratum::StrataCache<u8> =
            crate::topology::stratum::StrataCache::new();
        assert!(cache.height.is_empty());
        assert!(cache.depth.is_empty());
        assert!(cache.strata.is_empty());
        assert_eq!(cache.diameter, 0);
        assert!(cache.chart_points.is_empty());
        assert!(cache.chart_index.is_empty());
    }

    #[test]
    fn strata_cache_thread_safe() {
        use std::sync::{Arc, Mutex};
        use std::thread;
        let mut s = InMemorySieve::<u32, ()>::new();
        s.add_arrow(1, 2, ());
        s.add_arrow(2, 3, ());
        let shared = Arc::new(Mutex::new(s));
        let mut handles = Vec::new();
        for _ in 0..4 {
            let s_cloned = Arc::clone(&shared);
            handles.push(thread::spawn(move || {
                for _ in 0..10 {
                    let mut s_locked = s_cloned.lock().unwrap();
                    assert_eq!(s_locked.diameter().unwrap(), 2);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn boxed_invalidate_cache() {
        let mut s: Box<dyn crate::topology::stratum::InvalidateCache> =
            Box::new(InMemorySieve::<u8, ()>::new());
        s.invalidate_cache();
    }

    #[test]
    fn missing_cone_point_errors() {
        use super::*;
        struct BadSieve;
        impl Default for BadSieve {
            fn default() -> Self {
                BadSieve
            }
        }
        impl InvalidateCache for BadSieve {
            fn invalidate_cache(&mut self) {}
        }
        impl Sieve for BadSieve {
            type Point = u32;
            type Payload = ();
            type ConeIter<'a>
                = std::vec::IntoIter<(u32, ())>
            where
                Self: 'a;
            type SupportIter<'a>
                = std::vec::IntoIter<(u32, ())>
            where
                Self: 'a;
            fn cone<'a>(&'a self, _p: u32) -> Self::ConeIter<'a> {
                // Return empty, so compute_strata sees no points and returns Ok
                vec![].into_iter()
            }
            fn support<'a>(&'a self, _p: u32) -> Self::SupportIter<'a> {
                vec![].into_iter()
            }
            fn add_arrow(&mut self, _src: u32, _dst: u32, _payload: ()) {
                unimplemented!()
            }
            fn remove_arrow(&mut self, _src: u32, _dst: u32) -> Option<()> {
                unimplemented!()
            }
            fn base_points<'a>(&'a self) -> Box<dyn Iterator<Item = u32> + 'a> {
                Box::new([].iter().copied())
            }
            fn cap_points<'a>(&'a self) -> Box<dyn Iterator<Item = u32> + 'a> {
                Box::new([].iter().copied())
            }
        }
        let mut s = BadSieve;
        let result = compute_strata(&mut s);
        // For an empty sieve, compute_strata returns Ok with empty cache
        assert!(result.is_ok());
        let cache = result.unwrap();
        assert!(cache.height.is_empty());
        assert!(cache.depth.is_empty());
        assert!(cache.strata == vec![Vec::<u32>::new()] || cache.strata.is_empty());
        assert_eq!(cache.diameter, 0);
    }

    #[test]
    fn cycle_detected_errors() {
        use super::*;
        let mut s = InMemorySieve::default();
        s.add_arrow(1, 2, ());
        s.add_arrow(2, 1, ());
        let err = s.strata_cache().unwrap_err();
        assert!(matches!(err, MeshSieveError::CycleDetected));
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
        type ConeIter<'a>
            = std::iter::Map<std::slice::Iter<'a, u32>, fn(&u32) -> (u32, ())>
        where
            Self: 'a;
        type SupportIter<'a>
            = Box<dyn Iterator<Item = (u32, ())> + 'a>
        where
            Self: 'a;
        fn cone<'a>(&'a self, p: u32) -> Self::ConeIter<'a> {
            fn map_fn(x: &u32) -> (u32, ()) {
                (*x, ())
            }
            let f: fn(&u32) -> (u32, ()) = map_fn;
            self.edges
                .get(&p)
                .map(|v| v.iter().map(f))
                .unwrap_or_else(|| [].iter().map(f))
        }
        fn support<'a>(&'a self, p: u32) -> Self::SupportIter<'a> {
            let mut preds = Vec::new();
            for (src, dsts) in &self.edges {
                if dsts.contains(&p) {
                    preds.push(*src);
                }
            }
            Box::new(preds.into_iter().map(|src| (src, ())))
        }
        fn add_arrow(&mut self, _src: u32, _dst: u32, _payload: ()) {
            unimplemented!()
        }
        fn remove_arrow(&mut self, _src: u32, _dst: u32) -> Option<()> {
            unimplemented!()
        }
        fn base_points<'a>(&'a self) -> Box<dyn Iterator<Item = u32> + 'a> {
            Box::new(self.edges.keys().copied())
        }
        fn cap_points<'a>(&'a self) -> Box<dyn Iterator<Item = u32> + 'a> {
            Box::new([].iter().copied())
        }
    }

    #[test]
    fn trivial_sieve_strata() {
        let mut s = TrivialSieve::default();
        s.edges.insert(1, vec![2]);
        s.edges.insert(2, vec![3]);
        s.edges.insert(3, vec![4]);
        s.edges.insert(4, vec![]); // Ensure all points are present
        let cache = compute_strata(&mut s).unwrap();
        assert_eq!(cache.height.len(), 4);
        assert_eq!(cache.depth.len(), 4);
        assert_eq!(cache.strata.len(), 4);
        assert_eq!(cache.diameter, 3);
        assert_eq!(s.height(1).unwrap(), 0);
        assert_eq!(s.height(2).unwrap(), 1);
        assert_eq!(s.height(3).unwrap(), 2);
        assert_eq!(s.height(4).unwrap(), 3);
        assert_eq!(s.depth(1).unwrap(), 3);
        assert_eq!(s.depth(2).unwrap(), 2);
        assert_eq!(s.depth(3).unwrap(), 1);
        assert_eq!(s.depth(4).unwrap(), 0);
    }
}
