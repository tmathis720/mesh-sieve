//! Stratum computation: heights, depths, strata and diameter for a directed acyclic topology.
//!
//! This module provides:
//! 1. `StrataCache<P>`: stores precomputed height, depth, strata layers, and diameter for points of type `P`.
//! 2. `StratumHelpers`: an extension trait on `InMemorySieve` giving convenient access to stratum information.
//! 3. A cache invalidation mechanism in `InMemorySieve` and the core algorithm `compute_strata`.

use std::collections::HashMap;

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

// Extend InMemorySieve with stratum query methods.
/// Trait providing stratum queries on a `Sieve` implementation (only for InMemorySieve).
pub trait StratumHelpers: crate::topology::sieve::Sieve {
    /// Return the height (distance from sources) of point `p`.
    fn height(&self, p: Self::Point) -> u32;
    /// Return the depth  (distance to sinks)   of point `p`.
    fn depth(&self, p: Self::Point) -> u32;
    /// Return the diameter (maximum height) of the entire topology.
    fn diameter(&self) -> u32;
    /// Iterate over all points at given height `k`.
    fn height_stratum(&self, k: u32) -> Box<dyn Iterator<Item = Self::Point> + '_>;
    /// Iterate over all points at given depth `k`.
    fn depth_stratum(&self, k: u32) -> Box<dyn Iterator<Item = Self::Point> + '_>;
}

// Implement StratumHelpers for InMemorySieve
impl<P, T> StratumHelpers for crate::topology::sieve::InMemorySieve<P, T>
where
    P: Copy + Eq + std::hash::Hash + Ord,
    T: Clone,
{
    
    fn height(&self, p: P) -> u32 {
        // Use the cached height if available, otherwise return 0
        self.strata_cache().height.get(&p).copied().unwrap_or(0)
    }
    fn depth(&self, p: P) -> u32 {
        // Use the cached depth if available, otherwise return 0
        self.strata_cache().depth.get(&p).copied().unwrap_or(0)
    }
    fn diameter(&self) -> u32 {
        // Return the precomputed diameter from the cache
        self.strata_cache().diameter
    }
    fn height_stratum(&self, k: u32) -> Box<dyn Iterator<Item=P> + '_> {
        // Return an iterator over points at height k
        // If k is out of bounds, return an empty iterator
        let cache = self.strata_cache();
        if (k as usize) < cache.strata.len() {
            Box::new(cache.strata[k as usize].iter().copied())
        } else {
            Box::new(std::iter::empty())
        }
    }
    fn depth_stratum(&self, k: u32) -> Box<dyn Iterator<Item=P> + '_> {
        // Return an iterator over points at depth k
        let cache = self.strata_cache();
        // Build a reverse map: depth value -> Vec<P>
        let mut depth_map: std::collections::HashMap<u32, Vec<P>> = std::collections::HashMap::new();
        // Populate the map with points grouped by their depth
        for (&p, &d) in &cache.depth {
            depth_map.entry(d).or_default().push(p);
        }
        // Return the iterator for the requested depth k, or an empty iterator if k is not found
        let points = depth_map.get(&k).cloned().unwrap_or_default();
        // Convert Vec<P> into an iterator
        // Use Box to return a trait object for dynamic dispatch
        Box::new(points.into_iter())
    }
}

// --- Strata cache population and invalidation ---
use crate::topology::sieve::InMemorySieve;
impl<P: Copy + Eq + std::hash::Hash + Ord, T: Clone> InMemorySieve<P, T> {
    /// Lazily compute and cache the strata information.
    fn strata_cache(&self) -> &StrataCache<P> {
        self.strata.get_or_init(|| compute_strata(self))
    }
    /// Invalidate the cached strata information.
    /// This should be called whenever the sieve structure changes (e.g., arrows added/removed).
    fn invalidate_strata(&mut self) {
        self.strata.take();
    }
}

/// Compute the strata information using Kahn's algorithm for topological sorting.
/// This computes the height, depth, strata layers, and diameter of the directed acyclic graph (DAG).
/// It returns a `StrataCache` containing all the computed information.
fn compute_strata<P, T>(sieve: &InMemorySieve<P, T>) -> StrataCache<P>
where
    P: Copy + Eq + std::hash::Hash + Ord,
    T: Clone,
{
    // Kahn's algorithm for topological sort
    let mut in_deg = HashMap::new();
    // Count in-degrees for each point
    for (&p, outs) in &sieve.adjacency_out {
        // Initialize in-degree for point `p`
        in_deg.entry(p).or_insert(0);
        // For each outgoing arrow from `p`, increment the in-degree of the target point
        for (q, _) in outs {
            *in_deg.entry(*q).or_insert(0) += 1;
        }
    }
    // Initialize stack with all points that have in-degree 0 (sources)
    // These are the starting points for the topological sort
    // and will be processed first.
    // They are the "cells" in the context of topology.
    // This is the first step in Kahn's algorithm.
    let mut stack: Vec<P> = in_deg.iter().filter(|&(_, &d)| d == 0).map(|(&p, _)| p).collect();
    let mut topo = Vec::new();
    while let Some(p) = stack.pop() {
        topo.push(p);
        if let Some(outs) = sieve.adjacency_out.get(&p) {
            for (q, _) in outs {
                let deg = in_deg.get_mut(q).unwrap();
                *deg -= 1;
                if *deg == 0 {
                    stack.push(*q);
                }
            }
        }
    }
    // Compute height as distance from sources (cells) using adjacency_in
    let mut height = HashMap::new();
    // walk in topological order so parents are done before children
    // This ensures that when we compute the height of a point,
    // all its predecessors have already been processed.
    for &p in &topo {
        // If there are no incoming arrows, height is 0 (source)
        let h = if let Some(ins) = sieve.adjacency_in.get(&p) {
            if ins.is_empty() {
                0
            } else {
                // Otherwise, height is 1 + max height of predecessors
                // This computes the height as the maximum height of incoming points plus one.
                1 + ins.iter().map(|(pred, _)| *height.get(pred).unwrap_or(&0)).max().unwrap_or(0)
            }
        } else {
            0
        };
        height.insert(p, h);
    }
    // Compute strata
    let mut max_h = 0;
    // Find the maximum height to determine the number of strata
    for &h in height.values() { max_h = max_h.max(h); }
    // Initialize strata as a vector of empty vectors, one for each height level
    let mut strata = vec![Vec::new(); (max_h+1) as usize];
    // Populate strata with points grouped by their height
    for (&p, &h) in &height {
        strata[h as usize].push(p);
    }
    // Compute diameter
    let diameter = max_h;
    // Compute depth as distance to sinks (reverse‐cone pass)
    let mut depth = HashMap::new();
    // walk in reverse topological order so children are done before parents
    for &p in topo.iter().rev() {
        // If there are no outgoing arrows, depth is 0 (sink)
        // This computes the depth as the maximum depth of outgoing points plus one.
        // This ensures that when we compute the depth of a point,
        // all its successors have already been processed.
        let d = if let Some(outs) = sieve.adjacency_out.get(&p) {
            if outs.is_empty() {
                0
            } else {
                1 + outs
                    .iter()
                    .map(|(child, _)| *depth.get(child).unwrap_or(&0))
                    .max()
                    .unwrap_or(0)
            }
        } else {
            0
        };
        depth.insert(p, d);
    }
    StrataCache { height, depth, strata, diameter }
}

#[cfg(test)]
mod tests {
    use crate::topology::sieve::{InMemorySieve, Sieve};
    use crate::topology::point::PointId;
    use crate::topology::stratum::StratumHelpers;

    fn v(i: u64) -> PointId { PointId::new(i) }

    #[test]
    fn tetrahedral_block_heights_and_strata() {
        // Tetrahedron: 1 cell, 4 faces, 6 edges, 4 vertices
        // cell: 10
        // faces: 20,21,22,23
        // edges: 30,31,32,33,34,35
        // verts: 1,2,3,4
        let mut s = InMemorySieve::<PointId, ()>::default();
        // cell -> faces
        for f in [v(20), v(21), v(22), v(23)] { s.add_arrow(v(10), f, ()); }
        // faces -> edges
        s.add_arrow(v(20), v(30), ()); s.add_arrow(v(20), v(31), ()); s.add_arrow(v(20), v(32), ());
        s.add_arrow(v(21), v(32), ()); s.add_arrow(v(21), v(33), ()); s.add_arrow(v(21), v(34), ());
        s.add_arrow(v(22), v(34), ()); s.add_arrow(v(22), v(35), ()); s.add_arrow(v(22), v(30), ());
        s.add_arrow(v(23), v(31), ()); s.add_arrow(v(23), v(33), ()); s.add_arrow(v(23), v(35), ());
        // edges -> verts
        s.add_arrow(v(30), v(1), ()); s.add_arrow(v(30), v(2), ());
        s.add_arrow(v(31), v(1), ()); s.add_arrow(v(31), v(3), ());
        s.add_arrow(v(32), v(1), ()); s.add_arrow(v(32), v(4), ());
        s.add_arrow(v(33), v(2), ()); s.add_arrow(v(33), v(3), ());
        s.add_arrow(v(34), v(2), ()); s.add_arrow(v(34), v(4), ());
        s.add_arrow(v(35), v(3), ()); s.add_arrow(v(35), v(4), ());

        // Heights: cell=0, faces=1, edges=2, verts=3
        assert_eq!(s.height(v(10)), 0);
        for f in [20,21,22,23] { assert_eq!(s.height(v(f)), 1); }
        for e in [30,31,32,33,34,35] { assert_eq!(s.height(v(e)), 2); }
        for vert in [1,2,3,4] { assert_eq!(s.height(v(vert)), 3); }
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
        for f in [v(20), v(21), v(22), v(23)] { s.add_arrow(v(10), f, ()); }
        // Tet 11: faces 20,24,25,23
        for f in [v(20), v(24), v(25), v(23)] { s.add_arrow(v(11), f, ()); }
        // Faces to edges (arbitrary but consistent)
        for (f, es) in [
            (20, [30,31,32]), (21, [32,33,34]), (22, [34,35,30]), (23, [31,35,36]),
            (24, [36,37,38]), (25, [38,39,31])
        ] {
            for e in es { s.add_arrow(v(f), v(e), ()); }
        }
        // Edges to verts
        for (e, vs) in [
            (30, [1,2]), (31, [2,3]), (32, [3,4]), (33, [4,1]), (34, [1,5]),
            (35, [5,2]), (36, [3,5]), (37, [5,6]), (38, [6,2]), (39, [6,4])
        ] {
            for vtx in vs { s.add_arrow(v(e), v(vtx), ()); }
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
}
