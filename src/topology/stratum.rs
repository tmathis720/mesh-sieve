use std::collections::HashMap;

/// Cache for strata, heights, depths, and diameter.
#[derive(Clone, Debug)]
pub struct StrataCache<P> {
    pub height: HashMap<P, u32>,
    pub depth: HashMap<P, u32>,
    pub strata: Vec<Vec<P>>, // k -> points
    pub diameter: u32,
}

impl<P: Copy + Eq + std::hash::Hash + Ord> StrataCache<P> {
    pub fn new() -> Self {
        Self {
            height: HashMap::new(),
            depth: HashMap::new(),
            strata: Vec::new(),
            diameter: 0,
        }
    }
}

// Extension trait for Sieve: blanket impl for InMemorySieve only
pub trait StratumHelpers: crate::topology::sieve::Sieve {
    fn height(&self, p: Self::Point) -> u32;
    fn depth(&self, p: Self::Point) -> u32;
    fn diameter(&self) -> u32;
    fn height_stratum(&self, k: u32) -> Box<dyn Iterator<Item=Self::Point> + '_>;
    fn depth_stratum(&self, k: u32) -> Box<dyn Iterator<Item=Self::Point> + '_>;
}

impl<P, T> StratumHelpers for crate::topology::sieve::InMemorySieve<P, T>
where
    P: Copy + Eq + std::hash::Hash + Ord,
    T: Clone,
{
    fn height(&self, p: P) -> u32 {
        self.strata_cache().height.get(&p).copied().unwrap_or(0)
    }
    fn depth(&self, p: P) -> u32 {
        self.strata_cache().depth.get(&p).copied().unwrap_or(0)
    }
    fn diameter(&self) -> u32 {
        self.strata_cache().diameter
    }
    fn height_stratum(&self, k: u32) -> Box<dyn Iterator<Item=P> + '_> {
        let cache = self.strata_cache();
        if (k as usize) < cache.strata.len() {
            Box::new(cache.strata[k as usize].iter().copied())
        } else {
            Box::new(std::iter::empty())
        }
    }
    fn depth_stratum(&self, k: u32) -> Box<dyn Iterator<Item=P> + '_> {
        // Not implemented: for now, just return empty
        Box::new(std::iter::empty())
    }
}

// --- Strata cache population and invalidation ---
use crate::topology::sieve::InMemorySieve;
impl<P: Copy + Eq + std::hash::Hash + Ord, T: Clone> InMemorySieve<P, T> {
    fn strata_cache(&self) -> &StrataCache<P> {
        self.strata.get_or_init(|| compute_strata(self))
    }
    fn invalidate_strata(&mut self) {
        self.strata.take();
    }
}

fn compute_strata<P, T>(sieve: &InMemorySieve<P, T>) -> StrataCache<P>
where
    P: Copy + Eq + std::hash::Hash + Ord,
    T: Clone,
{
    // Kahn's algorithm for topological sort
    let mut in_deg = HashMap::new();
    for (&p, outs) in &sieve.adjacency_out {
        in_deg.entry(p).or_insert(0);
        for (q, _) in outs {
            *in_deg.entry(*q).or_insert(0) += 1;
        }
    }
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
    // Compute height (leaves = 0, parents = 1 + max child)
    let mut height = HashMap::new();
    for &p in topo.iter().rev() {
        let h = if let Some(outs) = sieve.adjacency_out.get(&p) {
            if outs.is_empty() {
                0
            } else {
                1 + outs.iter().map(|(q, _)| *height.get(q).unwrap_or(&0)).max().unwrap_or(0)
            }
        } else {
            0
        };
        height.insert(p, h);
    }
    // Compute strata
    let mut max_h = 0;
    for &h in height.values() { max_h = max_h.max(h); }
    let mut strata = vec![Vec::new(); (max_h+1) as usize];
    for (&p, &h) in &height {
        strata[h as usize].push(p);
    }
    // Compute diameter
    let diameter = max_h;
    // Compute depth (reverse pass)
    let mut depth = HashMap::new();
    for &p in topo.iter() {
        let d = if let Some(ins) = sieve.adjacency_in.get(&p) {
            1 + ins.iter().map(|(q, _)| *depth.get(q).unwrap_or(&0)).max().unwrap_or(0)
        } else {
            0
        };
        depth.insert(p, d);
    }
    StrataCache { height, depth, strata, diameter }
}

#[cfg(test)]
mod tests {
    use super::*;
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
}
