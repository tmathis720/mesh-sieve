// src/topology/sieve/trait.rs

/// Core bidirectional incidence API for mesh topology.
pub trait Sieve {
    type Point: Copy + Eq + std::hash::Hash;
    type Payload;

    type ConeIter<'a>: Iterator<Item = (Self::Point, &'a Self::Payload)> where Self: 'a;
    type SupportIter<'a>: Iterator<Item = (Self::Point, &'a Self::Payload)> where Self: 'a;

    /// Outgoing arrows from `p`.
    fn cone<'a>(&'a self, p: Self::Point) -> Self::ConeIter<'a>;
    /// Incoming arrows to `p`.
    fn support<'a>(&'a self, p: Self::Point) -> Self::SupportIter<'a>;

    /// Insert arrow `src → dst`.
    fn add_arrow(&mut self, src: Self::Point, dst: Self::Point, payload: Self::Payload);
    /// Remove arrow `src → dst`, returning its payload.
    fn remove_arrow(&mut self, src: Self::Point, dst: Self::Point) -> Option<Self::Payload>;

    /// Iterate all points in the domain (sources ∪ sinks).
    fn points<'a>(&'a self) -> Box<dyn Iterator<Item = Self::Point> + 'a>;
    /// All “base” points (with outgoing arrows).
    fn base_points<'a>(&'a self) -> Box<dyn Iterator<Item = Self::Point> + 'a>;
    /// All “cap” points (with incoming arrows).
    fn cap_points<'a>(&'a self) -> Box<dyn Iterator<Item = Self::Point> + 'a>;

    // --- graph traversals ---
    fn closure<'s, I>(&'s self, seeds: I) -> Box<dyn Iterator<Item=Self::Point> + 's>
    where
        I: IntoIterator<Item=Self::Point>,
    {
        use std::collections::HashSet;
        let mut stack: Vec<_> = seeds.into_iter().collect();
        let mut seen: HashSet<Self::Point> = stack.iter().copied().collect();
        Box::new(std::iter::from_fn(move || {
            if let Some(p) = stack.pop() {
                for (q, _) in self.cone(p) {
                    if seen.insert(q) {
                        stack.push(q);
                    }
                }
                Some(p)
            } else {
                None
            }
        }))
    }
    fn star<'s, I>(&'s self, seeds: I) -> Box<dyn Iterator<Item=Self::Point> + 's>
    where
        I: IntoIterator<Item=Self::Point>,
    {
        use std::collections::HashSet;
        let mut stack: Vec<_> = seeds.into_iter().collect();
        let mut seen: HashSet<Self::Point> = stack.iter().copied().collect();
        Box::new(std::iter::from_fn(move || {
            if let Some(p) = stack.pop() {
                for (q, _) in self.support(p) {
                    if seen.insert(q) {
                        stack.push(q);
                    }
                }
                Some(p)
            } else {
                None
            }
        }))
    }
    fn closure_both<'s, I>(&'s self, seeds: I) -> Box<dyn Iterator<Item=Self::Point> + 's>
    where
        I: IntoIterator<Item=Self::Point>,
    {
        use std::collections::HashSet;
        let mut stack: Vec<_> = seeds.into_iter().collect();
        let mut seen: HashSet<Self::Point> = stack.iter().copied().collect();
        Box::new(std::iter::from_fn(move || {
            if let Some(p) = stack.pop() {
                for (q, _) in self.cone(p) {
                    if seen.insert(q) {
                        stack.push(q)
                    }
                }
                for (q, _) in self.support(p) {
                    if seen.insert(q) {
                        stack.push(q)
                    }
                }
                Some(p)
            } else {
                None
            }
        }))
    }

    // --- lattice ops ---
    fn meet<'s>(&'s self, a: Self::Point, b: Self::Point) -> Box<dyn Iterator<Item=Self::Point> + 's>;
    fn join<'s>(&'s self, a: Self::Point, b: Self::Point) -> Box<dyn Iterator<Item=Self::Point> + 's>;

    // --- strata helpers ---
    fn height(&self, p: Self::Point) -> u32;
    fn depth(&self, p: Self::Point) -> u32;
    fn diameter(&self) -> u32;
    fn height_stratum(&self, k: u32) -> Box<dyn Iterator<Item=Self::Point> + '_>;
    fn depth_stratum(&self, k: u32) -> Box<dyn Iterator<Item=Self::Point> + '_>;
}
