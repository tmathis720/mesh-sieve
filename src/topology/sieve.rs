//! Sieve trait and in-memory implementation

use std::collections::HashMap;

/// The Sieve trait: bidirectional multimap for mesh topology.
pub trait Sieve {
    /// Handle that indexes the topology (usually `PointId`).
    type Point: Copy + Eq + std::hash::Hash;
    /// Per-arrow user payload.
    type Payload;
    /// Iterator of (dst, &payload) leaving `p` ("cone").
    type ConeIter<'a>: Iterator<Item = (Self::Point, &'a Self::Payload)> where Self: 'a;
    /// Iterator of (src, &payload) entering `p` ("support").
    type SupportIter<'a>: Iterator<Item = (Self::Point, &'a Self::Payload)> where Self: 'a;

    // --- Required methods ---
    fn cone<'a>(&'a self, p: Self::Point) -> Self::ConeIter<'a>;
    fn support<'a>(&'a self, p: Self::Point) -> Self::SupportIter<'a>;
    fn add_arrow(&mut self, src: Self::Point, dst: Self::Point, payload: Self::Payload);
    fn remove_arrow(&mut self, src: Self::Point, dst: Self::Point) -> Option<Self::Payload>;

    // --- Blanket default algorithms ---
    fn closure<'s>(&'s self, seeds: impl IntoIterator<Item=Self::Point>)
        -> impl Iterator<Item = Self::Point> + 's
    {
        use std::collections::HashSet;
        let mut stack: Vec<_> = seeds.into_iter().collect();
        let mut seen: HashSet<Self::Point> = stack.iter().copied().collect();
        std::iter::from_fn(move || {
            while let Some(p) = stack.pop() {
                for (q, _) in self.cone(p) {
                    if seen.insert(q) { stack.push(q) }
                }
                return Some(p);
            }
            None
        })
    }
    fn star<'s>(&'s self, seeds: impl IntoIterator<Item=Self::Point>)
        -> impl Iterator<Item = Self::Point> + 's
    {
        use std::collections::HashSet;
        let mut stack: Vec<_> = seeds.into_iter().collect();
        let mut seen: HashSet<Self::Point> = stack.iter().copied().collect();
        std::iter::from_fn(move || {
            while let Some(p) = stack.pop() {
                for (q, _) in self.support(p) {
                    if seen.insert(q) { stack.push(q) }
                }
                return Some(p);
            }
            None
        })
    }
}

/// In-memory implementation of Sieve using HashMaps.
pub struct InMemorySieve<P, T = ()> {
    adjacency_out: HashMap<P, Vec<(P, T)>>,
    adjacency_in: HashMap<P, Vec<(P, T)>>,
}

impl<P: Copy + Eq + std::hash::Hash, T> Default for InMemorySieve<P, T> {
    fn default() -> Self {
        Self {
            adjacency_out: HashMap::new(),
            adjacency_in: HashMap::new(),
        }
    }
}

impl<P: Copy + Eq + std::hash::Hash, T: Clone> InMemorySieve<P, T> {
    pub fn new() -> Self { Self::default() }
    pub fn from_arrows<I: IntoIterator<Item = (P, P, T)>>(arrows: I) -> Self {
        let mut sieve = Self::default();
        for (src, dst, payload) in arrows {
            sieve.add_arrow(src, dst, payload);
        }
        sieve
    }
}

type ConeMapIter<'a, P, T> = std::iter::Map<std::slice::Iter<'a, (P, T)>, fn(&'a (P, T)) -> (P, &'a T)>;

type EmptyMapIter<'a, P, T> = std::iter::Map<std::slice::Iter<'a, (P, T)>, fn(&'a (P, T)) -> (P, &'a T)>;

impl<P: Copy + Eq + std::hash::Hash, T: Clone> Sieve for InMemorySieve<P, T> {
    type Point = P;
    type Payload = T;
    type ConeIter<'a> = ConeMapIter<'a, P, T> where Self: 'a;
    type SupportIter<'a> = ConeMapIter<'a, P, T> where Self: 'a;

    fn cone<'a>(&'a self, p: P) -> Self::ConeIter<'a> {
        fn map_fn<P, T>((dst, payload): &(P, T)) -> (P, &T) where P: Copy { (*dst, payload) }
        let f: fn(&(P, T)) -> (P, &T) = map_fn::<P, T>;
        self.adjacency_out.get(&p)
            .map(|v| v.iter().map(f))
            .unwrap_or_else(|| [].iter().map(f))
    }
    fn support<'a>(&'a self, p: P) -> Self::SupportIter<'a> {
        fn map_fn<P, T>((src, payload): &(P, T)) -> (P, &T) where P: Copy { (*src, payload) }
        let f: fn(&(P, T)) -> (P, &T) = map_fn::<P, T>;
        self.adjacency_in.get(&p)
            .map(|v| v.iter().map(f))
            .unwrap_or_else(|| [].iter().map(f))
    }
    fn add_arrow(&mut self, src: P, dst: P, payload: T) {
        self.adjacency_out.entry(src).or_default().push((dst, payload.clone()));
        self.adjacency_in.entry(dst).or_default().push((src, payload));
    }
    fn remove_arrow(&mut self, src: P, dst: P) -> Option<T> {
        let mut removed = None;
        if let Some(vec) = self.adjacency_out.get_mut(&src) {
            if let Some(pos) = vec.iter().position(|(d, _)| *d == dst) {
                removed = Some(vec.remove(pos).1);
            }
        }
        if let Some(vec) = self.adjacency_in.get_mut(&dst) {
            if let Some(pos) = vec.iter().position(|(s, _)| *s == src) {
                vec.remove(pos);
            }
        }
        removed
    }
}
