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
    fn closure_both<'s>(&'s self, seeds: impl IntoIterator<Item=Self::Point>)
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

// --- Minimal separator meet (Algorithm 2, §3.3) ---
/// Returns the minimal separator set X such that closure(a) ∩ closure(b) = closure(X),
/// excluding a, b, and their direct cone/support neighbors.
pub fn meet_minimal_separator<S, P>(sieve: &S, a: P, b: P) -> Vec<P>
where
    S: Sieve<Point = P>,
    P: Copy + Ord + Eq + std::hash::Hash,
{
    // 1. Ca ← closure(a);  Cb ← closure(b)
    let mut ca: Vec<P> = sieve.closure([a]).collect();
    let mut cb: Vec<P> = sieve.closure([b]).collect();
    ca.sort_unstable(); ca.dedup();
    cb.sort_unstable(); cb.dedup();
    // 2. I ← Ca ∩ Cb
    let mut intersection = Vec::new();
    set_intersection(&ca, &cb, &mut intersection);
    // 3. F ← {a, b}
    let filter: Vec<P> = vec![a, b];
    // 4. result ← I \ closure(F)
    let mut to_remove: Vec<P> = sieve.closure(filter.iter().copied()).collect();
    to_remove.sort_unstable(); to_remove.dedup();
    intersection.retain(|x| !to_remove.binary_search(x).is_ok());
    // 5. Keep only maximal elements in the result
    let intersection_set = intersection.clone();
    intersection.retain(|&x| {
        !intersection_set.iter().any(|&y| y != x && sieve.closure([y]).any(|z| z == x))
    });
    intersection
}

/// Helper: set intersection of sorted, deduped Vecs
fn set_intersection<P: Ord + Copy>(a: &[P], b: &[P], out: &mut Vec<P>) {
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        if a[i] < b[j] { i += 1; }
        else if a[i] > b[j] { j += 1; }
        else { out.push(a[i]); i += 1; j += 1; }
    }
}

/// Helper: remove all points reachable from seeds (via closure) from set
fn exclude_closure<S, P>(sieve: &S, seeds: &[P], set: &mut Vec<P>)
where S: Sieve<Point = P>, P: Copy + Ord + Eq + std::hash::Hash {
    let mut to_remove: Vec<P> = sieve.closure(seeds.iter().copied()).collect();
    to_remove.sort_unstable(); to_remove.dedup();
    set.retain(|x| !to_remove.binary_search(x).is_ok());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::point::PointId;

    fn v(i: u64) -> PointId { PointId::new(i) }

    #[test]
    fn meet_two_triangles_shared_vertices() {
        // Two triangles: 10 (1,2,3), 11 (2,3,4), sharing vertices 2 and 3
        let mut s = InMemorySieve::<PointId, ()>::default();
        // triangle 10
        s.add_arrow(v(10), v(1), ());
        s.add_arrow(v(10), v(2), ());
        s.add_arrow(v(10), v(3), ());
        // triangle 11
        s.add_arrow(v(11), v(2), ());
        s.add_arrow(v(11), v(3), ());
        s.add_arrow(v(11), v(4), ());
        // support edges
        for (src, dsts) in [(v(10), [v(1),v(2),v(3)]), (v(11), [v(2),v(3),v(4)])] {
            for d in dsts { s.add_arrow(d, src, ()); }
        }
        let sep = meet_minimal_separator(&s, v(10), v(11));
        // Should be [] (no minimal separator in this model; shared vertices are in closure({a, b}))
        assert!(sep.is_empty());
    }

    #[test]
    fn meet_disjoint_cells() {
        let mut s = InMemorySieve::<PointId, ()>::default();
        s.add_arrow(v(10), v(1), ());
        s.add_arrow(v(10), v(2), ());
        s.add_arrow(v(11), v(3), ());
        s.add_arrow(v(11), v(4), ());
        for (src, dsts) in [(v(10), [v(1),v(2)]), (v(11), [v(3),v(4)])] {
            for d in dsts { s.add_arrow(d, src, ()); }
        }
        let sep = meet_minimal_separator(&s, v(10), v(11));
        assert!(sep.is_empty());
    }

    #[test]
    fn meet_same_cell() {
        let mut s = InMemorySieve::<PointId, ()>::default();
        s.add_arrow(v(10), v(1), ());
        s.add_arrow(v(10), v(2), ());
        for d in [v(1), v(2)] { s.add_arrow(d, v(10), ()); }
        let sep = meet_minimal_separator(&s, v(10), v(10));
        assert!(sep.is_empty());
    }

    #[test]
    fn meet_two_triangles_shared_edge_entity() {
        // Model: triangles 10 (1,2,3), 11 (2,3,4), sharing edge 20 (2,3)
        // Sieve: triangle -> edge(s), edge -> vertex(s)
        let mut s = InMemorySieve::<PointId, ()>::default();
        // triangle 10
        s.add_arrow(v(10), v(21), ()); // edge (1,2)
        s.add_arrow(v(10), v(20), ()); // edge (2,3)
        s.add_arrow(v(10), v(22), ()); // edge (3,1)
        // triangle 11
        s.add_arrow(v(11), v(20), ()); // edge (2,3)
        s.add_arrow(v(11), v(23), ()); // edge (3,4)
        s.add_arrow(v(11), v(24), ()); // edge (4,2)
        // edge to vertices
        s.add_arrow(v(20), v(2), ());
        s.add_arrow(v(20), v(3), ());
        s.add_arrow(v(21), v(1), ());
        s.add_arrow(v(21), v(2), ());
        s.add_arrow(v(22), v(3), ());
        s.add_arrow(v(22), v(1), ());
        s.add_arrow(v(23), v(3), ());
        s.add_arrow(v(23), v(4), ());
        s.add_arrow(v(24), v(4), ());
        s.add_arrow(v(24), v(2), ());
        // support arrows (optional, for bidirectionality)
        for (src, dsts) in [
            (v(10), vec![v(21), v(20), v(22)]),
            (v(11), vec![v(20), v(23), v(24)]),
            (v(20), vec![v(2), v(3)]),
            (v(21), vec![v(1), v(2)]),
            (v(22), vec![v(3), v(1)]),
            (v(23), vec![v(3), v(4)]),
            (v(24), vec![v(4), v(2)]),
        ] {
            for d in dsts { s.add_arrow(d, src, ()); }
        }
        let sep = meet_minimal_separator(&s, v(10), v(11));
        // Should be [] (empty, since closure(10) ∩ closure(11) = closure({10, 11}))
        // and the shared edge 20 is in closure({10, 11}).
        assert_eq!(sep, vec![]);
    }

    #[test]
    fn meet_two_triangles_shared_edge_entity_refined() {
        // Triangles 10 (1,2,3), 11 (2,3,4), sharing edge 20 (2,3)
        // Only triangles point to edges, and edges point to vertices. No support arrows.
        let mut s = InMemorySieve::<PointId, ()>::default();
        // triangle 10
        s.add_arrow(v(10), v(21), ()); // edge (1,2)
        s.add_arrow(v(10), v(20), ()); // edge (2,3)
        s.add_arrow(v(10), v(22), ()); // edge (3,1)
        // triangle 11
        s.add_arrow(v(11), v(20), ()); // edge (2,3)
        s.add_arrow(v(11), v(23), ()); // edge (3,4)
        s.add_arrow(v(11), v(24), ()); // edge (4,2)
        // edge to vertices
        s.add_arrow(v(20), v(2), ());
        s.add_arrow(v(20), v(3), ());
        s.add_arrow(v(21), v(1), ());
        s.add_arrow(v(21), v(2), ());
        s.add_arrow(v(22), v(3), ());
        s.add_arrow(v(22), v(1), ());
        s.add_arrow(v(23), v(3), ());
        s.add_arrow(v(23), v(4), ());
        s.add_arrow(v(24), v(4), ());
        s.add_arrow(v(24), v(2), ());
        // No support arrows!
        let sep = meet_minimal_separator(&s, v(10), v(11));
        // Should be [] (empty, since closure(10) ∩ closure(11) = closure({10, 11}))
        // and the shared edge 20 is not in closure({10, 11}).
        assert_eq!(sep, vec![]);
    }
}
