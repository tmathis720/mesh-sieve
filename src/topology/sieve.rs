//! # Sieve: Bidirectional Mesh Topology Interface
//!
//! This module defines the core `Sieve` trait for representing mesh topology
//! as a bidirectional multimap (`cone`/`support`), offers default graph
//! traversal algorithms (closure, star, closure_both), and provides a
//! convenient in-memory implementation `InMemorySieve` using `HashMap`s.

use std::collections::HashMap;

/// The `Sieve` trait models a directed incidence relation (arrows) over
/// mesh points.  It provides:
///
/// - **Forward** incidence (`cone`): all outgoing arrows from a point.
/// - **Backward** incidence (`support`): all incoming arrows to a point.
///
/// Users can insert and remove arrows, and benefit from default graph
/// algorithms: `closure`, `star`, and `closure_both`.
pub trait Sieve {
    /// Type for mesh points (e.g., `PointId`).  Must be `Copy + Eq + std::hash::Hash`.
    type Point: Copy + Eq + std::hash::Hash;
    /// Payload attached to each arrow; can carry orientation, weights, etc.
    type Payload;
    /// Iterator over `(dst, &payload)` for each arrow `p -> dst`.
    type ConeIter<'a>: Iterator<Item = (Self::Point, &'a Self::Payload)>
    where
        Self: 'a;

    /// Iterator over `(src, &payload)` for each arrow `src -> p`.
    type SupportIter<'a>: Iterator<Item = (Self::Point, &'a Self::Payload)>
    where
        Self: 'a;

    //=== Required methods ===

    /// Return all outgoing arrows from `p`.
    fn cone<'a>(&'a self, p: Self::Point) -> Self::ConeIter<'a>;

    /// Return all incoming arrows to `p`.
    fn support<'a>(&'a self, p: Self::Point) -> Self::SupportIter<'a>;

    /// Insert a new arrow `src -> dst` with given payload.
    fn add_arrow(&mut self, src: Self::Point, dst: Self::Point, payload: Self::Payload);

    /// Remove one arrow `src -> dst`, returning its payload if present.
    fn remove_arrow(&mut self, src: Self::Point, dst: Self::Point) -> Option<Self::Payload>;

    /// Returns an iterator over *all* points in this Sieve’s domain.
    /// (Needed for global algorithms: topology‐wide sorts, strata, partitioning, etc.)
    fn points<'a>(&'a self) -> Box<dyn Iterator<Item = Self::Point> + 'a>;

    /// Returns the set of “base” points: those which may have outgoing arrows.
    /// Often equals `self.adjacency_out.keys()`.
    fn base_points<'a>(&'a self) -> Box<dyn Iterator<Item = Self::Point> + 'a>;

    /// Returns the set of “cap” points: those which may have incoming arrows.
    /// Often equals `self.adjacency_in.keys()`.
    fn cap_points<'a>(&'a self) -> Box<dyn Iterator<Item = Self::Point> + 'a>;

    //=== Blanket default graph algorithms ===

    /// Compute the **closure** (transitive hull) following `cone` arrows
    /// from an initial set of `seeds`.  Yields each reachable point once.
    fn closure<'s>(
        &'s self,
        seeds: impl IntoIterator<Item = Self::Point>,
    ) -> impl Iterator<Item = Self::Point> + 's {
        use std::collections::HashSet;
        let mut stack: Vec<_> = seeds.into_iter().collect();
        let mut seen: HashSet<Self::Point> = stack.iter().copied().collect();
        std::iter::from_fn(move || {
            if let Some(p) = stack.pop() {
                // Explore forward neighbors
                for (q, _) in self.cone(p) {
                    if seen.insert(q) {
                        stack.push(q);
                    }
                }
                Some(p)
            } else {
                None
            }
        })
    }

    /// Compute the **star** (dual transitive hull) following `support` arrows
    /// from an initial set of `seeds`.
    fn star<'s>(
        &'s self,
        seeds: impl IntoIterator<Item = Self::Point>,
    ) -> impl Iterator<Item = Self::Point> + 's {
        use std::collections::HashSet;
        let mut stack: Vec<_> = seeds.into_iter().collect();
        let mut seen: HashSet<Self::Point> = stack.iter().copied().collect();
        std::iter::from_fn(move || {
            if let Some(p) = stack.pop() {
                // Explore backward neighbors
                for (q, _) in self.support(p) {
                    if seen.insert(q) {
                        stack.push(q);
                    }
                }
                Some(p)
            } else {
                None
            }
        })
    }

    /// Compute **both** closure and star simultaneously.
    /// Useful for undirected connectivity.
    fn closure_both<'s>(
        &'s self,
        seeds: impl IntoIterator<Item = Self::Point>,
    ) -> impl Iterator<Item = Self::Point> + 's {
        use std::collections::HashSet;
        let mut stack: Vec<_> = seeds.into_iter().collect();
        let mut seen: HashSet<Self::Point> = stack.iter().copied().collect();
        std::iter::from_fn(move || {
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
        })
    }

    //── Default lattice operations (minimal separator / dual separator) ────────
    /// The “meet” of two points: all points in closure(a) ∩ closure(b),
    /// excluding a,b themselves’ immediate cone/support.  
    #[inline]
    fn meet<'s>(&'s self, a: Self::Point, b: Self::Point) -> impl Iterator<Item = Self::Point> + 's
    where
        Self: 's + Sized,
        Self::Point: Ord,
    {
        // 1. collect and sort closures
        let mut ca: Vec<_> = self.closure(std::iter::once(a)).collect();
        let mut cb: Vec<_> = self.closure(std::iter::once(b)).collect();
        ca.sort_unstable();
        cb.sort_unstable();
        // 2. compute intersection
        let mut inter = Vec::with_capacity(ca.len().min(cb.len()));
        let (mut i, mut j) = (0, 0);
        while i < ca.len() && j < cb.len() {
            match ca[i].cmp(&cb[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    inter.push(ca[i]);
                    i += 1;
                    j += 1;
                }
            }
        }
        // 3. exclude closure({a,b})
        let mut to_rm: Vec<_> = self.closure([a, b]).collect();
        to_rm.sort_unstable();
        to_rm.dedup();
        inter
            .into_iter()
            .filter(move |x| to_rm.binary_search(x).is_err())
    }

    /// The “join” of two points: union of star(a) ∪ star(b)
    #[inline]
    fn join<'s>(&'s self, a: Self::Point, b: Self::Point) -> impl Iterator<Item = Self::Point> + 's
    where
        Self: 's + Sized,
        Self::Point: Ord,
    {
        // 1. collect & sort stars
        let mut sa: Vec<_> = self.star(std::iter::once(a)).collect();
        let mut sb: Vec<_> = self.star(std::iter::once(b)).collect();
        sa.sort_unstable();
        sb.sort_unstable();
        // 2. merge union
        let mut out = Vec::with_capacity(sa.len() + sb.len());
        let (mut i, mut j) = (0, 0);
        while i < sa.len() && j < sb.len() {
            match sa[i].cmp(&sb[j]) {
                std::cmp::Ordering::Less => {
                    out.push(sa[i]);
                    i += 1
                }
                std::cmp::Ordering::Greater => {
                    out.push(sb[j]);
                    j += 1
                }
                std::cmp::Ordering::Equal => {
                    out.push(sa[i]);
                    i += 1;
                    j += 1
                }
            }
        }
        out.extend_from_slice(&sa[i..]);
        out.extend_from_slice(&sb[j..]);
        out.into_iter()
    }
}

//-----------------------------------------------------------------------------
// In-memory `Sieve` implementation using HashMaps for fast prototyping.
//-----------------------------------------------------------------------------

/// `InMemorySieve<P, T>` stores two hash maps:
///
/// - `adjacency_out`: Vec of `(dst, payload)` per source point.
/// - `adjacency_in`: Vec of `(src, payload)` per destination point.
/// - `strata`: lazily computed cache of strata/height/depth.
#[derive(Clone, Debug)]
pub struct InMemorySieve<P, T = ()> {
    /// Outgoing edges: src -> [(dst, payload), ...]
    pub adjacency_out: HashMap<P, Vec<(P, T)>>,
    /// Incoming edges: dst -> [(src, payload), ...]
    pub adjacency_in: HashMap<P, Vec<(P, T)>>,
    /// Cached stratification (height/depth).  Invalidated on mutation.
    pub strata: once_cell::sync::OnceCell<crate::topology::stratum::StrataCache<P>>,
}

impl<P: Copy + Eq + std::hash::Hash, T> Default for InMemorySieve<P, T> {
    fn default() -> Self {
        Self {
            adjacency_out: HashMap::new(),
            adjacency_in: HashMap::new(),
            strata: once_cell::sync::OnceCell::new(),
        }
    }
}

impl<P: Copy + Eq + std::hash::Hash, T: Clone> InMemorySieve<P, T> {
    /// Create an empty sieve.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a sieve from a list of `(src, dst, payload)` triples.
    pub fn from_arrows<I: IntoIterator<Item = (P, P, T)>>(arrows: I) -> Self {
        let mut sieve = Self::default();
        for (src, dst, payload) in arrows {
            sieve.add_arrow(src, dst, payload);
        }
        sieve
    }
}

// Helper type for mapping Vec<(P,T)> -> Iterator<(P,&T)>
type ConeMapIter<'a, P, T> =
    std::iter::Map<std::slice::Iter<'a, (P, T)>, fn(&'a (P, T)) -> (P, &'a T)>;

impl<P: Copy + Eq + std::hash::Hash, T: Clone> Sieve for InMemorySieve<P, T> {
    type Point = P;
    type Payload = T;
    type ConeIter<'a>
        = ConeMapIter<'a, P, T>
    where
        Self: 'a;
    type SupportIter<'a>
        = ConeMapIter<'a, P, T>
    where
        Self: 'a;

    /// Return all `(dst, &payload)` for arrows src -> dst.
    fn cone<'a>(&'a self, p: P) -> Self::ConeIter<'a> {
        // Map Vec<(P,T)> to (P,&T)
        fn map_fn<P, T>((dst, pay): &(P, T)) -> (P, &T)
        where
            P: Copy,
        {
            (*dst, pay)
        }
        let f: fn(&(P, T)) -> (P, &T) = map_fn::<P, T>;
        self.adjacency_out
            .get(&p)
            .map(|v| v.iter().map(f))
            .unwrap_or_else(|| [].iter().map(f))
    }

    /// Return all `(src, &payload)` for arrows src -> dst = p.
    fn support<'a>(&'a self, p: P) -> Self::SupportIter<'a> {
        fn map_fn<P, T>((src, pay): &(P, T)) -> (P, &T)
        where
            P: Copy,
        {
            (*src, pay)
        }
        let f: fn(&(P, T)) -> (P, &T) = map_fn::<P, T>;
        self.adjacency_in
            .get(&p)
            .map(|v| v.iter().map(f))
            .unwrap_or_else(|| [].iter().map(f))
    }

    /// Insert an arrow; invalidates strata cache for recomputation.
    fn add_arrow(&mut self, src: P, dst: P, payload: T) {
        self.adjacency_out
            .entry(src)
            .or_default()
            .push((dst, payload.clone()));
        self.adjacency_in
            .entry(dst)
            .or_default()
            .push((src, payload));
        self.strata.take(); // drop cached strata
    }

    /// Remove one arrow, if exists, and return its payload; also invalidate strata.
    fn remove_arrow(&mut self, src: P, dst: P) -> Option<T> {
        let mut removed = None;
        if let Some(v) = self.adjacency_out.get_mut(&src) {
            if let Some(pos) = v.iter().position(|(d, _)| *d == dst) {
                removed = Some(v.remove(pos).1);
            }
        }
        if let Some(v) = self.adjacency_in.get_mut(&dst) {
            if let Some(pos) = v.iter().position(|(s, _)| *s == src) {
                v.remove(pos);
            }
        }
        self.strata.take();
        removed
    }

    /// Returns an iterator over *all* points in this Sieve’s domain.
    /// (Needed for global algorithms: topology‐wide sorts, strata, partitioning, etc.)
    fn points<'a>(&'a self) -> Box<dyn Iterator<Item = Self::Point> + 'a> {
        use std::collections::HashSet;
        let mut keys: HashSet<P> = HashSet::new();
        keys.extend(self.adjacency_out.keys().copied());
        keys.extend(self.adjacency_in.keys().copied());
        Box::new(keys.into_iter())
    }

    /// Returns the set of “base” points: those which may have outgoing arrows.
    /// Often equals `self.adjacency_out.keys()`.
    fn base_points<'a>(&'a self) -> Box<dyn Iterator<Item = Self::Point> + 'a> {
        Box::new(self.adjacency_out.keys().copied())
    }

    /// Returns the set of “cap” points: those which may have incoming arrows.
    /// Often equals `self.adjacency_in.keys()`.
    fn cap_points<'a>(&'a self) -> Box<dyn Iterator<Item = Self::Point> + 'a> {
        Box::new(self.adjacency_in.keys().copied())
    }
}

//----------------------------------------------------------------------------- 
// Minimal separator (Knepley & Karpeev, Alg 2 §3.3)
//-----------------------------------------------------------------------------

/// Compute a minimal separator `X` such that
/// `closure(a) ∩ closure(b) = closure(X)`, excluding direct neighbors of `a,b`.
///
/// Steps:
/// 1. Build closure sets `Ca`, `Cb`.
/// 2. Intersect `I = Ca ∩ Cb`.
/// 3. Remove `closure({a,b})` from `I`.
/// 4. Keep only maximal elements (no deeper closure relation).
pub fn meet_minimal_separator<S, P>(sieve: &S, a: P, b: P) -> Vec<P>
where
    S: Sieve<Point = P>,
    P: Copy + Ord + Eq + std::hash::Hash,
{
    let mut ca: Vec<P> = sieve.closure([a]).collect();
    let mut cb: Vec<P> = sieve.closure([b]).collect();
    ca.sort_unstable();
    ca.dedup();
    cb.sort_unstable();
    cb.dedup();

    // 2. I = intersection(Ca, Cb)
    let mut inter = Vec::new();
    set_intersection(&ca, &cb, &mut inter);

    // 3. Remove closure({a,b})
    let mut to_rm: Vec<P> = sieve.closure([a, b]).collect();
    to_rm.sort_unstable();
    to_rm.dedup();
    inter.retain(|x| to_rm.binary_search(x).is_err());

    // 4. Keep only maximal elements
    let original = inter.clone();
    inter.retain(|&x| {
        !original
            .iter()
            .any(|&y| y != x && sieve.closure([y]).any(|z| z == x))
    });
    inter
}

/// Helper: intersect two sorted, deduplicated slices into `out`.
fn set_intersection<P: Ord + Copy>(a: &[P], b: &[P], out: &mut Vec<P>) {
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        if a[i] < b[j] {
            i += 1;
        } else if a[i] > b[j] {
            j += 1;
        } else {
            out.push(a[i]);
            i += 1;
            j += 1;
        }
    }
}

//-----------------------------------------------------------------------------
// Tests for minimal separator computation
//-----------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::point::PointId;

    fn v(i: u64) -> PointId {
        PointId::new(i)
    }

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
        for (src, dsts) in [(v(10), [v(1), v(2), v(3)]), (v(11), [v(2), v(3), v(4)])] {
            for d in dsts {
                s.add_arrow(d, src, ());
            }
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
        for (src, dsts) in [(v(10), [v(1), v(2)]), (v(11), [v(3), v(4)])] {
            for d in dsts {
                s.add_arrow(d, src, ());
            }
        }
        let sep = meet_minimal_separator(&s, v(10), v(11));
        assert!(sep.is_empty());
    }

    #[test]
    fn meet_same_cell() {
        let mut s = InMemorySieve::<PointId, ()>::default();
        s.add_arrow(v(10), v(1), ());
        s.add_arrow(v(10), v(2), ());
        for d in [v(1), v(2)] {
            s.add_arrow(d, v(10), ());
        }
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
            for d in dsts {
                s.add_arrow(d, src, ());
            }
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
        let sep = meet_minimal_separator(&s, v(10), v(11));
        // Should be [] (empty, since closure(10) ∩ closure(11) = closure({10, 11}))
        // and the shared edge 20 is in closure({10, 11}).
        assert_eq!(sep, vec![]);
    }
}

#[cfg(test)]
mod lattice_tests {
    use super::*;
    use crate::topology::point::PointId as P;
    use crate::topology::sieve::InMemorySieve;

    fn p(x: u64) -> P {
        P::new(x)
    }

    /// Build the 1D chain 1→2→3→4
    fn chain() -> InMemorySieve<P, ()> {
        let mut s = InMemorySieve::default();
        for (u, v) in &[(1, 2), (2, 3), (3, 4)] {
            s.add_arrow(p(*u), p(*v), ());
        }
        s
    }

    #[test]
    fn meet_chain() {
        let s = chain();
        // closure(2) = {2,3,4}, closure(3) = {3,4}; so meet = [] (since closure({2,3}) = {2,3,4})
        let mut m: Vec<_> = s.meet(p(2), p(3)).collect();
        m.sort();
        m.dedup();
        assert_eq!(m, vec![]);
    }

    #[test]
    fn join_chain() {
        let s = chain();
        // star(2) = {2,1}, star(3) = {3,2,1}, union = {1,2,3}
        let mut j: Vec<_> = s.join(p(2), p(3)).collect();
        j.sort();
        j.dedup();
        assert_eq!(j, vec![p(1), p(2), p(3)]);
    }
}

#[cfg(test)]
mod pointset_tests {
    use super::*;
    use crate::topology::point::PointId as P;
    use crate::topology::sieve::InMemorySieve;

    fn s() -> InMemorySieve<P, ()> {
        let mut s = InMemorySieve::default();
        s.add_arrow(P::new(1), P::new(2), ());
        s.add_arrow(P::new(3), P::new(2), ());
        s
    }

    #[test]
    fn points_union_base_and_cap() {
        let s = s();
        let mut pts: Vec<_> = s.points().collect();
        pts.sort_unstable();
        assert_eq!(pts, vec![P::new(1), P::new(2), P::new(3)]);
    }

    #[test]
    fn base_vs_cap_points() {
        let s = s();
        let mut base: Vec<_> = s.base_points().collect();
        base.sort_unstable();
        assert_eq!(base, vec![P::new(1), P::new(3)]);

        let mut cap: Vec<_> = s.cap_points().collect();
        cap.sort_unstable();
        assert_eq!(cap, vec![P::new(2)]);
    }
}
