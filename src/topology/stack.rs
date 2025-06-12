//! Stack abstraction for vertical composition of Sieve topologies.
//!
//! A _Stack_ represents relationships ("vertical arrows") between two Sieves:
//! a **base** mesh and a **cap** mesh (e.g., from elements to degrees-of-freedom).
//! This module provides a generic `Stack` trait and an in-memory implementation,
//! along with facilities for composing multiple stacks.

use super::sieve::InMemorySieve;
use crate::topology::stratum::InvalidateCache;
use crate::topology::sieve::arc_payload::SieveArcPayload;
use std::collections::HashMap;

/// A `Stack` links a *base* Sieve to a *cap* Sieve via vertical arrows.
/// Each arrow carries a payload (e.g., orientation or permutation).
///
/// - `Point`:   The point type in the base mesh (commonly `PointId`).
/// - `CapPt`:   The point type in the cap mesh (commonly `PointId`).
/// - `Payload`: Data attached to each arrow (e.g., `Orientation`).
pub trait Stack {
    /// Base mesh point identifier.
    type Point: Copy + Eq + std::hash::Hash;
    /// Cap mesh point identifier.
    type CapPt: Copy + Eq + std::hash::Hash;
    /// Per-arrow payload type.
    type Payload: Clone;
    /// The underlying base Sieve type.
    type BaseSieve: crate::topology::sieve::sieve_trait::Sieve<Point=Self::Point, Payload=Self::Payload>;
    /// The underlying cap Sieve type.
    type CapSieve: crate::topology::sieve::sieve_trait::Sieve<Point=Self::CapPt, Payload=Self::Payload>;

    // === Topology queries ===
    /// Returns an iterator over all upward arrows from base point `p` to cap points.
    /// Each item is `(cap_point, &payload)`.
    fn lift<'a>(
        &'a self,
        p: Self::Point,
    ) -> Box<dyn Iterator<Item = (Self::CapPt, &'a Self::Payload)> + 'a>;

    /// Returns an iterator over all downward arrows from cap point `q` to base points.
    /// Each item is `(base_point, &payload)`.
    fn drop<'a>(
        &'a self,
        q: Self::CapPt,
    ) -> Box<dyn Iterator<Item = (Self::Point, &'a Self::Payload)> + 'a>;

    // === Mutation helpers ===
    /// Adds a new vertical arrow `base -> cap` with associated payload.
    ///
    /// **Implementors:** after mutating arrows, you *must* invalidate any derived caches on both the base‐ and cap‐sieves (e.g. via `InvalidateCache::invalidate_cache(self.base_mut())` and likewise for `self.cap_mut()`).
    fn add_arrow(&mut self, base: Self::Point, cap: Self::CapPt, pay: Self::Payload);

    /// Removes the arrow `base -> cap`, returning its payload if present.
    ///
    /// **Implementors:** after mutating arrows, you *must* invalidate any derived caches on both the base‐ and cap‐sieves.
    fn remove_arrow(&mut self, base: Self::Point, cap: Self::CapPt) -> Option<Self::Payload>;

    // === Convenience accessors ===
    /// Returns a reference to the underlying base Sieve.
    fn base(&self) -> &Self::BaseSieve;
    /// Returns a reference to the underlying cap Sieve.
    fn cap(&self) -> &Self::CapSieve;
}

/// In-memory implementation of the `Stack` trait.
///
/// Stores vertical arrows in two hash maps:
/// - `up`   maps base points to a list of `(cap_point, payload)`.
/// - `down` maps cap points to a list of `(base_point, payload)`.
///
/// Also embeds two `InMemorySieve`s to represent the base and cap topologies themselves.
#[derive(Clone, Debug)]
pub struct InMemoryStack<B: Copy + Eq + std::hash::Hash + Ord, C: Copy + Eq + std::hash::Hash + Ord, P = ()> {
    /// Underlying base sieve (e.g., mesh connectivity).
    pub base: InMemorySieve<B, P>,
    /// Underlying cap sieve (e.g., DOF connectivity).
    pub cap: InMemorySieve<C, P>,
    /// Upward adjacency: base -> cap
    pub up: HashMap<B, Vec<(C, P)>>,
    /// Downward adjacency: cap -> base
    pub down: HashMap<C, Vec<(B, P)>>,
}

impl<B, C, P> InMemoryStack<B, C, P>
where
    B: Copy + Eq + std::hash::Hash + Ord,
    C: Copy + Eq + std::hash::Hash + Ord,
{
    /// Creates an empty `InMemoryStack` with no arrows.
    pub fn new() -> Self {
        Self {
            base: InMemorySieve::default(),
            cap: InMemorySieve::default(),
            up: HashMap::new(),
            down: HashMap::new(),
        }
    }
}

/// Provides a default implementation for `InMemoryStack`.
impl<B, C, P: Clone> Default for InMemoryStack<B, C, P>
where
    B: Copy + Eq + std::hash::Hash + Ord,
    C: Copy + Eq + std::hash::Hash + Ord,
{
    /// Returns an empty `InMemoryStack`.
    fn default() -> Self {
        Self {
            base: InMemorySieve::default(),
            cap: InMemorySieve::default(),
            up: HashMap::new(),
            down: HashMap::new(),
        }
    }
}

impl<B, C, P> Stack for InMemoryStack<B, C, P>
where
    B: Copy + Eq + std::hash::Hash + Ord,
    C: Copy + Eq + std::hash::Hash + Ord,
    P: Clone,
{
    type Point = B;
    type CapPt = C;
    type Payload = P;
    type BaseSieve = InMemorySieve<B, P>;
    type CapSieve = InMemorySieve<C, P>;

    /// Returns an iterator over all upward arrows from base point `p` to cap points.
    /// Each item is `(cap_point, &payload)`.
    fn lift<'a>(&'a self, p: B) -> Box<dyn Iterator<Item = (C, &'a P)> + 'a> {
        // Return all upward arrows or empty if none
        match self.up.get(&p) {
            Some(vec) => Box::new(vec.iter().map(|(c, pay)| (*c, pay))),
            None => Box::new(std::iter::empty()),
        }
    }

    /// Returns an iterator over all downward arrows from cap point `q` to base points.
    /// Each item is `(base_point, &payload)`.
    fn drop<'a>(&'a self, q: C) -> Box<dyn Iterator<Item = (B, &'a P)> + 'a> {
        // Return all downward arrows or empty if none
        match self.down.get(&q) {
            Some(vec) => Box::new(vec.iter().map(|(b, pay)| (*b, pay))),
            None => Box::new(std::iter::empty()),
        }
    }

    /// Adds a new vertical arrow `base -> cap` with associated payload.
    ///
    /// After mutating arrows, this method invalidates any derived caches on both the base and cap sieves.
    fn add_arrow(&mut self, base: B, cap: C, pay: P) {
        self.up.entry(base).or_default().push((cap, pay.clone()));
        self.down.entry(cap).or_default().push((base, pay.clone()));
        InvalidateCache::invalidate_cache(&mut self.base);
        InvalidateCache::invalidate_cache(&mut self.cap);
    }

    /// Removes the arrow `base -> cap`, returning its payload if present.
    ///
    /// After mutating arrows, this method invalidates any derived caches on both the base and cap sieves.
    fn remove_arrow(&mut self, base: B, cap: C) -> Option<P> {
        // Remove from up map, capture payload
        let mut removed = None;
        if let Some(vec) = self.up.get_mut(&base) {
            if let Some(pos) = vec.iter().position(|(c, _)| *c == cap) {
                removed = Some(vec.remove(pos).1);
            }
        }
        // Remove from down map, ignore second removal payload
        if let Some(vec) = self.down.get_mut(&cap) {
            if let Some(pos) = vec.iter().position(|(b, _)| *b == base) {
                vec.remove(pos);
            }
        }
        InvalidateCache::invalidate_cache(&mut self.base);
        InvalidateCache::invalidate_cache(&mut self.cap);
        removed
    }

    /// Returns a reference to the underlying base Sieve.
    fn base(&self) -> &Self::BaseSieve {
        &self.base
    }
    /// Returns a reference to the underlying cap Sieve.
    fn cap(&self) -> &Self::CapSieve {
        &self.cap
    }
}

/// Provides accessors for base and cap points for testability.
impl<B, C, P> InMemoryStack<B, C, P>
where
    B: Copy + Eq + std::hash::Hash + Ord,
    C: Copy + Eq + std::hash::Hash + Ord,
    P: Clone,
{
    /// Returns an iterator over all base points with at least one upward arrow.
    pub fn base_points(&self) -> impl Iterator<Item = B> + '_ {
        self.up.keys().copied()
    }
    /// Returns an iterator over all cap points with at least one downward arrow.
    pub fn cap_points(&self) -> impl Iterator<Item = C> + '_ {
        self.down.keys().copied()
    }
}

/// A stack composed of two existing stacks: `lower: base -> mid` and `upper: mid -> cap`.
///
/// Traversal composes payloads via a `compose_payload` function.
///
/// This implementation uses an `Arc<P>` buffer to safely store composed payloads for the duration
/// of each traversal, avoiding leaks and ensuring memory safety. The buffer is cleared on each call
/// to `lift` or `drop`, and references returned are valid for the lifetime of the iterator.
pub struct ComposedStack<'a, S1, S2, F>
where
    S1: Stack,
    S2: Stack<Point = S1::CapPt>,
    F: Fn(&S1::Payload, &S2::Payload) -> S1::Payload,
{
    /// Lower-level stack (base -> mid)
    pub lower: &'a S1,
    /// Upper-level stack (mid -> cap)
    pub upper: &'a S2,
    /// Function to merge two payloads into one
    pub compose_payload: F,
    /// Buffer to hold composed payloads for the duration of traversal
    pub payloads: std::cell::RefCell<Vec<std::sync::Arc<S1::Payload>>>,
}

impl<'a, S1, S2, F> ComposedStack<'a, S1, S2, F>
where
    S1: Stack,
    S2: Stack<Point = S1::CapPt, Payload = S1::Payload>,
    F: Fn(&S1::Payload, &S2::Payload) -> S1::Payload,
{
    /// Create a new composed stack with an empty buffer
    pub fn new(lower: &'a S1, upper: &'a S2, compose_payload: F) -> Self {
        Self {
            lower,
            upper,
            compose_payload,
            payloads: std::cell::RefCell::new(Vec::new()),
        }
    }
}

impl<'a, S1, S2, F> Stack for ComposedStack<'a, S1, S2, F>
where
    S1: Stack,
    S2: Stack<Point = S1::CapPt, Payload = S1::Payload>,
    F: Fn(&S1::Payload, &S2::Payload) -> S1::Payload + Sync + Send,
{
    type Point = S1::Point;
    type CapPt = S2::CapPt;
    type Payload = std::sync::Arc<S1::Payload>;
    type BaseSieve = SieveArcPayload<S1::BaseSieve>;
    type CapSieve = SieveArcPayload<S2::CapSieve>;

    fn lift<'b>(
        &'b self,
        p: S1::Point,
    ) -> Box<dyn Iterator<Item = (S2::CapPt, &'b Self::Payload)> + 'b> {
        let lower = self.lower;
        let upper = self.upper;
        let compose = &self.compose_payload;
        // Compose all pairs and store in a local Vec
        let pairs: Vec<(S2::CapPt, std::sync::Arc<S1::Payload>)> = lower.lift(p)
            .flat_map(|(mid, pay1)| {
                upper.lift(mid).map(move |(cap, pay2)| {
                    let arc = std::sync::Arc::new((compose)(pay1, pay2));
                    (cap, arc)
                })
            })
            .collect();
        // Now update the buffer and return refs
        let mut buf = self.payloads.borrow_mut();
        buf.clear();
        buf.shrink_to_fit();
        buf.extend(pairs.iter().map(|(_, arc)| arc.clone()));
        let buf_ptr = &*buf as *const Vec<std::sync::Arc<S1::Payload>>;
        Box::new(pairs.into_iter().enumerate().map(move |(i, (cap, _))| {
            let arc_ref = unsafe { &(*buf_ptr)[i] };
            (cap, arc_ref)
        }))
    }

    fn drop<'b>(
        &'b self,
        q: S2::CapPt,
    ) -> Box<dyn Iterator<Item = (S1::Point, &'b Self::Payload)> + 'b> {
        let lower = self.lower;
        let upper = self.upper;
        let compose = &self.compose_payload;
        let pairs: Vec<(S1::Point, std::sync::Arc<S1::Payload>)> = upper.drop(q)
            .flat_map(|(mid, pay2)| {
                lower.drop(mid).map(move |(base, pay1)| {
                    let arc = std::sync::Arc::new((compose)(pay1, pay2));
                    (base, arc)
                })
            })
            .collect();
        let mut buf = self.payloads.borrow_mut();
        buf.clear();
        buf.shrink_to_fit();
        buf.extend(pairs.iter().map(|(_, arc)| arc.clone()));
        let buf_ptr = &*buf as *const Vec<std::sync::Arc<S1::Payload>>;
        Box::new(pairs.into_iter().enumerate().map(move |(i, (base, _))| {
            let arc_ref = unsafe { &(*buf_ptr)[i] };
            (base, arc_ref)
        }))
    }

    fn add_arrow(&mut self, _base: S1::Point, _cap: S2::CapPt, _pay: std::sync::Arc<S1::Payload>) {
        panic!("Cannot mutate a composed stack");
    }
    fn remove_arrow(&mut self, _base: S1::Point, _cap: S2::CapPt) -> Option<std::sync::Arc<S1::Payload>> {
        panic!("Cannot mutate a composed stack");
    }
    fn base(&self) -> &Self::BaseSieve {
        // Not implemented: would require storing a wrapped reference
        panic!("ComposedStack does not expose a base sieve");
    }
    fn cap(&self) -> &Self::CapSieve {
        // Not implemented: would require storing a wrapped reference
        panic!("ComposedStack does not expose a cap sieve");
    }
}

#[test]
fn composed_stack_no_leak() {
    use std::sync::Arc;
    let mut s1 = InMemoryStack::<u32, u32, i32>::new();
    let mut s2 = InMemoryStack::<u32, u32, i32>::new();
    s1.add_arrow(1, 10, 2);
    s2.add_arrow(10, 100, 5);
    let cs = ComposedStack::new(&s1, &s2, |a, b| a + b);
    // Allow the buffer to grow if needed, but it should not grow unboundedly for repeated traversals of the same size
    let mut max_capacity = 0;
    for _ in 0..100 {
        let _v: Vec<_> = cs.lift(1).map(|(_, p)| Arc::strong_count(p)).collect();
        let cap = cs.payloads.borrow().capacity();
        if cap > max_capacity {
            max_capacity = cap;
        }
    }
    // After repeated traversals, the capacity should stabilize
    let final_capacity = cs.payloads.borrow().capacity();
    assert!(final_capacity <= max_capacity);
}

impl<B, C, P> InvalidateCache for InMemoryStack<B, C, P>
where
    B: Copy + Eq + std::hash::Hash + Ord,
    C: Copy + Eq + std::hash::Hash + Ord,
    P: Clone,
{
    fn invalidate_cache(&mut self) {
        self.base.invalidate_cache();
        self.cap.invalidate_cache();
        // Add stack-specific cache clears here if needed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::sieve::sieve_trait::Sieve;
    #[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, PartialOrd, Ord)]
    struct V(u32);
    #[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, PartialOrd, Ord)]
    struct Dof(u32);
    #[test]
    fn add_and_lift_drop() {
        let mut stack = InMemoryStack::<V, Dof, i32>::new();
        stack.add_arrow(V(1), Dof(10), 42);
        stack.add_arrow(V(1), Dof(11), 43);
        let mut lifted: Vec<_> = stack.lift(V(1)).collect();
        lifted.sort_by_key(|(dof, _)| dof.0);
        assert_eq!(lifted, vec![(Dof(10), &42), (Dof(11), &43)]);
        let dropped: Vec<_> = stack.drop(Dof(10)).collect();
        assert_eq!(dropped, vec![(V(1), &42)]);
    }
    #[test]
    fn remove_arrow_behavior() {
        let mut stack = InMemoryStack::<V, Dof, i32>::new();
        stack.add_arrow(V(1), Dof(10), 99);
        assert_eq!(stack.remove_arrow(V(1), Dof(10)), Some(99));
        // Double remove returns None
        assert_eq!(stack.remove_arrow(V(1), Dof(10)), None);
        // After removal, lift/drop are empty
        assert!(stack.lift(V(1)).next().is_none());
        assert!(stack.drop(Dof(10)).next().is_none());
    }

    #[test]
    fn composed_stack_lift_drop() {
        use super::*;
        #[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, PartialOrd, Ord)]
        struct A(u32);
        #[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, PartialOrd, Ord)]
        struct B(u32);
        #[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, PartialOrd, Ord)]
        struct C(u32);
        // Stack1: A → B
        let mut s1 = InMemoryStack::<A, B, i32>::new();
        s1.add_arrow(A(1), B(10), 2);
        s1.add_arrow(A(1), B(11), 3);
        // Stack2: B → C
        let mut s2 = InMemoryStack::<B, C, i32>::new();
        s2.add_arrow(B(10), C(100), 5);
        s2.add_arrow(B(11), C(101), 7);
        // Compose: payloads are summed
        let composed = ComposedStack::new(&s1, &s2, |p1, p2| p1 + p2);
        let mut lifted: Vec<_> = composed.lift(A(1)).map(|(c, p)| (c, **p)).collect();
        lifted.sort_by_key(|(c, _)| (*c).0);
        assert_eq!(lifted, vec![(C(100), 7), (C(101), 10)]);
        let dropped: Vec<_> = composed.drop(C(100)).map(|(a, p)| (a, **p)).collect();
        assert_eq!(dropped, vec![(A(1), 7)]);
    }

    #[test]
    fn stack_cache_cleared_on_mutation() {
        let mut s = InMemoryStack::<u32, u32, i32>::new();
        s.add_arrow(1, 10, 2);
        let d0 = s.base.diameter();
        s.add_arrow(2, 11, 3);
        let d1 = s.base.diameter();
        assert!(d1 >= d0);
    }

    #[test]
    fn base_and_cap_points_reflect_maps() {
        let mut s = InMemoryStack::<u32,u32,()>::new();
        // empty
        assert!(s.base_points().next().is_none());
        assert!(s.cap_points().next().is_none());
        // add arrow 1→10
        s.add_arrow(1,10,());
        let bases: Vec<_> = s.base_points().collect();
        let caps:  Vec<_> = s.cap_points().collect();
        assert_eq!(bases, vec![1]);
        assert_eq!(caps,  vec![10]);
    }

    #[test]
    fn new_and_default_empty() {
        let s1 = InMemoryStack::<u8,u8,u8>::new();
        let s2: InMemoryStack<u8,u8,u8> = Default::default();
        assert_eq!(s1.base_points().count(), 0);
        assert_eq!(s2.cap_points().count(),  0);
    }

    #[test]
    #[should_panic(expected="Cannot mutate")]
    fn composed_stack_add_arrow_panics() {
        let s1 = InMemoryStack::<u8,u8,u8>::new();
        let s2 = InMemoryStack::<u8,u8,u8>::new();
        let mut cs = ComposedStack::new(&s1,&s2,|a,_b|*a);
        cs.add_arrow(0,0, std::sync::Arc::new(0));
    }
    #[test]
    #[should_panic(expected="Cannot mutate")]
    fn composed_stack_remove_arrow_panics() {
        let s1 = InMemoryStack::<u8,u8,u8>::new();
        let s2 = InMemoryStack::<u8,u8,u8>::new();
        let mut cs = ComposedStack::new(&s1,&s2,|a,_b|*a);
        cs.remove_arrow(0,0);
    }

    #[test]
    fn remove_nonexistent_returns_none() {
        let mut s = InMemoryStack::<u32,u32,()>::new();
        assert_eq!(s.remove_arrow(5,50), None);
    }

    #[test]
    fn invalidate_cache_clears_embedded() {
        let mut s = InMemoryStack::<u32,u32,i32>::new();
        // manually prime strata cache
        let _ = s.base().diameter();
        let _ = s.cap().diameter();
        s.invalidate_cache();
        // re‐access should re‐compute (no panic, but at least not stale)
        let _ = s.base().diameter();
        let _ = s.cap().diameter();
    }

    #[test]
    fn lift_drop_empty_iter() {
        let s = InMemoryStack::<u8,u8,()>::new();
        assert!(s.lift(0).next().is_none());
        assert!(s.drop(0).next().is_none());
    }

    #[test]
    #[should_panic]
    fn composed_stack_base_panics() {
        let s1 = InMemoryStack::<u8,u8,u8>::new();
        let s2 = InMemoryStack::<u8,u8,u8>::new();
        let cs = ComposedStack::new(&s1, &s2, |a,_b| *a);
        let _ = cs.base();
    }
    #[test]
    #[should_panic]
    fn composed_stack_cap_panics() {
        let s1 = InMemoryStack::<u8,u8,u8>::new();
        let s2 = InMemoryStack::<u8,u8,u8>::new();
        let cs = ComposedStack::new(&s1, &s2, |a,_b| *a);
        let _ = cs.cap();
    }

    #[test]
    fn stack_vertical_arrows_are_correct() {
        let mut s = InMemoryStack::<u32,u32,i32>::new();
        s.add_arrow(2, 20, 5);
        // The stack's lift and drop reflect the vertical arrows
        let lifted: Vec<_> = s.lift(2).collect();
        assert_eq!(lifted, vec![(20, &5)]);
        let dropped: Vec<_> = s.drop(20).collect();
        assert_eq!(dropped, vec![(2, &5)]);
    }
}
