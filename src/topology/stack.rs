//! Stack abstraction for vertical composition of Sieve topologies.
//!
//! A _Stack_ represents relationships ("vertical arrows") between two Sieves:
//! a **base** mesh and a **cap** mesh (e.g., from elements to degrees-of-freedom).
//! This module provides a generic `Stack` trait and an in-memory implementation,
//! along with facilities for composing multiple stacks.

use super::sieve::InMemorySieve;
use crate::mesh_error::MeshSieveError;
use crate::topology::cache::InvalidateCache;
use std::collections::HashMap;
use std::sync::Arc;

/// A `Stack` links a *base* Sieve to a *cap* Sieve via vertical arrows.
/// Each arrow carries a payload (e.g., orientation or permutation).
///
/// - `Point`:   The point type in the base mesh (commonly `PointId`).
/// - `CapPt`:   The point type in the cap mesh (commonly `PointId`).
/// - `Payload`: Data attached to each arrow (e.g., `Orientation`).
///
/// Some implementations (such as [`ComposedStack`]) do not expose a concrete
/// base or cap Sieve. Calling [`Stack::base`] or [`Stack::cap`] on such stacks
/// will panic. A future refactor may return `Option` or `Result` to make these
/// unsupported operations explicit.
pub trait Stack {
    /// Base mesh point identifier.
    type Point: Copy + Eq + std::hash::Hash;
    /// Cap mesh point identifier.
    type CapPt: Copy + Eq + std::hash::Hash;
    /// Per-arrow payload type.
    type Payload: Clone;
    /// The underlying base Sieve type.
    type BaseSieve: crate::topology::sieve::sieve_trait::Sieve<
        Point = Self::Point,
        Payload = Self::Payload,
    >;
    /// The underlying cap Sieve type.
    type CapSieve: crate::topology::sieve::sieve_trait::Sieve<
        Point = Self::CapPt,
        Payload = Self::Payload,
    >;

    // === Topology queries ===
    /// Returns an iterator over all upward arrows from base point `p` to cap points.
    /// Each item is `(cap_point, payload)`.
    fn lift<'a>(
        &'a self,
        p: Self::Point,
    ) -> Box<dyn Iterator<Item = (Self::CapPt, Self::Payload)> + 'a>;

    /// Returns an iterator over all downward arrows from cap point `q` to base points.
    /// Each item is `(base_point, payload)`.
    fn drop<'a>(
        &'a self,
        q: Self::CapPt,
    ) -> Box<dyn Iterator<Item = (Self::Point, Self::Payload)> + 'a>;

    // === Mutation helpers ===
    /// Adds a new vertical arrow `base -> cap` with associated payload.
    ///
    /// **Implementors:** after mutating arrows, you *must* invalidate any derived caches on both the base‐ and cap‐sieves (e.g. via `InvalidateCache::invalidate_cache(self.base_mut())` and likewise for `self.cap_mut()`).
    fn add_arrow(
        &mut self,
        base: Self::Point,
        cap: Self::CapPt,
        pay: Self::Payload,
    ) -> Result<(), MeshSieveError>;

    /// Removes the arrow `base -> cap`, returning its payload if present.
    ///
    /// **Implementors:** after mutating arrows, you *must* invalidate any derived caches on both the base‐ and cap‐sieves.
    fn remove_arrow(
        &mut self,
        base: Self::Point,
        cap: Self::CapPt,
    ) -> Result<Option<Self::Payload>, MeshSieveError>;

    // === Convenience accessors ===
    /// Returns a reference to the underlying base Sieve.
    ///
    /// # Panics
    /// Implementations may panic if the base Sieve is not exposed (e.g., [`ComposedStack`]).
    fn base(&self) -> &Self::BaseSieve;
    /// Returns a reference to the underlying cap Sieve.
    ///
    /// # Panics
    /// Implementations may panic if the cap Sieve is not exposed (e.g., [`ComposedStack`]).
    fn cap(&self) -> &Self::CapSieve;

    /// Returns a mutable reference to the underlying base Sieve.
    fn base_mut(&mut self) -> Result<&mut Self::BaseSieve, MeshSieveError>;
    /// Returns a mutable reference to the underlying cap Sieve.
    fn cap_mut(&mut self) -> Result<&mut Self::CapSieve, MeshSieveError>;
}

/// In-memory implementation of the `Stack` trait.
///
/// Stores vertical arrows in two hash maps:
/// - `up`   maps base points to a list of `(cap_point, payload)`.
/// - `down` maps cap points to a list of `(base_point, payload)`.
///
/// Also embeds two `InMemorySieve`s to represent the base and cap topologies themselves.
#[derive(Clone, Debug)]
pub struct InMemoryStack<
    B: Copy + Eq + std::hash::Hash + Ord + std::fmt::Debug,
    C: Copy + Eq + std::hash::Hash + Ord + std::fmt::Debug,
    P = (),
> {
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
    B: Copy + Eq + std::hash::Hash + Ord + std::fmt::Debug,
    C: Copy + Eq + std::hash::Hash + Ord + std::fmt::Debug,
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
    B: Copy + Eq + std::hash::Hash + Ord + std::fmt::Debug,
    C: Copy + Eq + std::hash::Hash + Ord + std::fmt::Debug,
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
    B: Copy + Eq + std::hash::Hash + Ord + std::fmt::Debug,
    C: Copy + Eq + std::hash::Hash + Ord + std::fmt::Debug,
    P: Clone,
{
    type Point = B;
    type CapPt = C;
    type Payload = P;
    type BaseSieve = InMemorySieve<B, P>;
    type CapSieve = InMemorySieve<C, P>;

    /// Returns an iterator over all upward arrows from base point `p` to cap points.
    /// Each item is `(cap_point, payload)`.
    fn lift<'a>(&'a self, p: B) -> Box<dyn Iterator<Item = (C, P)> + 'a> {
        // Return all upward arrows or empty if none
        match self.up.get(&p) {
            Some(vec) => Box::new(vec.iter().cloned()),
            None => Box::new(std::iter::empty()),
        }
    }

    /// Returns an iterator over all downward arrows from cap point `q` to base points.
    /// Each item is `(base_point, payload)`.
    fn drop<'a>(&'a self, q: C) -> Box<dyn Iterator<Item = (B, P)> + 'a> {
        // Return all downward arrows or empty if none
        match self.down.get(&q) {
            Some(vec) => Box::new(vec.iter().cloned()),
            None => Box::new(std::iter::empty()),
        }
    }

    /// Adds a new vertical arrow `base -> cap` with associated payload.
    ///
    /// After mutating arrows, this method invalidates any derived caches on both the base and cap sieves.
    fn add_arrow(&mut self, base: B, cap: C, pay: P) -> Result<(), MeshSieveError> {
        // Upsert in upward map
        let ups = self.up.entry(base).or_default();
        if let Some(slot) = ups.iter_mut().find(|(c, _)| *c == cap) {
            slot.1 = pay.clone();
        } else {
            ups.push((cap, pay.clone()));
        }

        // Upsert in downward map
        let downs = self.down.entry(cap).or_default();
        if let Some(slot) = downs.iter_mut().find(|(b, _)| *b == base) {
            slot.1 = pay.clone();
        } else {
            downs.push((base, pay.clone()));
        }

        InvalidateCache::invalidate_cache(&mut self.base);
        InvalidateCache::invalidate_cache(&mut self.cap);

        #[cfg(debug_assertions)]
        {
            self.debug_assert_no_parallel_edges_base(base);
            self.debug_assert_no_parallel_edges_cap(cap);
        }

        Ok(())
    }

    /// Removes the arrow `base -> cap`, returning its payload if present.
    ///
    /// After mutating arrows, this method invalidates any derived caches on both the base and cap sieves.
    fn remove_arrow(&mut self, base: B, cap: C) -> Result<Option<P>, MeshSieveError> {
        let mut removed = None;

        // Remove from the upward adjacency map and track whether the entry became empty.
        let remove_up = if let Some(vec) = self.up.get_mut(&base) {
            if let Some(pos) = vec.iter().position(|(c, _)| *c == cap) {
                removed = Some(vec.remove(pos).1);
            }
            vec.is_empty()
        } else {
            false
        };
        if remove_up {
            self.up.remove(&base);
        }

        // Remove from the downward adjacency map and track whether the entry became empty.
        let remove_down = if let Some(vec) = self.down.get_mut(&cap) {
            if let Some(pos) = vec.iter().position(|(b, _)| *b == base) {
                vec.remove(pos);
            }
            vec.is_empty()
        } else {
            false
        };
        if remove_down {
            self.down.remove(&cap);
        }
        InvalidateCache::invalidate_cache(&mut self.base);
        InvalidateCache::invalidate_cache(&mut self.cap);
        Ok(removed)
    }

    /// Returns a reference to the underlying base Sieve.
    fn base(&self) -> &Self::BaseSieve {
        &self.base
    }
    /// Returns a reference to the underlying cap Sieve.
    fn cap(&self) -> &Self::CapSieve {
        &self.cap
    }
    /// Returns a mutable reference to the underlying base Sieve.
    fn base_mut(&mut self) -> Result<&mut Self::BaseSieve, MeshSieveError> {
        Ok(&mut self.base)
    }
    /// Returns a mutable reference to the underlying cap Sieve.
    fn cap_mut(&mut self) -> Result<&mut Self::CapSieve, MeshSieveError> {
        Ok(&mut self.cap)
    }
}

impl<B, C, T> InMemoryStack<B, C, Arc<T>>
where
    B: Copy + Eq + std::hash::Hash + Ord + std::fmt::Debug,
    C: Copy + Eq + std::hash::Hash + Ord + std::fmt::Debug,
{
    /// Insert by value; wraps once into `Arc<T>`.
    #[inline]
    pub fn add_arrow_val(&mut self, base: B, cap: C, payload: T) -> Result<(), MeshSieveError> {
        self.add_arrow(base, cap, Arc::new(payload))
    }
}

/// Provides accessors for base and cap points for testability.
impl<B, C, P> InMemoryStack<B, C, P>
where
    B: Copy + Eq + std::hash::Hash + Ord + std::fmt::Debug,
    C: Copy + Eq + std::hash::Hash + Ord + std::fmt::Debug,
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

    #[cfg(debug_assertions)]
    fn debug_assert_no_parallel_edges_base(&self, b: B) {
        if let Some(v) = self.up.get(&b) {
            use std::collections::HashSet;
            let mut seen = HashSet::new();
            for (c, _) in v {
                assert!(seen.insert(*c), "duplicate base→cap ({:?}→{:?})", b, c);
            }
        }
    }

    #[cfg(debug_assertions)]
    fn debug_assert_no_parallel_edges_cap(&self, c: C) {
        if let Some(v) = self.down.get(&c) {
            use std::collections::HashSet;
            let mut seen = HashSet::new();
            for (b, _) in v {
                assert!(seen.insert(*b), "duplicate cap→base ({:?}→{:?})", c, b);
            }
        }
    }
}

/// A stack composed of two existing stacks: `lower: base -> mid` and `upper: mid -> cap`.
///
/// Traversal composes payloads via a `compose_payload` function.
///
pub struct ComposedStack<'a, S1, S2, F>
where
    S1: Stack,
    S2: Stack<Point = S1::CapPt, Payload = S1::Payload>,
    F: Fn(&S1::Payload, &S2::Payload) -> S1::Payload,
{
    pub lower: &'a S1,
    pub upper: &'a S2,
    pub compose_payload: F,
}

impl<'a, S1, S2, F> ComposedStack<'a, S1, S2, F>
where
    S1: Stack,
    S2: Stack<Point = S1::CapPt, Payload = S1::Payload>,
    F: Fn(&S1::Payload, &S2::Payload) -> S1::Payload,
{
    pub fn new(lower: &'a S1, upper: &'a S2, compose_payload: F) -> Self {
        Self {
            lower,
            upper,
            compose_payload,
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
    type Payload = S1::Payload;
    type BaseSieve = S1::BaseSieve;
    type CapSieve = S2::CapSieve;

    fn lift<'b>(&'b self, p: S1::Point) -> Box<dyn Iterator<Item = (S2::CapPt, S1::Payload)> + 'b> {
        let lower = self.lower.lift(p);
        let iter = lower.flat_map(move |(mid, pay1)| {
            self.upper
                .lift(mid)
                .map(move |(cap, pay2)| (cap, (self.compose_payload)(&pay1, &pay2)))
        });
        Box::new(iter)
    }

    fn drop<'b>(&'b self, q: S2::CapPt) -> Box<dyn Iterator<Item = (S1::Point, S1::Payload)> + 'b> {
        let upper = self.upper.drop(q);
        let iter = upper.flat_map(move |(mid, pay2)| {
            self.lower
                .drop(mid)
                .map(move |(base, pay1)| (base, (self.compose_payload)(&pay1, &pay2)))
        });
        Box::new(iter)
    }

    fn add_arrow(
        &mut self,
        _base: S1::Point,
        _cap: S2::CapPt,
        _pay: S1::Payload,
    ) -> Result<(), MeshSieveError> {
        Err(MeshSieveError::UnsupportedStackOperation(
            "add_arrow on ComposedStack",
        ))
    }
    fn remove_arrow(
        &mut self,
        _base: S1::Point,
        _cap: S2::CapPt,
    ) -> Result<Option<S1::Payload>, MeshSieveError> {
        Err(MeshSieveError::UnsupportedStackOperation(
            "remove_arrow on ComposedStack",
        ))
    }
    fn base_mut(&mut self) -> Result<&mut Self::BaseSieve, MeshSieveError> {
        Err(MeshSieveError::UnsupportedStackOperation(
            "base_mut on ComposedStack",
        ))
    }
    fn cap_mut(&mut self) -> Result<&mut Self::CapSieve, MeshSieveError> {
        Err(MeshSieveError::UnsupportedStackOperation(
            "cap_mut on ComposedStack",
        ))
    }
    /// Returns the base sieve of the composed stack.
    ///
    /// # Panics
    /// Panics because `ComposedStack` does not have direct access to a base sieve.
    fn base(&self) -> &Self::BaseSieve {
        panic!("base() is not supported on ComposedStack")
    }
    /// Returns the cap sieve of the composed stack.
    ///
    /// # Panics
    /// Panics because `ComposedStack` does not have direct access to a cap sieve.
    fn cap(&self) -> &Self::CapSieve {
        panic!("cap() is not supported on ComposedStack")
    }
}

#[test]
fn composed_stack_no_leak() {
    // This test is no longer needed: buffer reuse is gone, and all payloads are owned.
}

impl<B, C, P> InvalidateCache for InMemoryStack<B, C, P>
where
    B: Copy + Eq + std::hash::Hash + Ord + std::fmt::Debug,
    C: Copy + Eq + std::hash::Hash + Ord + std::fmt::Debug,
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
        let _ = stack.add_arrow(V(1), Dof(10), 42);
        let _ = stack.add_arrow(V(1), Dof(11), 43);
        let mut lifted: Vec<_> = stack.lift(V(1)).collect();
        lifted.sort_by_key(|(dof, _)| dof.0);
        assert_eq!(lifted, vec![(Dof(10), 42), (Dof(11), 43)]);
        let dropped: Vec<_> = stack.drop(Dof(10)).collect();
        assert_eq!(dropped, vec![(V(1), 42)]);
    }
    #[test]
    fn remove_arrow_behavior() {
        let mut stack = InMemoryStack::<V, Dof, i32>::new();
        let _ = stack.add_arrow(V(1), Dof(10), 99);
        assert_eq!(stack.remove_arrow(V(1), Dof(10)).unwrap(), Some(99));
        // Double remove returns None
        assert_eq!(stack.remove_arrow(V(1), Dof(10)).unwrap(), None);
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
        let _ = s1.add_arrow(A(1), B(10), 2);
        let _ = s1.add_arrow(A(1), B(11), 3);
        // Stack2: B → C
        let mut s2 = InMemoryStack::<B, C, i32>::new();
        let _ = s2.add_arrow(B(10), C(100), 5);
        let _ = s2.add_arrow(B(11), C(101), 7);
        // Compose: payloads are summed
        let composed = ComposedStack::new(&s1, &s2, |p1, p2| p1 + p2);
        let mut lifted: Vec<_> = composed.lift(A(1)).collect();
        lifted.sort_by_key(|(c, _)| (*c).0);
        assert_eq!(lifted, vec![(C(100), 7), (C(101), 10)]);
        let dropped: Vec<_> = composed.drop(C(100)).collect();
        assert_eq!(dropped, vec![(A(1), 7)]);
    }

    #[test]
    fn stack_cache_cleared_on_mutation() {
        let mut s = InMemoryStack::<u32, u32, i32>::new();
        let _ = s.add_arrow(1, 10, 2);
        let d0 = s.base_mut().unwrap().diameter().unwrap();
        let _ = s.add_arrow(2, 11, 3);
        let d1 = s.base_mut().unwrap().diameter().unwrap();
        assert!(d1 >= d0);
    }

    #[test]
    fn base_and_cap_points_reflect_maps() {
        let mut s = InMemoryStack::<u32, u32, ()>::new();
        // empty
        assert!(s.base_points().next().is_none());
        assert!(s.cap_points().next().is_none());
        // add arrow 1→10
        let _ = s.add_arrow(1, 10, ());
        let bases: Vec<_> = s.base_points().collect();
        let caps: Vec<_> = s.cap_points().collect();
        assert_eq!(bases, vec![1]);
        assert_eq!(caps, vec![10]);
    }

    #[test]
    fn new_and_default_empty() {
        let s1 = InMemoryStack::<u8, u8, u8>::new();
        let s2: InMemoryStack<u8, u8, u8> = Default::default();
        assert_eq!(s1.base_points().count(), 0);
        assert_eq!(s2.cap_points().count(), 0);
    }

    #[test]
    fn composed_stack_add_arrow_error() {
        let s1 = InMemoryStack::<u8, u8, u8>::new();
        let s2 = InMemoryStack::<u8, u8, u8>::new();
        let mut cs = ComposedStack::new(&s1, &s2, |a, _b| *a);
        let err = cs.add_arrow(0, 0, 0).unwrap_err();
        assert_eq!(
            err.to_string(),
            MeshSieveError::UnsupportedStackOperation("add_arrow on ComposedStack").to_string()
        );
    }
    #[test]
    fn composed_stack_remove_arrow_error() {
        let s1 = InMemoryStack::<u8, u8, u8>::new();
        let s2 = InMemoryStack::<u8, u8, u8>::new();
        let mut cs = ComposedStack::new(&s1, &s2, |a, _b| *a);
        let err = cs.remove_arrow(0, 0).unwrap_err();
        assert_eq!(
            err.to_string(),
            MeshSieveError::UnsupportedStackOperation("remove_arrow on ComposedStack").to_string()
        );
    }
    #[test]
    fn composed_stack_base_mut_error() {
        let s1 = InMemoryStack::<u8, u8, u8>::new();
        let s2 = InMemoryStack::<u8, u8, u8>::new();
        let mut cs = ComposedStack::new(&s1, &s2, |a, _b| *a);
        let err = cs.base_mut().unwrap_err();
        assert_eq!(
            err.to_string(),
            MeshSieveError::UnsupportedStackOperation("base_mut on ComposedStack").to_string()
        );
    }
    #[test]
    fn composed_stack_cap_mut_error() {
        let s1 = InMemoryStack::<u8, u8, u8>::new();
        let s2 = InMemoryStack::<u8, u8, u8>::new();
        let mut cs = ComposedStack::new(&s1, &s2, |a, _b| *a);
        let err = cs.cap_mut().unwrap_err();
        assert_eq!(
            err.to_string(),
            MeshSieveError::UnsupportedStackOperation("cap_mut on ComposedStack").to_string()
        );
    }

    #[test]
    fn remove_nonexistent_returns_none() {
        let mut s = InMemoryStack::<u32, u32, ()>::new();
        assert_eq!(s.remove_arrow(5, 50).unwrap(), None);
    }

    #[test]
    fn remove_arrow_cleans_empty_maps() {
        let mut s = InMemoryStack::<u32, u32, ()>::new();
        let _ = s.add_arrow(1, 10, ());
        // Remove the only arrow and ensure maps no longer report the points
        assert_eq!(s.remove_arrow(1, 10).unwrap(), Some(()));
        assert!(s.base_points().next().is_none());
        assert!(s.cap_points().next().is_none());
    }

    #[test]
    fn invalidate_cache_clears_embedded() {
        let mut s = InMemoryStack::<u32, u32, i32>::new();
        // manually prime strata cache
        let _ = s.base_mut().unwrap().diameter();
        let _ = s.cap_mut().unwrap().diameter();
        s.invalidate_cache();
        // re‐access should re‐compute (no panic, but at least not stale)
        let _ = s.base_mut().unwrap().diameter();
        let _ = s.cap_mut().unwrap().diameter();
    }

    #[test]
    fn lift_drop_empty_iter() {
        let s = InMemoryStack::<u8, u8, ()>::new();
        assert!(s.lift(0).next().is_none());
        assert!(s.drop(0).next().is_none());
    }

    // #[test]
    // #[should_panic]
    // fn composed_stack_base_panics() {
    //     let s1 = InMemoryStack::<u8,u8,u8>::new();
    //     let s2 = InMemoryStack::<u8,u8,u8>::new();
    //     let cs = ComposedStack::new(&s1, &s2, |a,_b| *a);
    //     let _ = cs.base();
    // }
    // #[test]
    // #[should_panic]
    // fn composed_stack_cap_panics() {
    //     let s1 = InMemoryStack::<u8,u8,u8>::new();
    //     let s2 = InMemoryStack::<u8,u8,u8>::new();
    //     let cs = ComposedStack::new(&s1, &s2, |a,_b| *a);
    //     let _ = cs.cap();
    // }

    #[test]
    fn stack_vertical_arrows_are_correct() {
        let mut s = InMemoryStack::<u32, u32, i32>::new();
        let _ = s.add_arrow(2, 20, 5);
        // The stack's lift and drop reflect the vertical arrows
        let lifted: Vec<_> = s.lift(2).collect();
        assert_eq!(lifted, vec![(20, 5)]);
        let dropped: Vec<_> = s.drop(20).collect();
        assert_eq!(dropped, vec![(2, 5)]);
    }
}
