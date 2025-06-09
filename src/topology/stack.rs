//! Stack abstraction for vertical composition of Sieve topologies.
//!
//! A _Stack_ represents relationships ("vertical arrows") between two Sieves:
//! a **base** mesh and a **cap** mesh (e.g., from elements to degrees-of-freedom).
//! This module provides a generic `Stack` trait and an in-memory implementation,
//! along with facilities for composing multiple stacks.

use std::collections::HashMap;
use super::sieve::InMemorySieve;

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
    type BaseSieve;
    /// The underlying cap Sieve type.
    type CapSieve;

    // === Topology queries ===
    /// Returns an iterator over all upward arrows from base point `p` to cap points.
    /// Each item is `(cap_point, &payload)`.
    fn lift<'a>(&'a self, p: Self::Point) -> Box<dyn Iterator<Item = (Self::CapPt, &'a Self::Payload)> + 'a>;

    /// Returns an iterator over all downward arrows from cap point `q` to base points.
    /// Each item is `(base_point, &payload)`.
    fn drop<'a>(&'a self, q: Self::CapPt) -> Box<dyn Iterator<Item = (Self::Point, &'a Self::Payload)> + 'a>;

    // === Mutation helpers ===
    /// Adds a new vertical arrow `base -> cap` with associated payload.
    fn add_arrow(&mut self, base: Self::Point, cap: Self::CapPt, pay: Self::Payload);

    /// Removes the arrow `base -> cap`, returning its payload if present.
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
pub struct InMemoryStack<
    B: Copy + Eq + std::hash::Hash,
    C: Copy + Eq + std::hash::Hash,
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
    B: Copy + Eq + std::hash::Hash,
    C: Copy + Eq + std::hash::Hash,
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

impl<B, C, P: Clone> Default for InMemoryStack<B, C, P>
where
    B: Copy + Eq + std::hash::Hash,
    C: Copy + Eq + std::hash::Hash,
{
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
    B: Copy + Eq + std::hash::Hash,
    C: Copy + Eq + std::hash::Hash,
    P: Clone,
{
    type Point = B;
    type CapPt = C;
    type Payload = P;
    type BaseSieve = InMemorySieve<B, P>;
    type CapSieve = InMemorySieve<C, P>;

    fn lift<'a>(&'a self, p: B) -> Box<dyn Iterator<Item = (C, &'a P)> + 'a> {
        // Return all upward arrows or empty if none
        match self.up.get(&p) {
            Some(vec) => Box::new(vec.iter().map(|(c, pay)| (*c, pay))),
            None => Box::new(std::iter::empty()),
        }
    }

    fn drop<'a>(&'a self, q: C) -> Box<dyn Iterator<Item = (B, &'a P)> + 'a> {
        // Return all downward arrows or empty if none
        match self.down.get(&q) {
            Some(vec) => Box::new(vec.iter().map(|(b, pay)| (*b, pay))),
            None => Box::new(std::iter::empty()),
        }
    }

    fn add_arrow(&mut self, base: B, cap: C, pay: P) {
        // Insert into both up and down maps
        self.up.entry(base).or_default().push((cap, pay.clone()));
        self.down.entry(cap).or_default().push((base, pay));
    }

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
        removed
    }

    fn base(&self) -> &Self::BaseSieve { &self.base }
    fn cap(&self) -> &Self::CapSieve { &self.cap }
}

impl<B, C, P> InMemoryStack<B, C, P>
where
    B: Copy + Eq + std::hash::Hash,
    C: Copy + Eq + std::hash::Hash,
    P: Clone,
{
    /// Build a Sifter for a given base point: all (cap, payload) pairs for that base.
    pub fn sifter(&self, base: B) -> Vec<(C, P)> {
        self.up.get(&base)
            .map(|v| v.iter().cloned().collect())
            .unwrap_or_default()
    }
}

/// A stack composed of two existing stacks: `lower: base -> mid` and `upper: mid -> cap`.
///
/// Traversal composes payloads via a `compose_payload` function.
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

    fn lift<'b>(&'b self, p: S1::Point) -> Box<dyn Iterator<Item = (S2::CapPt, &'b S1::Payload)> + 'b> {
        let lower = self.lower;
        let upper = self.upper;
        let compose = &self.compose_payload;
        // First lift through lower, then through upper, composing payloads
        Box::new(
            lower.lift(p)
                .flat_map(move |(mid, pay1)| {
                    upper.lift(mid).map(move |(cap, pay2)| {
                        // Leak a composed payload to extend its lifetime
                        let payload: &'b S1::Payload = Box::leak(Box::new((compose)(pay1, pay2)));
                        (cap, payload)
                    })
                })
        )
    }

    fn drop<'b>(&'b self, q: S2::CapPt) -> Box<dyn Iterator<Item = (S1::Point, &'b S1::Payload)> + 'b> {
        let lower = self.lower;
        let upper = self.upper;
        let compose = &self.compose_payload;
        Box::new(
            upper.drop(q)
                .flat_map(move |(mid, pay2)| {
                    lower.drop(mid).map(move |(base, pay1)| {
                        let payload: &'b S1::Payload = Box::leak(Box::new((compose)(pay1, pay2)));
                        (base, payload)
                    })
                })
        )
    }

    fn add_arrow(&mut self, _base: S1::Point, _cap: S2::CapPt, _pay: S1::Payload) {
        panic!("Cannot mutate a composed stack");
    }
    fn remove_arrow(&mut self, _base: S1::Point, _cap: S2::CapPt) -> Option<S1::Payload> {
        panic!("Cannot mutate a composed stack");
    }
    fn base(&self) -> &Self::BaseSieve { self.lower.base() }
    fn cap(&self) -> &Self::CapSieve { self.upper.cap() }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
    struct V(u32);
    #[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
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
        #[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
        struct A(u32);
        #[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
        struct B(u32);
        #[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
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
        let composed = ComposedStack {
            lower: &s1,
            upper: &s2,
            compose_payload: |p1, p2| p1 + p2,
        };
        let mut lifted: Vec<_> = composed.lift(A(1)).map(|(c, p)| (c, *p)).collect();
        lifted.sort_by_key(|(c, _)| c.0);
        assert_eq!(lifted, vec![(C(100), 7), (C(101), 10)]);
        let dropped: Vec<_> = composed.drop(C(100)).map(|(a, p)| (a, *p)).collect();
        assert_eq!(dropped, vec![(A(1), 7)]);
    }
}
