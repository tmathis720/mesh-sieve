//! Stack abstraction for vertical composition of Sieve topologies.
//!
//! A _Stack_ represents relationships ("vertical arrows") between two Sieves:
//! a **base** mesh and a **cap** mesh (e.g., from elements to degrees-of-freedom).
//! This module provides a generic `Stack` trait and an in-memory implementation,
//! along with facilities for composing multiple stacks.
//!
//! [`Stack::add_arrow`] requires that `base` is present in the base sieve and `cap`
//! is present in the cap sieve. Insert points first via the sieve mutators
//! (`add_point`, `add_base_point`, `add_cap_point`). In debug and
//! `strict-invariants` builds, violating this invariant panics; in all builds the
//! method returns `Err(MeshSieveError::StackMissingPoint{..})` when either point
//! is missing.

use super::sieve::InMemorySieve;
use crate::mesh_error::MeshSieveError;
use crate::topology::_debug_invariants::{debug_invariants, inv_assert};
use crate::topology::bounds::{PayloadLike, PointLike};
use crate::topology::cache::InvalidateCache;
use std::collections::HashMap;
use std::sync::Arc;

/// A `Stack` links a *base* Sieve to a *cap* Sieve via vertical arrows.
/// Each vertical arrow carries its own payload (e.g., polarity or permutation),
/// independent of the horizontal Sieve payloads.
///
/// - `Point`:            The point type in the base mesh (commonly `PointId`).
/// - `CapPt`:            The point type in the cap mesh (commonly `PointId`).
/// - `VerticalPayload`:  Data attached to each vertical arrow (e.g., `Polarity`).
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
    /// Vertical arrow payload type.
    type VerticalPayload: Clone;
    /// Underlying base Sieve type (horizontal), payload unconstrained.
    type BaseSieve: crate::topology::sieve::sieve_trait::Sieve<Point = Self::Point>;
    /// Underlying cap Sieve type (horizontal), payload unconstrained.
    type CapSieve: crate::topology::sieve::sieve_trait::Sieve<Point = Self::CapPt>;

    // === Topology queries ===
    /// Returns an iterator over all upward arrows from base point `p` to cap points.
    /// Each item is `(cap_point, vertical_payload)`.
    fn lift<'a>(
        &'a self,
        p: Self::Point,
    ) -> Box<dyn Iterator<Item = (Self::CapPt, Self::VerticalPayload)> + 'a>;

    /// Returns an iterator over all downward arrows from cap point `q` to base points.
    /// Each item is `(base_point, vertical_payload)`.
    fn drop<'a>(
        &'a self,
        q: Self::CapPt,
    ) -> Box<dyn Iterator<Item = (Self::Point, Self::VerticalPayload)> + 'a>;

    // === Mutation helpers ===
    /// Adds a new vertical arrow `base -> cap` with associated payload.
    fn add_arrow(
        &mut self,
        base: Self::Point,
        cap: Self::CapPt,
        pay: Self::VerticalPayload,
    ) -> Result<(), MeshSieveError>;

    /// Removes the arrow `base -> cap`, returning its payload if present.
    fn remove_arrow(
        &mut self,
        base: Self::Point,
        cap: Self::CapPt,
    ) -> Result<Option<Self::VerticalPayload>, MeshSieveError>;

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
pub struct InMemoryStack<B: PointLike, C: PointLike, V = (), PB = (), PC = ()> {
    /// Underlying base sieve (e.g., mesh connectivity).
    base: InMemorySieve<B, PB>,
    /// Underlying cap sieve (e.g., DOF connectivity).
    cap: InMemorySieve<C, PC>,
    /// Upward adjacency: base -> cap
    pub up: HashMap<B, Vec<(C, V)>>,
    /// Downward adjacency: cap -> base
    pub down: HashMap<C, Vec<(B, V)>>,
}

impl<B, C, V, PB, PC> InMemoryStack<B, C, V, PB, PC>
where
    B: PointLike,
    C: PointLike,
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

    /// Hint: preallocate additional upward slots for `base`.
    #[inline]
    pub fn reserve_lift(&mut self, base: B, additional: usize) {
        self.up.entry(base).or_default().reserve(additional);
    }

    /// Hint: preallocate additional downward slots for `cap`.
    #[inline]
    pub fn reserve_drop(&mut self, cap: C, additional: usize) {
        self.down.entry(cap).or_default().reserve(additional);
    }

    /// Optionally release excess capacity after bulk construction.
    pub fn shrink_to_fit(&mut self) {
        for v in self.up.values_mut() {
            v.shrink_to_fit();
        }
        for v in self.down.values_mut() {
            v.shrink_to_fit();
        }
    }
}

/// Provides a default implementation for `InMemoryStack`.
impl<B, C, V, PB, PC> Default for InMemoryStack<B, C, V, PB, PC>
where
    B: PointLike,
    C: PointLike,
    V: PayloadLike,
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

impl<B, C, V, PB, PC> Stack for InMemoryStack<B, C, V, PB, PC>
where
    B: PointLike,
    C: PointLike,
    V: PayloadLike,
    PB: PayloadLike,
    PC: PayloadLike,
{
    type Point = B;
    type CapPt = C;
    type VerticalPayload = V;
    type BaseSieve = InMemorySieve<B, PB>;
    type CapSieve = InMemorySieve<C, PC>;

    fn lift<'a>(&'a self, p: B) -> Box<dyn Iterator<Item = (C, V)> + 'a> {
        match self.up.get(&p) {
            Some(vec) => Box::new(vec.iter().cloned()),
            None => Box::new(std::iter::empty()),
        }
    }

    fn drop<'a>(&'a self, q: C) -> Box<dyn Iterator<Item = (B, V)> + 'a> {
        match self.down.get(&q) {
            Some(vec) => Box::new(vec.iter().cloned()),
            None => Box::new(std::iter::empty()),
        }
    }

    fn add_arrow(&mut self, base: B, cap: C, pay: V) -> Result<(), MeshSieveError> {
        if !self.base.contains_point(base) {
            inv_assert!(false, "stack add_arrow: base point missing: {base:?}");
            return Err(MeshSieveError::StackMissingPoint {
                role: "base",
                point: format!("{base:?}"),
            });
        }
        if !self.cap.contains_point(cap) {
            inv_assert!(false, "stack add_arrow: cap point missing: {cap:?}");
            return Err(MeshSieveError::StackMissingPoint {
                role: "cap",
                point: format!("{cap:?}"),
            });
        }

        let ups = self.up.entry(base).or_default();
        if let Some(slot) = ups.iter_mut().find(|(c, _)| *c == cap) {
            slot.1 = pay.clone();
        } else {
            ups.push((cap, pay.clone()));
        }

        let downs = self.down.entry(cap).or_default();
        if let Some(slot) = downs.iter_mut().find(|(b, _)| *b == base) {
            slot.1 = pay.clone();
        } else {
            downs.push((base, pay.clone()));
        }

        InvalidateCache::invalidate_cache(&mut self.base);
        InvalidateCache::invalidate_cache(&mut self.cap);
        debug_invariants!(self);
        Ok(())
    }

    fn remove_arrow(&mut self, base: B, cap: C) -> Result<Option<V>, MeshSieveError> {
        if !self.base.contains_point(base) {
            inv_assert!(false, "stack remove_arrow: base point missing: {base:?}");
        }
        if !self.cap.contains_point(cap) {
            inv_assert!(false, "stack remove_arrow: cap point missing: {cap:?}");
        }

        let mut removed = None;

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
        debug_invariants!(self);
        Ok(removed)
    }

    fn base(&self) -> &Self::BaseSieve {
        &self.base
    }
    fn cap(&self) -> &Self::CapSieve {
        &self.cap
    }
    fn base_mut(&mut self) -> Result<&mut Self::BaseSieve, MeshSieveError> {
        Ok(&mut self.base)
    }
    fn cap_mut(&mut self) -> Result<&mut Self::CapSieve, MeshSieveError> {
        Ok(&mut self.cap)
    }
}

impl<B, C, T, PB, PC> InMemoryStack<B, C, Arc<T>, PB, PC>
where
    B: PointLike,
    C: PointLike,
    PB: PayloadLike,
    PC: PayloadLike,
{
    #[inline]
    pub fn add_arrow_val(&mut self, base: B, cap: C, payload: T) -> Result<(), MeshSieveError> {
        self.add_arrow(base, cap, Arc::new(payload))
    }
}

/// Provides accessors for base and cap points for testability.
impl<B, C, V, PB, PC> InMemoryStack<B, C, V, PB, PC>
where
    B: PointLike,
    C: PointLike,
    V: PayloadLike,
{
    pub fn base_points(&self) -> impl Iterator<Item = B> + '_ {
        self.up.keys().copied()
    }
    pub fn cap_points(&self) -> impl Iterator<Item = C> + '_ {
        self.down.keys().copied()
    }
    #[cfg(any(debug_assertions, feature = "strict-invariants"))]
    pub(crate) fn debug_assert_invariants(&self) {
        use std::collections::HashSet;

        for (b, v) in &self.up {
            let mut seen = HashSet::new();
            for (c, _) in v {
                crate::topology::_debug_invariants::inv_assert!(
                    seen.insert(*c),
                    "duplicate vertical arrow base={b:?} cap={c:?}"
                );
            }
        }
        for (c, v) in &self.down {
            let mut seen = HashSet::new();
            for (b, _) in v {
                crate::topology::_debug_invariants::inv_assert!(
                    seen.insert(*b),
                    "duplicate vertical arrow cap={c:?} base={b:?}"
                );
            }
        }

        let out_total: usize = self.up.values().map(|v| v.len()).sum();
        let in_total: usize = self.down.values().map(|v| v.len()).sum();
        crate::topology::_debug_invariants::inv_assert_eq!(
            out_total,
            in_total,
            "stack up/down totals differ",
        );

        for (b, ups) in &self.up {
            crate::topology::_debug_invariants::inv_assert!(
                self.base.adjacency_out.contains_key(b) || self.base.adjacency_in.contains_key(b),
                "vertical base point {b:?} not present in base sieve"
            );
            for (c, _) in ups {
                crate::topology::_debug_invariants::inv_assert!(
                    self.cap.adjacency_out.contains_key(c) || self.cap.adjacency_in.contains_key(c),
                    "vertical cap point {c:?} not present in cap sieve"
                );
                let has = self
                    .down
                    .get(c)
                    .is_some_and(|v| v.iter().any(|(bb, _)| *bb == *b));
                crate::topology::_debug_invariants::inv_assert!(
                    has,
                    "stack mirror missing: up {b:?}->{c:?} has no down",
                );
            }
        }
        for (c, downs) in &self.down {
            crate::topology::_debug_invariants::inv_assert!(
                self.cap.adjacency_out.contains_key(c) || self.cap.adjacency_in.contains_key(c),
                "vertical cap point {c:?} not present in cap sieve"
            );
            for (b, _) in downs {
                crate::topology::_debug_invariants::inv_assert!(
                    self.base.adjacency_out.contains_key(b)
                        || self.base.adjacency_in.contains_key(b),
                    "vertical base point {b:?} not present in base sieve"
                );
                let has = self
                    .up
                    .get(b)
                    .is_some_and(|v| v.iter().any(|(cc, _)| *cc == *c));
                crate::topology::_debug_invariants::inv_assert!(
                    has,
                    "stack mirror missing: down {b:?}->{c:?} has no up",
                );
            }
        }
    }
}

/// A stack composed of two existing stacks: `lower: base -> mid` and `upper: mid -> cap`.
///
/// Traversal composes payloads via a `compose_payload` function.
///
/// # Examples
///
/// Using [`Polarity`] payloads (XOR composition):
/// ```
/// use mesh_sieve::topology::arrow::Polarity;
/// use mesh_sieve::topology::stack::{ComposedStack, InMemoryStack, Stack};
/// use mesh_sieve::topology::point::PointId;
/// let s1 = InMemoryStack::<PointId, PointId, Polarity>::new();
/// let s2 = InMemoryStack::<PointId, PointId, Polarity>::new();
/// let _cs = ComposedStack::new(&s1, &s2, |a, b| (*a) ^ (*b));
/// ```
///
/// Using group-valued [`orientation::Sign`] with trait composition:
/// ```
/// use mesh_sieve::topology::orientation::Sign;
/// use mesh_sieve::topology::sieve::oriented::Orientation as _;
/// use mesh_sieve::topology::stack::{ComposedStack, InMemoryStack, Stack};
/// use mesh_sieve::topology::point::PointId;
/// let s1 = InMemoryStack::<PointId, PointId, Sign>::new();
/// let s2 = InMemoryStack::<PointId, PointId, Sign>::new();
/// let _cs = ComposedStack::new(&s1, &s2, |a, b| Sign::compose(*a, *b));
/// ```
pub struct ComposedStack<'a, S1, S2, F, VO>
where
    S1: Stack,
    S2: Stack<Point = S1::CapPt>,
    F: Fn(&S1::VerticalPayload, &S2::VerticalPayload) -> VO,
    VO: Clone,
{
    pub lower: &'a S1,
    pub upper: &'a S2,
    pub compose_payload: F,
    _phantom: core::marker::PhantomData<VO>,
}

impl<'a, S1, S2, F, VO> ComposedStack<'a, S1, S2, F, VO>
where
    S1: Stack,
    S2: Stack<Point = S1::CapPt>,
    F: Fn(&S1::VerticalPayload, &S2::VerticalPayload) -> VO,
    VO: Clone,
{
    pub fn new(lower: &'a S1, upper: &'a S2, compose_payload: F) -> Self {
        Self {
            lower,
            upper,
            compose_payload,
            _phantom: core::marker::PhantomData,
        }
    }
}

impl<'a, S1, S2, F, VO> Stack for ComposedStack<'a, S1, S2, F, VO>
where
    S1: Stack,
    S2: Stack<Point = S1::CapPt>,
    F: Fn(&S1::VerticalPayload, &S2::VerticalPayload) -> VO + Sync + Send,
    VO: Clone,
{
    type Point = S1::Point;
    type CapPt = S2::CapPt;
    type VerticalPayload = VO;
    type BaseSieve = S1::BaseSieve;
    type CapSieve = S2::CapSieve;

    fn lift<'b>(&'b self, p: S1::Point) -> Box<dyn Iterator<Item = (S2::CapPt, VO)> + 'b> {
        let lower = self.lower.lift(p);
        let iter = lower.flat_map(move |(mid, pay1)| {
            self.upper
                .lift(mid)
                .map(move |(cap, pay2)| (cap, (self.compose_payload)(&pay1, &pay2)))
        });
        Box::new(iter)
    }

    fn drop<'b>(&'b self, q: S2::CapPt) -> Box<dyn Iterator<Item = (S1::Point, VO)> + 'b> {
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
        _pay: VO,
    ) -> Result<(), MeshSieveError> {
        Err(MeshSieveError::UnsupportedStackOperation(
            "add_arrow on ComposedStack",
        ))
    }
    fn remove_arrow(
        &mut self,
        _base: S1::Point,
        _cap: S2::CapPt,
    ) -> Result<Option<VO>, MeshSieveError> {
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

impl<B, C, V, PB, PC> InvalidateCache for InMemoryStack<B, C, V, PB, PC>
where
    B: PointLike,
    C: PointLike,
    V: PayloadLike,
{
    fn invalidate_cache(&mut self) {
        // stack-local caches only; base and cap left untouched
    }
}

impl<B, C, V, PB, PC> InMemoryStack<B, C, V, PB, PC>
where
    B: PointLike,
    C: PointLike,
    PB: PayloadLike,
    PC: PayloadLike,
{
    #[inline]
    pub fn invalidate_base_and_cap(&mut self) {
        self.base.invalidate_cache();
        self.cap.invalidate_cache();
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
    fn stack_payloads_decoupled() {
        // Base payload u8, Cap payload String, Vertical payload bool
        type S = InMemoryStack<u32, u32, bool, u8, String>;
        let mut st = S::default();
        st.base.add_arrow(1, 2, 7u8);
        st.cap.add_arrow(10, 20, "x".to_string());
        st.add_arrow(1, 10, true).unwrap();
        let v: Vec<_> = st.lift(1).collect();
        assert_eq!(v, vec![(10, true)]);
    }
    #[test]
    fn add_and_lift_drop() {
        let mut stack = InMemoryStack::<V, Dof, i32>::new();
        stack.base.add_arrow(V(1), V(1), ());
        stack.cap.add_arrow(Dof(10), Dof(10), ());
        stack.cap.add_arrow(Dof(11), Dof(11), ());
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
        stack.base.add_arrow(V(1), V(1), ());
        stack.cap.add_arrow(Dof(10), Dof(10), ());
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
        s1.base.add_arrow(A(1), A(1), ());
        s1.cap.add_arrow(B(10), B(10), ());
        s1.cap.add_arrow(B(11), B(11), ());
        let _ = s1.add_arrow(A(1), B(10), 2);
        let _ = s1.add_arrow(A(1), B(11), 3);
        // Stack2: B → C
        let mut s2 = InMemoryStack::<B, C, i32>::new();
        s2.base.add_arrow(B(10), B(10), ());
        s2.base.add_arrow(B(11), B(11), ());
        s2.cap.add_arrow(C(100), C(100), ());
        s2.cap.add_arrow(C(101), C(101), ());
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
    fn vertical_edits_invalidate_horizontal_caches() {
        let mut s = InMemoryStack::<u32, u32, (), (), ()>::default();
        s.base.add_arrow(1, 2, ());
        s.cap.add_arrow(10, 20, ());
        assert!(s.base.strata.get().is_none());
        assert!(s.cap.strata.get().is_none());
        let _ = s.base.chart_points();
        let _ = s.cap.chart_points();
        assert!(s.base.strata.get().is_some());
        assert!(s.cap.strata.get().is_some());
        s.add_arrow(1, 10, ()).unwrap();
        assert!(s.base.strata.get().is_none());
        assert!(s.cap.strata.get().is_none());
    }

    #[test]
    fn base_and_cap_points_reflect_maps() {
        let mut s = InMemoryStack::<u32, u32, ()>::new();
        s.base.add_arrow(1, 1, ());
        s.cap.add_arrow(10, 10, ());
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
        use crate::topology::sieve::MutableSieve;
        let mut s = InMemoryStack::<u32, u32, ()>::new();
        MutableSieve::add_base_point(s.base_mut().unwrap(), 5);
        MutableSieve::add_cap_point(s.cap_mut().unwrap(), 50);
        assert_eq!(s.remove_arrow(5, 50).unwrap(), None);
    }

    #[test]
    fn remove_arrow_cleans_empty_maps() {
        let mut s = InMemoryStack::<u32, u32, ()>::new();
        s.base.add_arrow(1, 1, ());
        s.cap.add_arrow(10, 10, ());
        let _ = s.add_arrow(1, 10, ());
        // Remove the only arrow and ensure maps no longer report the points
        assert_eq!(s.remove_arrow(1, 10).unwrap(), Some(()));
        assert!(s.base_points().next().is_none());
        assert!(s.cap_points().next().is_none());
    }

    #[test]
    fn invalidate_cache_noop() {
        let mut s = InMemoryStack::<u32, u32, i32>::new();
        s.base.add_arrow(1, 2, ());
        s.cap.add_arrow(10, 20, ());
        let _ = s.base.chart_points();
        let _ = s.cap.chart_points();
        assert!(s.base.strata.get().is_some());
        assert!(s.cap.strata.get().is_some());
        s.invalidate_cache();
        assert!(s.base.strata.get().is_some());
        assert!(s.cap.strata.get().is_some());
    }

    #[test]
    fn lift_drop_empty_iter() {
        let s = InMemoryStack::<u8, u8, ()>::new();
        assert!(s.lift(0).next().is_none());
        assert!(s.drop(0).next().is_none());
    }

    #[test]
    fn stack_add_arrow_missing_base_rejected() {
        use crate::topology::sieve::MutableSieve;
        let mut st = InMemoryStack::<u32, u32, ()>::new();
        MutableSieve::add_cap_point(st.cap_mut().unwrap(), 20);
        let res =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| st.add_arrow(10, 20, ())));
        match res {
            Ok(Err(e)) => assert!(format!("{e}").contains("base")),
            Err(_) => (), // panic expected in debug/strict builds
            Ok(Ok(_)) => panic!("expected failure"),
        }
    }

    #[test]
    fn stack_add_arrow_ok_when_points_exist() {
        use crate::topology::sieve::MutableSieve;
        let mut st = InMemoryStack::<u32, u32, ()>::new();
        MutableSieve::add_base_point(st.base_mut().unwrap(), 10);
        MutableSieve::add_cap_point(st.cap_mut().unwrap(), 20);
        st.add_arrow(10, 20, ()).unwrap();
    }

    #[cfg(any(debug_assertions, feature = "strict-invariants"))]
    #[test]
    #[should_panic]
    fn strict_build_panics_on_missing_cap() {
        use crate::topology::sieve::MutableSieve;
        let mut st = InMemoryStack::<u32, u32, ()>::new();
        MutableSieve::add_base_point(st.base_mut().unwrap(), 10);
        let _ = st.add_arrow(10, 20, ());
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
        s.base.add_arrow(2, 2, ());
        s.cap.add_arrow(20, 20, ());
        let _ = s.add_arrow(2, 20, 5);
        // The stack's lift and drop reflect the vertical arrows
        let lifted: Vec<_> = s.lift(2).collect();
        assert_eq!(lifted, vec![(20, 5)]);
        let dropped: Vec<_> = s.drop(20).collect();
        assert_eq!(dropped, vec![(2, 5)]);
    }
}
