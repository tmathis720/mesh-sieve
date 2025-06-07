//! Stack trait and in-memory implementation for vertical Sieve arrows.

use std::collections::HashMap;
use super::sieve::InMemorySieve;

/// A Stack links a *base* Sieve to a *cap* Sieve with vertical arrows.
pub trait Stack {
    type Point   : Copy + Eq + std::hash::Hash;      // in the base Sieve
    type CapPt   : Copy + Eq + std::hash::Hash;      // in the cap  Sieve
    type Payload : Clone;
    type BaseSieve;
    type CapSieve;

    // --- topology queries ---
    /// Upward arrows  (base → cap)   – `capCone`
    fn lift<'a>(&'a self, p: Self::Point) -> Box<dyn Iterator<Item=(Self::CapPt,&'a Self::Payload)> + 'a>;
    /// Downward arrows (cap  → base) – `baseSupport`
    fn drop<'a>(&'a self, q: Self::CapPt) -> Box<dyn Iterator<Item=(Self::Point,&'a Self::Payload)> + 'a>;

    // --- mutation helpers ---
    fn add_arrow(&mut self, base: Self::Point, cap: Self::CapPt, pay: Self::Payload);
    fn remove_arrow(&mut self, base: Self::Point, cap: Self::CapPt) -> Option<Self::Payload>;

    // --- convenience ---
    fn base(&self) -> &Self::BaseSieve;
    fn cap(&self) -> &Self::CapSieve;
}

#[derive(Clone, Debug, Default)]
pub struct InMemoryStack<
    B: Copy + Eq + std::hash::Hash,
    C: Copy + Eq + std::hash::Hash,
    P = (),
> {
    pub base: InMemorySieve<B, P>,
    pub cap: InMemorySieve<C, P>,
    up: HashMap<B, Vec<(C, P)>>,
    down: HashMap<C, Vec<(B, P)>>,
}

impl<B, C, P: Clone> InMemoryStack<B, C, P>
where
    B: Copy + Eq + std::hash::Hash,
    C: Copy + Eq + std::hash::Hash,
{
    pub fn new() -> Self {
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
        match self.up.get(&p) {
            Some(vec) => Box::new(vec.iter().map(|(c, pay)| (*c, pay))),
            None => Box::new(std::iter::empty()),
        }
    }
    fn drop<'a>(&'a self, q: C) -> Box<dyn Iterator<Item = (B, &'a P)> + 'a> {
        match self.down.get(&q) {
            Some(vec) => Box::new(vec.iter().map(|(b, pay)| (*b, pay))),
            None => Box::new(std::iter::empty()),
        }
    }
    fn add_arrow(&mut self, base: B, cap: C, pay: P) {
        self.up.entry(base).or_default().push((cap, pay.clone()));
        self.down.entry(cap).or_default().push((base, pay));
    }
    fn remove_arrow(&mut self, base: B, cap: C) -> Option<P> {
        let mut removed = None;
        if let Some(vec) = self.up.get_mut(&base) {
            if let Some(pos) = vec.iter().position(|(c, _)| *c == cap) {
                removed = Some(vec.remove(pos).1);
            }
        }
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

/// Orientation of a vertical arrow in a Stack.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Orientation {
    Positive, // e.g., aligned or +1
    Negative, // e.g., flipped or -1
    Permutation(Vec<usize>), // e.g., for permuting DOF order
}

impl Orientation {
    /// Compose two orientations (sign multiplication or permutation composition).
    pub fn compose(&self, other: &Orientation) -> Orientation {
        match (self, other) {
            (Orientation::Positive, o) | (o, Orientation::Positive) => o.clone(),
            (Orientation::Negative, Orientation::Negative) => Orientation::Positive,
            (Orientation::Negative, o) | (o, Orientation::Negative) => {
                match o {
                    Orientation::Positive => Orientation::Negative,
                    Orientation::Negative => Orientation::Positive,
                    Orientation::Permutation(p) => {
                        let mut rev = p.clone();
                        rev.reverse();
                        Orientation::Permutation(rev)
                    }
                }
            }
            (Orientation::Permutation(p1), Orientation::Permutation(p2)) => {
                let composed = p1.iter().map(|&i| p2[i]).collect();
                Orientation::Permutation(composed)
            }
        }
    }
}

/// Compose two stacks: base → mid and mid → cap, yielding a stack base → cap.
pub struct ComposedStack<'a, S1, S2, F>
where
    S1: Stack,
    S2: Stack<Point = S1::CapPt>,
    F: Fn(&S1::Payload, &S2::Payload) -> S1::Payload,
{
    pub lower: &'a S1,
    pub upper: &'a S2,
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
        Box::new(
            lower.lift(p).flat_map(move |(mid, pay1)| {
                upper.lift(mid).map(move |(cap, pay2)| {
                    // Compose payloads and store in a Box for correct lifetime
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
            upper.drop(q).flat_map(move |(mid, pay2)| {
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
    fn orientation_compose_signs() {
        use super::Orientation::*;
        assert_eq!(Positive.compose(&Positive), Positive);
        assert_eq!(Positive.compose(&Negative), Negative);
        assert_eq!(Negative.compose(&Positive), Negative);
        assert_eq!(Negative.compose(&Negative), Positive);
    }

    #[test]
    fn orientation_compose_permutations() {
        use super::Orientation::*;
        let p1 = Permutation(vec![2, 0, 1]); // maps 0->2, 1->0, 2->1
        let p2 = Permutation(vec![1, 2, 0]); // maps 0->1, 1->2, 2->0
        // Compose: p2(p1[i])
        let composed = p1.compose(&p2);
        assert_eq!(composed, Permutation(vec![0, 1, 2]));
    }

    #[test]
    fn orientation_negate_permutation() {
        use super::Orientation::*;
        let p = Permutation(vec![1, 0, 2]);
        let neg = Negative.compose(&p);
        assert_eq!(neg, Permutation(vec![2, 0, 1])); // reversed
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
