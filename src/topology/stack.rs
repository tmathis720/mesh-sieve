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

pub struct InMemoryStack<B, C, P = ()> {
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
}
