//! In-memory oriented Sieve: stores (dst, payload, orientation) per arrow.
//! Implements both `Sieve` (payload-only view) and `OrientedSieve`.

use once_cell::sync::OnceCell;
use std::collections::HashMap;

use super::sieve_trait::Sieve;
use super::super::stratum::{InvalidateCache, StrataCache, compute_strata};
use super::oriented::{OrientedSieve, Orientation};
use crate::mesh_error::MeshSieveError;

#[derive(Clone, Debug)]
pub struct InMemoryOrientedSieve<P, T=(), O=i32>
where
    P: Ord + std::fmt::Debug,
    O: Orientation,
{
    pub adjacency_out: HashMap<P, Vec<(P, T, O)>>,
    pub adjacency_in:  HashMap<P, Vec<(P, T, O)>>,
    pub strata: OnceCell<StrataCache<P>>,
}

impl<P, T, O> Default for InMemoryOrientedSieve<P, T, O>
where
    P: Copy + Eq + std::hash::Hash + Ord + std::fmt::Debug,
    O: Orientation,
{
    fn default() -> Self {
        Self { adjacency_out: HashMap::new(),
               adjacency_in: HashMap::new(),
               strata: OnceCell::new() }
    }
}

impl<P, T, O> InMemoryOrientedSieve<P, T, O>
where
    P: Copy + Eq + std::hash::Hash + Ord + std::fmt::Debug,
    T: Clone,
    O: Orientation,
{
    pub fn new() -> Self { Self::default() }

    pub fn from_arrows<I: IntoIterator<Item=(P, P, T, O)>>(arrows: I) -> Self {
        let mut s = Self::default();
        for (src, dst, pay, ori) in arrows {
            s.add_arrow_o(src, dst, pay, ori);
        }
        s
    }

    fn rebuild_support_from_out(&mut self) {
        self.adjacency_in.clear();
        for (&src, outs) in &self.adjacency_out {
            for &(dst, ref pay, ori) in outs {
                self.adjacency_in.entry(dst).or_default().push((src, pay.clone(), ori));
            }
        }
    }

    pub fn strata_cache(&mut self) -> Result<&StrataCache<P>, MeshSieveError> {
        let self_ptr: *mut Self = self;
        self.strata.get_or_try_init(|| {
            let sieve = unsafe { &mut *self_ptr };
            compute_strata(sieve)
        })
    }

    pub fn invalidate_strata(&mut self) { self.strata.take(); }
}

// ----------- Sieve (payload-only view) -----------
type MapOut<'a, P, T, O> = std::iter::Map<
    std::slice::Iter<'a, (P, T, O)>,
    fn(&'a (P, T, O)) -> (P, T)
>;

type MapOOut<'a, P, T, O> = std::iter::Map<
    std::slice::Iter<'a, (P, T, O)>,
    fn(&'a (P, T, O)) -> (P, O)
>;

impl<P, T, O> Sieve for InMemoryOrientedSieve<P, T, O>
where
    P: Copy + Eq + std::hash::Hash + Ord + std::fmt::Debug,
    T: Clone,
    O: Orientation,
{
    type Point = P;
    type Payload = T;

    type ConeIter<'a> = MapOut<'a, P, T, O> where Self: 'a;
    type SupportIter<'a> = MapOut<'a, P, T, O> where Self: 'a;

    fn cone<'a>(&'a self, p: P) -> Self::ConeIter<'a> {
        fn map_fn<P: Copy, T: Clone, O: Copy>((dst, pay, _): &(P, T, O)) -> (P, T) { (*dst, pay.clone()) }
        let f: fn(&(P, T, O)) -> (P, T) = map_fn::<P, T, O>;
        self.adjacency_out.get(&p).map(|v| v.iter().map(f)).unwrap_or_else(|| [].iter().map(f))
    }

    fn support<'a>(&'a self, p: P) -> Self::SupportIter<'a> {
        fn map_fn<P: Copy, T: Clone, O: Copy>((src, pay, _): &(P, T, O)) -> (P, T) { (*src, pay.clone()) }
        let f: fn(&(P, T, O)) -> (P, T) = map_fn::<P, T, O>;
        self.adjacency_in.get(&p).map(|v| v.iter().map(f)).unwrap_or_else(|| [].iter().map(f))
    }

    fn add_arrow(&mut self, src: P, dst: P, payload: T) {
        self.add_arrow_o(src, dst, payload, O::default());
    }

    fn remove_arrow(&mut self, src: P, dst: P) -> Option<T> {
        let mut removed = None;
        if let Some(v) = self.adjacency_out.get_mut(&src) {
            if let Some(pos) = v.iter().position(|(d,_,_)| *d == dst) {
                removed = Some(v.remove(pos).1);
            }
        }
        if let Some(v) = self.adjacency_in.get_mut(&dst) {
            if let Some(pos) = v.iter().position(|(s,_,_)| *s == src) {
                v.remove(pos);
            }
        }
        self.invalidate_cache();
        removed
    }

    fn base_points<'a>(&'a self) -> Box<dyn Iterator<Item = P> + 'a> {
        Box::new(self.adjacency_out.keys().copied())
    }

    fn cap_points<'a>(&'a self) -> Box<dyn Iterator<Item = P> + 'a> {
        Box::new(self.adjacency_in.keys().copied())
    }
}

impl<P, T, O> OrientedSieve for InMemoryOrientedSieve<P, T, O>
where
    P: Copy + Eq + std::hash::Hash + Ord + std::fmt::Debug,
    T: Clone,
    O: Orientation,
{
    type Orient = O;
    type ConeOIter<'a> = MapOOut<'a, P, T, O> where Self: 'a;
    type SupportOIter<'a> = MapOOut<'a, P, T, O> where Self: 'a;

    fn cone_o<'a>(&'a self, p: P) -> Self::ConeOIter<'a> {
        fn map_fn<P: Copy, T, O: Copy>((dst, _, ori): &(P, T, O)) -> (P, O) { (*dst, *ori) }
        let f: fn(&(P, T, O)) -> (P, O) = map_fn::<P, T, O>;
        self.adjacency_out.get(&p).map(|v| v.iter().map(f)).unwrap_or_else(|| [].iter().map(f))
    }

    fn support_o<'a>(&'a self, p: P) -> Self::SupportOIter<'a> {
        fn map_fn<P: Copy, T, O: Copy>((src, _, ori): &(P, T, O)) -> (P, O) { (*src, *ori) }
        let f: fn(&(P, T, O)) -> (P, O) = map_fn::<P, T, O>;
        self.adjacency_in.get(&p).map(|v| v.iter().map(f)).unwrap_or_else(|| [].iter().map(f))
    }

    fn add_arrow_o(&mut self, src: P, dst: P, payload: T, orient: O) {
        self.adjacency_out.entry(src).or_default().push((dst, payload.clone(), orient));
        self.adjacency_in.entry(dst).or_default().push((src, payload, orient));
        self.invalidate_cache();
    }
}

impl<P, T, O> InvalidateCache for InMemoryOrientedSieve<P, T, O>
where
    P: Copy + Eq + std::hash::Hash + Ord + std::fmt::Debug,
    T: Clone,
    O: Orientation,
{
    fn invalidate_cache(&mut self) {
        self.strata.take();
    }
}

