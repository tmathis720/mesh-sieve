// src/topology/sieve/in_memory.rs

use super::sieve_trait::Sieve;
use once_cell::sync::OnceCell;
use std::collections::HashMap;
use crate::topology::stratum::StrataCache;

#[derive(Clone, Debug)]
pub struct InMemorySieve<P, T=()>
where
    P: Ord,
{
    pub adjacency_out: HashMap<P, Vec<(P,T)>>,
    pub adjacency_in:  HashMap<P, Vec<(P,T)>>,
    pub strata: OnceCell<StrataCache<P>>,
}

impl<P: Copy+Eq+std::hash::Hash+Ord, T> Default for InMemorySieve<P,T> {
    fn default() -> Self {
        Self {
            adjacency_out: HashMap::new(),
            adjacency_in: HashMap::new(),
            strata: OnceCell::new(),
        }
    }
}

impl<P: Copy+Eq+std::hash::Hash+Ord, T:Clone> InMemorySieve<P,T> {
    pub fn new() -> Self { Self::default() }
    pub fn from_arrows<I:IntoIterator<Item=(P,P,T)>>(arrows:I) -> Self {
        let mut sieve = Self::default();
        for (src, dst, payload) in arrows {
            sieve.add_arrow(src, dst, payload);
        }
        sieve
    }
}

type ConeMapIter<'a, P, T> = std::iter::Map<std::slice::Iter<'a, (P, T)>, fn(&'a (P, T)) -> (P, &'a T)>;

impl<P: Copy+Eq+std::hash::Hash+Ord, T:Clone> Sieve for InMemorySieve<P,T> {
    type Point = P;
    type Payload = T;
    type ConeIter<'a> = ConeMapIter<'a, P, T> where Self: 'a;
    type SupportIter<'a> = ConeMapIter<'a, P, T> where Self: 'a;

    fn cone<'a>(&'a self, p: P) -> Self::ConeIter<'a> {
        fn map_fn<P, T>((dst, pay): &(P, T)) -> (P, &T) where P: Copy { (*dst, pay) }
        let f: fn(&(P, T)) -> (P, &T) = map_fn::<P, T>;
        self.adjacency_out.get(&p).map(|v| v.iter().map(f)).unwrap_or_else(|| [].iter().map(f))
    }
    fn support<'a>(&'a self, p: P) -> Self::SupportIter<'a> {
        fn map_fn<P, T>((src, pay): &(P, T)) -> (P, &T) where P: Copy { (*src, pay) }
        let f: fn(&(P, T)) -> (P, &T) = map_fn::<P, T>;
        self.adjacency_in.get(&p).map(|v| v.iter().map(f)).unwrap_or_else(|| [].iter().map(f))
    }
    fn add_arrow(&mut self, src: P, dst: P, payload: T) {
        self.adjacency_out.entry(src).or_default().push((dst, payload.clone()));
        self.adjacency_in.entry(dst).or_default().push((src, payload));
        self.strata.take();
    }
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
    fn points<'a>(&'a self) -> Box<dyn Iterator<Item=P> + 'a> {
        let mut set = std::collections::HashSet::new();
        set.extend(self.adjacency_out.keys().copied());
        set.extend(self.adjacency_in.keys().copied());
        Box::new(set.into_iter())
    }
    fn base_points<'a>(&'a self) -> Box<dyn Iterator<Item=P> + 'a> {
        Box::new(self.adjacency_out.keys().copied())
    }
    fn cap_points<'a>(&'a self) -> Box<dyn Iterator<Item=P> + 'a> {
        Box::new(self.adjacency_in.keys().copied())
    }
    // override strata‐helpers using `self.strata_cache()`
    fn height(&self,p:P)->u32{ self.strata_cache().height.get(&p).copied().unwrap_or(0) }
    fn depth(&self,p:P)->u32{ self.strata_cache().depth.get(&p).copied().unwrap_or(0) }
    fn diameter(&self)->u32{ self.strata_cache().diameter }
    fn height_stratum(&self,k:u32)->Box<dyn Iterator<Item=P> + '_> {
        let cache = self.strata_cache();
        if let Some(v) = cache.strata.get(k as usize) {
            Box::new(v.iter().copied())
        } else {
            Box::new(std::iter::empty())
        }
    }
    fn depth_stratum(&self,k:u32)->Box<dyn Iterator<Item=P> + '_> {
        let cache = self.strata_cache();
        let points: Vec<_> = cache.depth.iter().filter(|(_, d)| **d == k).map(|(&p, _)| p).collect();
        Box::new(points.into_iter())
    }
    // Implement meet and join by delegating to LatticeOps
    fn meet<'s>(&'s self, a: P, b: P) -> Box<dyn Iterator<Item=P> + 's> {
        crate::topology::sieve::lattice::LatticeOps::meet(self, a, b)
    }
    fn join<'s>(&'s self, a: P, b: P) -> Box<dyn Iterator<Item=P> + 's> {
        crate::topology::sieve::lattice::LatticeOps::join(self, a, b)
    }
}
