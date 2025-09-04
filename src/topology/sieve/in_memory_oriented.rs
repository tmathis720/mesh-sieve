//! In-memory oriented Sieve: stores (dst, payload, orientation) per arrow.
//! Implements both `Sieve` (payload-only view) and `OrientedSieve`.

use once_cell::sync::OnceCell;
use std::collections::HashMap;

use super::oriented::{Orientation, OrientedSieve};
use super::sieve_trait::Sieve;
use crate::mesh_error::MeshSieveError;
use crate::topology::cache::InvalidateCache;
use crate::topology::sieve::strata::{compute_strata, StrataCache};

#[derive(Clone, Debug)]
pub struct InMemoryOrientedSieve<P, T = (), O = i32>
where
    P: Ord + std::fmt::Debug,
    O: Orientation,
{
    pub adjacency_out: HashMap<P, Vec<(P, T, O)>>,
    pub adjacency_in: HashMap<P, Vec<(P, T, O)>>,
    pub strata: OnceCell<StrataCache<P>>,
}

impl<P, T, O> Default for InMemoryOrientedSieve<P, T, O>
where
    P: Copy + Eq + std::hash::Hash + Ord + std::fmt::Debug,
    O: Orientation,
{
    fn default() -> Self {
        Self {
            adjacency_out: HashMap::new(),
            adjacency_in: HashMap::new(),
            strata: OnceCell::new(),
        }
    }
}

impl<P, T, O> InMemoryOrientedSieve<P, T, O>
where
    P: Copy + Eq + std::hash::Hash + Ord + std::fmt::Debug,
    T: Clone,
    O: Orientation,
{
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_arrows<I: IntoIterator<Item = (P, P, T, O)>>(arrows: I) -> Self {
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
                self.adjacency_in
                    .entry(dst)
                    .or_default()
                    .push((src, pay.clone(), ori));
            }
        }
    }

    #[inline]
    pub fn strata_cache(&self) -> Result<&StrataCache<P>, MeshSieveError> {
        self.strata.get_or_try_init(|| compute_strata(self))
    }

    #[inline]
    pub fn invalidate_strata(&mut self) {
        self.strata.take();
    }

    #[inline]
    fn scrub_outgoing_only(&mut self, src: P) {
        let old: Vec<(P, T, O)> = std::mem::take(self.adjacency_out.entry(src).or_default());
        for (dst, _, _) in old {
            if let Some(ins) = self.adjacency_in.get_mut(&dst) {
                ins.retain(|(s, _, _)| *s != src);
            }
        }
    }

    #[inline]
    fn scrub_incoming_only(&mut self, dst: P) {
        let old: Vec<(P, T, O)> = std::mem::take(self.adjacency_in.entry(dst).or_default());
        for (src, _, _) in old {
            if let Some(outs) = self.adjacency_out.get_mut(&src) {
                outs.retain(|(d, _, _)| *d != dst);
            }
        }
    }

    #[cfg(debug_assertions)]
    pub fn debug_assert_consistent(&self) {
        for (src, outs) in &self.adjacency_out {
            for (dst, _, _) in outs {
                let ok = self
                    .adjacency_in
                    .get(dst)
                    .map_or(false, |ins| ins.iter().any(|(s, _, _)| s == src));
                debug_assert!(
                    ok,
                    "Missing mirror in[{dst:?}] for out edge ({src:?} -> {dst:?})"
                );
            }
        }
        for (dst, ins) in &self.adjacency_in {
            for (src, _, _) in ins {
                let ok = self
                    .adjacency_out
                    .get(src)
                    .map_or(false, |outs| outs.iter().any(|(d, _, _)| d == dst));
                debug_assert!(
                    ok,
                    "Missing mirror out[{src:?}] for in edge ({src:?} -> {dst:?})"
                );
            }
        }
    }

    #[inline]
    pub fn has_arrow(&self, src: P, dst: P) -> bool {
        self.adjacency_out
            .get(&src)
            .map_or(false, |v| v.iter().any(|(d, _, _)| *d == dst))
    }

    #[cfg(debug_assertions)]
    pub fn debug_assert_no_parallel_edges_src(&self, src: P) {
        if let Some(v) = self.adjacency_out.get(&src) {
            use std::collections::HashSet;
            let mut seen = HashSet::new();
            for (dst, _, _) in v {
                assert!(
                    seen.insert(*dst),
                    "duplicate edges out of {:?} to {:?}",
                    src,
                    dst
                );
            }
        }
    }

    #[cfg(debug_assertions)]
    pub fn debug_assert_no_parallel_edges_dst(&self, dst: P) {
        if let Some(v) = self.adjacency_in.get(&dst) {
            use std::collections::HashSet;
            let mut seen = HashSet::new();
            for (src, _, _) in v {
                assert!(
                    seen.insert(*src),
                    "duplicate edges into {:?} from {:?}",
                    dst,
                    src
                );
            }
        }
    }
}

// ----------- Sieve (payload-only view) -----------
type MapOut<'a, P, T, O> =
    std::iter::Map<std::slice::Iter<'a, (P, T, O)>, fn(&'a (P, T, O)) -> (P, T)>;

type MapOOut<'a, P, T, O> =
    std::iter::Map<std::slice::Iter<'a, (P, T, O)>, fn(&'a (P, T, O)) -> (P, O)>;

impl<P, T, O> Sieve for InMemoryOrientedSieve<P, T, O>
where
    P: Copy + Eq + std::hash::Hash + Ord + std::fmt::Debug,
    T: Clone,
    O: Orientation,
{
    type Point = P;
    type Payload = T;

    type ConeIter<'a>
        = MapOut<'a, P, T, O>
    where
        Self: 'a;
    type SupportIter<'a>
        = MapOut<'a, P, T, O>
    where
        Self: 'a;

    fn cone<'a>(&'a self, p: P) -> Self::ConeIter<'a> {
        fn map_fn<P: Copy, T: Clone, O: Copy>((dst, pay, _): &(P, T, O)) -> (P, T) {
            (*dst, pay.clone())
        }
        let f: fn(&(P, T, O)) -> (P, T) = map_fn::<P, T, O>;
        self.adjacency_out
            .get(&p)
            .map(|v| v.iter().map(f))
            .unwrap_or_else(|| [].iter().map(f))
    }

    fn support<'a>(&'a self, p: P) -> Self::SupportIter<'a> {
        fn map_fn<P: Copy, T: Clone, O: Copy>((src, pay, _): &(P, T, O)) -> (P, T) {
            (*src, pay.clone())
        }
        let f: fn(&(P, T, O)) -> (P, T) = map_fn::<P, T, O>;
        self.adjacency_in
            .get(&p)
            .map(|v| v.iter().map(f))
            .unwrap_or_else(|| [].iter().map(f))
    }

    fn add_arrow(&mut self, src: P, dst: P, payload: T) {
        self.add_arrow_o(src, dst, payload, O::default());
    }

    fn remove_arrow(&mut self, src: P, dst: P) -> Option<T> {
        let mut removed = None;
        if let Some(v) = self.adjacency_out.get_mut(&src) {
            if let Some(pos) = v.iter().position(|(d, _, _)| *d == dst) {
                removed = Some(v.remove(pos).1);
            }
        }
        if let Some(v) = self.adjacency_in.get_mut(&dst) {
            if let Some(pos) = v.iter().position(|(s, _, _)| *s == src) {
                v.remove(pos);
            }
        }
        self.invalidate_cache();
        #[cfg(debug_assertions)]
        self.debug_assert_consistent();
        removed
    }

    fn base_points<'a>(&'a self) -> Box<dyn Iterator<Item = P> + 'a> {
        Box::new(self.adjacency_out.keys().copied())
    }

    fn cap_points<'a>(&'a self) -> Box<dyn Iterator<Item = P> + 'a> {
        Box::new(self.adjacency_in.keys().copied())
    }

    fn height(&mut self, p: P) -> Result<u32, MeshSieveError> {
        let cache = self.strata_cache()?;
        Ok(cache.height.get(&p).copied().unwrap_or(0))
    }

    fn depth(&mut self, p: P) -> Result<u32, MeshSieveError> {
        let cache = self.strata_cache()?;
        Ok(cache.depth.get(&p).copied().unwrap_or(0))
    }

    fn diameter(&mut self) -> Result<u32, MeshSieveError> {
        Ok(self.strata_cache()?.diameter)
    }

    fn height_stratum<'a>(
        &'a mut self,
        k: u32,
    ) -> Result<Box<dyn Iterator<Item = P> + 'a>, MeshSieveError> {
        let items = self
            .strata_cache()?
            .strata
            .get(k as usize)
            .cloned()
            .unwrap_or_default();
        Ok(Box::new(items.into_iter()))
    }

    fn depth_stratum<'a>(
        &'a mut self,
        k: u32,
    ) -> Result<Box<dyn Iterator<Item = P> + 'a>, MeshSieveError> {
        let cache = self.strata_cache()?;
        let pts: Vec<_> = cache
            .depth
            .iter()
            .filter_map(|(&p, &d)| if d == k { Some(p) } else { None })
            .collect();
        Ok(Box::new(pts.into_iter()))
    }

    fn chart_index(&mut self, p: P) -> Result<Option<usize>, MeshSieveError> {
        Ok(self.strata_cache()?.index_of(p))
    }

    fn chart_points(&mut self) -> Result<Vec<P>, MeshSieveError> {
        Ok(self.strata_cache()?.chart_points.clone())
    }

    fn reserve_cone(&mut self, p: P, additional: usize) {
        self.adjacency_out.entry(p).or_default().reserve(additional);
    }

    fn reserve_support(&mut self, q: P, additional: usize) {
        self.adjacency_in.entry(q).or_default().reserve(additional);
    }

    fn add_point(&mut self, p: P) {
        self.adjacency_out.entry(p).or_default();
        self.adjacency_in.entry(p).or_default();
        self.invalidate_cache();
        #[cfg(debug_assertions)]
        self.debug_assert_consistent();
    }

    fn remove_point(&mut self, p: P) {
        if self.adjacency_out.contains_key(&p) {
            self.scrub_outgoing_only(p);
        }
        if self.adjacency_in.contains_key(&p) {
            self.scrub_incoming_only(p);
        }
        self.invalidate_cache();
        #[cfg(debug_assertions)]
        self.debug_assert_consistent();
    }

    fn add_base_point(&mut self, p: P) {
        self.adjacency_out.entry(p).or_default();
        self.invalidate_cache();
        #[cfg(debug_assertions)]
        self.debug_assert_consistent();
    }

    fn add_cap_point(&mut self, p: P) {
        self.adjacency_in.entry(p).or_default();
        self.invalidate_cache();
        #[cfg(debug_assertions)]
        self.debug_assert_consistent();
    }

    fn remove_base_point(&mut self, p: P) {
        if self.adjacency_out.contains_key(&p) {
            self.scrub_outgoing_only(p);
            self.adjacency_out.remove(&p);
        }
        self.invalidate_cache();
        #[cfg(debug_assertions)]
        self.debug_assert_consistent();
    }

    fn remove_cap_point(&mut self, p: P) {
        if self.adjacency_in.contains_key(&p) {
            self.scrub_incoming_only(p);
            self.adjacency_in.remove(&p);
        }
        self.invalidate_cache();
        #[cfg(debug_assertions)]
        self.debug_assert_consistent();
    }
}

impl<P, T, O> OrientedSieve for InMemoryOrientedSieve<P, T, O>
where
    P: Copy + Eq + std::hash::Hash + Ord + std::fmt::Debug,
    T: Clone,
    O: Orientation,
{
    type Orient = O;
    type ConeOIter<'a>
        = MapOOut<'a, P, T, O>
    where
        Self: 'a;
    type SupportOIter<'a>
        = MapOOut<'a, P, T, O>
    where
        Self: 'a;

    fn cone_o<'a>(&'a self, p: P) -> Self::ConeOIter<'a> {
        fn map_fn<P: Copy, T, O: Copy>((dst, _, ori): &(P, T, O)) -> (P, O) {
            (*dst, *ori)
        }
        let f: fn(&(P, T, O)) -> (P, O) = map_fn::<P, T, O>;
        self.adjacency_out
            .get(&p)
            .map(|v| v.iter().map(f))
            .unwrap_or_else(|| [].iter().map(f))
    }

    fn support_o<'a>(&'a self, p: P) -> Self::SupportOIter<'a> {
        fn map_fn<P: Copy, T, O: Copy>((src, _, ori): &(P, T, O)) -> (P, O) {
            (*src, *ori)
        }
        let f: fn(&(P, T, O)) -> (P, O) = map_fn::<P, T, O>;
        self.adjacency_in
            .get(&p)
            .map(|v| v.iter().map(f))
            .unwrap_or_else(|| [].iter().map(f))
    }

    fn add_arrow_o(&mut self, src: P, dst: P, payload: T, orient: O) {
        // Upsert outgoing
        let outs = self.adjacency_out.entry(src).or_default();
        if let Some(slot) = outs.iter_mut().find(|(d, _, _)| *d == dst) {
            slot.1 = payload.clone();
            slot.2 = orient;
        } else {
            outs.push((dst, payload.clone(), orient));
        }

        // Upsert incoming
        let ins = self.adjacency_in.entry(dst).or_default();
        if let Some(slot) = ins.iter_mut().find(|(s, _, _)| *s == src) {
            slot.1 = payload;
            slot.2 = orient;
        } else {
            ins.push((src, payload, orient));
        }

        self.invalidate_cache();

        #[cfg(debug_assertions)]
        {
            self.debug_assert_consistent();
            self.debug_assert_no_parallel_edges_src(src);
            self.debug_assert_no_parallel_edges_dst(dst);
        }
    }
}

impl<P, T, O> InvalidateCache for InMemoryOrientedSieve<P, T, O>
where
    P: Copy + Eq + std::hash::Hash + Ord + std::fmt::Debug,
    T: Clone,
    O: Orientation,
{
    #[inline]
    fn invalidate_cache(&mut self) {
        self.strata.take();
    }
}
