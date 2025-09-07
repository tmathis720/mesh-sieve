use once_cell::sync::OnceCell;
use std::collections::BTreeMap;

use super::mutable::MutableSieve;
use super::sieve_trait::Sieve;
use crate::mesh_error::MeshSieveError;
use crate::topology::_debug_invariants::debug_invariants;
use crate::topology::bounds::{PayloadLike, PointLike};
use crate::topology::cache::InvalidateCache;
use crate::topology::sieve::strata::{StrataCache, compute_strata};

/// Deterministic in-memory sieve backed by `BTreeMap` and sorted neighbor lists.
#[derive(Clone, Debug)]
pub struct InMemorySieveDeterministic<P, T = ()>
where
    P: PointLike,
{
    pub adjacency_out: BTreeMap<P, Vec<(P, T)>>,
    pub adjacency_in: BTreeMap<P, Vec<(P, T)>>,
    pub strata: OnceCell<StrataCache<P>>,
}

impl<P, T> Default for InMemorySieveDeterministic<P, T>
where
    P: PointLike,
{
    fn default() -> Self {
        Self {
            adjacency_out: BTreeMap::new(),
            adjacency_in: BTreeMap::new(),
            strata: OnceCell::new(),
        }
    }
}

impl<P: PointLike, T: PayloadLike> InMemorySieveDeterministic<P, T> {
    #[inline]
    fn upsert(vec: &mut Vec<(P, T)>, key: P, payload: T) -> bool {
        match vec.binary_search_by_key(&key, |(q, _)| *q) {
            Ok(pos) => {
                vec[pos].1 = payload;
                false
            }
            Err(pos) => {
                vec.insert(pos, (key, payload));
                true
            }
        }
    }

    #[inline]
    fn remove(vec: &mut Vec<(P, T)>, key: P) -> Option<T> {
        match vec.binary_search_by_key(&key, |(q, _)| *q) {
            Ok(pos) => Some(vec.remove(pos).1),
            Err(_) => None,
        }
    }

    #[inline]
    fn scrub_outgoing_only(&mut self, src: P) {
        if let Some(outs) = self.adjacency_out.remove(&src) {
            for (dst, _) in outs {
                if let Some(ins) = self.adjacency_in.get_mut(&dst) {
                    let _ = Self::remove(ins, src);
                }
            }
        }
    }

    #[inline]
    fn scrub_incoming_only(&mut self, dst: P) {
        if let Some(ins) = self.adjacency_in.remove(&dst) {
            for (src, _) in ins {
                if let Some(outs) = self.adjacency_out.get_mut(&src) {
                    let _ = Self::remove(outs, dst);
                }
            }
        }
    }

    #[inline]
    fn strata_cache(&self) -> Result<&StrataCache<P>, MeshSieveError> {
        self.strata.get_or_try_init(|| compute_strata(self))
    }
}

impl<P: PointLike, T: PayloadLike> InvalidateCache for InMemorySieveDeterministic<P, T> {
    #[inline]
    fn invalidate_cache(&mut self) {
        self.strata.take();
    }
}

impl<P: PointLike, T: PayloadLike> Sieve for InMemorySieveDeterministic<P, T> {
    type Point = P;
    type Payload = T;

    type ConeIter<'a>
        = std::iter::Cloned<std::slice::Iter<'a, (P, T)>>
    where
        Self: 'a;
    type SupportIter<'a>
        = std::iter::Cloned<std::slice::Iter<'a, (P, T)>>
    where
        Self: 'a;

    fn cone<'a>(&'a self, p: P) -> Self::ConeIter<'a> {
        self.adjacency_out
            .get(&p)
            .map(|v| v.iter().cloned())
            .unwrap_or_else(|| [].iter().cloned())
    }

    fn support<'a>(&'a self, p: P) -> Self::SupportIter<'a> {
        self.adjacency_in
            .get(&p)
            .map(|v| v.iter().cloned())
            .unwrap_or_else(|| [].iter().cloned())
    }

    fn add_arrow(&mut self, src: P, dst: P, payload: T) {
        let ins = Self::upsert(
            self.adjacency_out.entry(src).or_default(),
            dst,
            payload.clone(),
        );
        let _ = Self::upsert(self.adjacency_in.entry(dst).or_default(), src, payload);
        if ins {
            self.invalidate_cache();
        }
        debug_invariants!(self);
    }

    fn remove_arrow(&mut self, src: P, dst: P) -> Option<T> {
        let removed = self
            .adjacency_out
            .get_mut(&src)
            .and_then(|v| Self::remove(v, dst));
        if removed.is_some() {
            if let Some(v) = self.adjacency_in.get_mut(&dst) {
                let _ = Self::remove(v, src);
            }
            self.invalidate_cache();
            debug_invariants!(self);
        }
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
        let cache = self.strata_cache()?;
        let items = cache.strata.get(k as usize).cloned().unwrap_or_default();
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
}

impl<P: PointLike, T: PayloadLike> MutableSieve for InMemorySieveDeterministic<P, T> {
    fn add_point(&mut self, p: P) {
        self.adjacency_out.entry(p).or_default();
        self.adjacency_in.entry(p).or_default();
        self.invalidate_cache();
    }
    fn remove_point(&mut self, p: P) {
        if self.adjacency_out.contains_key(&p) {
            self.scrub_outgoing_only(p);
        }
        if self.adjacency_in.contains_key(&p) {
            self.scrub_incoming_only(p);
        }
        self.invalidate_cache();
    }
    fn add_base_point(&mut self, p: P) {
        self.adjacency_out.entry(p).or_default();
        self.invalidate_cache();
    }
    fn add_cap_point(&mut self, p: P) {
        self.adjacency_in.entry(p).or_default();
        self.invalidate_cache();
    }
    fn remove_base_point(&mut self, p: P) {
        if self.adjacency_out.contains_key(&p) {
            self.scrub_outgoing_only(p);
        }
        self.invalidate_cache();
    }
    fn remove_cap_point(&mut self, p: P) {
        if self.adjacency_in.contains_key(&p) {
            self.scrub_incoming_only(p);
        }
        self.invalidate_cache();
    }
    fn reserve_cone(&mut self, p: P, additional: usize) {
        self.adjacency_out.entry(p).or_default().reserve(additional);
    }
    fn reserve_support(&mut self, p: P, additional: usize) {
        self.adjacency_in.entry(p).or_default().reserve(additional);
    }
}

impl<P: PointLike, T: PayloadLike> InMemorySieveDeterministic<P, T> {
    #[cfg(debug_assertions)]
    pub(crate) fn debug_assert_invariants(&self) {
        use std::collections::HashMap;
        let out_view: HashMap<P, Vec<(P, ())>> = self
            .adjacency_out
            .iter()
            .map(|(&src, v)| (src, v.iter().map(|(dst, _)| (*dst, ())).collect()))
            .collect();
        crate::topology::_debug_invariants::assert_no_dups_per_src(&out_view);
        let out_total: usize = self.adjacency_out.values().map(|v| v.len()).sum();
        let in_total: usize = self.adjacency_in.values().map(|v| v.len()).sum();
        debug_assert_eq!(out_total, in_total, "out != in");
    }
}

impl<P: PointLike, T: PayloadLike> super::query_ext::SieveQueryExt
    for InMemorySieveDeterministic<P, T>
{
    #[inline]
    fn out_degree(&self, p: P) -> usize {
        self.adjacency_out.get(&p).map_or(0, |v| v.len())
    }
    #[inline]
    fn in_degree(&self, p: P) -> usize {
        self.adjacency_in.get(&p).map_or(0, |v| v.len())
    }
}
