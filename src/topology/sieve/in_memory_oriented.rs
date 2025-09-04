//! In-memory oriented Sieve: stores (dst, payload, orientation) per arrow.
//! Implements both `Sieve` (payload-only view) and `OrientedSieve`.

use once_cell::sync::OnceCell;
use std::collections::HashMap;
use std::sync::Arc;

use super::mutable::MutableSieve;
use super::oriented::{Orientation, OrientedSieve};
use super::sieve_trait::Sieve;
use crate::mesh_error::MeshSieveError;
use crate::topology::cache::InvalidateCache;
use crate::topology::orientation::Sign;
use crate::topology::sieve::strata::{StrataCache, compute_strata};
use crate::topology::_debug_invariants::debug_invariants;

#[derive(Clone, Debug)]
pub struct InMemoryOrientedSieve<P, T = (), O = Sign>
where
    P: Ord + std::fmt::Debug,
    O: Orientation,
{
    pub adjacency_out: HashMap<P, Vec<(P, T, O)>>,
    pub adjacency_in: HashMap<P, Vec<(P, T, O)>>,
    pub strata: OnceCell<StrataCache<P>>,
}

impl<P, T, O> InMemoryOrientedSieve<P, Arc<T>, O>
where
    P: Copy + Eq + std::hash::Hash + Ord + std::fmt::Debug,
    O: super::oriented::Orientation + PartialEq + std::fmt::Debug,
{
    /// Insert by value; wraps once into `Arc<T>`.
    #[inline]
    pub fn add_arrow_val(&mut self, src: P, dst: P, payload: T) {
        self.add_arrow(src, dst, Arc::new(payload));
    }

    /// Oriented insert by value; wraps once into `Arc<T>`.
    #[inline]
    pub fn add_arrow_o_val(&mut self, src: P, dst: P, payload: T, orient: O) {
        self.add_arrow_o(src, dst, Arc::new(payload), orient);
    }
}

impl<P, T, O> Default for InMemoryOrientedSieve<P, T, O>
where
    P: Copy + Eq + std::hash::Hash + Ord + std::fmt::Debug,
    O: Orientation + PartialEq + std::fmt::Debug,
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
    O: Orientation + PartialEq + std::fmt::Debug,
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

    #[cfg(any(debug_assertions, feature = "strict-invariants"))]
    pub(crate) fn debug_assert_invariants(&self)
    where
        O: super::oriented::Orientation + PartialEq + std::fmt::Debug,
    {
        use crate::topology::_debug_invariants as dbg;

        let out_view: std::collections::HashMap<P, Vec<(P, ())>> = self
            .adjacency_out
            .iter()
            .map(|(&src, v)| (src, v.iter().map(|(dst, _, _)| (*dst, ())).collect()))
            .collect();
        dbg::assert_no_dups_per_src(&out_view);

        let out_pairs = dbg::count_pairs(self.adjacency_out.iter().flat_map(|(&src, vec)| {
            vec.iter().map(move |(dst, _, _)| (src, *dst))
        }));
        let in_pairs = dbg::count_pairs(self.adjacency_in.iter().flat_map(|(&dst, vec)| {
            vec.iter().map(move |(src, _, _)| (*src, dst))
        }));
        dbg::counts_equal(&out_pairs, &in_pairs, "adjacency_out", "adjacency_in");

        fn check_orient<P, T, O>(s: &InMemoryOrientedSieve<P, T, O>)
        where
            P: Copy + Eq + std::hash::Hash + Ord + std::fmt::Debug,
            O: super::oriented::Orientation + PartialEq + std::fmt::Debug,
        {
            use std::collections::HashMap;
            let mut out_map: HashMap<(P, P), O> = HashMap::new();
            for (&src, v) in &s.adjacency_out {
                for (dst, _, o) in v {
                    let prev = out_map.insert((src, *dst), *o);
                    debug_assert!(prev.is_none(), "duplicate (src,dst) unexpectedly seen");
                }
            }
            for (&dst, v) in &s.adjacency_in {
                for (src, _, o_in) in v {
                    let Some(o_out) = out_map.get(&(*src, dst)) else {
                        debug_assert!(false, "in mirror without out entry: ({:?}->{:?})", src, dst);
                        continue;
                    };
                    debug_assert_eq!(
                        o_in, o_out,
                        "orientation mismatch for ({:?}->{:?}): in={:?}, out={:?}",
                        src, dst, o_in, o_out
                    );
                }
            }
        }

        check_orient(self);
    }

    /// Pre-size cone/support capacities from `(src, dst, count)` tuples.
    pub fn reserve_from_edge_counts(
        &mut self,
        counts: impl IntoIterator<Item = (P, P, usize)>,
    ) {
        use std::collections::HashMap;
        let mut by_src: HashMap<P, usize> = HashMap::new();
        let mut by_dst: HashMap<P, usize> = HashMap::new();
        for (src, dst, k) in counts {
            *by_src.entry(src).or_default() += k;
            *by_dst.entry(dst).or_default() += k;
        }
        for (src, k) in by_src {
            MutableSieve::reserve_cone(self, src, k);
        }
        for (dst, k) in by_dst {
            MutableSieve::reserve_support(self, dst, k);
        }
    }

    /// Convenience helper to preallocate from a raw edge list.
    pub fn reserve_from_edges(&mut self, edges: impl IntoIterator<Item = (P, P)>) {
        use std::collections::HashMap;
        let mut by_src: HashMap<P, usize> = HashMap::new();
        let mut by_dst: HashMap<P, usize> = HashMap::new();
        for (s, d) in edges {
            *by_src.entry(s).or_default() += 1;
            *by_dst.entry(d).or_default() += 1;
        }
        for (s, k) in by_src {
            MutableSieve::reserve_cone(self, s, k);
        }
        for (d, k) in by_dst {
            MutableSieve::reserve_support(self, d, k);
        }
    }

    /// Optional: compact any excess capacity after bulk construction.
    pub fn shrink_to_fit(&mut self) {
        for v in self.adjacency_out.values_mut() {
            v.shrink_to_fit();
        }
        for v in self.adjacency_in.values_mut() {
            v.shrink_to_fit();
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
    O: Orientation + PartialEq + std::fmt::Debug,
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
        debug_invariants!(self);
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
}

impl<P, T, O> MutableSieve for InMemoryOrientedSieve<P, T, O>
where
    P: Copy + Eq + std::hash::Hash + Ord + std::fmt::Debug,
    T: Clone,
    O: Orientation + PartialEq + std::fmt::Debug,
{
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
        debug_invariants!(self);
    }

    fn remove_point(&mut self, p: P) {
        if self.adjacency_out.contains_key(&p) {
            self.scrub_outgoing_only(p);
        }
        if self.adjacency_in.contains_key(&p) {
            self.scrub_incoming_only(p);
        }
        self.invalidate_cache();
        debug_invariants!(self);
    }

    fn add_base_point(&mut self, p: P) {
        self.adjacency_out.entry(p).or_default();
        self.invalidate_cache();
        debug_invariants!(self);
    }

    fn add_cap_point(&mut self, p: P) {
        self.adjacency_in.entry(p).or_default();
        self.invalidate_cache();
        debug_invariants!(self);
    }

    fn remove_base_point(&mut self, p: P) {
        if self.adjacency_out.contains_key(&p) {
            self.scrub_outgoing_only(p);
            self.adjacency_out.remove(&p);
        }
        self.invalidate_cache();
        debug_invariants!(self);
    }

    fn remove_cap_point(&mut self, p: P) {
        if self.adjacency_in.contains_key(&p) {
            self.scrub_incoming_only(p);
            self.adjacency_in.remove(&p);
        }
        self.invalidate_cache();
        debug_invariants!(self);
    }

    fn set_cone(&mut self, p: P, chain: impl IntoIterator<Item = (P, T)>) {
        let mut new_cone: Vec<(P, T)> = chain.into_iter().collect();

        // Dedup by destination, last wins
        {
            use std::collections::HashSet;
            let mut seen = HashSet::new();
            let mut dedup = Vec::with_capacity(new_cone.len());
            for (dst, pay) in new_cone.into_iter().rev() {
                if seen.insert(dst) {
                    dedup.push((dst, pay));
                }
            }
            dedup.reverse();
            new_cone = dedup;
        }

        self.adjacency_out.entry(p).or_default().reserve(new_cone.len());
        {
            use std::collections::HashMap;
            let mut dst_counts: HashMap<P, usize> = HashMap::new();
            for (dst, _) in &new_cone {
                *dst_counts.entry(*dst).or_default() += 1;
            }
            for (dst, cnt) in dst_counts {
                MutableSieve::reserve_support(self, dst, cnt);
            }
        }

        let old_cone: Vec<(P, T, O)> = std::mem::take(self.adjacency_out.entry(p).or_default());
        for (dst, _, _) in &old_cone {
            if let Some(ins) = self.adjacency_in.get_mut(dst) {
                ins.retain(|(src, _, _)| *src != p);
            }
        }

        let out = self.adjacency_out.entry(p).or_default();
        *out = new_cone
            .iter()
            .map(|(d, pay)| (*d, pay.clone(), O::default()))
            .collect();
        for (dst, pay) in new_cone {
            self.adjacency_in
                .entry(dst)
                .or_default()
                .push((p, pay, O::default()));
        }

        self.invalidate_cache();
        debug_invariants!(self);
    }

    fn add_cone(&mut self, p: P, chain: impl IntoIterator<Item = (P, T)>) {
        let add: Vec<(P, T)> = chain.into_iter().collect();

        self.adjacency_out.entry(p).or_default().reserve(add.len());
        {
            use std::collections::HashMap;
            let mut dst_counts: HashMap<P, usize> = HashMap::new();
            for (dst, _) in &add {
                *dst_counts.entry(*dst).or_default() += 1;
            }
            for (dst, cnt) in dst_counts {
                MutableSieve::reserve_support(self, dst, cnt);
            }
        }

        let out = self.adjacency_out.entry(p).or_default();
        for (dst, pay) in add {
            if let Some(slot) = out.iter_mut().find(|(d, _, _)| *d == dst) {
                slot.1 = pay.clone();
                slot.2 = O::default();
            } else {
                out.push((dst, pay.clone(), O::default()));
            }
            let ins = self.adjacency_in.entry(dst).or_default();
            if let Some(slot) = ins.iter_mut().find(|(s, _, _)| *s == p) {
                slot.1 = pay.clone();
                slot.2 = O::default();
            } else {
                ins.push((p, pay.clone(), O::default()));
            }
        }
        self.invalidate_cache();
        debug_invariants!(self);
    }

    fn set_support(&mut self, q: P, chain: impl IntoIterator<Item = (P, T)>) {
        let mut new_sup: Vec<(P, T)> = chain.into_iter().collect();

        // Dedup by source, last wins
        {
            use std::collections::HashSet;
            let mut seen = HashSet::new();
            let mut dedup = Vec::with_capacity(new_sup.len());
            for (src, pay) in new_sup.into_iter().rev() {
                if seen.insert(src) {
                    dedup.push((src, pay));
                }
            }
            dedup.reverse();
            new_sup = dedup;
        }

        self.adjacency_in.entry(q).or_default().reserve(new_sup.len());
        {
            use std::collections::HashMap;
            let mut src_counts: HashMap<P, usize> = HashMap::new();
            for (src, _) in &new_sup {
                *src_counts.entry(*src).or_default() += 1;
            }
            for (src, cnt) in src_counts {
                MutableSieve::reserve_cone(self, src, cnt);
            }
        }

        let old_sup: Vec<(P, T, O)> = std::mem::take(self.adjacency_in.entry(q).or_default());
        for (src, _, _) in &old_sup {
            if let Some(outs) = self.adjacency_out.get_mut(src) {
                outs.retain(|(dst, _, _)| *dst != q);
            }
        }

        let ins = self.adjacency_in.entry(q).or_default();
        *ins = new_sup
            .iter()
            .map(|(s, pay)| (*s, pay.clone(), O::default()))
            .collect();
        for (src, pay) in new_sup {
            self.adjacency_out
                .entry(src)
                .or_default()
                .push((q, pay, O::default()));
        }
        self.invalidate_cache();
        debug_invariants!(self);
    }

    fn add_support(&mut self, q: P, chain: impl IntoIterator<Item = (P, T)>) {
        let add: Vec<(P, T)> = chain.into_iter().collect();

        self.adjacency_in.entry(q).or_default().reserve(add.len());
        {
            use std::collections::HashMap;
            let mut src_counts: HashMap<P, usize> = HashMap::new();
            for (src, _) in &add {
                *src_counts.entry(*src).or_default() += 1;
            }
            for (src, cnt) in src_counts {
                MutableSieve::reserve_cone(self, src, cnt);
            }
        }

        let ins = self.adjacency_in.entry(q).or_default();
        for (src, pay) in add {
            if let Some(slot) = ins.iter_mut().find(|(s, _, _)| *s == src) {
                slot.1 = pay.clone();
                slot.2 = O::default();
            } else {
                ins.push((src, pay.clone(), O::default()));
            }
            let outs = self.adjacency_out.entry(src).or_default();
            if let Some(slot) = outs.iter_mut().find(|(d, _, _)| *d == q) {
                slot.1 = pay.clone();
                slot.2 = O::default();
            } else {
                outs.push((q, pay.clone(), O::default()));
            }
        }
        self.invalidate_cache();
        debug_invariants!(self);
    }
}

impl<P, T, O> OrientedSieve for InMemoryOrientedSieve<P, T, O>
where
    P: Copy + Eq + std::hash::Hash + Ord + std::fmt::Debug,
    T: Clone,
    O: Orientation + PartialEq + std::fmt::Debug,
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
        debug_invariants!(self);
    }
}

impl<P, T, O> InvalidateCache for InMemoryOrientedSieve<P, T, O>
where
    P: Copy + Eq + std::hash::Hash + Ord + std::fmt::Debug,
    T: Clone,
    O: Orientation + PartialEq + std::fmt::Debug,
{
    #[inline]
    fn invalidate_cache(&mut self) {
        self.strata.take();
    }
}
