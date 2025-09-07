//! Frozen CSR (Compressed Sparse Row) representation of a [`Sieve`] topology.
//!
//! Immutable, cache-friendly adjacency structure with deterministic iteration
//! order. All neighbor lists and the global chart are sorted; `cone`/`support`
//! walk contiguous slices, and global iteration is deterministic. Built from
//! any [`Sieve`] implementation and intended for read-only traversal workloads.

use std::collections::HashMap;
use std::sync::Arc;

use super::query_ext::SieveQueryExt;
use super::sieve_ref::SieveRef;
use super::sieve_trait::Sieve;
use crate::topology::bounds::{PayloadLike, PointLike};
use crate::topology::cache::InvalidateCache;

/// Immutable sieve backed by a pair of CSR adjacency graphs.
#[derive(Clone, Debug)]
pub struct FrozenSieveCsr<P, T>
where
    P: PointLike,
{
    /// Dense index → point mapping in deterministic order.
    pub point_of: Arc<[P]>,
    /// Point → dense index map.
    pub index_of: HashMap<P, u32>,

    /// CSR arrays for outgoing edges.
    pub out_offsets: Arc<[u32]>,
    pub out_dsts: Arc<[u32]>,
    pub out_pay: Arc<[T]>,

    /// CSR arrays for incoming edges (mirrors).
    pub in_offsets: Arc<[u32]>,
    pub in_srcs: Arc<[u32]>,
    pub in_pay: Arc<[T]>,
}

impl<P, T> Default for FrozenSieveCsr<P, T>
where
    P: PointLike,
{
    fn default() -> Self {
        Self {
            point_of: Arc::from([]),
            index_of: HashMap::new(),
            out_offsets: Arc::from([0]),
            out_dsts: Arc::from([]),
            out_pay: Arc::from([]),
            in_offsets: Arc::from([0]),
            in_srcs: Arc::from([]),
            in_pay: Arc::from([]),
        }
    }
}

impl<P, T> InvalidateCache for FrozenSieveCsr<P, T>
where
    P: PointLike,
{
    fn invalidate_cache(&mut self) {}
}

impl<P, T> FrozenSieveCsr<P, T>
where
    P: PointLike,
    T: PayloadLike,
{
    /// Build from any [`Sieve`], enforcing deterministic order.
    pub fn from_sieve<S>(mut s: S) -> Self
    where
        S: Sieve<Point = P, Payload = T>,
    {
        // 1) global point order
        let point_of: Vec<P> = match s.points_chart_order() {
            Ok(v) => v,
            Err(_) => s.points_sorted(),
        };
        let n = point_of.len();
        let mut index_of = HashMap::with_capacity(n);
        for (i, p) in point_of.iter().copied().enumerate() {
            index_of.insert(p, i as u32);
        }

        // 2) degree counts
        let mut out_deg = vec![0u32; n];
        let mut in_deg = vec![0u32; n];
        for (i, p) in point_of.iter().copied().enumerate() {
            for (q, _) in s.cone(p) {
                let qi = *index_of.get(&q).expect("destination must exist") as usize;
                out_deg[i] += 1;
                in_deg[qi] += 1;
            }
        }

        // prefix sums
        let mut out_offsets = vec![0u32; n + 1];
        let mut in_offsets = vec![0u32; n + 1];
        for i in 0..n {
            out_offsets[i + 1] = out_offsets[i] + out_deg[i];
            in_offsets[i + 1] = in_offsets[i] + in_deg[i];
        }
        let m = out_offsets[n] as usize;

        // allocate adjacency arrays
        let mut out_dsts = vec![0u32; m];
        let mut out_pay: Vec<Option<T>> = vec![None; m];
        let mut in_srcs = vec![0u32; m];
        let mut in_pay: Vec<Option<T>> = vec![None; m];
        let mut out_write = out_offsets.clone();
        let mut in_write = in_offsets.clone();

        // 3) populate arrays
        for (i, p) in point_of.iter().copied().enumerate() {
            let mut neigh: Vec<(u32, T)> = s
                .cone(p)
                .map(|(q, pay)| (*index_of.get(&q).unwrap(), pay))
                .collect();
            neigh.sort_unstable_by_key(|(qi, _)| *qi);
            for (qi, pay) in neigh {
                let pos = out_write[i] as usize;
                out_dsts[pos] = qi;
                out_pay[pos] = Some(pay.clone());
                out_write[i] += 1;

                let pos_in = in_write[qi as usize] as usize;
                in_srcs[pos_in] = i as u32;
                in_pay[pos_in] = Some(pay);
                in_write[qi as usize] += 1;
            }
        }

        let out_pay: Vec<T> = out_pay.into_iter().map(|o| o.unwrap()).collect();
        let in_pay: Vec<T> = in_pay.into_iter().map(|o| o.unwrap()).collect();

        Self {
            point_of: point_of.into(),
            index_of,
            out_offsets: out_offsets.into(),
            out_dsts: out_dsts.into(),
            out_pay: out_pay.into(),
            in_offsets: in_offsets.into(),
            in_srcs: in_srcs.into(),
            in_pay: in_pay.into(),
        }
    }
}

/// Freeze any [`Sieve`] into a [`FrozenSieveCsr`].
pub fn freeze_csr<S, P, T>(s: S) -> FrozenSieveCsr<P, T>
where
    S: Sieve<Point = P, Payload = T>,
    P: PointLike,
    T: PayloadLike,
{
    FrozenSieveCsr::from_sieve(s)
}

// --- iterators --------------------------------------------------------------------

pub struct ConeIter<'a, P, T>
where
    P: PointLike,
    T: PayloadLike,
{
    sieve: &'a FrozenSieveCsr<P, T>,
    pos: usize,
    end: usize,
}

impl<'a, P, T> Iterator for ConeIter<'a, P, T>
where
    P: PointLike,
    T: PayloadLike,
{
    type Item = (P, T);
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos < self.end {
            let k = self.pos;
            self.pos += 1;
            let qi = self.sieve.out_dsts[k] as usize;
            Some((self.sieve.point_of[qi], self.sieve.out_pay[k].clone()))
        } else {
            None
        }
    }
}

pub struct ConeRefIter<'a, P, T>
where
    P: PointLike,
{
    sieve: &'a FrozenSieveCsr<P, T>,
    pos: usize,
    end: usize,
}

impl<'a, P, T> Iterator for ConeRefIter<'a, P, T>
where
    P: PointLike,
{
    type Item = (P, &'a T);
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos < self.end {
            let k = self.pos;
            self.pos += 1;
            let qi = self.sieve.out_dsts[k] as usize;
            Some((self.sieve.point_of[qi], &self.sieve.out_pay[k]))
        } else {
            None
        }
    }
}

pub struct SupportIter<'a, P, T>
where
    P: PointLike,
    T: PayloadLike,
{
    sieve: &'a FrozenSieveCsr<P, T>,
    pos: usize,
    end: usize,
}

impl<'a, P, T> Iterator for SupportIter<'a, P, T>
where
    P: PointLike,
    T: PayloadLike,
{
    type Item = (P, T);
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos < self.end {
            let k = self.pos;
            self.pos += 1;
            let si = self.sieve.in_srcs[k] as usize;
            Some((self.sieve.point_of[si], self.sieve.in_pay[k].clone()))
        } else {
            None
        }
    }
}

pub struct SupportRefIter<'a, P, T>
where
    P: PointLike,
{
    sieve: &'a FrozenSieveCsr<P, T>,
    pos: usize,
    end: usize,
}

impl<'a, P, T> Iterator for SupportRefIter<'a, P, T>
where
    P: PointLike,
{
    type Item = (P, &'a T);
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos < self.end {
            let k = self.pos;
            self.pos += 1;
            let si = self.sieve.in_srcs[k] as usize;
            Some((self.sieve.point_of[si], &self.sieve.in_pay[k]))
        } else {
            None
        }
    }
}

// --- trait impls ----------------------------------------------------------------

impl<P, T> Sieve for FrozenSieveCsr<P, T>
where
    P: PointLike,
    T: PayloadLike,
{
    type Point = P;
    type Payload = T;

    type ConeIter<'a>
        = ConeIter<'a, P, T>
    where
        Self: 'a;
    type SupportIter<'a>
        = SupportIter<'a, P, T>
    where
        Self: 'a;

    fn cone<'a>(&'a self, p: P) -> Self::ConeIter<'a> {
        let i = *self.index_of.get(&p).expect("unknown point") as usize;
        let lo = self.out_offsets[i] as usize;
        let hi = self.out_offsets[i + 1] as usize;
        ConeIter {
            sieve: self,
            pos: lo,
            end: hi,
        }
    }

    fn support<'a>(&'a self, p: P) -> Self::SupportIter<'a> {
        let i = *self.index_of.get(&p).expect("unknown point") as usize;
        let lo = self.in_offsets[i] as usize;
        let hi = self.in_offsets[i + 1] as usize;
        SupportIter {
            sieve: self,
            pos: lo,
            end: hi,
        }
    }

    fn add_arrow(&mut self, _src: P, _dst: P, _payload: T) {
        unreachable!("frozen sieve is immutable");
    }

    fn remove_arrow(&mut self, _src: P, _dst: P) -> Option<T> {
        unreachable!("frozen sieve is immutable");
    }

    fn base_points<'a>(&'a self) -> Box<dyn Iterator<Item = P> + 'a> {
        Box::new(
            self.point_of
                .iter()
                .copied()
                .enumerate()
                .filter(|(i, _)| self.out_offsets[i + 1] > self.out_offsets[*i])
                .map(|(_, p)| p),
        )
    }

    fn cap_points<'a>(&'a self) -> Box<dyn Iterator<Item = P> + 'a> {
        Box::new(
            self.point_of
                .iter()
                .copied()
                .enumerate()
                .filter(|(i, _)| self.in_offsets[i + 1] > self.in_offsets[*i])
                .map(|(_, p)| p),
        )
    }
}

impl<P, T> SieveRef for FrozenSieveCsr<P, T>
where
    P: PointLike,
    T: PayloadLike,
{
    type ConeRefIter<'a>
        = ConeRefIter<'a, P, T>
    where
        Self: 'a;
    type SupportRefIter<'a>
        = SupportRefIter<'a, P, T>
    where
        Self: 'a;

    fn cone_ref<'a>(&'a self, p: P) -> Self::ConeRefIter<'a> {
        let i = *self.index_of.get(&p).expect("unknown point") as usize;
        let lo = self.out_offsets[i] as usize;
        let hi = self.out_offsets[i + 1] as usize;
        ConeRefIter {
            sieve: self,
            pos: lo,
            end: hi,
        }
    }

    fn support_ref<'a>(&'a self, p: P) -> Self::SupportRefIter<'a> {
        let i = *self.index_of.get(&p).expect("unknown point") as usize;
        let lo = self.in_offsets[i] as usize;
        let hi = self.in_offsets[i + 1] as usize;
        SupportRefIter {
            sieve: self,
            pos: lo,
            end: hi,
        }
    }
}

impl<P, T> SieveQueryExt for FrozenSieveCsr<P, T>
where
    P: PointLike,
    T: PayloadLike,
{
    #[inline]
    fn out_degree(&self, p: P) -> usize {
        let i = *self.index_of.get(&p).expect("unknown point") as usize;
        (self.out_offsets[i + 1] - self.out_offsets[i]) as usize
    }
    #[inline]
    fn in_degree(&self, p: P) -> usize {
        let i = *self.index_of.get(&p).expect("unknown point") as usize;
        (self.in_offsets[i + 1] - self.in_offsets[i]) as usize
    }
}
