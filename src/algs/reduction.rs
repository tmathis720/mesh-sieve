use crate::mesh_error::MeshSieveError;
use crate::topology::sieve::{Sieve, SieveRef};

/// Abstraction over reachability rows used in transitive algorithms.
///
/// Default implementation is dense and backed by `Vec<u64>`. Enable the
/// `sparse-bitset` feature for a chunked sparse representation suitable for
/// very large, sparse graphs.
trait ReachRow {
    /// Create a row able to track `n` bits.
    fn with_size(n: usize) -> Self;
    /// Set bit `i`.
    fn set(&mut self, i: usize);
    /// Read bit `i`.
    fn get(&self, i: usize) -> bool;
    /// Bitwise OR assignment with `other`.
    fn or_assign_from(&mut self, other: &Self);
}

/// Dense bitset implementation using `Vec<u64>` words.
#[derive(Clone)]
struct DenseRow {
    words: Vec<u64>,
}

impl ReachRow for DenseRow {
    #[inline]
    fn with_size(n: usize) -> Self {
        Self {
            words: vec![0; n.div_ceil(64)],
        }
    }
    #[inline]
    fn set(&mut self, i: usize) {
        self.words[i / 64] |= 1u64 << (i % 64);
    }
    #[inline]
    fn get(&self, i: usize) -> bool {
        (self.words[i / 64] >> (i % 64)) & 1 == 1
    }
    #[inline]
    fn or_assign_from(&mut self, other: &Self) {
        for (a, b) in self.words.iter_mut().zip(&other.words) {
            *a |= *b;
        }
    }
}

#[cfg(feature = "sparse-bitset")]
mod sparse {
    use super::ReachRow;
    use std::collections::BTreeMap;

    /// Sparse bitset chunked by 64-bit words.
    pub struct SparseRow {
        chunks: BTreeMap<usize, u64>,
    }

    impl ReachRow for SparseRow {
        fn with_size(_n: usize) -> Self {
            Self {
                chunks: BTreeMap::new(),
            }
        }
        #[inline]
        fn set(&mut self, i: usize) {
            let w = i / 64;
            let b = 1u64 << (i % 64);
            *self.chunks.entry(w).or_insert(0) |= b;
        }
        #[inline]
        fn get(&self, i: usize) -> bool {
            let w = i / 64;
            let b = 1u64 << (i % 64);
            self.chunks.get(&w).map_or(0, |&x| x) & b != 0
        }
        fn or_assign_from(&mut self, other: &Self) {
            for (&w, &bits) in &other.chunks {
                *self.chunks.entry(w).or_insert(0) |= bits;
            }
        }
    }
}

#[cfg(feature = "sparse-bitset")]
type Row = sparse::SparseRow;
#[cfg(not(feature = "sparse-bitset"))]
type Row = DenseRow;

#[inline]
#[cfg(any(
    debug_assertions,
    feature = "strict-invariants",
    feature = "check-invariants"
))]
fn is_acyclic_by_chart<S>(s: &S, chart: &[S::Point]) -> bool
where
    S: Sieve + SieveRef,
{
    use std::collections::HashMap;
    let mut idx = HashMap::with_capacity(chart.len());
    for (i, &p) in chart.iter().enumerate() {
        idx.insert(p, i);
    }
    for &u in chart {
        let ui = idx[&u];
        for (v, _) in s.cone_ref(u) {
            if idx[&v] <= ui {
                return false;
            }
        }
    }
    true
}

/// Remove all transitive edges in a **DAG**. Returns number of removed edges.
///
/// # Preconditions
/// - `s` must be acyclic (DAG). We rely on [`chart_points`](Sieve::chart_points)
///   for a topological order and return `Err(MeshSieveError::CycleDetected)` on
///   cycles.
///
/// # Complexity (dense bitset)
/// - Time: ~`O(E + V * (V/64) + Σ_u deg(u)^2 / W)` where `W = 64`.
/// - Memory: `O(V * ⌈V/64⌉)` words. Enable the `sparse-bitset` feature for a
///   memory-saving sparse representation.
pub fn transitive_reduction_dag<S>(s: &mut S) -> Result<usize, MeshSieveError>
where
    S: Sieve + SieveRef,
    S::Point: Copy + Eq + std::hash::Hash + Ord + std::fmt::Debug,
{
    use std::collections::HashMap;
    let chart = s.chart_points()?;
    #[cfg(any(
        debug_assertions,
        feature = "strict-invariants",
        feature = "check-invariants"
    ))]
    debug_assert!(is_acyclic_by_chart(s, &chart), "chart must be acyclic");
    let n = chart.len();
    let mut idx = HashMap::with_capacity(n);
    for (i, &p) in chart.iter().enumerate() {
        idx.insert(p, i);
    }
    let mut reach: Vec<Row> = (0..n).map(|_| Row::with_size(n)).collect();
    for &u in chart.iter().rev() {
        let ui = idx[&u];
        for (v, _) in s.cone_ref(u) {
            let vi = idx[&v];
            // vi > ui in a topological order
            let (row_u, row_v) = {
                let (pre, suf) = reach.split_at_mut(vi);
                (&mut pre[ui], &suf[0])
            };
            row_u.or_assign_from(row_v);
            row_u.set(vi);
        }
    }
    let mut to_remove = Vec::new();
    for &u in &chart {
        let mut neigh: Vec<_> = SieveRef::cone_points(s, u).collect();
        neigh.sort_unstable();
        neigh.dedup();
        for &v in &neigh {
            let vi = idx[&v];
            let implied = neigh
                .iter()
                .copied()
                .any(|w| w != v && reach[idx[&w]].get(vi));
            if implied {
                to_remove.push((u, v));
            }
        }
    }
    to_remove.sort_unstable_by_key(|&(u, v)| (u, v));
    for (u, v) in &to_remove {
        let _ = s.remove_arrow(*u, *v);
    }
    Ok(to_remove.len())
}

/// Compute missing transitive-closure edges of a DAG (`u ⇒ v` without a direct edge`).
/// Does not modify the sieve.
///
/// # Preconditions
/// - `s` must be acyclic (DAG); cycles yield `Err(MeshSieveError::CycleDetected)`.
///
/// # Complexity (dense bitset)
/// - Time: ~`O(E + V * (V/64))` to build reachability plus membership checks.
/// - Memory: `O(V * ⌈V/64⌉)` words.
///
/// Returned edge order is deterministic, following the chart order.
pub fn transitive_closure_edges<S>(s: &mut S) -> Result<Vec<(S::Point, S::Point)>, MeshSieveError>
where
    S: Sieve + SieveRef,
    S::Point: Copy + Eq + std::hash::Hash + Ord + std::fmt::Debug,
{
    use std::collections::{HashMap, HashSet};
    let chart = s.chart_points()?;
    #[cfg(any(
        debug_assertions,
        feature = "strict-invariants",
        feature = "check-invariants"
    ))]
    debug_assert!(is_acyclic_by_chart(s, &chart), "chart must be acyclic");
    let n = chart.len();
    let mut idx = HashMap::with_capacity(n);
    for (i, &p) in chart.iter().enumerate() {
        idx.insert(p, i);
    }
    let mut reach: Vec<Row> = (0..n).map(|_| Row::with_size(n)).collect();
    let mut direct = HashSet::new();
    for &u in chart.iter().rev() {
        let ui = idx[&u];
        let mut neigh: Vec<_> = s.cone_ref(u).map(|(v, _)| idx[&v]).collect();
        neigh.sort_unstable();
        neigh.dedup();
        for &vi in &neigh {
            direct.insert((ui, vi));
            let (row_u, row_v) = {
                let (pre, suf) = reach.split_at_mut(vi);
                (&mut pre[ui], &suf[0])
            };
            row_u.or_assign_from(row_v);
            row_u.set(vi);
        }
    }
    let mut out = Vec::new();
    for (ui, &u) in chart.iter().enumerate() {
        for vi in 0..n {
            if ui == vi {
                continue;
            }
            if reach[ui].get(vi) && !direct.contains(&(ui, vi)) {
                out.push((u, chart[vi]));
            }
        }
    }
    Ok(out)
}

/// Summary statistics from [`transitive_reduction_dag_stats`].
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct ReductionStats {
    pub removed: usize,
    pub remaining: usize,
}

/// Perform [`transitive_reduction_dag`] and report removed/remaining edge counts.
pub fn transitive_reduction_dag_stats<S>(s: &mut S) -> Result<ReductionStats, MeshSieveError>
where
    S: Sieve + SieveRef,
    S::Point: Copy + Eq + std::hash::Hash + Ord + std::fmt::Debug,
{
    fn arrow_count<S2>(s: &S2) -> usize
    where
        S2: Sieve + SieveRef,
        S2::Point: Copy + Eq + std::hash::Hash + Ord + std::fmt::Debug,
    {
        s.base_points()
            .map(|u| SieveRef::cone_points(s, u).count())
            .sum()
    }

    let before = arrow_count(s);
    let removed = transitive_reduction_dag(s)?;
    let after = arrow_count(s);
    debug_assert_eq!(before - removed, after);
    Ok(ReductionStats {
        removed,
        remaining: after,
    })
}
