use crate::mesh_error::MeshSieveError;
use crate::topology::sieve::{Sieve, SieveRef};

/// Bitset helper used for transitive algorithms.
#[derive(Clone)]
struct BitSet { words: Vec<u64> }
impl BitSet {
    fn new(n: usize) -> Self { Self { words: vec![0; (n + 63) / 64] } }
    #[inline] fn set(&mut self, i: usize) { self.words[i / 64] |= 1u64 << (i % 64); }
    #[inline] fn get(&self, i: usize) -> bool { (self.words[i / 64] >> (i % 64)) & 1 == 1 }
    #[inline] fn or_assign(&mut self, other: &BitSet) {
        for (a, b) in self.words.iter_mut().zip(&other.words) { *a |= *b; }
    }
}

/// Remove all transitive edges in a DAG. Returns number of removed edges.
/// Returns an error if a cycle is detected.
pub fn transitive_reduction_dag<S>(s: &mut S) -> Result<usize, MeshSieveError>
where
    S: Sieve + SieveRef,
    S::Point: Copy + Eq + std::hash::Hash + Ord + std::fmt::Debug,
{
    use std::collections::HashMap;
    let chart = s.chart_points()?;
    let n = chart.len();
    let mut idx = HashMap::with_capacity(n);
    for (i, &p) in chart.iter().enumerate() { idx.insert(p, i); }
    let mut reach: Vec<BitSet> = (0..n).map(|_| BitSet::new(n)).collect();
    for &u in chart.iter().rev() {
        let ui = idx[&u];
        for (v, _) in s.cone_ref(u) {
            let vi = idx[&v];
            let bs = reach[vi].clone();
            reach[ui].or_assign(&bs);
            reach[ui].set(vi);
        }
    }
    let mut to_remove = Vec::new();
    for &u in &chart {
        let neigh: Vec<_> = SieveRef::cone_points(s, u).collect();
        for &v in &neigh {
            let vi = idx[&v];
            let implied = neigh.iter().copied().any(|w| w != v && reach[idx[&w]].get(vi));
            if implied { to_remove.push((u, v)); }
        }
    }
    for (u, v) in &to_remove { let _ = s.remove_arrow(*u, *v); }
    Ok(to_remove.len())
}

/// Compute missing transitive-closure edges of a DAG (u⇒v without direct edge).
/// Does not modify the sieve; returns edges in deterministic chart order.
pub fn transitive_closure_edges<S>(s: &mut S) -> Result<Vec<(S::Point, S::Point)>, MeshSieveError>
where
    S: Sieve + SieveRef,
    S::Point: Copy + Eq + std::hash::Hash + Ord + std::fmt::Debug,
{
    use std::collections::{HashMap, HashSet};
    let chart = s.chart_points()?;
    let n = chart.len();
    let mut idx = HashMap::with_capacity(n);
    for (i, &p) in chart.iter().enumerate() { idx.insert(p, i); }
    let mut reach: Vec<BitSet> = (0..n).map(|_| BitSet::new(n)).collect();
    let mut direct = HashSet::new();
    for &u in chart.iter().rev() {
        let ui = idx[&u];
        let mut neigh: Vec<_> = s.cone_ref(u).map(|(v, _)| idx[&v]).collect();
        neigh.sort_unstable();
        neigh.dedup();
        for &vi in &neigh {
            direct.insert((ui, vi));
            let bs = reach[vi].clone();
            reach[ui].or_assign(&bs);
            reach[ui].set(vi);
        }
    }
    let mut out = Vec::new();
    for (ui, &u) in chart.iter().enumerate() {
        for vi in 0..n {
            if ui == vi { continue; }
            if reach[ui].get(vi) && !direct.contains(&(ui, vi)) {
                out.push((u, chart[vi]));
            }
        }
    }
    Ok(out)
}
