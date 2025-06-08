//! Utility helpers for topology, including DAG assertion.
use crate::topology::sieve::InMemorySieve;
use std::collections::{HashMap, VecDeque, HashSet};

/// Panics if the sieve contains a cycle (not a DAG).
pub fn assert_dag<P: Copy + Eq + std::hash::Hash, T>(s: &InMemorySieve<P, T>) {
    // Kahn's algorithm: count in-degrees
    let mut in_deg = HashMap::new();
    for (&src, outs) in &s.adjacency_out {
        in_deg.entry(src).or_insert(0);
        for (dst, _) in outs {
            *in_deg.entry(*dst).or_insert(0) += 1;
        }
    }
    let mut queue: VecDeque<_> = in_deg.iter().filter(|&(_, &d)| d == 0).map(|(&p, _)| p).collect();
    let mut seen = HashSet::new();
    while let Some(p) = queue.pop_front() {
        seen.insert(p);
        if let Some(outs) = s.adjacency_out.get(&p) {
            for (dst, _) in outs {
                if let Some(d) = in_deg.get_mut(dst) {
                    *d -= 1;
                    if *d == 0 {
                        queue.push_back(*dst);
                    }
                }
            }
        }
    }
    if seen.len() != in_deg.len() {
        panic!("Sieve contains a cycle: not a DAG");
    }
}
