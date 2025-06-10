//! Utility helpers for topology, including DAG assertion.
use crate::topology::sieve::InMemorySieve;
use std::collections::{HashMap, HashSet, VecDeque};

/// Panics if the sieve contains a cycle (not a DAG).
pub fn assert_dag<P: Copy + Eq + std::hash::Hash + Ord, T>(s: &InMemorySieve<P, T>) {
    // Kahn's algorithm: count in-degrees
    let mut in_deg = HashMap::new();
    // Initialize in-degrees to 0 for all vertices
    for (&src, outs) in &s.adjacency_out {
        // Ensure src is in in_deg with 0 in-degree
        in_deg.entry(src).or_insert(0);
        // Count in-degrees for each destination vertex
        for (dst, _) in outs {
            *in_deg.entry(*dst).or_insert(0) += 1;
        }
    }
    // Add any vertices that have no outgoing edges
    let mut queue: VecDeque<_> = in_deg
        .iter()
        .filter(|&(_, &d)| d == 0)
        .map(|(&p, _)| p)
        .collect();
    // If no vertices have 0 in-degree, the sieve is not a DAG
    let mut seen = HashSet::new();
    // Process vertices with 0 in-degree
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
    // If we have seen all vertices, then the sieve is a DAG
    // If not, it contains a cycle
    if seen.len() != in_deg.len() {
        panic!("Sieve contains a cycle: not a DAG");
    }
}
