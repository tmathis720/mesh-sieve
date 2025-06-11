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

#[cfg(test)]
mod assert_dag_tests {
    use super::assert_dag;
    use crate::topology::sieve::InMemorySieve;
    use crate::topology::point::PointId;
    use crate::topology::sieve::sieve_trait::Sieve;

    fn v(x: u64) -> PointId { PointId::new(x) }

    #[test]
    fn empty_sieve_is_dag() {
        let s = InMemorySieve::<PointId, ()>::default();
        assert_dag(&s);
    }

    #[test]
    fn singleton_node_is_dag() {
        let mut s = InMemorySieve::<PointId, ()>::default();
        s.adjacency_out.insert(v(1), Vec::new());
        assert_dag(&s);
    }

    #[test]
    fn simple_chain_is_dag() {
        let mut s = InMemorySieve::<u32, ()>::new();
        s.add_arrow(1,2,());
        s.add_arrow(2,3,());
        assert_dag(&s);
    }

    #[test]
    #[should_panic(expected = "Sieve contains a cycle")]
    fn two_node_cycle_panics() {
        let mut s = InMemorySieve::<u32, ()>::new();
        s.add_arrow(1,2,());
        s.add_arrow(2,1,());
        assert_dag(&s);
    }

    #[test]
    #[should_panic(expected = "Sieve contains a cycle")]
    fn self_loop_panics() {
        let mut s = InMemorySieve::<u32, ()>::new();
        s.add_arrow(5,5,());
        assert_dag(&s);
    }

    #[test]
    fn disconnected_dag_is_ok() {
        let mut s = InMemorySieve::<u32, ()>::new();
        s.add_arrow(1,2,());
        s.add_arrow(3,4,());
        s.add_arrow(6,5,());
        assert_dag(&s);
    }

    #[test]
    #[should_panic(expected = "Sieve contains a cycle")]
    fn embedded_cycle_panics() {
        let mut s = InMemorySieve::<u32, ()>::new();
        s.add_arrow(1,2,());
        s.add_arrow(2,3,());
        s.add_arrow(4,5,());
        s.add_arrow(5,6,());
        s.add_arrow(6,4,());
        assert_dag(&s);
    }

    #[test]
    fn repeated_assert_dag_no_panic() {
        let mut s = InMemorySieve::<u32, ()>::new();
        s.add_arrow(1,2,());
        assert_dag(&s);
        assert_dag(&s);
    }
}
