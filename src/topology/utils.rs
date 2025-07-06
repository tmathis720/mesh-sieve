//! Utility helpers for topology, including DAG assertion.
use crate::topology::sieve::InMemorySieve;
use crate::mesh_error::MeshSieveError;
use std::collections::{HashMap, HashSet, VecDeque};

/// Checks if the sieve is a DAG. Returns Err if a cycle is found.
pub fn check_dag<P: Copy + Eq + std::hash::Hash + Ord + std::fmt::Debug, T>(s: &InMemorySieve<P, T>) -> Result<(), MeshSieveError> {
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
        return Err(MeshSieveError::CycleDetected);
    }
    Ok(())
}

#[cfg(test)]
mod assert_dag_tests {
    use super::check_dag;
    use crate::topology::sieve::InMemorySieve;
    use crate::topology::point::PointId;
    use crate::topology::sieve::sieve_trait::Sieve;
    use crate::mesh_error::MeshSieveError;

    fn v(x: u64) -> PointId { PointId::new(x).unwrap() }

    #[test]
    fn empty_sieve_is_dag() {
        let s = InMemorySieve::<PointId, ()>::default();
        check_dag(&s).unwrap();
    }

    #[test]
    fn singleton_node_is_dag() {
        let mut s = InMemorySieve::<PointId, ()>::default();
        s.adjacency_out.insert(v(1), Vec::new());
        check_dag(&s).unwrap();
    }

    #[test]
    fn simple_chain_is_dag() {
        let mut s = InMemorySieve::<u32, ()>::new();
        s.add_arrow(1,2,());
        s.add_arrow(2,3,());
        check_dag(&s).unwrap();
    }

    #[test]
    fn two_node_cycle_errors() {
        let mut s = InMemorySieve::<u32, ()>::new();
        s.add_arrow(1,2,());
        s.add_arrow(2,1,());
        let err = check_dag(&s).unwrap_err();
        assert!(matches!(err, MeshSieveError::CycleDetected));
    }

    #[test]
    fn self_loop_errors() {
        let mut s = InMemorySieve::<u32, ()>::new();
        s.add_arrow(5,5,());
        let err = check_dag(&s).unwrap_err();
        assert!(matches!(err, MeshSieveError::CycleDetected));
    }

    #[test]
    fn disconnected_dag_is_ok() {
        let mut s = InMemorySieve::<u32, ()>::new();
        s.add_arrow(1,2,());
        s.add_arrow(3,4,());
        s.add_arrow(6,5,());
        check_dag(&s).unwrap();
    }

    #[test]
    fn embedded_cycle_errors() {
        let mut s = InMemorySieve::<u32, ()>::new();
        s.add_arrow(1,2,());
        s.add_arrow(2,3,());
        s.add_arrow(4,5,());
        s.add_arrow(5,6,());
        s.add_arrow(6,4,());
        let err = check_dag(&s).unwrap_err();
        assert!(matches!(err, MeshSieveError::CycleDetected));
    }

    #[test]
    fn repeated_check_dag_no_error() {
        let mut s = InMemorySieve::<u32, ()>::new();
        s.add_arrow(1,2,());
        check_dag(&s).unwrap();
        check_dag(&s).unwrap();
    }
}
