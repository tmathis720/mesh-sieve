//! Utility helpers for topology, including DAG assertion.
use crate::mesh_error::MeshSieveError;
use crate::topology::sieve::Sieve;
use std::collections::{HashMap, VecDeque};

/// Generic DAG check for any `S: Sieve`.
///
/// Uses Kahn's algorithm. Returns:
/// - `Ok(())` if the topology is acyclic,
/// - `Err(MeshSieveError::CycleDetected)` if a cycle is detected.
///
/// Robust across backends: operates via `Sieve::points()` and `Sieve::cone(..)`.
pub fn check_dag<S>(s: &S) -> Result<(), MeshSieveError>
where
    S: Sieve,
    S::Point: Copy + Eq + std::hash::Hash + Ord + std::fmt::Debug,
{
    // 1) Build in-degree map over all points we know about,
    //    and insert any cone destinations that were not listed by `points()`.
    let mut in_deg: HashMap<S::Point, usize> = HashMap::new();

    for p in s.points() {
        in_deg.entry(p).or_insert(0);
        // Count in-degrees for each destination
        for (dst, _) in s.cone(p) {
            *in_deg.entry(dst).or_insert(0) += 1;
        }
    }

    // 2) Seed queue with all zero in-degree vertices.
    let mut q: VecDeque<S::Point> = in_deg
        .iter()
        .filter_map(|(&p, &d)| (d == 0).then_some(p))
        .collect();

    // 3) Kahn’s algorithm
    let mut visited = 0usize;
    while let Some(p) = q.pop_front() {
        visited += 1;
        for (dst, _) in s.cone(p) {
            if let Some(d) = in_deg.get_mut(&dst) {
                *d -= 1;
                if *d == 0 {
                    q.push_back(dst);
                }
            }
        }
    }

    // 4) If not all nodes were visited, we have a cycle.
    if visited != in_deg.len() {
        return Err(MeshSieveError::CycleDetected);
    }
    Ok(())
}

/// Optional zero-clone variant for backends that implement `SieveRef`.
#[cfg(feature = "sieve_ref_fast_dag")]
pub fn check_dag_ref<S>(s: &S) -> Result<(), MeshSieveError>
where
    S: Sieve + crate::topology::sieve::SieveRef,
    S::Point: Copy + Eq + std::hash::Hash + Ord + std::fmt::Debug,
{
    use std::collections::{HashMap, VecDeque};
    let mut in_deg: HashMap<S::Point, usize> = HashMap::new();

    for p in s.points() {
        in_deg.entry(p).or_insert(0);
        for (dst, _) in s.cone_ref(p) {
            *in_deg.entry(dst).or_insert(0) += 1;
        }
    }

    let mut q: VecDeque<S::Point> = in_deg
        .iter()
        .filter_map(|(&p, &d)| (d == 0).then_some(p))
        .collect();

    let mut visited = 0usize;
    while let Some(p) = q.pop_front() {
        visited += 1;
        for (dst, _) in s.cone_ref(p) {
            if let Some(d) = in_deg.get_mut(&dst) {
                *d -= 1;
                if *d == 0 {
                    q.push_back(dst);
                }
            }
        }
    }

    if visited != in_deg.len() {
        return Err(MeshSieveError::CycleDetected);
    }
    Ok(())
}
