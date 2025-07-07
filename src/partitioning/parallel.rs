//! Parallel utilities for partitioning algorithms.
//!
//! This module provides helpers for thread-local random number generation and
//! parallel execution patterns, supporting deterministic and efficient parallel partitioning.

use ahash::AHasher;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rayon::prelude::*;
use std::cell::RefCell;
use std::hash::Hasher;

thread_local! {
    /// Each thread gets its own SmallRng seeded from global seed.
    static THREAD_RNG: RefCell<Option<SmallRng>> = RefCell::new(None);
}

/// Initializes the thread’s RNG from a global seed.
///
/// Should be called once per thread before using [`with_thread_rng`].
pub fn init_thread_rng(global_seed: u64) {
    let mut hasher = AHasher::default();
    let thread_idx = rayon::current_thread_index().unwrap_or(0) as u64;
    hasher.write_u64(global_seed ^ thread_idx);
    let seed = hasher.finish();
    THREAD_RNG.with(|cell| {
        *cell.borrow_mut() = Some(SmallRng::seed_from_u64(seed));
    });
}

/// Returns a mutable reference to the SmallRng for this thread by running a closure.
///
/// # Panics
/// Panics if the thread-local RNG has not been initialized.
pub fn with_thread_rng<F, R>(f: F) -> R
where
    F: FnOnce(&mut SmallRng) -> R,
{
    THREAD_RNG.with(|cell| {
        let mut opt = cell.borrow_mut();
        let rng = opt
            .as_mut()
            .expect("Thread‐local RNG not initialized: call init_thread_rng() first");
        f(rng)
    })
}

/// A helper to spawn a Rayon parallel scope with each worker’s RNG initialized.
///
/// The closure `f` is executed within the parallel scope.
pub fn parallel_scope_with_rng<F: FnOnce() + Send + Sync>(global_seed: u64, f: F) {
    rayon::scope(|s| {
        s.spawn(|_| init_thread_rng(global_seed));
        f();
    });
}

/// Executes `func(i, &item)` in parallel over `0..n`.
///
/// Calls the provided function for each item in the slice, passing the index and a reference.
pub fn par_for_each_mutex<T, F>(data: &[T], func: F)
where
    T: Sync,
    F: Fn(usize, &T) + Send + Sync,
{
    data.par_iter().enumerate().for_each(|(i, item)| {
        func(i, item);
    });
}

#[cfg(feature = "mpi-support")]
pub mod onizuka_partitioning {
    //! Parallel partitioning routines for Onizuka algorithms.
    use super::*;
    use crate::partitioning::binpack::Item;
    use crate::partitioning::{PartitionerConfig, PartitionMap};
    use crate::partitioning::graph_traits::PartitionableGraph;
    use crate::partitioning::PartitionerError;

    /// Phase 2 parallel merge that seeds each thread’s RNG and then
    /// does the adjacency-aware cluster merge.
    pub fn parallel_merge_clusters_into_parts(
        items: &[Item],
        cfg: &PartitionerConfig,
    ) -> Result<Vec<usize>, PartitionerError> {
        let mut part_assignment = None;
        let mut error = None;
        parallel_scope_with_rng(cfg.rng_seed, || {
            match crate::partitioning::binpack::merge_clusters_into_parts(
                items,
                cfg.n_parts,
                cfg.epsilon,
            ) {
                Ok(assign) => part_assignment = Some(assign),
                Err(e) => error = Some(e),
            }
        });
        if let Some(e) = error {
            Err(e)
        } else {
            Ok(part_assignment.expect("Parallel merge did not produce a result"))
        }
    }

    /// Phase 3 parallel vertex-cut: seeds each worker’s RNG with salt,
    /// then performs load-aware owner selection in parallel.
    pub fn parallel_build_vertex_cuts<G>(
        graph: &G,
        pm: &PartitionMap<G::VertexId>,
        salt: u64,
    ) -> (Vec<usize>, Vec<Vec<(G::VertexId, usize)>>)
    where
        G: PartitionableGraph<VertexId = usize> + Sync,
    {
        let mut result = None;
        parallel_scope_with_rng(salt, || {
            result = Some(crate::partitioning::vertex_cut::build_vertex_cuts(graph, pm, salt));
        });
        result.expect("Parallel vertex cut did not produce a result").expect("vertex cut failed")
    }
}
