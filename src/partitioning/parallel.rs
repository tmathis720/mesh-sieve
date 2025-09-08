//! Parallel utilities for partitioning algorithms.
//!
//! This module provides helpers for deterministic, thread‑local random number
//! generation and ergonomic wrappers around Rayon’s thread pools. Each worker
//! in a pool built via [`build_rng_thread_pool`] is seeded deterministically
//! from a `global_seed`, enabling reproducible parallel algorithms.

use rand::rngs::SmallRng;
use rand::{RngCore, SeedableRng};
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::cell::{Cell, RefCell};

thread_local! {
    /// Thread‑local storage for each worker’s RNG.
    static TLS_RNG: RefCell<Option<SmallRng>> = RefCell::new(None);
    /// Thread‑local storage for the worker index.
    static TLS_WORKER_IDX: Cell<Option<usize>> = Cell::new(None);
}

// ---------------------------------------------------------------------------
// Stable seed mixers
// ---------------------------------------------------------------------------

/// Steele/Vigna SplitMix64 mixer used for deterministic seeding.
#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline]
fn mix_seed(global_seed: u64, worker_index: usize) -> u64 {
    let x = global_seed
        ^ 0xD6E8_FEB8_6659_FD93u64
        ^ (worker_index as u64).wrapping_mul(0x9E37_79B1_85EB_CA87);
    splitmix64(x)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Build a Rayon thread pool whose workers are deterministically seeded from
/// `global_seed`. The returned pool must be used with [`ThreadPool::install`].
///
/// If `num_threads` is `None`, Rayon selects its default thread count.
pub fn build_rng_thread_pool(global_seed: u64, num_threads: Option<usize>) -> ThreadPool {
    let mut builder = ThreadPoolBuilder::new();
    if let Some(n) = num_threads {
        builder = builder.num_threads(n);
    }

    builder
        .start_handler(move |idx| {
            let seed = mix_seed(global_seed, idx);
            TLS_RNG.with(|cell| {
                *cell.borrow_mut() = Some(SmallRng::seed_from_u64(seed));
            });
            TLS_WORKER_IDX.with(|c| c.set(Some(idx)));
        })
        .exit_handler(|_| {
            TLS_RNG.with(|cell| *cell.borrow_mut() = None);
            TLS_WORKER_IDX.with(|c| c.set(None));
        })
        .build()
        .expect("Failed to build deterministic RNG thread pool")
}

/// Install a deterministic RNG pool and execute `f` within it. All Rayon
/// parallelism inside `f` uses this pool.
pub fn install_with_rng_pool<F, R>(global_seed: u64, num_threads: Option<usize>, f: F) -> R
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    let pool = build_rng_thread_pool(global_seed, num_threads);
    pool.install(f)
}

/// Borrow the worker’s thread‑local RNG, panicking if called outside a seeded
/// pool (i.e. not inside [`install_with_rng_pool`] or
/// [`ThreadPool::install`] on a pool built via [`build_rng_thread_pool`]).
pub fn with_thread_rng<T>(f: impl FnOnce(&mut SmallRng) -> T) -> T {
    TLS_RNG.with(|cell| {
        let mut opt = cell.borrow_mut();
        let rng = opt
            .as_mut()
            .expect("with_thread_rng() called outside a seeded pool. Use install_with_rng_pool().");
        f(rng)
    })
}

/// Returns the current Rayon worker index, if inside a seeded pool.
pub fn worker_index() -> Option<usize> {
    TLS_WORKER_IDX.with(|c| c.get())
}

/// Deterministically derive a new 64‑bit salt from `(label, extra)` using a
/// stable mixer.
pub fn derive_salt(global_seed: u64, label: &str, extra: u64) -> u64 {
    let mut h = global_seed ^ 0xA076_1D64_78BD_642F;
    h = splitmix64(h ^ extra.rotate_left(17));
    for &b in label.as_bytes() {
        h = splitmix64(h ^ b as u64);
    }
    h
}

/// Parallel for‑each over a slice with stable indexing. Should be invoked
/// from within a seeded pool.
pub fn par_for_each_indexed<T, F>(data: &[T], func: F)
where
    T: Sync,
    F: Fn(usize, &T) + Send + Sync,
{
    data.par_iter()
        .enumerate()
        .for_each(|(i, item)| func(i, item));
}

/// Produce a deterministic sub‑seed from the worker’s RNG.
pub fn next_subseed() -> u64 {
    with_thread_rng(|rng| rng.next_u64())
}

// ---------------------------------------------------------------------------
// Deprecated API
// ---------------------------------------------------------------------------

#[deprecated(
    note = "Use build_rng_thread_pool(...).install(|| ...) or install_with_rng_pool(...). \n            init_thread_rng seeded only one worker and was racy across pools."
)]
pub fn init_thread_rng(_global_seed: u64) {
    // intentionally a no‑op
}

#[deprecated(
    note = "Use install_with_rng_pool(global_seed, num_threads, f). \n            The old function seeded only one worker in the scope."
)]
pub fn parallel_scope_with_rng<F: FnOnce() + Send + Sync>(_global_seed: u64, f: F) {
    // Backward‑compat shim: run without deterministic seeding.
    rayon::scope(|_| f());
}

// ---------------------------------------------------------------------------
// Convenience wrappers for existing partitioning routines
// ---------------------------------------------------------------------------

#[cfg(feature = "mpi-support")]
pub mod onizuka_partitioning {
    //! Parallel partitioning routines for Onizuka algorithms.
    use super::*;
    use crate::partitioning::binpack::Item;
    use crate::partitioning::graph_traits::PartitionableGraph;
    use crate::partitioning::{PartitionMap, PartitionerConfig, PartitionerError};

    /// Phase 2 parallel merge that seeds each thread’s RNG and then performs the
    /// adjacency‑aware cluster merge.
    pub fn parallel_merge_clusters_into_parts(
        items: &[Item],
        cfg: &PartitionerConfig,
    ) -> Result<Vec<usize>, PartitionerError> {
        let mut part_assignment = None;
        let mut error = None;
        install_with_rng_pool(cfg.rng_seed, None, || {
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

    /// Phase 3 parallel vertex‑cut: seeds each worker’s RNG with `salt`, then
    /// performs load‑aware owner selection in parallel.
    pub fn parallel_build_vertex_cuts<G>(
        graph: &G,
        pm: &PartitionMap<G::VertexId>,
        salt: u64,
    ) -> (Vec<usize>, Vec<Vec<(G::VertexId, usize)>>)
    where
        G: PartitionableGraph<VertexId = usize> + Sync,
    {
        let mut result = None;
        install_with_rng_pool(salt, None, || {
            result = Some(crate::partitioning::vertex_cut::build_vertex_cuts(
                graph, pm, salt,
            ));
        });
        result
            .expect("Parallel vertex cut did not produce a result")
            .expect("vertex cut failed")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tls_rng_initialized_on_all_workers() {
        install_with_rng_pool(12345, Some(4), || {
            (0..10_000).into_par_iter().for_each(|_| {
                let _ = with_thread_rng(|rng| rng.next_u64());
            });
        });
    }

    #[test]
    fn deterministic_across_runs_same_threads() {
        let run = |seed| -> Vec<u64> {
            install_with_rng_pool(seed, Some(1), || {
                (0..1000)
                    .map(|_| with_thread_rng(|rng| rng.next_u64()))
                    .collect::<Vec<_>>()
            })
        };
        let a = run(777);
        let b = run(777);
        assert_eq!(a, b);
    }

    #[test]
    fn worker_index_exposed() {
        install_with_rng_pool(1, Some(2), || {
            (0..1000).into_par_iter().for_each(|_| {
                assert!(worker_index().is_some());
            });
        });
        assert!(worker_index().is_none());
    }
}
