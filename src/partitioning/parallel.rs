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
pub fn parallel_scope_with_rng<F: FnOnce() + Send + Sync>(global_seed: u64, f: F) {
    rayon::scope(|s| {
        s.spawn(|_| init_thread_rng(global_seed));
        f();
    });
}

/// Executes `func(i, &item)` in parallel over `0..n`.
pub fn par_for_each_mutex<T, F>(data: &[T], func: F)
where
    T: Sync,
    F: Fn(usize, &T) + Send + Sync,
{
    data.par_iter().enumerate().for_each(|(i, item)| {
        func(i, item);
    });
}
