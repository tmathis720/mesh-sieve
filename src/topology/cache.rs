//! Cache invalidation utilities shared across topology structures.

/// Anything that caches derived topology (strata, overlaps, dual graphs, …)
/// should implement this.
pub trait InvalidateCache {
    /// Invalidate *all* internal caches so future queries recompute correctly.
    fn invalidate_cache(&mut self);
}

// Blanket impl for Box<T>
impl<T: InvalidateCache + ?Sized> InvalidateCache for Box<T> {
    #[inline]
    fn invalidate_cache(&mut self) {
        (**self).invalidate_cache();
    }
}

