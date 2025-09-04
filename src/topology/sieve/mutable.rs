use super::sieve_trait::Sieve;
use crate::topology::cache::InvalidateCache;

/// Trait for sieves that support full topology mutation.
///
/// [`Sieve`] provides traversal and arrow-level mutation. `MutableSieve`
/// extends it with point/role mutators and higher-level convenience routines
/// that maintain mirror invariants and invalidate caches.
pub trait MutableSieve: Sieve + InvalidateCache {
    /// Insert a brand-new point (both roles appear, initially empty).
    fn add_point(&mut self, p: Self::Point);

    /// Remove a point and **all** incident arrows (both roles).
    fn remove_point(&mut self, p: Self::Point);

    /// Ensure `p` appears in the base set (outgoing role).
    fn add_base_point(&mut self, p: Self::Point);

    /// Ensure `p` appears in the cap set (incoming role).
    fn add_cap_point(&mut self, p: Self::Point);

    /// Remove `p` from base role (drop all outgoing arrows of `p`).
    fn remove_base_point(&mut self, p: Self::Point);

    /// Remove `p` from cap role (drop all incoming arrows to `p`).
    fn remove_cap_point(&mut self, p: Self::Point);

    // ---------- Generic convenience mutators (correct by construction) ----------

    /// Replace the entire cone of `p` with `chain` ("last wins" per dst).
    fn set_cone(
        &mut self,
        p: Self::Point,
        chain: impl IntoIterator<Item = (Self::Point, Self::Payload)>,
    )
    where
        Self::Payload: Clone,
    {
        let old_dsts: Vec<_> = self.cone_points(p).collect();
        for dst in old_dsts {
            let _ = self.remove_arrow(p, dst);
        }
        for (dst, pay) in chain {
            self.add_arrow(p, dst, pay);
        }
        self.invalidate_cache();
    }

    /// Append to the cone of `p` (upsert per edge).
    fn add_cone(
        &mut self,
        p: Self::Point,
        chain: impl IntoIterator<Item = (Self::Point, Self::Payload)>,
    )
    where
        Self::Payload: Clone,
    {
        for (dst, pay) in chain {
            self.add_arrow(p, dst, pay);
        }
        self.invalidate_cache();
    }

    /// Replace the entire support of `q` with `chain` ("last wins" per src).
    fn set_support(
        &mut self,
        q: Self::Point,
        chain: impl IntoIterator<Item = (Self::Point, Self::Payload)>,
    )
    where
        Self::Payload: Clone,
    {
        let old_srcs: Vec<_> = self.support_points(q).collect();
        for src in old_srcs {
            let _ = self.remove_arrow(src, q);
        }
        for (src, pay) in chain {
            self.add_arrow(src, q, pay);
        }
        self.invalidate_cache();
    }

    /// Append to the support of `q` (upsert per edge).
    fn add_support(
        &mut self,
        q: Self::Point,
        chain: impl IntoIterator<Item = (Self::Point, Self::Payload)>,
    )
    where
        Self::Payload: Clone,
    {
        for (src, pay) in chain {
            self.add_arrow(src, q, pay);
        }
        self.invalidate_cache();
    }

    /// Optional preallocation hints; default no-ops are fine.
    fn reserve_cone(&mut self, _p: Self::Point, _additional: usize) {}
    fn reserve_support(&mut self, _q: Self::Point, _additional: usize) {}
}

#[cfg(test)]
mod docs {
    /// Calling mutators on a type that is only bound by [`Sieve`] fails to compile.
    ///
    /// ```compile_fail
    /// use mesh_sieve::prelude::Sieve;
    /// fn needs_mutation<S: Sieve<Point = u32>>(s: &mut S) {
    ///     s.add_point(1);
    /// }
    /// ```
    #[allow(dead_code)]
    fn bound_enforced() {}
}
