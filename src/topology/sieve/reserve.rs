//! Extension trait providing bulk preallocation helpers for sieves.

use super::{InMemoryOrientedSieve, InMemorySieve};

/// Helpers for bulk preallocation based on edge lists or counts.
pub trait SieveReserveExt<P> {
    /// Pre-size cone and support capacities given `(src, dst, count)` tuples.
    fn reserve_from_edge_counts(&mut self, counts: impl IntoIterator<Item = (P, P, usize)>);

    /// Convenience: preallocate from raw `(src, dst)` edge lists.
    fn reserve_from_edges(&mut self, edges: impl IntoIterator<Item = (P, P)>);
}

impl<P, T> SieveReserveExt<P> for InMemorySieve<P, T>
where
    P: Copy + Eq + std::hash::Hash + Ord + std::fmt::Debug,
    T: Clone,
{
    fn reserve_from_edge_counts(&mut self, counts: impl IntoIterator<Item = (P, P, usize)>) {
        InMemorySieve::reserve_from_edge_counts(self, counts);
    }

    fn reserve_from_edges(&mut self, edges: impl IntoIterator<Item = (P, P)>) {
        InMemorySieve::reserve_from_edges(self, edges);
    }
}

impl<P, T, O> SieveReserveExt<P> for InMemoryOrientedSieve<P, T, O>
where
    P: Copy + Eq + std::hash::Hash + Ord + std::fmt::Debug,
    T: Clone,
    O: super::oriented::Orientation,
{
    fn reserve_from_edge_counts(&mut self, counts: impl IntoIterator<Item = (P, P, usize)>) {
        InMemoryOrientedSieve::reserve_from_edge_counts(self, counts);
    }

    fn reserve_from_edges(&mut self, edges: impl IntoIterator<Item = (P, P)>) {
        InMemoryOrientedSieve::reserve_from_edges(self, edges);
    }
}

