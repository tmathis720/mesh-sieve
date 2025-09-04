//! Sieve module: provides the core [`Sieve`] trait and in-memory implementations.
//!
//! This module defines the main interface for sieve data structures, which are used for efficient set membership queries.
//! It also provides several implementations, including in-memory variants and helpers for orientation-aware storage.
//!
//! # Shared payloads
//! When payloads are large or reused across many arrows, instantiate a sieve with `Payload = Arc<T>`.
//! Traversals clone the `Arc` handle without cloning `T`. Use the [`InMemorySieveArc`],
//! [`InMemoryOrientedSieveArc`], and [`InMemoryStackArc`] type aliases for convenience.
//! Avoid wrappers that convert between `T` and `Arc<T>` on the fly; they add allocations and defeat sharing.

/// In-memory implementation of the [`Sieve`] trait.
pub mod in_memory;
/// In-memory implementation storing per-arrow orientations.
pub mod in_memory_oriented;
/// Trait for sieves that support full topology mutation.
pub mod mutable;
/// Orientation-aware extensions to [`Sieve`].
pub mod oriented;
/// Reference-returning extensions to [`Sieve`].
pub mod sieve_ref;
/// Core trait for sieve data structures.
pub mod sieve_trait;
/// Strata sieve implementation.
pub mod strata;
/// Bulk preallocation helpers.
pub mod reserve;
/// Concrete traversal iterators without dynamic dispatch.
pub mod traversal_iter;
/// Frozen CSR representation for deterministic, cache-friendly traversal.
pub mod frozen_csr;

// Re-export the core trait and in‐memory impl at top level
pub use in_memory::InMemorySieve;
pub use in_memory_oriented::InMemoryOrientedSieve;
pub use mutable::MutableSieve;
pub use oriented::{Orientation, OrientedSieve};
pub use sieve_ref::SieveRef;
pub use sieve_trait::Sieve;
pub use reserve::SieveReserveExt;
pub use traversal_iter::{
    ClosureBothIter, ClosureBothIterRef, ClosureIter, ClosureIterRef, StarIter, StarIterRef,
};
pub use frozen_csr::{freeze_csr, FrozenSieveCsr};

use std::sync::Arc;

/// In-memory sieve storing `Arc<T>` payloads for shared ownership.
pub type InMemorySieveArc<P, T> = in_memory::InMemorySieve<P, Arc<T>>;
/// Oriented in-memory sieve storing `Arc<T>` payloads.
pub type InMemoryOrientedSieveArc<P, T, O = crate::topology::orientation::Sign> =
    in_memory_oriented::InMemoryOrientedSieve<P, Arc<T>, O>;
/// Vertical stack with `Arc<T>` payload sharing.
pub type InMemoryStackArc<B, C, T> = crate::topology::stack::InMemoryStack<B, C, Arc<T>>;

// Helpful aliases for common orientation groups
pub type OrientedTriSieve<P, T> =
    in_memory_oriented::InMemoryOrientedSieve<P, T, crate::topology::orientation::D3>;
pub type OrientedQuadSieve<P, T> =
    in_memory_oriented::InMemoryOrientedSieve<P, T, crate::topology::orientation::D4>;
pub type OrientedSignSieve<P, T> =
    in_memory_oriented::InMemoryOrientedSieve<P, T, crate::topology::orientation::Sign>;

#[cfg(test)]
mod tests;
