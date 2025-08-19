//! Sieve module: provides the core [`Sieve`] trait and implementations for in-memory, strata, and arc payload sieves.
//!
//! This module defines the main interface for sieve data structures, which are used for efficient set membership queries.
//! It also provides several implementations, including an in-memory version and others for specialized use cases.

/// Core trait for sieve data structures.
pub mod sieve_trait;
/// Orientation-aware extensions to [`Sieve`].
pub mod oriented;
/// In-memory implementation of the [`Sieve`] trait.
pub mod in_memory;
/// In-memory implementation storing per-arrow orientations.
pub mod in_memory_oriented;
/// Reference-returning extensions to [`Sieve`].
pub mod sieve_ref;
/// Strata sieve implementation.
pub mod strata;
/// Sieve implementation using arc payloads.
pub mod arc_payload;
/// Concrete traversal iterators without dynamic dispatch.
pub mod traversal_iter;

// Re-export the core trait and in‐memory impl at top level
pub use sieve_trait::Sieve;
pub use oriented::{Orientation, OrientedSieve};
pub use in_memory::InMemorySieve;
pub use in_memory_oriented::InMemoryOrientedSieve;
pub use sieve_ref::SieveRef;
pub use traversal_iter::{ClosureBothIter, ClosureIter, StarIter};
