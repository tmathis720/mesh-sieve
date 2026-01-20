//! Top-level module for mesh topology abstractions.
//!
//! This module provides the core types and traits for representing mesh topologies using the Sieve abstraction.
//! It includes:
//! - Arrow and point types for mesh connectivity
//! - The Sieve trait and in-memory implementation
//! - Stack and stratum utilities for advanced mesh operations
//!
//! Most users will interact with the `Sieve` trait and the `InMemorySieve` implementation for building and traversing mesh topologies.

//! ## Edge uniqueness
//! A topology stores a *set* of arrows: for any `(src, dst)` pair there is at most one edge.
//! Insertion behaves as an upsert, replacing existing payloads (and orientations) in place.
//! Duplicate edges are forbidden; debug builds assert that the outgoing and incoming maps
//! remain perfect mirrors and contain no parallel edges.

mod _debug_invariants;
pub mod arrow;
pub mod bounds;
pub mod cache;
pub mod cell_type;
pub mod coarsen;
pub mod labels;
pub mod ownership;
pub mod orientation;
pub mod point;
pub mod refine;
pub mod adapt;
pub mod sieve;
pub mod stack;
pub mod utils;

pub use cache::InvalidateCache;
pub use cell_type::CellType;
pub use labels::LabelSet;
pub use ownership::{OwnershipEntry, PointOwnership};
pub use orientation::*;
pub use sieve::*;

#[cfg(test)]
mod tests;
