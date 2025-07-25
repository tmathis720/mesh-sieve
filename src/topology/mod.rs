//! Top-level module for mesh topology abstractions.
//!
//! This module provides the core types and traits for representing mesh topologies using the Sieve abstraction.
//! It includes:
//! - Arrow and point types for mesh connectivity
//! - The Sieve trait and in-memory implementation
//! - Stack and stratum utilities for advanced mesh operations
//!
//! Most users will interact with the `Sieve` trait and the `InMemorySieve` implementation for building and traversing mesh topologies.

pub mod arrow;
pub mod point;
pub mod sieve;
pub mod stack;
pub mod stratum;
pub mod utils;

pub use sieve::*;
