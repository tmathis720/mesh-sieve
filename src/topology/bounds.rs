//! Common bound aliases used across topology code.
//!
//! These traits have blanket impls, so any type satisfying the underlying
//! bounds will automatically implement them. They are zero-cost and only
//! reduce duplication in `where` clauses.

/// Canonical bound set for point identifiers.
///
/// Rationale:
/// - `Copy` for cheap pass-by-value in tight loops
/// - `Eq + Hash` for `HashMap`-backed adjacencies
/// - `Ord` to allow deterministic ordering (sort strata/neighbors)
/// - `Debug` for diagnostics and invariant checks
pub trait PointLike: Copy + Eq + std::hash::Hash + Ord + std::fmt::Debug {}
impl<T> PointLike for T where T: Copy + Eq + std::hash::Hash + Ord + std::fmt::Debug {}

/// Minimal bound we expect for per-arrow payloads in in-memory backends.
/// Keep this deliberately small to avoid over-constraining higher layers.
pub trait PayloadLike: Clone {}
impl<T: Clone> PayloadLike for T {}
