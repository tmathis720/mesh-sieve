//! MeshSieveError: Unified error type for mesh-sieve public APIs
//!
//! This error type is used throughout the mesh-sieve library to provide robust,
//! non-panicking error handling for all public APIs.

use thiserror::Error;
use std::fmt::Debug;

/// Unified error type for mesh-sieve operations.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum MeshSieveError {
    /// Attempted to construct a PointId with a zero value (invalid).
    #[error("PointId must be non-zero (0 is reserved as invalid/sentinel)")]
    InvalidPointId,
    /// Mutation on a read-only stack (e.g., `ComposedStack`) is not allowed.
    #[error("Unsupported stack operation: {0}")]
    UnsupportedStackOperation(&'static str),
    /// A point appeared in a cone but wasn’t in the initial point set.
    #[error("Topology error: point `{0}` found in cone but not in point set")]
    MissingPointInCone(String),
    /// The mesh topology contains a cycle; expected a DAG.
    #[error("Topology error: cycle detected in mesh (expected DAG)")]
    CycleDetected,
    /// Attempt to insert a zero-length slice, which is invalid.
    #[error("Atlas error: zero-length slice is not allowed")]
    ZeroLengthSlice,
    /// Attempt to insert a point that’s already in the atlas.
    #[error("Atlas error: point {0:?} already present")]
    DuplicatePoint(crate::topology::point::PointId),
    /// Internal invariant broken: point in order but missing from map.
    #[error("Atlas internal error: missing length for point {0:?}")]
    MissingAtlasPoint(crate::topology::point::PointId),
    // TODO: Add more error variants as needed for other modules.
}
