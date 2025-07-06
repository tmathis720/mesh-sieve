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
    // TODO: Add more error variants as needed for other modules.
}
