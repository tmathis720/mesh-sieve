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
    /// Attempt to access data for a point not in the atlas.
    #[error("Section error: point {0:?} not found in atlas")]
    PointNotInAtlas(crate::topology::point::PointId),
    /// Attempt to access a point not in the atlas.
    #[error("SievedArray error: point {0:?} not found in atlas")]
    SievedArrayPointNotInAtlas(crate::topology::point::PointId),

    /// Mismatch between expected and provided slice length for a point.
    #[error("Section error: slice length mismatch for {point:?}: expected {expected}, got {found}")]
    SliceLengthMismatch {
        point: crate::topology::point::PointId,
        expected: usize,
        found: usize,
    },
    /// Mismatch between expected and provided slice length for a point (SievedArray).
    #[error("SievedArray error: slice length mismatch at {point:?}: expected {expected}, got {found}")]
    SievedArraySliceLengthMismatch {
        point: crate::topology::point::PointId,
        expected: usize,
        found: usize,
    },

    /// Attempt to add a point to the section failed at atlas insertion.
    #[error("Section error: failed to add point {0:?} to atlas: {1}")]
    AtlasInsertionFailed(crate::topology::point::PointId, #[source] Box<MeshSieveError>),
    /// Attempt to remove or copy data for a point that disappeared.
    #[error("Section internal error: missing data for point {0:?}")]
    MissingSectionPoint(crate::topology::point::PointId),
    /// Bulk scatter mismatch between total lengths.
    #[error("Section error: scatter source length mismatch: expected {expected}, got {found}")]
    ScatterLengthMismatch { expected: usize, found: usize },
    /// One of the scatter chunks did not fit.
    #[error("Section error: scatter chunk at offset {offset} of length {len} out of bounds")]
    ScatterChunkMismatch { offset: usize, len: usize },
    /// Failure converting count to primitive (should never happen if FromPrimitive is well-behaved).
    #[error("SievedArray error: cannot convert count {0} via FromPrimitive")]
    SievedArrayPrimitiveConversionFailure(usize),
    // TODO: Add more error variants as needed for other modules.
}
