//! MeshSieveError: Unified error type for mesh-sieve public APIs
//!
//! This error type is used throughout the mesh-sieve library to provide robust,
//! non-panicking error handling for all public APIs.

use thiserror::Error;
use std::fmt::Debug;

/// Unified error type for mesh-sieve operations.
#[derive(Debug, Error)]
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
    /// Delta application failed because source and dest slices had different lengths.
    #[error("Delta error: slice length mismatch (src.len={expected}, dest.len={found})")]
    DeltaLengthMismatch {
        expected: usize,
        found: usize,
    },
    /// Partition‐point computation overflowed to zero (invalid owner).
    #[error("Invalid partition owner: computed raw ID = 0")]
    PartitionPointOverflow,
    /// The `parts` slice is missing an entry for point `{0}`.
    #[error("No partition mapping for point ID {0}")]
    PartitionIndexOutOfBounds(usize),
    /// Error sending or receiving data over the wire.
    #[error("communication error: {0}")]
    Communication(#[from] CommError),

    /// Missing expected recv count for a neighbor during data exchange.
    #[error("Missing recv count for neighbor {neighbor}")]
    MissingRecvCount { neighbor: usize },
    /// Error accessing a section for a given point.
    #[error("Section access error at point {point:?}: {source}")]
    SectionAccess {
        point: crate::topology::point::PointId,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    /// Communication error for a specific neighbor.
    #[error("Communication error with neighbor {neighbor}: {source}")]
    CommError {
        neighbor: usize,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    /// Buffer size mismatch during data exchange.
    #[error("Buffer size mismatch for neighbor {neighbor}: expected {expected}, got {got}")]
    BufferSizeMismatch {
        neighbor: usize,
        expected: usize,
        got: usize,
    },
    /// Part count mismatch during data exchange.
    #[error("Part count mismatch for neighbor {neighbor}: expected {expected}, got {got}")]
    PartCountMismatch {
        neighbor: usize,
        expected: usize,
        got: usize,
    },
}

impl PartialEq for MeshSieveError {
    fn eq(&self, other: &MeshSieveError) -> bool {
        use MeshSieveError::*;
        match (self, other) {
            (InvalidPointId, InvalidPointId)
            | (CycleDetected, CycleDetected)
            | (ZeroLengthSlice, ZeroLengthSlice)
            | (PartitionPointOverflow, PartitionPointOverflow) => true,
            (UnsupportedStackOperation(a), UnsupportedStackOperation(b)) => a == b,
            (MissingPointInCone(a), MissingPointInCone(b)) => a == b,
            (DuplicatePoint(a), DuplicatePoint(b)) => a == b,
            (MissingAtlasPoint(a), MissingAtlasPoint(b)) => a == b,
            (PointNotInAtlas(a), PointNotInAtlas(b)) => a == b,
            (SievedArrayPointNotInAtlas(a), SievedArrayPointNotInAtlas(b)) => a == b,
            (SliceLengthMismatch { point: p1, expected: e1, found: f1 },
             SliceLengthMismatch { point: p2, expected: e2, found: f2 }) => p1 == p2 && e1 == e2 && f1 == f2,
            (SievedArraySliceLengthMismatch { point: p1, expected: e1, found: f1 },
             SievedArraySliceLengthMismatch { point: p2, expected: e2, found: f2 }) => p1 == p2 && e1 == e2 && f1 == f2,
            (AtlasInsertionFailed(p1, _), AtlasInsertionFailed(p2, _)) => p1 == p2,
            (MissingSectionPoint(a), MissingSectionPoint(b)) => a == b,
            (ScatterLengthMismatch { expected: e1, found: f1 }, ScatterLengthMismatch { expected: e2, found: f2 }) => e1 == e2 && f1 == f2,
            (ScatterChunkMismatch { offset: o1, len: l1 }, ScatterChunkMismatch { offset: o2, len: l2 }) => o1 == o2 && l1 == l2,
            (SievedArrayPrimitiveConversionFailure(a), SievedArrayPrimitiveConversionFailure(b)) => a == b,
            (DeltaLengthMismatch { expected: e1, found: f1 }, DeltaLengthMismatch { expected: e2, found: f2 }) => e1 == e2 && f1 == f2,
            (PartitionIndexOutOfBounds(a), PartitionIndexOutOfBounds(b)) => a == b,
            (MissingRecvCount { neighbor: n1 }, MissingRecvCount { neighbor: n2 }) => n1 == n2,
            (SectionAccess { point: p1, .. }, SectionAccess { point: p2, .. }) => p1 == p2,
            (CommError { neighbor: n1, .. }, CommError { neighbor: n2, .. }) => n1 == n2,
            (BufferSizeMismatch { neighbor: n1, expected: e1, got: g1 }, BufferSizeMismatch { neighbor: n2, expected: e2, got: g2 }) => n1 == n2 && e1 == e2 && g1 == g2,
            (PartCountMismatch { neighbor: n1, expected: e1, got: g1 }, PartCountMismatch { neighbor: n2, expected: e2, got: g2 }) => n1 == n2 && e1 == e2 && g1 == g2,
            (Communication(a), Communication(b)) => a == b,
            _ => false,
        }
    }
}
impl Eq for MeshSieveError {}

/// Low-level communicator failure.
#[derive(Debug, thiserror::Error, Clone)]
#[error("{0}")]
pub struct CommError(pub String);

impl PartialEq for CommError {
    fn eq(&self, other: &CommError) -> bool {
        self.0 == other.0
    }
}
impl Eq for CommError {}

