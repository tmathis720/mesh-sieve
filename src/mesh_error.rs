//! MeshSieveError: Unified error type for mesh-sieve public APIs
//!
//! This error type is used throughout the mesh-sieve library to provide robust,
//! non-panicking error handling for all public APIs.

use std::fmt::Debug;
use thiserror::Error;

/// Unified error type for mesh-sieve operations.
#[derive(Debug, Error)]
pub enum MeshSieveError {
    /// Error indicating that the overlap graph is missing required neighbor links.
    #[error("Missing overlap: {source}")]
    MissingOverlap {
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    /// Generic mesh error (for internal use, e.g. error propagation)
    #[error("mesh error: {0}")]
    MeshError(Box<MeshSieveError>),
    /// GPU buffer mapping failed.
    #[error("GPU buffer mapping failed")]
    GpuMappingFailed,
    /// Attempted to construct a PointId with a zero value (invalid).
    #[error("PointId must be non-zero (0 is reserved as invalid/sentinel)")]
    InvalidPointId,
    /// Vertical arrow references a point missing from the base or cap sieve.
    #[error("stack arrow references missing point in {role}: {point}")]
    StackMissingPoint { role: &'static str, point: String },
    /// Mutation on a read-only stack (e.g., `ComposedStack`) is not allowed.
    #[error("Unsupported stack operation: {0}")]
    UnsupportedStackOperation(&'static str),
    /// A point appeared in a cone but wasn’t in the initial point set.
    #[error("Topology error: point `{0}` found in cone but not in point set")]
    MissingPointInCone(String),
    /// Attempted to access a point that is not present in the CSR chart.
    #[error("Topology error: point `{0}` not present in chart")]
    UnknownPoint(String),
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
    /// Slice length changed for a point during atlas mutation.
    #[deprecated(note = "Use AtlasPointLengthChanged")]
    #[error("atlas slice length changed for {point:?}: {old} -> {new}")]
    AtlasSliceLengthChanged {
        point: crate::topology::point::PointId,
        old: usize,
        new: usize,
    },
    /// Mismatch between expected and provided slice length for a point (SievedArray).
    #[error(
        "SievedArray error: slice length mismatch at {point:?}: expected {expected}, got {found}"
    )]
    SievedArraySliceLengthMismatch {
        point: crate::topology::point::PointId,
        expected: usize,
        found: usize,
    },

    /// Refinement attempted to map multiple coarse points into the same fine point.
    #[error("Refinement maps more than one coarse point into the same fine point {fine:?}")]
    DuplicateRefinementTarget {
        fine: crate::topology::point::PointId,
    },

    /// Attempt to add a point to the section failed at atlas insertion.
    #[error("Section error: failed to add point {0:?} to atlas: {1}")]
    AtlasInsertionFailed(
        crate::topology::point::PointId,
        #[source] Box<MeshSieveError>,
    ),
    /// Attempt to remove or copy data for a point that disappeared.
    #[error("Section internal error: missing data for point {0:?}")]
    MissingSectionPoint(crate::topology::point::PointId),
    /// Bulk scatter mismatch between total lengths.
    #[error("scatter total length mismatch (expected={expected}, found={found})")]
    ScatterLengthMismatch { expected: usize, found: usize },
    /// One of the scatter chunks did not fit.
    #[error("scatter chunk out of bounds or overflow (offset={offset}, len={len})")]
    ScatterChunkMismatch { offset: usize, len: usize },
    /// Attempted to use a scatter plan built from an outdated atlas.
    #[error("plan stale: built for atlas version {expected}, current {found}")]
    AtlasPlanStale { expected: u64, found: u64 },
    /// Missing ownership metadata for a point.
    #[error("Ownership missing for point {0:?}")]
    MissingOwnership(crate::topology::point::PointId),
    /// Point's slice length changed across atlas rebuild.
    #[error("atlas point length changed for {point:?} (expected={expected}, found={found})")]
    AtlasPointLengthChanged {
        point: crate::topology::point::PointId,
        expected: usize,
        found: usize,
    },
    /// Reducer or path length mismatch when no concrete point exists.
    #[error("reduction length mismatch (expected={expected}, found={found})")]
    ReducerLengthMismatch { expected: usize, found: usize },
    /// Atlas slices are not contiguous as expected.
    #[error(
        "atlas contiguity mismatch for {point:?} (expected_offset={expected}, found_offset={found})"
    )]
    AtlasContiguityMismatch {
        point: crate::topology::point::PointId,
        expected: usize,
        found: usize,
    },
    /// Failure converting count to primitive (should never happen if FromPrimitive is well-behaved).
    #[error("SievedArray error: cannot convert count {0} via FromPrimitive")]
    SievedArrayPrimitiveConversionFailure(usize),
    /// Delta application failed because source and dest slices had different lengths.
    #[error("Delta error: slice length mismatch (src.len={expected}, dest.len={found})")]
    DeltaLengthMismatch { expected: usize, found: usize },
    /// Constraint index is out of bounds for a point slice.
    #[error("Constraint error at point {point:?}: DOF index {index} out of bounds (len={len})")]
    ConstraintIndexOutOfBounds {
        point: crate::topology::point::PointId,
        index: usize,
        len: usize,
    },
    /// Refinement template did not match the cell cone size.
    #[error(
        "Refinement topology mismatch for cell {cell:?}: template {template} expects {expected} vertices, found {found}"
    )]
    RefinementTopologyMismatch {
        cell: crate::topology::point::PointId,
        template: &'static str,
        expected: usize,
        found: usize,
    },
    /// Unsupported cell type encountered during refinement.
    #[error("Unsupported refinement cell type {cell_type:?} at cell {cell:?}")]
    UnsupportedRefinementCellType {
        cell: crate::topology::point::PointId,
        cell_type: crate::topology::cell_type::CellType,
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
    /// Lengths array count mismatch during data exchange.
    #[error("Lengths count mismatch for neighbor {neighbor}: expected {expected}, got {got}")]
    LengthsCountMismatch {
        neighbor: usize,
        expected: usize,
        got: usize,
    },
    /// Payload element count mismatch during data exchange.
    #[error("Payload count mismatch for neighbor {neighbor}: expected {expected}, got {got}")]
    PayloadCountMismatch {
        neighbor: usize,
        expected: usize,
        got: usize,
    },

    #[error("Overlap link not found for (local={0}, rank={1})")]
    OverlapLinkMissing(crate::topology::point::PointId, usize),

    #[error("Overlap: edge must be Local->Part, found {src:?} -> {dst:?}")]
    OverlapNonBipartite {
        src: crate::overlap::overlap::OvlId,
        dst: crate::overlap::overlap::OvlId,
    },
    #[error("expected Local(_), found {found:?}")]
    OverlapExpectedLocal {
        found: crate::overlap::overlap::OvlId,
    },
    #[error("expected Part(_), found {found:?}")]
    OverlapExpectedPart {
        found: crate::overlap::overlap::OvlId,
    },

    /// Generic I/O error for mesh readers/writers.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// Invalid or inverted geometry detected.
    #[error("Geometry error: {0}")]
    InvalidGeometry(String),
    /// Parse error while reading mesh formats.
    #[error("Mesh I/O parse error: {0}")]
    MeshIoParse(String),
    #[error("Overlap: payload.rank {found} != Part({expected})")]
    OverlapRankMismatch { expected: usize, found: usize },
    #[error("Overlap: Part node found in base_points()")]
    OverlapPartInBasePoints,
    #[error("Overlap: Local node found in cap_points()")]
    OverlapLocalInCapPoints,
    #[error("Overlap: duplicate edge {src:?} -> {dst:?}")]
    OverlapDuplicateEdge {
        src: crate::overlap::overlap::OvlId,
        dst: crate::overlap::overlap::OvlId,
    },
    #[error("Overlap: empty Part({rank}) node (no incoming edges)")]
    OverlapEmptyPart { rank: usize },

    /// Out edge exists but its mirrored in edge is missing.
    #[error("overlap in/out mirror missing: {src:?} -> {dst:?}")]
    OverlapInOutMirrorMissing {
        src: crate::overlap::overlap::OvlId,
        dst: crate::overlap::overlap::OvlId,
    },

    /// Out/in edges exist but payloads differ (rank/remote_point mismatch).
    #[error("overlap in/out payload mismatch on {src:?} -> {dst:?}: out={out:?} in={inn:?}")]
    OverlapInOutPayloadMismatch {
        src: crate::overlap::overlap::OvlId,
        dst: crate::overlap::overlap::OvlId,
        out: crate::overlap::overlap::Remote,
        inn: crate::overlap::overlap::Remote,
    },

    /// Counts of edges in out vs. in adjacency differ.
    #[error("overlap in/out edge count mismatch: out={out_edges}, in={in_edges}")]
    OverlapInOutEdgeCountMismatch { out_edges: usize, in_edges: usize },

    #[error(
        "Overlap resolution conflict for (local={local}, rank={rank}): existing={existing:?}, new={new:?}"
    )]
    OverlapResolutionConflict {
        local: crate::topology::point::PointId,
        rank: usize,
        existing: Option<crate::topology::point::PointId>,
        new: crate::topology::point::PointId,
    },
}

impl PartialEq for MeshSieveError {
    fn eq(&self, other: &MeshSieveError) -> bool {
        use MeshSieveError::*;
        match (self, other) {
            (InvalidPointId, InvalidPointId)
            | (CycleDetected, CycleDetected)
            | (ZeroLengthSlice, ZeroLengthSlice)
            | (PartitionPointOverflow, PartitionPointOverflow)
            | (GpuMappingFailed, GpuMappingFailed) => true,
            (InvalidGeometry(a), InvalidGeometry(b)) => a == b,
            (UnsupportedStackOperation(a), UnsupportedStackOperation(b)) => a == b,
            (MissingPointInCone(a), MissingPointInCone(b)) => a == b,
            (UnknownPoint(a), UnknownPoint(b)) => a == b,
            (DuplicatePoint(a), DuplicatePoint(b)) => a == b,
            (MissingAtlasPoint(a), MissingAtlasPoint(b)) => a == b,
            (PointNotInAtlas(a), PointNotInAtlas(b)) => a == b,
            (SievedArrayPointNotInAtlas(a), SievedArrayPointNotInAtlas(b)) => a == b,
            (
                SliceLengthMismatch {
                    point: p1,
                    expected: e1,
                    found: f1,
                },
                SliceLengthMismatch {
                    point: p2,
                    expected: e2,
                    found: f2,
                },
            ) => p1 == p2 && e1 == e2 && f1 == f2,
            (
                AtlasSliceLengthChanged {
                    point: p1,
                    old: o1,
                    new: n1,
                },
                AtlasSliceLengthChanged {
                    point: p2,
                    old: o2,
                    new: n2,
                },
            ) => p1 == p2 && o1 == o2 && n1 == n2,
            (
                AtlasPointLengthChanged {
                    point: p1,
                    expected: e1,
                    found: f1,
                },
                AtlasPointLengthChanged {
                    point: p2,
                    expected: e2,
                    found: f2,
                },
            ) => p1 == p2 && e1 == e2 && f1 == f2,
            (
                SievedArraySliceLengthMismatch {
                    point: p1,
                    expected: e1,
                    found: f1,
                },
                SievedArraySliceLengthMismatch {
                    point: p2,
                    expected: e2,
                    found: f2,
                },
            ) => p1 == p2 && e1 == e2 && f1 == f2,
            (DuplicateRefinementTarget { fine: f1 }, DuplicateRefinementTarget { fine: f2 }) => {
                f1 == f2
            }
            (AtlasInsertionFailed(p1, _), AtlasInsertionFailed(p2, _)) => p1 == p2,
            (MissingSectionPoint(a), MissingSectionPoint(b)) => a == b,
            (
                ScatterLengthMismatch {
                    expected: e1,
                    found: f1,
                },
                ScatterLengthMismatch {
                    expected: e2,
                    found: f2,
                },
            ) => e1 == e2 && f1 == f2,
            (
                ReducerLengthMismatch {
                    expected: e1,
                    found: f1,
                },
                ReducerLengthMismatch {
                    expected: e2,
                    found: f2,
                },
            ) => e1 == e2 && f1 == f2,
            (
                ScatterChunkMismatch {
                    offset: o1,
                    len: l1,
                },
                ScatterChunkMismatch {
                    offset: o2,
                    len: l2,
                },
            ) => o1 == o2 && l1 == l2,
            (
                AtlasContiguityMismatch {
                    point: p1,
                    expected: e1,
                    found: f1,
                },
                AtlasContiguityMismatch {
                    point: p2,
                    expected: e2,
                    found: f2,
                },
            ) => p1 == p2 && e1 == e2 && f1 == f2,
            (
                AtlasPlanStale {
                    expected: e1,
                    found: f1,
                },
                AtlasPlanStale {
                    expected: e2,
                    found: f2,
                },
            ) => e1 == e2 && f1 == f2,
            (MissingOwnership(a), MissingOwnership(b)) => a == b,
            (
                SievedArrayPrimitiveConversionFailure(a),
                SievedArrayPrimitiveConversionFailure(b),
            ) => a == b,
            (
                DeltaLengthMismatch {
                    expected: e1,
                    found: f1,
                },
                DeltaLengthMismatch {
                    expected: e2,
                    found: f2,
                },
            ) => e1 == e2 && f1 == f2,
            (PartitionIndexOutOfBounds(a), PartitionIndexOutOfBounds(b)) => a == b,
            (MissingRecvCount { neighbor: n1 }, MissingRecvCount { neighbor: n2 }) => n1 == n2,
            (SectionAccess { point: p1, .. }, SectionAccess { point: p2, .. }) => p1 == p2,
            (CommError { neighbor: n1, .. }, CommError { neighbor: n2, .. }) => n1 == n2,
            (
                BufferSizeMismatch {
                    neighbor: n1,
                    expected: e1,
                    got: g1,
                },
                BufferSizeMismatch {
                    neighbor: n2,
                    expected: e2,
                    got: g2,
                },
            ) => n1 == n2 && e1 == e2 && g1 == g2,
            (
                PartCountMismatch {
                    neighbor: n1,
                    expected: e1,
                    got: g1,
                },
                PartCountMismatch {
                    neighbor: n2,
                    expected: e2,
                    got: g2,
                },
            ) => n1 == n2 && e1 == e2 && g1 == g2,
            (
                LengthsCountMismatch {
                    neighbor: n1,
                    expected: e1,
                    got: g1,
                },
                LengthsCountMismatch {
                    neighbor: n2,
                    expected: e2,
                    got: g2,
                },
            ) => n1 == n2 && e1 == e2 && g1 == g2,
            (
                PayloadCountMismatch {
                    neighbor: n1,
                    expected: e1,
                    got: g1,
                },
                PayloadCountMismatch {
                    neighbor: n2,
                    expected: e2,
                    got: g2,
                },
            ) => n1 == n2 && e1 == e2 && g1 == g2,
            (Communication(a), Communication(b)) => a == b,
            (OverlapLinkMissing(a1, b1), OverlapLinkMissing(a2, b2)) => a1 == a2 && b1 == b2,
            (
                OverlapNonBipartite { src: s1, dst: d1 },
                OverlapNonBipartite { src: s2, dst: d2 },
            ) => s1 == s2 && d1 == d2,
            (OverlapExpectedLocal { found: f1 }, OverlapExpectedLocal { found: f2 }) => f1 == f2,
            (OverlapExpectedPart { found: f1 }, OverlapExpectedPart { found: f2 }) => f1 == f2,
            (
                OverlapRankMismatch {
                    expected: e1,
                    found: f1,
                },
                OverlapRankMismatch {
                    expected: e2,
                    found: f2,
                },
            ) => e1 == e2 && f1 == f2,
            (OverlapPartInBasePoints, OverlapPartInBasePoints) => true,
            (OverlapLocalInCapPoints, OverlapLocalInCapPoints) => true,
            (
                OverlapDuplicateEdge { src: s1, dst: d1 },
                OverlapDuplicateEdge { src: s2, dst: d2 },
            ) => s1 == s2 && d1 == d2,
            (OverlapEmptyPart { rank: r1 }, OverlapEmptyPart { rank: r2 }) => r1 == r2,
            (
                OverlapInOutMirrorMissing { src: s1, dst: d1 },
                OverlapInOutMirrorMissing { src: s2, dst: d2 },
            ) => s1 == s2 && d1 == d2,
            (
                OverlapInOutPayloadMismatch {
                    src: s1,
                    dst: d1,
                    out: o1,
                    inn: i1,
                },
                OverlapInOutPayloadMismatch {
                    src: s2,
                    dst: d2,
                    out: o2,
                    inn: i2,
                },
            ) => s1 == s2 && d1 == d2 && o1 == o2 && i1 == i2,
            (
                OverlapInOutEdgeCountMismatch {
                    out_edges: o1,
                    in_edges: i1,
                },
                OverlapInOutEdgeCountMismatch {
                    out_edges: o2,
                    in_edges: i2,
                },
            ) => o1 == o2 && i1 == i2,
            (
                OverlapResolutionConflict {
                    local: l1,
                    rank: r1,
                    existing: ex1,
                    new: n1,
                },
                OverlapResolutionConflict {
                    local: l2,
                    rank: r2,
                    existing: ex2,
                    new: n2,
                },
            ) => l1 == l2 && r1 == r2 && ex1 == ex2 && n1 == n2,
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
