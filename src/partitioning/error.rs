//! Partitioning errors for mesh-sieve

use thiserror::Error;

/// Errors from vertex-cut construction and partitioning routines
#[derive(Debug, Error)]
pub enum PartitionError {
    /// A vertex returned by the graph was not found in the vertex-index map
    #[error("Owner lookup failed for vertex {0}")]
    VertexNotFound(usize),
    /// Partition map did not contain an entry for a vertex
    #[error("PartitionMap missing part for vertex {0}")]
    MissingPartition(usize),
    /// Unexpected condition: no parts in the map
    #[error("Empty partition map: no parts available")]
    NoParts,
    /// Other errors (e.g. METIS wrapper failures)
    #[error("Partitioner error: {0}")]
    Other(String),
}
