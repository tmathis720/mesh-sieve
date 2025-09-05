//! High‐level “complete_section” that runs neighbour_links → exchange_sizes → exchange_data.
//!
//! This module provides the top-level routine for distributed section completion,
//! orchestrating neighbor-link discovery, size exchange, and data exchange phases.

use std::collections::HashSet;

use crate::mesh_error::MeshSieveError;
use crate::algs::communicator::Communicator;
use crate::algs::completion::{
    data_exchange,
    neighbour_links::neighbour_links,
    size_exchange::exchange_sizes_symmetric,
};
use crate::data::section::Section;
use crate::overlap::delta::ValueDelta;
use crate::overlap::overlap::Overlap;

pub fn complete_section<V, D, C>(
    section: &mut Section<V>,
    overlap: &mut Overlap,
    comm: &C,
    _delta: &D,
    my_rank: usize,
    _n_ranks: usize,
) -> Result<(), MeshSieveError>
where
    V: Clone + Default + Send + PartialEq + 'static,
    D: ValueDelta<V> + Send + Sync + 'static,
    D::Part: bytemuck::Pod + Default,
    C: Communicator + Sync,
{
    const BASE_TAG: u16 = 0xBEEF;

    // 1) discover which points each neighbor needs
    let links = neighbour_links(section, overlap, my_rank)
        .map_err(|e| {
            MeshSieveError::CommError {
                neighbor: my_rank, // context: error during neighbour_links
                source: format!("neighbour_links failed: {}", e).into(),
            }
        })?;

    // 2) Build true neighbor set from actual overlap (both outgoing and incoming)
    let mut all_neighbors = HashSet::new();
    
    // outgoing neighbors (who want data from us)
    all_neighbors.extend(links.keys().copied());
    
    // incoming neighbors (from whom we want data)
    all_neighbors.extend(overlap.neighbor_ranks());

    // 3) exchange the item counts
    let counts = exchange_sizes_symmetric(&links, comm, BASE_TAG, &all_neighbors)
        .map_err(|e| {
            MeshSieveError::CommError {
                neighbor: my_rank, // context: error during size exchange
                source: format!("exchange_sizes_symmetric failed: {}", e).into(),
            }
        })?;

    // 4) exchange the actual data parts & fuse into our section (MPI needs the symmetric handshake)
    data_exchange::exchange_data_symmetric::<V, D, C>(
        &links,
        &counts,
        comm,
        BASE_TAG + 1,
        section,
        &all_neighbors,
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    // TODO: add unit tests for complete_section with a mock Communicator
}
