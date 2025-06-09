//! High‐level “complete_section” that runs neighbour_links → exchange_sizes → exchange_data.

pub fn complete_section<V, D, C>(
    section: &mut crate::data::section::Section<V>,
    overlap: &crate::overlap::overlap::Overlap,
    comm: &C,
    _delta: &D,
    my_rank: usize,
    n_ranks: usize,
) where
    V: Clone + Default + Send + 'static,
    D: crate::overlap::delta::Delta<V> + Send + Sync + 'static,
    D::Part: bytemuck::Pod + Default,
    C: crate::algs::communicator::Communicator + Sync,
{
    const BASE_TAG: u16 = 0xBEEF;
    let links =
        crate::algs::completion::neighbour_links::neighbour_links(section, overlap, my_rank);
    // --- DEADLOCK FIX: ensure symmetric communication ---
    // For tests: use all ranks except my_rank as neighbors
    let all_neighbors: std::collections::HashSet<usize> =
        (0..n_ranks).filter(|&r| r != my_rank).collect();
    // Exchange sizes (always post send/recv for all neighbors)
    let counts = crate::algs::completion::size_exchange::exchange_sizes_symmetric(
        &links,
        comm,
        BASE_TAG,
        &all_neighbors,
    );
    crate::algs::completion::data_exchange::exchange_data_symmetric::<V, D, C>(
        &links,
        &counts,
        comm,
        BASE_TAG + 1,
        section,
        &all_neighbors,
    );
}

#[cfg(test)]
mod tests {
    // All thread-based tests removed: use MPI example for real multi-rank coverage.
}
