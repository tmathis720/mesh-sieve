//! High‐level “complete_section” that runs neighbour_links → exchange_sizes → exchange_data.

use crate::data::section::Section;
use crate::overlap::overlap::Overlap;
use crate::algs::communicator::Communicator;
use super::{neighbour_links, size_exchange, data_exchange};
use crate::overlap::delta::Delta;
use bytemuck::Pod;

pub fn complete_section<V, D, C>(
    section: &mut crate::data::section::Section<V>,
    overlap: &crate::overlap::overlap::Overlap,
    comm: &C,
    delta: &D,
    my_rank: usize,
) where
    V: Clone + Default + Send + 'static,
    D: crate::overlap::delta::Delta<V> + Send + Sync + 'static,
    D::Part: bytemuck::Pod,
    C: crate::algs::communicator::Communicator + Sync,
{
    const BASE_TAG: u16 = 0xBEEF;
    let links = crate::algs::completion::neighbour_links::neighbour_links(section, overlap, my_rank);
    let counts = crate::algs::completion::size_exchange::exchange_sizes(&links, comm, BASE_TAG);
    crate::algs::completion::data_exchange::exchange_data::<V, D, C>(&links, &counts, comm, BASE_TAG+1, section);
}

// Optionally, add a test that wires all three helpers together under #[cfg(test)]
