pub mod data_exchange;
pub mod neighbour_links;
pub mod section_completion;
pub mod sieve_completion;
pub mod size_exchange;
pub mod stack_completion;

pub use section_completion::complete_section;
pub use sieve_completion::complete_sieve;
pub use stack_completion::complete_stack;

pub fn partition_point(rank: usize) -> crate::topology::point::PointId {
    crate::topology::point::PointId::new((rank as u64) + 1)
        .expect("Failed to create PointId in partition_point")
}
