//! High-level assembly orchestration across overlaps.
//!
//! This module provides helpers to:
//! 1. reduce/accumulate shared DOFs using ownership metadata,
//! 2. apply constraints, and
//! 3. complete (copy) values back to ghosts.

use crate::algs::communicator::{CommTag, Communicator, SectionCommTags};
use crate::algs::completion::section_completion::complete_section_with_tags_and_ownership;
use crate::data::constrained_section::{apply_constraints_to_section, ConstraintSet};
use crate::data::section::Section;
use crate::data::storage::Storage;
use crate::mesh_error::MeshSieveError;
use crate::overlap::delta::{CopyDelta, ValueDelta};
use crate::overlap::overlap::Overlap;
use crate::topology::ownership::PointOwnership;

/// Communication tags for the two-phase assembly workflow.
#[derive(Copy, Clone, Debug)]
pub struct AssemblyCommTags {
    /// Tags for the reduction/accumulation phase.
    pub reduce: SectionCommTags,
    /// Tags for the owner-to-ghost completion phase.
    pub complete: SectionCommTags,
}

impl AssemblyCommTags {
    /// Construct tags from a base, assigning deterministic offsets per phase.
    #[inline]
    pub const fn from_base(base: CommTag) -> Self {
        Self {
            reduce: SectionCommTags::from_base(base),
            complete: SectionCommTags::from_base(base.offset(2)),
        }
    }
}

/// High-level assembly across overlap with ownership and constraints.
///
/// # Steps
/// 1. Reduce/accumulate shared DOFs using `ValueDelta` and ownership metadata.
/// 2. Apply constraints locally.
/// 3. Complete (copy) owned values back to ghosts.
pub fn assemble_section_with_tags_and_ownership<V, S, D, C, Con>(
    section: &mut Section<V, S>,
    overlap: &Overlap,
    ownership: &PointOwnership,
    comm: &C,
    my_rank: usize,
    tags: AssemblyCommTags,
    constraints: &Con,
) -> Result<(), MeshSieveError>
where
    V: Clone + Default + Send + PartialEq + bytemuck::Pod + Copy + 'static,
    S: Storage<V>,
    D: ValueDelta<V> + Send + Sync + 'static,
    D::Part: bytemuck::Pod + Default + Copy,
    C: Communicator + Sync,
    Con: ConstraintSet<V>,
{
    complete_section_with_tags_and_ownership::<V, S, D, C>(
        section, overlap, ownership, comm, my_rank, tags.reduce,
    )?;

    apply_constraints_to_section(section, constraints)?;

    complete_section_with_tags_and_ownership::<V, S, CopyDelta, C>(
        section, overlap, ownership, comm, my_rank, tags.complete,
    )?;

    Ok(())
}

/// Convenience wrapper using a legacy default tag (0xBEEF).
pub fn assemble_section_with_ownership<V, S, D, C, Con>(
    section: &mut Section<V, S>,
    overlap: &Overlap,
    ownership: &PointOwnership,
    comm: &C,
    my_rank: usize,
    constraints: &Con,
) -> Result<(), MeshSieveError>
where
    V: Clone + Default + Send + PartialEq + bytemuck::Pod + Copy + 'static,
    S: Storage<V>,
    D: ValueDelta<V> + Send + Sync + 'static,
    D::Part: bytemuck::Pod + Default + Copy,
    C: Communicator + Sync,
    Con: ConstraintSet<V>,
{
    let tags = AssemblyCommTags::from_base(CommTag::new(0xBEEF));
    assemble_section_with_tags_and_ownership::<V, S, D, C, Con>(
        section, overlap, ownership, comm, my_rank, tags, constraints,
    )
}
