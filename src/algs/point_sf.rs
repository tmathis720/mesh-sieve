//! PointSF: overlap + communicator + ownership metadata for ghost updates.

use crate::algs::communicator::Communicator;
use crate::algs::completion::{complete_section, complete_section_with_ownership};
use crate::data::section::Section;
use crate::data::storage::Storage;
use crate::mesh_error::MeshSieveError;
use crate::overlap::delta::CopyDelta;
use crate::overlap::overlap::Overlap;
use crate::topology::ownership::PointOwnership;

/// Lightweight wrapper for overlap-based communication.
#[derive(Clone, Copy, Debug)]
pub struct PointSF<'a, C> {
    overlap: &'a Overlap,
    comm: &'a C,
    ownership: Option<&'a PointOwnership>,
    my_rank: usize,
}

impl<'a, C> PointSF<'a, C>
where
    C: Communicator + Sync,
{
    /// Create a PointSF without ownership metadata.
    pub fn new(overlap: &'a Overlap, comm: &'a C, my_rank: usize) -> Self {
        Self {
            overlap,
            comm,
            ownership: None,
            my_rank,
        }
    }

    /// Create a PointSF with ownership metadata.
    pub fn with_ownership(
        overlap: &'a Overlap,
        ownership: &'a PointOwnership,
        comm: &'a C,
        my_rank: usize,
    ) -> Self {
        Self {
            overlap,
            comm,
            ownership: Some(ownership),
            my_rank,
        }
    }

    /// Borrow the underlying overlap graph.
    pub fn overlap(&self) -> &'a Overlap {
        self.overlap
    }

    /// Borrow the communicator.
    pub fn comm(&self) -> &'a C {
        self.comm
    }

    /// Borrow optional ownership metadata.
    pub fn ownership(&self) -> Option<&'a PointOwnership> {
        self.ownership
    }

    /// Rank for this PointSF.
    pub fn rank(&self) -> usize {
        self.my_rank
    }

    /// Validate overlap/ownership consistency in debug builds.
    pub fn validate(&self) -> Result<(), MeshSieveError> {
        #[cfg(any(
            debug_assertions,
            feature = "strict-invariants",
            feature = "check-invariants"
        ))]
        {
            self.overlap.validate_invariants()?;
            if let Some(ownership) = self.ownership {
                for src in self.overlap.base_points() {
                    if let Some(point) = src.as_local() {
                        if ownership.entry(point).is_none() {
                            return Err(MeshSieveError::OverlapPointMissingOwnership { point });
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Complete a section using CopyDelta and optional ownership metadata.
    pub fn complete_section<V, S>(&self, section: &mut Section<V, S>) -> Result<(), MeshSieveError>
    where
        V: Clone + Default + Send + PartialEq + 'static,
        S: Storage<V>,
    {
        self.validate()?;
        if let Some(ownership) = self.ownership {
            complete_section_with_ownership::<V, S, CopyDelta, C>(
                section,
                self.overlap,
                ownership,
                self.comm,
                self.my_rank,
            )
        } else {
            complete_section::<V, S, CopyDelta, C>(section, self.overlap, self.comm, self.my_rank)
        }
    }
}
