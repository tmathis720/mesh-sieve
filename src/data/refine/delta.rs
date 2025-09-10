//! A “delta” maps sources→dest slices, e.g. via polarity permutation.
//!
//! This module defines the [`SliceDelta`] trait for slice-level transformations and
//! provides an implementation for [`Polarity`] to permute or reverse slices.
//!
//! # Naming
//! [`SliceDelta`] is the preferred name. A deprecated [`Delta`] alias is provided
//! for backward compatibility and will be removed in a future release.
//!
//! # Choosing the right `Delta`
//! - [`crate::data::refine::delta::SliceDelta`]: transforms one *slice* into another
//!   (e.g. reverse for orientation).
//! - [`crate::overlap::delta::ValueDelta`]: describes *communication/merge semantics*
//!   for overlap/exchange between parts.
//!
//! ## Disambiguation tip
//! ```rust
//! use mesh_sieve::data::refine::delta::SliceDelta;            // slice semantics
//! use mesh_sieve::overlap::delta::ValueDelta as OverlapDelta; // communication semantics
//! ```

use crate::mesh_error::MeshSieveError;
use crate::topology::arrow::Polarity;

/// Trait for applying a transformation (delta) from a source slice to a destination slice.
///
/// Implementors define how to map or permute values from `src` to `dest`.
pub trait SliceDelta<V: Clone>: Sync {
    /// Apply the transformation from `src` to `dest`.
    ///
    /// Returns an error if lengths differ.
    fn apply(&self, src: &[V], dest: &mut [V]) -> Result<(), MeshSieveError>;
}

/// Backward-compatible re-export: external code can still name this trait `Delta`.
#[deprecated(note = "Renamed to SliceDelta; Delta will be removed in a future major release")]
pub use SliceDelta as Delta;

impl<V: Clone> SliceDelta<V> for Polarity {
    fn apply(&self, src: &[V], dest: &mut [V]) -> Result<(), MeshSieveError> {
        let expected = src.len();
        let found = dest.len();
        if expected != found {
            return Err(MeshSieveError::DeltaLengthMismatch { expected, found });
        }
        match self {
            Polarity::Forward => {
                dest.clone_from_slice(src);
            }
            Polarity::Reverse => {
                for (d, s) in dest.iter_mut().zip(src.iter().rev()) {
                    *d = s.clone();
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::arrow::Polarity;

    #[test]
    fn orientation_forward_noop() {
        let src = vec![1, 2, 3];
        let mut dst = src.clone();
        assert!(Polarity::Forward.apply(&src, &mut dst).is_ok());
        assert_eq!(dst, src);
    }

    #[test]
    fn orientation_reverse_reverses() {
        let src = vec![1, 2, 3, 4];
        let mut dst = vec![0; 4];
        assert!(Polarity::Reverse.apply(&src, &mut dst).is_ok());
        assert_eq!(dst, vec![4, 3, 2, 1]);
    }

    #[test]
    fn orientation_reverse_empty() {
        let src: Vec<i32> = vec![];
        let mut dst: Vec<i32> = vec![];
        assert!(Polarity::Reverse.apply(&src, &mut dst).is_ok());
        assert!(dst.is_empty());
    }

    #[test]
    fn orientation_mismatch_errors() {
        let src = vec![1, 2, 3];
        let mut dst = vec![0; 2];
        let err = Polarity::Forward.apply(&src, &mut dst).unwrap_err();
        assert_eq!(
            err,
            crate::mesh_error::MeshSieveError::DeltaLengthMismatch {
                expected: 3,
                found: 2
            }
        );
    }
}
