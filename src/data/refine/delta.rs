//! A “delta” maps sources→dest slices, e.g. via Orientation permutation.
//!
//! This module defines the [`Delta`] trait for slice transformations and provides
//! an implementation for [`Orientation`] to permute or reverse slices.

use crate::topology::arrow::Orientation;

/// Trait for applying a transformation (delta) from a source slice to a destination slice.
///
/// Implementors define how to map or permute values from `src` to `dest`.
pub trait Delta<V: Clone + Default>: Sync {
    /// Apply the transformation from `src` to `dest`.
    ///
    /// Returns an error if `src.len() != dest.len()`.
    fn apply(&self, src: &[V], dest: &mut [V]) -> Result<(), crate::mesh_error::MeshSieveError>;
}

impl<V: Clone + Default> Delta<V> for Orientation {
    fn apply(&self, src: &[V], dest: &mut [V]) -> Result<(), crate::mesh_error::MeshSieveError> {
        let expected = src.len();
        let found = dest.len();
        if expected != found {
            return Err(crate::mesh_error::MeshSieveError::DeltaLengthMismatch { expected, found });
        }
        match self {
            Orientation::Forward => {
                dest.clone_from_slice(src);
            }
            Orientation::Reverse => {
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
    use crate::topology::arrow::Orientation;

    #[test]
    fn orientation_forward_noop() {
        let src = vec![1, 2, 3];
        let mut dst = src.clone();
        assert!(Orientation::Forward.apply(&src, &mut dst).is_ok());
        assert_eq!(dst, src);
    }

    #[test]
    fn orientation_reverse_reverses() {
        let src = vec![1, 2, 3, 4];
        let mut dst = vec![0; 4];
        assert!(Orientation::Reverse.apply(&src, &mut dst).is_ok());
        assert_eq!(dst, vec![4, 3, 2, 1]);
    }

    #[test]
    fn orientation_reverse_empty() {
        let src: Vec<i32> = vec![];
        let mut dst: Vec<i32> = vec![];
        assert!(Orientation::Reverse.apply(&src, &mut dst).is_ok());
        assert!(dst.is_empty());
    }

    #[test]
    fn orientation_mismatch_errors() {
        let src = vec![1, 2, 3];
        let mut dst = vec![0; 2];
        let err = Orientation::Forward
            .apply(&src, &mut dst)
            .unwrap_err();
        assert_eq!(
            err,
            crate::mesh_error::MeshSieveError::DeltaLengthMismatch {
                expected: 3,
                found: 2
            }
        );
    }
}
