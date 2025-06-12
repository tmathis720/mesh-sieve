//! A “delta” maps sources→dest slices, e.g. via Orientation permutation.
//!
//! This module defines the [`Delta`] trait for slice transformations and provides
//! an implementation for [`Orientation`] to permute or reverse slices.

use crate::topology::arrow::Orientation;

/// Trait for applying a transformation (delta) from a source slice to a destination slice.
///
/// Implementors define how to map or permute values from `src` to `dest`.
pub trait Delta<V: Clone + Default>: Sync {
    /// Applies the delta transformation from `src` to `dest`.
    fn apply(&self, src: &[V], dest: &mut [V]);
}

impl<V: Clone + Default> Delta<V> for Orientation {
    /// Applies the orientation to map `src` to `dest`.
    ///
    /// - `Forward`: copies `src` to `dest` as-is.
    /// - `Reverse`: copies `src` to `dest` in reverse order.
    fn apply(&self, src: &[V], dest: &mut [V]) {
        match self {
            Orientation::Forward => dest.clone_from_slice(src),
            Orientation::Reverse => {
                for (d, s) in dest.iter_mut().zip(src.iter().rev()) {
                    *d = s.clone();
                }
            }
        }
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
        Orientation::Forward.apply(&src, &mut dst);
        assert_eq!(dst, src);
    }

    #[test]
    fn orientation_reverse_reverses() {
        let src = vec![1, 2, 3, 4];
        let mut dst = vec![0; 4];
        Orientation::Reverse.apply(&src, &mut dst);
        assert_eq!(dst, vec![4, 3, 2, 1]);
    }

    #[test]
    fn orientation_reverse_empty() {
        let src: Vec<i32> = vec![];
        let mut dst: Vec<i32> = vec![];
        Orientation::Reverse.apply(&src, &mut dst);
        assert!(dst.is_empty());
    }
}
