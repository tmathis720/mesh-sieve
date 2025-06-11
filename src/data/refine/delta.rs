//! A “delta” maps sources→dest slices, e.g. via Orientation permutation.

use crate::topology::arrow::Orientation;

pub trait Delta<V: Clone + Default>: Sync {
    fn apply(&self, src: &[V], dest: &mut [V]);
}

impl<V: Clone + Default> Delta<V> for Orientation {
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
