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
// no tests needed here beyond Orientation’s own tests
