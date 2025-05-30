//! Topologically-indexed handle (cell, face, edge, vertex, …)

use std::{
    fmt,
    num::NonZeroU64,
};

#[repr(transparent)]
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, serde::Serialize, serde::Deserialize)]
pub struct PointId(NonZeroU64);

impl PointId {
    /// Creates a new `PointId`.
    ///
    /// # Panics
    /// Panics if `raw` is zero – we reserve 0 as *invalid/sentinel*.
    #[inline]
    pub fn new(raw: u64) -> Self {
        Self(NonZeroU64::new(raw).expect("PointId must be non-zero"))
    }

    /// Returns the raw integer value.
    #[inline]
    pub const fn get(self) -> u64 { self.0.get() }
}

// ----- Display / Debug -----
impl fmt::Debug for PointId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("PointId").field(&self.get()).finish()
    }
}
impl fmt::Display for PointId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.get())
    }
}

// ----- Safety guarantees for zero-cost FFI -----
#[cfg(feature = "mpi-support")]
unsafe impl mpi::datatype::Equivalence for PointId {
    type Out = <u64 as mpi::datatype::Equivalence>::Out;
    fn equivalent_datatype() -> Self::Out {
        u64::equivalent_datatype()
    }
}

// Static layout check (optional dev-dependency)
#[cfg(test)]
mod layout_tests {
    use super::*;
    use static_assertions::assert_eq_size;
    assert_eq_size!(PointId, u64);
}

// Unit tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_zero_guard() {
        // should panic on zero
        assert!(std::panic::catch_unwind(|| PointId::new(0)).is_err());
        let p = PointId::new(42);
        assert_eq!(p.get(), 42);
    }

    #[test]
    fn ordering_and_hash() {
        let a = PointId::new(1);
        let b = PointId::new(2);

        assert!(a < b);
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(a);
        set.insert(b);
        assert_eq!(set.len(), 2);
    }
}
