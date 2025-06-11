//! `PointId`: a strong, zero-cost handle for mesh entities
//!
//! In a mesh topology, every element (cell, face, edge, vertex, etc.) is
//! represented by a unique, opaque identifier. `PointId` wraps a nonzero
//! `u64` to enforce at compile- and runtime that 0 is reserved as an
//! invalid or sentinel value.
//!
//! This module provides:
//! - A transparent `PointId` newtype around `NonZeroU64` for zero-cost FFI and
//!   memory layout guarantees.
//! - Constructors and accessors with safety checks.
//! - Implementations of common traits (`Debug`, `Display`, ordering,
//!   hashing) so `PointId` can be used in maps, sets, and printed easily.

use std::{fmt, num::NonZeroU64};
///
/// # PETSc SF semantics
/// In the context of parallel mesh distribution (see Knepley & Karpeev 2009),
/// a `PointId` is used as the root/leaf in the Star Forest (SF) model.
///
/// # Memory layout
/// This type is `repr(transparent)`, meaning it has the same ABI and
/// alignment as its single field (`NonZeroU64`) and can be passed to FFI
/// exactly like a `u64`.
#[derive(
    Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[repr(transparent)]
pub struct PointId(NonZeroU64);

impl PointId {
    /// Creates a new `PointId` from a raw `u64` value.
    ///
    /// # Panics
    ///
    /// Panics if `raw == 0`. We reserve 0 as an invalid or sentinel value.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use sieve_rs::topology::point::PointId;
    /// let p = PointId::new(1);
    /// assert_eq!(p.get(), 1);
    /// ```
    #[inline]
    pub fn new(raw: u64) -> Self {
        // NonZeroU64::new returns Option; `expect` panics if raw == 0
        PointId(NonZeroU64::new(raw).expect("PointId must be non-zero"))
    }

    /// Returns the inner `u64` value of this `PointId`.
    ///
    /// This is a cheap, const-time getter. Use it when you need to inspect
    /// or print the raw integer, but prefer to work with `PointId` otherwise
    /// for type safety.
    #[inline]
    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

// -----------------------------------------------------------------------------
// Formatting traits
// -----------------------------------------------------------------------------

/// Custom `Debug` implementation to display as `PointId(raw_value)`.
impl fmt::Debug for PointId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Use tuple debug formatting for clarity
        f.debug_tuple("PointId").field(&self.get()).finish()
    }
}

/// Custom `Display` implementation to print only the raw integer.
///
/// Prints the numeric ID without any wrapper text.
impl fmt::Display for PointId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.get())
    }
}

// -----------------------------------------------------------------------------
// FFI and layout guarantees
// -----------------------------------------------------------------------------

/// Provide MPI compatibility: `PointId` can be sent over MPI as a `u64`.
///
/// We declare that `PointId` has the same MPI datatype as `u64`, ensuring
/// zero-cost, layout-safe interop.
unsafe impl mpi::datatype::Equivalence for PointId {
    type Out = <u64 as mpi::datatype::Equivalence>::Out;

    fn equivalent_datatype() -> Self::Out {
        // Delegate to u64's MPI equivalence
        u64::equivalent_datatype()
    }
}

// -----------------------------------------------------------------------------
// Testing and assertions
// -----------------------------------------------------------------------------

#[cfg(test)]
mod layout_tests {
    //! Compile-time assertion that `PointId` has the same size as `u64`.
    use super::*;
    use static_assertions::assert_eq_size;

    // If this fails, our repr(transparent) guarantee is broken!
    assert_eq_size!(PointId, u64);
}

#[cfg(test)]
mod tests {
    //! Unit tests for `PointId` functionality.
    use super::*;

    #[test]
    fn new_zero_panics() {
        // Attempting to create with 0 should panic
        assert!(std::panic::catch_unwind(|| PointId::new(0)).is_err());
    }

    #[test]
    fn new_and_get() {
        let p = PointId::new(42);
        assert_eq!(p.get(), 42);
    }

    #[test]
    fn debug_and_display() {
        let p = PointId::new(7);
        assert_eq!(format!("{:?}", p), "PointId(7)");
        assert_eq!(format!("{}", p), "7");
    }

    #[test]
    fn ordering_and_hash() {
        let a = PointId::new(1);
        let b = PointId::new(2);
        // Ordering
        assert!(a < b);
        // HashSet support
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(a);
        set.insert(b);
        assert_eq!(set.len(), 2);
    }
}

#[cfg(test)]
mod serde_tests {
    use super::*;
    #[test]
    fn json_roundtrip() {
        let p = PointId::new(123);
        let s = serde_json::to_string(&p).unwrap();
        let p2: PointId = serde_json::from_str(&s).unwrap();
        assert_eq!(p2, p);
    }
    #[test]
    fn bincode_roundtrip() {
        let p = PointId::new(456);
        let bytes = bincode::serialize(&p).unwrap();
        let p2: PointId = bincode::deserialize(&bytes).unwrap();
        assert_eq!(p2, p);
    }
}

#[cfg(test)]
mod abi_tests {
    use super::*;
    use static_assertions::{assert_eq_align, assert_eq_size};
    #[test]
    fn alignment_matches_u64() {
        assert_eq_align!(PointId, u64);
    }
    #[test]
    fn size_matches_u64() {
        assert_eq_size!(PointId, u64);
    }
}

#[cfg(test)]
mod copy_clone_eq_tests {
    use super::*;
    #[test]
    fn copy_and_clone() {
        let p = PointId::new(5);
        let q = p;
        let r = p.clone();
        assert_eq!(p, q);
        assert_eq!(p, r);
    }
    #[test]
    fn eq_and_neq() {
        let p = PointId::new(8);
        let q = PointId::new(8);
        let r = PointId::new(9);
        assert_eq!(p, p);
        assert_eq!(p, q);
        assert_ne!(p, r);
    }
}

#[cfg(test)]
mod edge_case_tests {
    use super::*;
    #[test]
    fn max_value() {
        let p = PointId::new(u64::MAX);
        assert_eq!(p.get(), u64::MAX);
    }
}

#[cfg(all(test, feature = "metis-support"))]
mod mpi_equivalence_tests {
    use super::*;
    use mpi::datatype::Equivalence;
    #[test]
    fn pointid_equivalence_matches_u64() {
        let t1 = <PointId as Equivalence>::equivalent_datatype();
        let t2 = <u64 as Equivalence>::equivalent_datatype();
        assert_eq!(t1, t2);
    }
}

// Add these to dev-dependencies in Cargo.toml:
// serde_json = "1.0"
// bincode = "1.3"
