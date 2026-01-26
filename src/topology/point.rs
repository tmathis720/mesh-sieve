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

use crate::mesh_error::MeshSieveError;
use std::{convert::TryFrom, fmt, num::NonZeroU64};
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
    /// Returns an error if `raw == 0`. We reserve 0 as an invalid or sentinel value.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use mesh_sieve::topology::point::PointId;
    /// # fn try_example() -> Result<(), mesh_sieve::mesh_error::MeshSieveError> {
    /// let p = PointId::new(1)?;
    /// assert_eq!(p.get(), 1);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn new(raw: u64) -> Result<Self, MeshSieveError> {
        NonZeroU64::new(raw)
            .map(PointId)
            .ok_or(MeshSieveError::InvalidPointId)
    }

    /// Returns the underlying `u64` value of this `PointId`.
    #[inline]
    pub fn get(&self) -> u64 {
        self.0.get()
    }

    /// # Safety
    /// Caller must ensure `raw != 0`.
    #[inline]
    pub unsafe fn new_unchecked(raw: u64) -> Self {
        // SAFETY: caller must guarantee raw != 0
        PointId(unsafe { NonZeroU64::new_unchecked(raw) })
    }
}

impl TryFrom<u64> for PointId {
    type Error = MeshSieveError;
    #[inline]
    fn try_from(raw: u64) -> Result<Self, MeshSieveError> {
        PointId::new(raw)
    }
}

impl TryFrom<usize> for PointId {
    type Error = MeshSieveError;
    #[inline]
    fn try_from(raw: usize) -> Result<Self, MeshSieveError> {
        let raw64 = u64::try_from(raw).map_err(|_| MeshSieveError::InvalidPointId)?;
        PointId::new(raw64)
    }
}

// -----------------------------------------------------------------------------
// Infallible conversions
// -----------------------------------------------------------------------------

/// Convert to raw `u64`.
///
/// ```rust
/// # use mesh_sieve::topology::point::PointId;
/// let p = PointId::new(42).unwrap();
/// let raw: u64 = p.into();
/// assert_eq!(raw, 42);
/// ```
impl From<PointId> for u64 {
    #[inline]
    fn from(p: PointId) -> Self {
        p.get()
    }
}

/// Convert to the inner `NonZeroU64`.
///
/// ```rust
/// # use mesh_sieve::topology::point::PointId;
/// # use std::num::NonZeroU64;
/// let p = PointId::new(7).unwrap();
/// let nz: NonZeroU64 = p.into();
/// assert_eq!(nz.get(), 7);
/// ```
impl From<PointId> for NonZeroU64 {
    #[inline]
    fn from(p: PointId) -> Self {
        p.0
    }
}

/// Wrap an existing `NonZeroU64` without checks.
///
/// ```rust
/// # use mesh_sieve::topology::point::PointId;
/// # use std::num::NonZeroU64;
/// let nz = NonZeroU64::new(9).unwrap();
/// let p: PointId = nz.into();
/// assert_eq!(p.get(), 9);
/// ```
impl From<NonZeroU64> for PointId {
    #[inline]
    fn from(nz: NonZeroU64) -> Self {
        PointId(nz)
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
#[cfg(feature = "mpi-support")]
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
    fn new_zero_returns_error() {
        // Attempting to create with 0 should return an error
        assert!(PointId::new(0).is_err());
    }

    #[test]
    fn new_and_get() {
        let p = PointId::new(42).unwrap();
        assert_eq!(p.get(), 42);
    }

    #[test]
    fn try_from_u64_success() -> Result<(), MeshSieveError> {
        let p = PointId::try_from(42u64)?;
        assert_eq!(p.get(), 42);
        Ok(())
    }

    #[test]
    fn try_from_u64_zero_fails() {
        assert_eq!(
            PointId::try_from(0u64).unwrap_err(),
            MeshSieveError::InvalidPointId
        );
    }

    #[test]
    fn try_from_usize_success() -> Result<(), MeshSieveError> {
        let p = PointId::try_from(123usize)?;
        assert_eq!(p.get(), 123);
        Ok(())
    }

    #[test]
    fn try_from_usize_zero_fails() {
        assert_eq!(
            PointId::try_from(0usize).unwrap_err(),
            MeshSieveError::InvalidPointId
        );
    }

    #[test]
    fn debug_and_display() {
        let p = PointId::new(7).unwrap();
        assert_eq!(format!("{:?}", p), "PointId(7)");
        assert_eq!(format!("{}", p), "7");
    }

    #[test]
    fn ordering_and_hash() {
        let a = PointId::new(1).unwrap();
        let b = PointId::new(2).unwrap();
        // Ordering
        assert!(a < b);
        // HashSet support
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(a);
        set.insert(b);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn roundtrip_conversions() {
        use std::num::NonZeroU64;

        let p = PointId::new(1).unwrap();
        let raw: u64 = p.into();
        assert_eq!(raw, 1);

        let nz = NonZeroU64::new(123).unwrap();
        let p2: PointId = nz.into();
        assert_eq!(u64::from(p2), 123);

        let nz2: NonZeroU64 = p2.into();
        assert_eq!(nz2.get(), 123);
    }
}

#[cfg(test)]
mod serde_tests {
    use super::*;
    #[test]
    fn json_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
        let p = PointId::new(123)?;
        let s = serde_json::to_string(&p)?;
        let p2: PointId = serde_json::from_str(&s)?;
        assert_eq!(p2, p);
        Ok(())
    }
    #[test]
    fn bincode_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
        let p = PointId::new(456)?;
        let bytes = bincode::serialize(&p)?;
        let p2: PointId = bincode::deserialize(&bytes)?;
        assert_eq!(p2, p);
        Ok(())
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

#[cfg(all(test, feature = "mpi-support"))]
mod mpi_tests {
    use super::*;
    use mpi::datatype::Equivalence;

    #[test]
    fn pointid_mpi_equivalence_type_is_u64() {
        fn _assert_equiv<T: Equivalence>() {}
        _assert_equiv::<PointId>();
    }
}

#[cfg(test)]
mod copy_clone_eq_tests {
    use super::*;
    #[test]
    fn copy_and_clone() {
        let p = PointId::new(5).unwrap();
        let q = p;
        let r = p.clone();
        assert_eq!(p, q);
        assert_eq!(p, r);
    }
    #[test]
    fn eq_and_neq() {
        let p = PointId::new(8).unwrap();
        let q = PointId::new(8).unwrap();
        let r = PointId::new(9).unwrap();
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
        let p = PointId::new(u64::MAX).unwrap();
        assert_eq!(p.get(), u64::MAX);
    }
}

// Add these to dev-dependencies in Cargo.toml:
// serde_json = "1.0"
// bincode = "1.3"
// thiserror = "1.0"
