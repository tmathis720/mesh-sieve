//! Reference-based extensions for `Sieve` to avoid cloning payloads during traversal.
//!
//! This trait is additive: existing code using `Sieve` keeps working.
//! Algorithms that want zero-clone traversal can require `SieveRef`.

use super::sieve_trait::Sieve;

pub trait SieveRef: Sieve {
    /// Iterator over (dst, &payload) without cloning.
    type ConeRefIter<'a>: Iterator<Item = (Self::Point, &'a Self::Payload)>
    where
        Self: 'a;

    /// Iterator over (src, &payload) without cloning.
    type SupportRefIter<'a>: Iterator<Item = (Self::Point, &'a Self::Payload)>
    where
        Self: 'a;

    /// Borrowing cone.
    fn cone_ref<'a>(&'a self, p: Self::Point) -> Self::ConeRefIter<'a>;

    /// Borrowing support.
    fn support_ref<'a>(&'a self, p: Self::Point) -> Self::SupportRefIter<'a>;

    /// Point-only adapters (never touch payloads).
    #[inline]
    fn cone_points<'a>(&'a self, p: Self::Point) -> impl Iterator<Item = Self::Point> + 'a {
        self.cone_ref(p).map(|(q, _)| q)
    }

    #[inline]
    fn support_points<'a>(&'a self, p: Self::Point) -> impl Iterator<Item = Self::Point> + 'a {
        self.support_ref(p).map(|(q, _)| q)
    }
}
