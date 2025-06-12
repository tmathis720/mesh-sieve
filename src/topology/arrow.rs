//! Arrow: Directed incidence (mesh topology edge) with payload and orientation.
//!
//! In mesh algorithms (Knepley & Karpeev, 2009), an *arrow* represents an incidence
//! relation between two topological entities (e.g., cell → face, face → edge),
//! optionally carrying data (payload). This module defines the `Arrow` type,
//! utility constructors, payload mapping, and a simple `Orientation` enum for
//! vertical arrows in a `Stack`.

use crate::topology::point::PointId;

/// A directed connection from `src` to `dst` carrying an arbitrary `payload`.
///
/// # Type Parameters
/// - `P`: The type of per-arrow payload. Defaults to `()` for payload-free arrows.
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Arrow<P = ()> {
    /// Source topological point (e.g., a cell or face handle).
    pub src: PointId,
    /// Destination topological point (e.g., a face or edge handle).
    pub dst: PointId,
    /// User-defined payload data attached to this incidence.
    pub payload: P,
}

impl<P> Arrow<P> {
    /// Construct a new `Arrow` from `src` → `dst` with given `payload`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use mesh_sieve::topology::arrow::Arrow;
    /// use mesh_sieve::topology::point::PointId;
    /// let a = Arrow::new(PointId::new(1), PointId::new(2), 3u32);
    /// assert_eq!(a.src.get(), 1);
    /// assert_eq!(a.dst.get(), 2);
    /// assert_eq!(a.payload, 3);
    /// ```
    #[inline]
    pub fn new(src: PointId, dst: PointId, payload: P) -> Self {
        Arrow { src, dst, payload }
    }

    /// Returns the `(src, dst)` endpoints, dropping the payload.
    ///
    /// Useful when you only care about connectivity.
    #[inline]
    pub fn endpoints(&self) -> (PointId, PointId) {
        (self.src, self.dst)
    }

    /// Transform the payload `P` to a new type `Q` by applying `f`.
    ///
    /// The source and destination remain unchanged. This is handy for
    /// deriving new arrow views without mutating the original.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use mesh_sieve::topology::arrow::Arrow;
    /// use mesh_sieve::topology::point::PointId;
    /// let a = Arrow::new(PointId::new(1), PointId::new(2), 10);
    /// // Double the payload
    /// let b = a.clone().map(|v| v * 2);
    /// assert_eq!(b.payload, 20);
    /// assert_eq!(b.src, a.src);
    /// assert_eq!(b.dst, a.dst);
    /// ```
    pub fn map<Q>(self, f: impl FnOnce(P) -> Q) -> Arrow<Q> {
        Arrow::new(self.src, self.dst, f(self.payload))
    }
}

//------------------------------------------------------------------------------
// Convenience for empty-payload arrows
//------------------------------------------------------------------------------

impl Arrow<()> {
    /// Create an arrow with no payload (`()`), i.e., a bare connectivity edge.
    ///
    /// This is equivalent to `Arrow::new(src, dst, ())`.
    #[inline]
    pub fn unit(src: PointId, dst: PointId) -> Self {
        Arrow::new(src, dst, ())
    }
}

/// Provide a default only for `Arrow<()>` so you can write `Arrow::default()`.
///
/// The default arrow points from PointId(1) → PointId(1) and carries `()`.
impl Default for Arrow<()> {
    fn default() -> Self {
        // We pick a dummy sentinel id `1` for default; users should override.
        Arrow::unit(PointId::new(1), PointId::new(1))
    }
}

//------------------------------------------------------------------------------
// Orientation: for vertical arrows in a Stack
//------------------------------------------------------------------------------

/// Sign or permutation for vertical incidence arrows in a `Stack`.
///
/// Used to record orientation when lifting/pulling degrees-of-freedom.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Orientation {
    /// No change in orientation (e.g., aligned sign +1).
    Forward,
    /// Opposite orientation (e.g., sign flip -1).
    Reverse,
}

//------------------------------------------------------------------------------
// Unit Tests
//------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::point::PointId;

    #[test]
    fn build_and_endpoints() {
        let src = PointId::new(1);
        let dst = PointId::new(2);
        let arrow = Arrow::unit(src, dst);
        assert_eq!(arrow.endpoints(), (src, dst));
    }

    #[test]
    fn payload_map() {
        let arrow = Arrow::new(PointId::new(3), PointId::new(4), 10u32);
        let arrow2 = arrow.map(|v| v * 2);
        assert_eq!(arrow2.payload, 20);
    }

    #[test]
    fn equality_and_hash() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let a1 = Arrow::unit(PointId::new(5), PointId::new(6));
        let a2 = Arrow::unit(PointId::new(5), PointId::new(6));

        // Ensure hashing equal arrows yields same hash
        let mut h1 = DefaultHasher::new();
        let mut h2 = DefaultHasher::new();
        a1.hash(&mut h1);
        a2.hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish());
    }

    #[test]
    fn new_stores_payload() {
        let a = Arrow::new(PointId::new(7), PointId::new(8), "foo");
        assert_eq!(a.src.get(), 7);
        assert_eq!(a.dst.get(), 8);
        assert_eq!(a.payload, "foo");
    }

    #[test]
    fn default_is_unit_on_1() {
        let d = Arrow::<()>::default();
        assert_eq!(d, Arrow::unit(PointId::new(1), PointId::new(1)));
    }

    #[test]
    fn debug_prints_struct() {
        let a = Arrow::new(PointId::new(1), PointId::new(2), 99u8);
        let dbg = format!("{:?}", a);
        assert!(dbg.contains("Arrow") && dbg.contains("99"));
    }

    #[test]
    fn clone_eqs_original() {
        let a = Arrow::new(PointId::new(3), PointId::new(4), vec![1,2,3]);
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn orientation_traits() {
        let f = Orientation::Forward;
        let r = Orientation::Reverse;
        let f2 = f;
        assert_eq!(f, f2);
        use std::collections::HashSet;
        let mut s = HashSet::new();
        s.insert(f);
        s.insert(r);
        assert!(s.contains(&Orientation::Forward));
    }

    #[test]
    fn arrow_with_zero_payload() {
        let a = Arrow::unit(PointId::new(1), PointId::new(2));
        let b = a.clone().map(|()| ());
        assert_eq!(b.payload, ());
    }

    #[test]
    fn serde_arrow_roundtrip() {
        let a = Arrow::new(PointId::new(1), PointId::new(2), vec![10]);
        let json = serde_json::to_string(&a).unwrap();
        let a2: Arrow<Vec<u8>> = serde_json::from_str(&json).unwrap();
        assert_eq!(a, a2);
    }
}
