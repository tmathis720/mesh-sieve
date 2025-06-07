//! Directed incidence relation (“Arrow” in Knepley & Karpeev, 2009).

use crate::topology::point::PointId;

/// A directed edge from `src → dst` carrying opaque payload `P`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Arrow<P = ()> {
    pub src: PointId,
    pub dst: PointId,
    pub payload: P,
}

impl<P> Arrow<P> {
    /// General constructor.
    #[inline]
    pub fn new(src: PointId, dst: PointId, payload: P) -> Self {
        Self { src, dst, payload }
    }

    /// Drop payload, returning `(src, dst)`.
    #[inline]
    pub fn endpoints(&self) -> (PointId, PointId) {
        (self.src, self.dst)
    }

    /// Map payload via closure – handy for building derived views.
    pub fn map<Q>(self, f: impl FnOnce(P) -> Q) -> Arrow<Q> {
        Arrow::new(self.src, self.dst, f(self.payload))
    }
}

// Convenience constructor for empty-payload arrows.
impl Arrow<()> {
    #[inline]
    pub fn unit(src: PointId, dst: PointId) -> Self {
        Self::new(src, dst, ())
    }
}

// Re-use `Default` only for the zero-payload specialization.
impl Default for Arrow<()> {
    fn default() -> Self { Arrow::unit(PointId::new(1), PointId::new(1)) }
}

/// Orientation for vertical arrows in a stack (for sign/permutation, etc.)
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Orientation {
    Forward,
    Reverse,
}

// Unit tests
#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::point::PointId;

    #[test]
    fn build_and_endpoints() {
        let (src, dst) = (PointId::new(1), PointId::new(2));
        let arw = Arrow::unit(src, dst);
        assert_eq!(arw.endpoints(), (src, dst));
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

        let mut h1 = DefaultHasher::new();
        let mut h2 = DefaultHasher::new();
        a1.hash(&mut h1);
        a2.hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish());
    }
}
