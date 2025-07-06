//! Wrapper sieve that presents Payload = Arc<P> for any Sieve<P>.
//!
//! This module provides [`SieveArcPayload`], a wrapper that adapts any [`Sieve`] implementation
//! to use `Arc<P>` as its payload type, enabling shared ownership and efficient cloning of payloads.
//!
//! # Example
//! ```rust
//! use mesh_sieve::topology::sieve::arc_payload::SieveArcPayload;
//! use mesh_sieve::topology::sieve::in_memory::InMemorySieve;
//! use mesh_sieve::topology::sieve::Sieve;
//! use mesh_sieve::topology::point::PointId;
//! let mut s = SieveArcPayload::new(InMemorySieve::<PointId, u32>::default());
//! s.add_arrow(PointId::new(1).unwrap(), PointId::new(2).unwrap(), std::sync::Arc::new(42));
//! let d = s.diameter()?;
//! assert_eq!(d, 1);
//! # Ok::<(), mesh_sieve::mesh_error::MeshSieveError>(())
//! ```

use super::sieve_trait::Sieve;
use crate::topology::stratum::InvalidateCache;
use std::cell::RefCell;

/// A wrapper sieve that presents `Payload = Arc<P>` for any inner sieve with payload `P`.
///
/// This allows payloads to be shared efficiently between multiple references,
/// reducing unnecessary cloning and enabling shared ownership semantics.
#[derive(Clone)]
pub struct SieveArcPayload<S: Sieve> {
    /// The inner sieve being wrapped.
    pub inner: S,
    /// Persistent buffer for Arc payloads (cleared and reused per query)
    pub buffer: RefCell<Vec<std::sync::Arc<S::Payload>>>,
    /// Persistent buffer for points (cleared and reused per query)
    pub point_buffer: RefCell<Vec<S::Point>>,
}

impl<S: Sieve> SieveArcPayload<S> {
    /// Creates a new `SieveArcPayload` wrapping the given sieve.
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            buffer: RefCell::new(Vec::new()),
            point_buffer: RefCell::new(Vec::new()),
        }
    }
}

impl<S: Sieve> Default for SieveArcPayload<S> {
    fn default() -> Self {
        Self {
            inner: S::default(),
            buffer: RefCell::new(Vec::new()),
            point_buffer: RefCell::new(Vec::new()),
        }
    }
}

impl<S: Sieve> InvalidateCache for SieveArcPayload<S> {
    /// Invalidates the cache of the inner sieve.
    fn invalidate_cache(&mut self) {
        self.inner.invalidate_cache();
        self.buffer.borrow_mut().clear();
        self.point_buffer.borrow_mut().clear();
    }
}

impl<S: Sieve> Sieve for SieveArcPayload<S>
where
    S::Payload: Clone,
{
    type Point = S::Point;
    type Payload = std::sync::Arc<S::Payload>;
    type ConeIter<'a> = Box<dyn Iterator<Item = (S::Point, std::sync::Arc<S::Payload>)> + 'a> where Self: 'a;
    type SupportIter<'a> = Box<dyn Iterator<Item = (S::Point, std::sync::Arc<S::Payload>)> + 'a> where Self: 'a;

    fn cone<'a>(&'a self, p: S::Point) -> Self::ConeIter<'a> {
        Box::new(self.inner.cone(p).map(|(q, pay)| (q, std::sync::Arc::new(pay.clone()))))
    }
    fn support<'a>(&'a self, p: S::Point) -> Self::SupportIter<'a> {
        Box::new(self.inner.support(p).map(|(q, pay)| (q, std::sync::Arc::new(pay.clone()))))
    }
    fn add_arrow(&mut self, src: S::Point, dst: S::Point, payload: std::sync::Arc<S::Payload>) {
        self.inner.add_arrow(src, dst, (*payload).clone());
    }
    fn remove_arrow(&mut self, src: S::Point, dst: S::Point) -> Option<std::sync::Arc<S::Payload>> {
        self.inner.remove_arrow(src, dst).map(std::sync::Arc::new)
    }
    fn base_points<'a>(&'a self) -> Box<dyn Iterator<Item = S::Point> + 'a> {
        self.inner.base_points()
    }
    fn cap_points<'a>(&'a self) -> Box<dyn Iterator<Item = S::Point> + 'a> {
        self.inner.cap_points()
    }
}
