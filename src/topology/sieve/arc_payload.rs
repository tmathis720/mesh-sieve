//! Wrapper sieve that presents Payload = Arc<P> for any Sieve<P>.
//!
//! This module provides [`SieveArcPayload`], a wrapper that adapts any [`Sieve`] implementation
//! to use `Arc<P>` as its payload type, enabling shared ownership and efficient cloning of payloads.

use super::sieve_trait::Sieve;
use crate::topology::stratum::InvalidateCache;
use std::sync::Arc;

/// A wrapper sieve that presents `Payload = Arc<P>` for any inner sieve with payload `P`.
///
/// This allows payloads to be shared efficiently between multiple references,
/// reducing unnecessary cloning and enabling shared ownership semantics.
#[derive(Default, Clone)]
pub struct SieveArcPayload<S: Sieve> {
    /// The inner sieve being wrapped.
    pub inner: S,
}

impl<S: Sieve> SieveArcPayload<S> {
    /// Creates a new `SieveArcPayload` wrapping the given sieve.
    pub fn new(inner: S) -> Self {
        Self { inner }
    }
}

impl<S: Sieve> InvalidateCache for SieveArcPayload<S> {
    /// Invalidates the cache of the inner sieve.
    fn invalidate_cache(&mut self) {
        self.inner.invalidate_cache();
    }
}

impl<S: Sieve> Sieve for SieveArcPayload<S>
where
    S::Payload: Clone,
{
    type Point = S::Point;
    type Payload = Arc<S::Payload>;
    type ConeIter<'a> = Box<dyn Iterator<Item = (S::Point, &'a Arc<S::Payload>)> + 'a> where Self: 'a;
    type SupportIter<'a> = Box<dyn Iterator<Item = (S::Point, &'a Arc<S::Payload>)> + 'a> where Self: 'a;

    fn cone<'a>(&'a self, p: S::Point) -> Self::ConeIter<'a> {
        // Compose a buffer of Arc-wrapped payloads for this call
        let buf: Vec<(S::Point, Arc<S::Payload>)> = self.inner.cone(p)
            .map(|(q, pay)| (q, Arc::new(pay.clone())))
            .collect();
        // Store in a Box to extend lifetime
        let buf = Box::new(buf);
        let ptr = &*buf as *const Vec<(S::Point, Arc<S::Payload>)>;
        Box::new((0..buf.len()).map(move |i| {
            let (q, _arc) = &buf[i];
            (*q, unsafe { &(*ptr)[i].1 })
        }))
    }
    fn support<'a>(&'a self, p: S::Point) -> Self::SupportIter<'a> {
        let buf: Vec<(S::Point, Arc<S::Payload>)> = self.inner.support(p)
            .map(|(q, pay)| (q, Arc::new(pay.clone())))
            .collect();
        let buf = Box::new(buf);
        let ptr = &*buf as *const Vec<(S::Point, Arc<S::Payload>)>;
        Box::new((0..buf.len()).map(move |i| {
            let (q, _arc) = &buf[i];
            (*q, unsafe { &(*ptr)[i].1 })
        }))
    }
    fn add_arrow(&mut self, src: S::Point, dst: S::Point, payload: Arc<S::Payload>) {
        self.inner.add_arrow(src, dst, (*payload).clone());
    }
    fn remove_arrow(&mut self, src: S::Point, dst: S::Point) -> Option<Arc<S::Payload>> {
        self.inner.remove_arrow(src, dst).map(Arc::new)
    }
    fn base_points<'a>(&'a self) -> Box<dyn Iterator<Item = S::Point> + 'a> {
        self.inner.base_points()
    }
    fn cap_points<'a>(&'a self) -> Box<dyn Iterator<Item = S::Point> + 'a> {
        self.inner.cap_points()
    }
}
