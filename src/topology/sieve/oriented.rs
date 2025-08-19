//! Oriented variants of Sieve traversals in the spirit of PETSc DMPlex.
//!
//! - `Orientation` models a tiny group with `compose` and `inverse`,
//!    so we can accumulate per-arrow orientations along transitive closure.
//! - `OrientedSieve` extends `Sieve` with orientation-aware cone/support
//!    and a constructor for arrows that includes an orientation.
//!
//! Default `closure_o`/`star_o` accumulate orientations via `compose`,
//! and return a stable, point-sorted vector for deterministic behavior.

use super::sieve_trait::Sieve;

/// A minimal orientation "group".
/// Implementations should satisfy:
///   compose(id, a) = a, compose(a, id) = a, compose(a, inverse(a)) = id,
///   and compose is associative.
pub trait Orientation: Copy + Default + std::fmt::Debug + 'static {
    fn compose(a: Self, b: Self) -> Self;
    fn inverse(a: Self) -> Self;
}

/// Trivial orientation (no-op)
impl Orientation for () {
    #[inline] fn compose(_: (), _: ()) -> () { () }
    #[inline] fn inverse(_: ()) -> () { () }
}

/// Common integer orientation (e.g. flips/rotations encoded as ints).
/// Default composition is additive; customize with your own type if needed.
impl Orientation for i32 {
    #[inline] fn compose(a: i32, b: i32) -> i32 { a + b }
    #[inline] fn inverse(a: i32) -> i32 { -a }
}

/// Sieve extension with orientation-aware incidence.
pub trait OrientedSieve: Sieve {
    type Orient: Orientation;

    type ConeOIter<'a>: Iterator<Item = (Self::Point, Self::Orient)> where Self: 'a;
    type SupportOIter<'a>: Iterator<Item = (Self::Point, Self::Orient)> where Self: 'a;

    /// Oriented outgoing incidence from `p`. Pairs `(dst, orient(p->dst))`.
    fn cone_o<'a>(&'a self, p: Self::Point) -> Self::ConeOIter<'a>;

    /// Oriented incoming incidence to `p`. Pairs `(src, orient(src->p))`.
    fn support_o<'a>(&'a self, p: Self::Point) -> Self::SupportOIter<'a>;

    /// Insert an oriented arrow `src -> dst` with payload and orientation.
    fn add_arrow_o(&mut self, src: Self::Point, dst: Self::Point,
                   payload: Self::Payload, orient: Self::Orient);

    /// Transitive closure with accumulated orientations (downward).
    ///
    /// Returns a **stable, point-sorted** `Vec<(point, orientation_from_seed)>`.
    fn closure_o<'s, I>(&'s self, seeds: I) -> Vec<(Self::Point, Self::Orient)>
    where
        I: IntoIterator<Item = Self::Point>,
    {
        use std::collections::HashMap;
        let mut stack: Vec<(Self::Point, Self::Orient)> =
            seeds.into_iter().map(|p| (p, Self::Orient::default())).collect();

        // first-arrival wins to keep deterministic, minimal composition
        let mut best: HashMap<Self::Point, Self::Orient> = HashMap::new();

        while let Some((p, acc)) = stack.pop() {
            if best.contains_key(&p) { continue; }
            best.insert(p, acc);
            for (q, o) in self.cone_o(p) {
                let nxt = <Self::Orient as Orientation>::compose(acc, o);
                if !best.contains_key(&q) {
                    stack.push((q, nxt));
                }
            }
        }
        let mut out: Vec<_> = best.into_iter().collect();
        out.sort_unstable_by_key(|(pt, _)| *pt);
        out
    }

    /// Transitive star with accumulated orientations (upward).
    fn star_o<'s, I>(&'s self, seeds: I) -> Vec<(Self::Point, Self::Orient)>
    where
        I: IntoIterator<Item = Self::Point>,
    {
        use std::collections::HashMap;
        let mut stack: Vec<(Self::Point, Self::Orient)> =
            seeds.into_iter().map(|p| (p, Self::Orient::default())).collect();
        let mut best: HashMap<Self::Point, Self::Orient> = HashMap::new();

        while let Some((p, acc)) = stack.pop() {
            if best.contains_key(&p) { continue; }
            best.insert(p, acc);
            for (q, o) in self.support_o(p) {
                let nxt = <Self::Orient as Orientation>::compose(acc, o);
                if !best.contains_key(&q) {
                    stack.push((q, nxt));
                }
            }
        }
        let mut out: Vec<_> = best.into_iter().collect();
        out.sort_unstable_by_key(|(pt, _)| *pt);
        out
    }
}

