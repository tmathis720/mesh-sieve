//! Oriented variants of Sieve traversals in the spirit of PETSc DMPlex.
//!
//! - [`Orientation`] models a finite group with `compose` and `inverse`,
//!   so we can accumulate per-arrow orientations along transitive closure.
//! - [`OrientedSieve`] extends [`Sieve`] with orientation-aware cone/support
//!   and a constructor for arrows that includes an orientation.
//!
//! `compose(a, b)` is interpreted as "do `a`, then `b`" while traversing a
//! path (left-accumulating). `inverse(a)` is the orientation used when the
//! same arrow is traversed in reverse.
//!
//! ## Choosing an orientation type
//!
//! | Use case                    | Type |
//! |----------------------------|------|
//! | 1D edge flip / sign        | [`Sign`](crate::topology::orientation::Sign) |
//! | Triangle face orientation  | [`D3`](crate::topology::orientation::D3) |
//! | Quad face orientation      | [`D4`](crate::topology::orientation::D4) |
//! | Arbitrary k-permutation    | [`Perm<K>`](crate::topology::orientation::Perm) |
//!
//! Default [`closure_o`](OrientedSieve::closure_o)/[`star_o`](OrientedSieve::star_o)
//! accumulate orientations via [`Orientation::compose`] and return a stable,
//! point-sorted vector for deterministic behavior.

use super::sieve_trait::Sieve;

/// A finite group capturing per-arrow orientations/permutations.
/// Implementations **must** satisfy for all `a`, `b`, `c`:
///   - associativity: `compose(a, compose(b, c)) == compose(compose(a, b), c)`
///   - identity:      `compose(id, a) == a == compose(a, id)` where `id = Default::default()`
///   - inverse:       `compose(a, inverse(a)) == id == compose(inverse(a), a)`
///
/// Semantics:
/// - `compose(a, b)` = "do `a`, then `b`" along a path (left-accumulating).
/// - `inverse(a)`    = orientation used when traversing the same arrow in reverse.
pub trait Orientation: Copy + Default + std::fmt::Debug + 'static {
    fn compose(a: Self, b: Self) -> Self;
    fn inverse(a: Self) -> Self;
}

/// Trivial orientation (no-op)
impl Orientation for () {
    #[inline]
    fn compose(_: (), _: ()) -> () {
        ()
    }
    #[inline]
    fn inverse(_: ()) -> () {
        ()
    }
}

/// Legacy integer orientation (additive). Prefer explicit types from
/// [`crate::topology::orientation`].
impl Orientation for i32 {
    #[inline]
    fn compose(a: i32, b: i32) -> i32 {
        a + b
    }
    #[inline]
    fn inverse(a: i32) -> i32 {
        -a
    }
}

/// Sieve extension with orientation-aware incidence.
pub trait OrientedSieve: Sieve {
    type Orient: Orientation;

    type ConeOIter<'a>: Iterator<Item = (Self::Point, Self::Orient)>
    where
        Self: 'a;
    type SupportOIter<'a>: Iterator<Item = (Self::Point, Self::Orient)>
    where
        Self: 'a;

    /// Oriented outgoing incidence from `p`. Pairs `(dst, orient(p->dst))`.
    fn cone_o<'a>(&'a self, p: Self::Point) -> Self::ConeOIter<'a>;

    /// Oriented incoming incidence to `p`. Pairs `(src, orient(src->p))`.
    ///
    /// The orientation returned here corresponds to the stored **forward**
    /// arrow from `src` to `p`. When traversing "upward" against that arrow
    /// (e.g. in [`star_o`](OrientedSieve::star_o)), the orientation must be
    /// inverted.
    fn support_o<'a>(&'a self, p: Self::Point) -> Self::SupportOIter<'a>;

    /// Insert an oriented arrow `src -> dst` with payload and orientation.
    fn add_arrow_o(
        &mut self,
        src: Self::Point,
        dst: Self::Point,
        payload: Self::Payload,
        orient: Self::Orient,
    );

    /// Transitive closure with accumulated orientations (downward).
    ///
    /// Returns a **stable, point-sorted** `Vec<(point, orientation_from_seed)>`.
    fn closure_o<'s, I>(&'s self, seeds: I) -> Vec<(Self::Point, Self::Orient)>
    where
        I: IntoIterator<Item = Self::Point>,
    {
        use std::collections::HashMap;
        let mut stack: Vec<(Self::Point, Self::Orient)> = seeds
            .into_iter()
            .map(|p| (p, Self::Orient::default()))
            .collect();

        // first-arrival wins to keep deterministic, minimal composition
        let mut best: HashMap<Self::Point, Self::Orient> = HashMap::new();

        while let Some((p, acc)) = stack.pop() {
            if best.contains_key(&p) {
                continue;
            }
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
    ///
    /// Because `support_o` reports the orientation of the forward arrow
    /// (`src -> dst`), following the arrow in reverse requires composing
    /// with its inverse at each step.
    fn star_o<'s, I>(&'s self, seeds: I) -> Vec<(Self::Point, Self::Orient)>
    where
        I: IntoIterator<Item = Self::Point>,
    {
        use std::collections::HashMap;
        let mut stack: Vec<(Self::Point, Self::Orient)> = seeds
            .into_iter()
            .map(|p| (p, Self::Orient::default()))
            .collect();
        let mut best: HashMap<Self::Point, Self::Orient> = HashMap::new();

        while let Some((p, acc)) = stack.pop() {
            if best.contains_key(&p) {
                continue;
            }
            best.insert(p, acc);
            for (q, o_src_p) in self.support_o(p) {
                // Walking "up" the arrow means composing with the inverse
                // of the stored orientation.
                let step = <Self::Orient as Orientation>::inverse(o_src_p);
                let nxt = <Self::Orient as Orientation>::compose(acc, step);
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
