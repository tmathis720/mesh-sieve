//! Minimal PetscSection-like interface to describe DOF layout on points.
//!
//! This lightweight trait provides just enough information to map mesh
//! points to contiguous spans in a global array. It can be paired with
//! [`OrientedSieve`](crate::topology::sieve::oriented::OrientedSieve) to
//! iterate over degrees of freedom on transitive closures.

/// Minimal interface describing degrees of freedom associated with points.
pub trait Section {
    /// Point identifier type.
    type Point: Copy + Ord;

    /// Number of degrees of freedom for point `p`.
    fn dof(&self, p: Self::Point) -> usize;
    /// Offset where point `p`'s DOFs begin in a contiguous array.
    fn offset(&self, p: Self::Point) -> usize;
}

/// Iterate `(offset, dof)` pairs over the closure of `cell`.
///
/// The iterator order mirrors [`closure_o`](crate::topology::sieve::oriented::OrientedSieve::closure_o):
/// points are sorted by their identifier. If a different ordering is
/// required (e.g. by topological dimension), callers should sort the
/// resulting collection accordingly.
pub fn section_on_closure<S, Sec>(
    sieve: &S,
    sec: &Sec,
    cell: S::Point,
) -> impl Iterator<Item = (usize, usize)>
where
    S: crate::topology::sieve::oriented::OrientedSieve,
    Sec: Section<Point = S::Point>,
{
    sieve
        .closure_o([cell])
        .into_iter()
        .map(move |(p, _)| (sec.offset(p), sec.dof(p)))
}

/// Same as [`section_on_closure`] but also returns the accumulated
/// orientation from the seed to each point.
pub fn section_on_closure_o<S, Sec>(
    sieve: &S,
    sec: &Sec,
    cell: S::Point,
) -> impl Iterator<Item = (usize, usize, S::Orient)>
where
    S: crate::topology::sieve::oriented::OrientedSieve,
    Sec: Section<Point = S::Point>,
{
    sieve
        .closure_o([cell])
        .into_iter()
        .map(move |(p, o)| (sec.offset(p), sec.dof(p), o))
}

/// Optional hook for applying orientations to local element buffers.
pub trait ApplyOrientation {
    /// The orientation type used by the Sieve.
    type Orient: crate::topology::sieve::oriented::Orientation;

    /// Apply orientation `o` to the degrees-of-freedom slice `buf`.
    /// Implementations may permute indices or apply sign changes.
    fn apply(&self, o: Self::Orient, buf: &mut [f64]);
}
