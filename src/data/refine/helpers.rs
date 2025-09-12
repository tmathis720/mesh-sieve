//! Helpers for pulling per-point slices out along a Sieve.
//! (Previously lived in section.rs)
//!
//! This module provides utility functions for extracting slices of data
//! associated with points in a mesh, following closure or star traversals
//! of a [`Sieve`]. It also provides a read-only wrapper for section data.
//!
//! The legacy infallible helpers and [`Map`] adapter are gated behind the
//! `map-adapter` feature (off by default). Prefer the fallible [`FallibleMap`]
//! trait and `try_*` helpers.

use crate::data::section::FallibleMap;
#[cfg(feature = "map-adapter")]
use crate::data::section::Map;
use crate::data::storage::{Storage, VecStorage};
use core::marker::PhantomData;
use crate::mesh_error::MeshSieveError;
use crate::topology::point::PointId;
use crate::topology::sieve::Sieve;

#[cfg(feature = "map-adapter")]
#[cfg_attr(docsrs, doc(cfg(feature = "map-adapter")))]
/// Restrict a map along the closure of the given seed points.
///
/// Returns an iterator over `(PointId, &[V])` for all points in the closure.
///
/// # Migration
/// Prefer [`try_restrict_closure`] with [`FallibleMap`] for robust error handling;
/// this helper panics if a point is missing.
pub fn restrict_closure<'s, M, V: 's>(
    sieve: &'s impl Sieve<Point = PointId>,
    map: &'s M,
    seeds: impl IntoIterator<Item = PointId>,
) -> impl Iterator<Item = (PointId, &'s [V])> + 's
where
    M: Map<V> + 's,
{
    sieve.closure(seeds).map(move |p| (p, map.get(p)))
}

#[cfg(feature = "map-adapter")]
#[cfg_attr(docsrs, doc(cfg(feature = "map-adapter")))]
/// Restrict a map along the star of the given seed points.
///
/// Returns an iterator over `(PointId, &[V])` for all points in the star.
///
/// # Migration
/// Prefer [`try_restrict_star`] with [`FallibleMap`] for robust error handling;
/// this helper panics if a point is missing.
pub fn restrict_star<'s, M, V: 's>(
    sieve: &'s impl Sieve<Point = PointId>,
    map: &'s M,
    seeds: impl IntoIterator<Item = PointId>,
) -> impl Iterator<Item = (PointId, &'s [V])> + 's
where
    M: Map<V> + 's,
{
    sieve.star(seeds).map(move |p| (p, map.get(p)))
}

#[cfg(feature = "map-adapter")]
#[cfg_attr(docsrs, doc(cfg(feature = "map-adapter")))]
/// Restrict a map along the closure of the given seed points, collecting results into a vector.
///
/// # Migration
/// Prefer [`try_restrict_closure_vec`] with [`FallibleMap`] for robust error handling;
/// this helper panics if a point is missing.
pub fn restrict_closure_vec<'s, M, V: 's>(
    sieve: &'s impl Sieve<Point = PointId>,
    map: &'s M,
    seeds: impl IntoIterator<Item = PointId>,
) -> Vec<(PointId, &'s [V])>
where
    M: Map<V> + 's,
{
    restrict_closure(sieve, map, seeds).collect()
}

#[cfg(feature = "map-adapter")]
#[cfg_attr(docsrs, doc(cfg(feature = "map-adapter")))]
/// Restrict a map along the star of the given seed points, collecting results into a vector.
///
/// # Migration
/// Prefer [`try_restrict_star_vec`] with [`FallibleMap`] for robust error handling;
/// this helper panics if a point is missing.
pub fn restrict_star_vec<'s, M, V: 's>(
    sieve: &'s impl Sieve<Point = PointId>,
    map: &'s M,
    seeds: impl IntoIterator<Item = PointId>,
) -> Vec<(PointId, &'s [V])>
where
    M: Map<V> + 's,
{
    restrict_star(sieve, map, seeds).collect()
}

/// Restrict a map along the closure of the given seed points, propagating errors.
///
/// # Errors
/// Returns [`MeshSieveError`] if any point is missing in the underlying map.
///
/// # Complexity
/// **O(n)** in the size of the closure.
pub fn try_restrict_closure<'s, M, V: 's>(
    sieve: &'s impl Sieve<Point = PointId>,
    map: &'s M,
    seeds: impl IntoIterator<Item = PointId>,
) -> impl Iterator<Item = Result<(PointId, &'s [V]), MeshSieveError>> + 's
where
    M: FallibleMap<V> + 's,
{
    sieve
        .closure(seeds)
        .map(move |p| map.try_get(p).map(|sl| (p, sl)))
}

/// Restrict a map along the star of the given seed points, propagating errors.
///
/// # Errors
/// Returns [`MeshSieveError`] if any point is missing in the underlying map.
///
/// # Complexity
/// **O(n)** in the size of the star.
pub fn try_restrict_star<'s, M, V: 's>(
    sieve: &'s impl Sieve<Point = PointId>,
    map: &'s M,
    seeds: impl IntoIterator<Item = PointId>,
) -> impl Iterator<Item = Result<(PointId, &'s [V]), MeshSieveError>> + 's
where
    M: FallibleMap<V> + 's,
{
    sieve
        .star(seeds)
        .map(move |p| map.try_get(p).map(|sl| (p, sl)))
}

/// Collects the closure restriction into a vector, short-circuiting on error.
///
/// # Errors
/// Propagates the first error encountered while traversing the closure.
pub fn try_restrict_closure_vec<'s, M, V: 's>(
    sieve: &'s impl Sieve<Point = PointId>,
    map: &'s M,
    seeds: impl IntoIterator<Item = PointId>,
) -> Result<Vec<(PointId, &'s [V])>, MeshSieveError>
where
    M: FallibleMap<V> + 's,
{
    try_restrict_closure(sieve, map, seeds).collect()
}

/// Collects the star restriction into a vector, short-circuiting on error.
///
/// # Errors
/// Propagates the first error encountered while traversing the star.
pub fn try_restrict_star_vec<'s, M, V: 's>(
    sieve: &'s impl Sieve<Point = PointId>,
    map: &'s M,
    seeds: impl IntoIterator<Item = PointId>,
) -> Result<Vec<(PointId, &'s [V])>, MeshSieveError>
where
    M: FallibleMap<V> + 's,
{
    try_restrict_star(sieve, map, seeds).collect()
}

#[cfg(feature = "rayon")]
pub fn try_restrict_closure_vec_parallel<'s, M, V: Send + Sync + 's>(
    sieve: &impl Sieve<Point = PointId>,
    map: &'s M,
    seeds: impl IntoIterator<Item = PointId>,
) -> Result<Vec<(PointId, &'s [V])>, MeshSieveError>
where
    M: FallibleMap<V> + Sync + 's,
{
    use rayon::prelude::*;
    let pts: Vec<_> = sieve.closure(seeds).collect();
    pts.par_iter()
        .map(|&p| map.try_get(p).map(|sl| (p, sl)))
        .collect::<Result<Vec<_>, MeshSieveError>>()
}

#[cfg(feature = "rayon")]
pub fn try_restrict_star_vec_parallel<'s, M, V: Send + Sync + 's>(
    sieve: &impl Sieve<Point = PointId>,
    map: &'s M,
    seeds: impl IntoIterator<Item = PointId>,
) -> Result<Vec<(PointId, &'s [V])>, MeshSieveError>
where
    M: FallibleMap<V> + Sync + 's,
{
    use rayon::prelude::*;
    let pts: Vec<_> = sieve.star(seeds).collect();
    pts.par_iter()
        .map(|&p| map.try_get(p).map(|sl| (p, sl)))
        .collect::<Result<Vec<_>, MeshSieveError>>()
}

/// Read-only wrapper for a section.
pub struct ReadOnlyMap<'a, V, S: Storage<V> = VecStorage<V>> {
    /// Underlying [`Section`] providing slice data.
    pub section: &'a crate::data::section::Section<V, S>,
    _marker: PhantomData<V>,
}

impl<'a, V, S> FallibleMap<V> for ReadOnlyMap<'a, V, S>
where
    S: Storage<V>,
{
    #[inline]
    fn try_get(&self, p: crate::topology::point::PointId) -> Result<&[V], MeshSieveError> {
        self.section.try_restrict(p)
    }

    #[inline]
    fn try_get_mut(
        &mut self,
        _p: crate::topology::point::PointId,
    ) -> Result<&mut [V], MeshSieveError> {
        Err(MeshSieveError::UnsupportedStackOperation(
            "ReadOnlyMap::try_get_mut",
        ))
    }
}

#[cfg(feature = "map-adapter")]
#[cfg_attr(docsrs, doc(cfg(feature = "map-adapter")))]
impl<'a, V, S> Map<V> for ReadOnlyMap<'a, V, S>
where
    S: Storage<V>,
{
    fn get(&self, p: crate::topology::point::PointId) -> &[V] {
        self.section
            .try_restrict(p)
            .unwrap_or_else(|e| panic!("ReadOnlyMap::get({p:?}) failed: {e}"))
    }
    // get_mut left as default (None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::atlas::Atlas;
#[cfg(feature = "map-adapter")]
use crate::data::section::Map;
use crate::data::section::{FallibleMap, Section};
use crate::data::storage::VecStorage;
    use crate::topology::point::PointId;
    use crate::topology::sieve::in_memory::InMemorySieve;

    fn v(i: u64) -> PointId {
        PointId::new(i).unwrap()
    }

    #[test]
    fn restrict_helpers_basic() {
        // Build tiny mesh: 1→2→3
        let mut s = InMemorySieve::<PointId, ()>::default();
        s.add_arrow(v(1), v(2), ());
        s.add_arrow(v(2), v(3), ());
        let mut atlas = Atlas::default();
        atlas.try_insert(v(1), 1).unwrap();
        atlas.try_insert(v(2), 1).unwrap();
        atlas.try_insert(v(3), 1).unwrap();
        let mut sec = Section::<i32, VecStorage<i32>>::new(atlas);
        sec.try_set(v(1), &[10]).unwrap();
        sec.try_set(v(2), &[20]).unwrap();
        sec.try_set(v(3), &[30]).unwrap();

        // fallible helpers succeed
        let expected: Vec<_> = s
            .closure([v(1)])
            .map(|p| (p, sec.try_restrict(p).unwrap()))
            .collect();
        assert_eq!(
            try_restrict_closure_vec(&s, &sec, [v(1)]).unwrap(),
            expected
        );

        // fallible helpers return Err on missing point
        assert!(matches!(
            try_restrict_closure_vec(&s, &sec, [v(99)]),
            Err(MeshSieveError::PointNotInAtlas(pid)) if pid == v(99)
        ));

        // ReadOnlyMap fallible
        #[allow(unused_mut)]
        let mut rom = ReadOnlyMap { section: &sec, _marker: PhantomData };
        assert_eq!(
            <ReadOnlyMap<'_, i32, VecStorage<i32>> as FallibleMap<i32>>::try_get(&rom, v(3)).unwrap(),
            sec.try_restrict(v(3)).unwrap()
        );
        assert!(<ReadOnlyMap<'_, i32, VecStorage<i32>> as FallibleMap<i32>>::try_get(&rom, v(99)).is_err());

        #[cfg(feature = "map-adapter")]
        assert!(<ReadOnlyMap<'_, i32, VecStorage<i32>> as Map<i32>>::get_mut(&mut rom, v(3)).is_none());
    }

    #[cfg(feature = "map-adapter")]
    #[test]
    fn restrict_helpers_legacy() {
        // Build tiny mesh: 1→2→3
        let mut s = InMemorySieve::<PointId, ()>::default();
        s.add_arrow(v(1), v(2), ());
        s.add_arrow(v(2), v(3), ());
        let mut atlas = Atlas::default();
        atlas.try_insert(v(1), 1).unwrap();
        atlas.try_insert(v(2), 1).unwrap();
        atlas.try_insert(v(3), 1).unwrap();
        let mut sec = Section::<i32, VecStorage<i32>>::new(atlas);
        sec.try_set(v(1), &[10]).unwrap();
        sec.try_set(v(2), &[20]).unwrap();
        sec.try_set(v(3), &[30]).unwrap();

        // restrict_closure
        let out: Vec<_> = restrict_closure(&s, &sec, [v(1)]).collect();
        let expected: Vec<_> = s
            .closure([v(1)])
            .map(|p| (p, sec.try_restrict(p).unwrap()))
            .collect();
        assert_eq!(out, expected);
        // empty
        let empty: Vec<_> = restrict_star(&s, &sec, std::iter::empty()).collect();
        assert!(empty.is_empty());
        // vec variants match
        assert_eq!(
            restrict_closure_vec(&s, &sec, [v(2)]),
            restrict_closure(&s, &sec, [v(2)]).collect::<Vec<_>>()
        );
        // fallible helpers match legacy
        assert_eq!(
            try_restrict_closure_vec(&s, &sec, [v(1)]).unwrap(),
            restrict_closure_vec(&s, &sec, [v(1)])
        );
        // legacy helper panics on missing point
        assert!(std::panic::catch_unwind(|| restrict_closure_vec(&s, &sec, [v(99)])).is_err());
    }
}
