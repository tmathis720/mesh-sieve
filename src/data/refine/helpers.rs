//! Helpers for pulling per-point slices out along a Sieve.
//! (Previously lived in section.rs)
//!
//! This module provides utility functions for extracting slices of data
//! associated with points in a mesh, following closure or star traversals
//! of a [`Sieve`]. It also provides a read-only wrapper for section data.

use crate::data::section::Map;
use crate::topology::point::PointId;
use crate::topology::sieve::Sieve;

/// Restrict a map along the closure of the given seed points.
///
/// Returns an iterator over `(PointId, &[V])` for all points in the closure.
pub fn restrict_closure<'s, M, V: Clone + Default + 's>(
    sieve: &impl Sieve<Point = PointId>,
    map: &'s M,
    seeds: impl IntoIterator<Item = PointId>,
) -> impl Iterator<Item = (PointId, &'s [V])>
where
    M: Map<V> + 's,
{
    sieve.closure(seeds).map(move |p| (p, map.get(p)))
}

/// Restrict a map along the star of the given seed points.
///
/// Returns an iterator over `(PointId, &[V])` for all points in the star.
pub fn restrict_star<'s, M, V: Clone + Default + 's>(
    sieve: &impl Sieve<Point = PointId>,
    map: &'s M,
    seeds: impl IntoIterator<Item = PointId>,
) -> impl Iterator<Item = (PointId, &'s [V])>
where
    M: Map<V> + 's,
{
    sieve.star(seeds).map(move |p| (p, map.get(p)))
}

/// Restrict a map along the closure of the given seed points, collecting results into a vector.
pub fn restrict_closure_vec<'s, M, V: Clone + Default + 's>(
    sieve: &impl Sieve<Point = PointId>,
    map: &'s M,
    seeds: impl IntoIterator<Item = PointId>,
) -> Vec<(PointId, &'s [V])>
where
    M: Map<V> + 's,
{
    restrict_closure(sieve, map, seeds).collect()
}

/// Restrict a map along the star of the given seed points, collecting results into a vector.
pub fn restrict_star_vec<'s, M, V: Clone + Default + 's>(
    sieve: &impl Sieve<Point = PointId>,
    map: &'s M,
    seeds: impl IntoIterator<Item = PointId>,
) -> Vec<(PointId, &'s [V])>
where
    M: Map<V> + 's,
{
    restrict_star(sieve, map, seeds).collect()
}

/// Read-only wrapper for a section, implementing the [`Map`] trait.
pub struct ReadOnlyMap<'a, V: Clone + Default> {
    pub section: &'a crate::data::section::Section<V>,
}

impl<'a, V: Clone + Default> crate::data::section::Map<V> for ReadOnlyMap<'a, V> {
    fn get(&self, p: crate::topology::point::PointId) -> &[V] {
        self.section.restrict(p)
    }
    // get_mut left as default (None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::point::PointId;
    use crate::data::atlas::Atlas;
    use crate::data::section::Section;
    use crate::topology::sieve::in_memory::InMemorySieve;
    use crate::data::section::Map;

    fn v(i: u64) -> PointId { PointId::new(i).unwrap() }

    #[test]
    fn restrict_helpers_basic() {
        // Build tiny mesh: 1→2→3
        let mut s = InMemorySieve::<PointId, ()>::default();
        s.add_arrow(v(1), v(2), ());
        s.add_arrow(v(2), v(3), ());
        let mut atlas = Atlas::default();
        atlas.try_insert(v(1),1).unwrap();
        atlas.try_insert(v(2),1).unwrap();
        atlas.try_insert(v(3),1).unwrap();
        let mut sec = Section::<i32>::new(atlas);
        sec.set(v(1), &[10]); sec.set(v(2), &[20]); sec.set(v(3), &[30]);
        // restrict_closure
        let out: Vec<_> = restrict_closure(&s, &sec, [v(1)]).collect();
        let expected: Vec<_> = s.closure([v(1)]).map(|p| (p, sec.restrict(p))).collect();
        assert_eq!(out, expected);
        // empty
        let empty: Vec<_> = restrict_star(&s, &sec, std::iter::empty()).collect();
        assert!(empty.is_empty());
        // vec variants match
        assert_eq!(restrict_closure_vec(&s,&sec,[v(2)]), restrict_closure(&s,&sec,[v(2)]).collect::<Vec<_>>());
        // ReadOnlyMap
        let mut rom = ReadOnlyMap { section: &sec };
        assert_eq!(rom.get(v(3)), sec.restrict(v(3)));
        assert!(<ReadOnlyMap<'_, i32> as Map<i32>>::get_mut(&mut rom, v(3)).is_none());
    }
}
