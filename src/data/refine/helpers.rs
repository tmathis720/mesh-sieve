//! Helpers for pulling per-point slices out along a Sieve.
//! (Previously lived in section.rs under #[cfg(feature="data_refine")])

use crate::data::section::Map;
use crate::topology::point::PointId;
use crate::topology::sieve::Sieve;

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

// #[cfg(test)] module here for testing these four helpers in isolation
