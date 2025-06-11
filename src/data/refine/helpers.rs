//! Helpers for pulling per-point slices out along a Sieve.
//! (Previously lived in section.rs)

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

// Remove or update #[cfg(feature = "data_refine")] to avoid warning
// #[cfg(feature = "data_refine")]
pub struct ReadOnlyMap<'a, V: Clone + Default> {
    pub section: &'a crate::data::section::Section<V>,
}
// #[cfg(feature = "data_refine")]
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

    fn v(i: u64) -> PointId { PointId::new(i) }

    #[test]
    fn restrict_helpers_basic() {
        // Build tiny mesh: 1→2→3
        let mut s = InMemorySieve::<PointId, ()>::default();
        s.add_arrow(v(1), v(2), ());
        s.add_arrow(v(2), v(3), ());
        let mut atlas = Atlas::default();
        atlas.insert(v(1),1); atlas.insert(v(2),1); atlas.insert(v(3),1);
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
