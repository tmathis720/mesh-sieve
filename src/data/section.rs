//! Field data over a topology atlas.

use crate::data::atlas::Atlas;
use crate::topology::point::PointId;

#[derive(Clone, Debug)]
pub struct Section<V> {
    atlas: Atlas,
    data: Vec<V>,
}

impl<V: Clone + Default> Section<V> {
    /// Creates an empty section with an existing atlas.
    pub fn new(atlas: Atlas) -> Self {
        Self {
            data: vec![V::default(); atlas.total_len()],
            atlas,
        }
    }

    /// Immutable restriction (read-only slice).
    #[inline] pub fn restrict(&self, p: PointId) -> &[V] {
        let (off, len) = self.atlas.get(p)
            .expect("point not in atlas");
        &self.data[off .. off + len]
    }

    /// Mutable restriction (write-capable slice).
    #[inline] pub fn restrict_mut(&mut self, p: PointId) -> &mut [V] {
        let (off, len) = self.atlas.get(p)
            .expect("point not in atlas");
        &mut self.data[off .. off + len]
    }

    /// Copies values into the storage for point `p`.
    pub fn set(&mut self, p: PointId, val: &[V]) {
        let tgt = self.restrict_mut(p);
        assert_eq!(tgt.len(), val.len());
        tgt.clone_from_slice(val);
    }

    /// Iterate `(PointId, &[V])` pairs in atlas order.
    pub fn iter<'s>(&'s self) -> impl Iterator<Item=(PointId, &'s [V])> {
        self.atlas.points()
            .map(move |p| (p, self.restrict(p)))
    }
}

impl<V: Clone + Send> Section<V> {
    /// Place `other` into slice of *this* determined by `atlas_map`.
    /// (General building block for MPI receive or threaded gather.)
    pub fn scatter_from(&mut self,
                        other: &[V],
                        atlas_map: &[(usize, usize)])
    {
        for ((off,len), src_chunk) in
            atlas_map.iter().zip(other.chunks_exact(atlas_map[0].1)) {
            self.data[*off .. off+*len].clone_from_slice(src_chunk);
        }
    }
}

#[cfg(feature = "data_refine")]
pub trait Map<V: Clone + Default> {
    /// Return immutable slice bound to the mesh point.
    fn get(&self, p: PointId) -> &[V];
    /// Return mutable slice (optional; not all maps are mutable).
    fn get_mut(&mut self, _p: PointId) -> Option<&mut [V]> { None }
}

#[cfg(feature = "data_refine")]
impl<V: Clone + Default> Map<V> for Section<V> {
    fn get(&self, p: PointId) -> &[V] { self.restrict(p) }
    fn get_mut(&mut self, p: PointId) -> Option<&mut [V]> {
        Some(self.restrict_mut(p))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::atlas::Atlas;
    use crate::topology::point::PointId;

    fn make_section() -> Section<f64> {
        let mut atlas = Atlas::default();
        atlas.insert(PointId::new(1), 2); // 2 dof
        atlas.insert(PointId::new(2), 1);
        Section::<f64>::new(atlas)
    }

    #[test]
    fn restrict_and_set() {
        let mut s = make_section();
        s.set(PointId::new(1), &[1.0,2.0]);
        s.set(PointId::new(2), &[3.5]);

        assert_eq!(s.restrict(PointId::new(1)), &[1.0,2.0]);
        assert_eq!(s.restrict(PointId::new(2)), &[3.5]);
    }

    #[test]
    fn iter_order() {
        let mut s = make_section();
        s.set(PointId::new(1), &[9.0,8.0]);
        s.set(PointId::new(2), &[7.0]);

        let collected: Vec<_> = s.iter()
                                 .map(|(_,sl)| sl[0])
                                 .collect();
        assert_eq!(collected, vec![9.0,7.0]); // atlas order
    }

    #[cfg(feature = "data_refine")]
    #[test]
    fn map_trait_section_get_and_mut() {
        use super::Map;
        let mut s = make_section();
        s.set(PointId::new(1), &[1.0,2.0]);
        s.set(PointId::new(2), &[3.5]);
        // get == restrict
        assert_eq!(<Section<f64> as Map<f64>>::get(&s, PointId::new(1)), s.restrict(PointId::new(1)));
        // get_mut returns Some for Section
        assert!(<Section<f64> as Map<f64>>::get_mut(&mut s, PointId::new(1)).is_some());
    }

    #[cfg(feature = "data_refine")]
    struct ReadOnlyMap<'a, V: Clone + Default> { section: &'a Section<V> }
    #[cfg(feature = "data_refine")]
    impl<'a, V: Clone + Default> Map<V> for ReadOnlyMap<'a, V> {
        fn get(&self, p: PointId) -> &[V] { self.section.restrict(p) }
        // get_mut left as default (None)
    }
    #[cfg(feature = "data_refine")]
    #[test]
    fn map_trait_readonly_get_mut_none() {
        let s = make_section();
        let mut ro = ReadOnlyMap { section: &s };
        assert!(ro.get_mut(PointId::new(1)).is_none());
    }

    #[cfg(feature = "data_refine")]
    pub(crate) fn restrict_closure<'s, M, V: Clone + Default + 's>(
        sieve: &impl crate::topology::sieve::Sieve<Point = PointId>,
        map:   &'s M,
        seeds: impl IntoIterator<Item = PointId>,
    ) -> impl Iterator<Item = (PointId, &'s [V])>
    where
        M: Map<V> + 's,
    {
        sieve.closure(seeds).map(move |p| (p, map.get(p)))
    }

    #[cfg(feature = "data_refine")]
    pub(crate) fn restrict_star<'s, M, V: Clone + Default + 's>(
        sieve: &impl crate::topology::sieve::Sieve<Point = PointId>,
        map:   &'s M,
        seeds: impl IntoIterator<Item = PointId>,
    ) -> impl Iterator<Item = (PointId, &'s [V])>
    where
        M: Map<V> + 's,
    {
        sieve.star(seeds).map(move |p| (p, map.get(p)))
    }

    #[cfg(feature = "data_refine")]
    pub(crate) fn restrict_closure_vec<'s, M, V: Clone + Default + 's>(
        sieve: &impl crate::topology::sieve::Sieve<Point = PointId>,
        map:   &'s M,
        seeds: impl IntoIterator<Item = PointId>,
    ) -> Vec<(PointId, &'s [V])>
    where
        M: Map<V> + 's,
    {
        restrict_closure(sieve, map, seeds).collect()
    }

    #[cfg(feature = "data_refine")]
    pub(crate) fn restrict_star_vec<'s, M, V: Clone + Default + 's>(
        sieve: &impl crate::topology::sieve::Sieve<Point = PointId>,
        map:   &'s M,
        seeds: impl IntoIterator<Item = PointId>,
    ) -> Vec<(PointId, &'s [V])>
    where
        M: Map<V> + 's,
    {
        restrict_star(sieve, map, seeds).collect()
    }

    #[cfg(feature = "data_refine")]
    #[test]
    fn restrict_closure_and_star_helpers() {
        use super::*;
        use crate::topology::sieve::{InMemorySieve, Sieve};
        let mut atlas = Atlas::default();
        atlas.insert(PointId::new(1), 1);
        atlas.insert(PointId::new(2), 1);
        atlas.insert(PointId::new(3), 1);
        let mut s = Section::<i32>::new(atlas);
        s.set(PointId::new(1), &[10]);
        s.set(PointId::new(2), &[20]);
        s.set(PointId::new(3), &[30]);
        // Toy mesh: 1 -> 2, 2 -> 3
        let mut sieve = InMemorySieve::<PointId, ()>::default();
        Sieve::add_arrow(&mut sieve, PointId::new(1), PointId::new(2), ());
        Sieve::add_arrow(&mut sieve, PointId::new(2), PointId::new(3), ());
        // Closure from 1: [1,2,3]
        let closure = restrict_closure_vec(&sieve, &s, [PointId::new(1)]);
        let mut vals: Vec<_> = closure.iter().map(|(_, v)| v[0]).collect();
        vals.sort();
        assert_eq!(vals, vec![10, 20, 30]);
        // Star from 3: [3,2,1]
        let star = restrict_star_vec(&sieve, &s, [PointId::new(3)]);
        let mut vals: Vec<_> = star.iter().map(|(_, v)| v[0]).collect();
        vals.sort();
        assert_eq!(vals, vec![10, 20, 30]);
    }
}
