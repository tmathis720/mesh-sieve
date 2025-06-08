//! Section: Field data storage over a topology atlas.
//!
//! The `Section<V>` type couples an `Atlas` (mapping points to slices in a
//! contiguous array) with a `Vec<V>` to hold the actual data. It provides
//! methods for inserting, accessing, and iterating per-point data slices.

use crate::data::atlas::Atlas;
use crate::topology::point::PointId;

/// Storage for per-point field data, backed by an `Atlas`.
#[derive(Clone, Debug)]
pub struct Section<V> {
    /// Atlas mapping each `PointId` to (offset, length) in `data`.
    atlas: Atlas,
    /// Contiguous storage of values for all points.
    data: Vec<V>,
}

impl<V: Clone + Default> Section<V> {
    /// Construct a new `Section` given an existing `Atlas`.
    ///
    /// Initializes the data buffer with `V::default()` repeated for each
    /// degree of freedom in the atlas.
    pub fn new(atlas: Atlas) -> Self {
        // Fill `data` with default values up to total_len from atlas.
        let data = vec![V::default(); atlas.total_len()];
        Section { atlas, data }
    }

    /// Read-only view of the data slice for a given point `p`.
    ///
    /// # Panics
    /// Panics if `p` is not registered in the atlas.
    #[inline]
    pub fn restrict(&self, p: PointId) -> &[V] {
        // Look up offset and length in the atlas.
        let (offset, len) = self.atlas.get(p)
            .expect("PointId not found in atlas");
        &self.data[offset .. offset + len]
    }

    /// Mutable view of the data slice for a given point `p`.
    ///
    /// # Panics
    /// Panics if `p` is not registered in the atlas.
    #[inline]
    pub fn restrict_mut(&mut self, p: PointId) -> &mut [V] {
        let (offset, len) = self.atlas.get(p)
            .expect("PointId not found in atlas");
        &mut self.data[offset .. offset + len]
    }

    /// Overwrite the data slice at point `p` with the values in `val`.
    ///
    /// # Panics
    /// Panics if the length of `val` does not match the slice length for `p`.
    pub fn set(&mut self, p: PointId, val: &[V]) {
        let target = self.restrict_mut(p);
        assert_eq!(target.len(), val.len(),
            "Input slice length must match point's DOF count");
        // Clone values into the section's buffer.
        target.clone_from_slice(val);
    }

    /// Iterate over `(PointId, &[V])` for all points in atlas order.
    ///
    /// Useful for serializing or visiting all data in a deterministic order.
    pub fn iter<'s>(&'s self) -> impl Iterator<Item = (PointId, &'s [V])> {
        // Use the atlas's point order for deterministic iteration.
        self.atlas.points()
            .map(move |pid| (pid, self.restrict(pid)))
    }
}

impl<V: Clone + Send> Section<V> {
    /// Scatter values from an external buffer `other` into this section.
    ///
    /// `atlas_map` provides a list of (offset, length) pairs corresponding to
    /// where each chunk of `other` should be copied in.
    ///
    /// # Panics
    /// Panics if `other` length does not match expected total length or if
    /// chunk sizes mismatch.
    pub fn scatter_from(&mut self,
                        other: &[V],
                        atlas_map: &[(usize, usize)])
    {
        // Iterate over each (offset,len) and the corresponding chunk.
        for ((offset, len), chunk) in atlas_map.iter()
                                                .zip(other.chunks_exact(atlas_map[0].1)) {
            // Copy the chunk into our data buffer.
            self.data[*offset .. offset + *len]
                .clone_from_slice(chunk);
        }
    }
}

/// Trait for read-only or mutable views of per-point data.
///
/// Provides a zero-cost abstraction over types that can supply a slice
/// for each `PointId`. Used in data-refinement algorithms.
pub trait Map<V: Clone + Default> {
    /// Immutable access to the data slice for `p`.
    fn get(&self, p: PointId) -> &[V];

    /// Optional mutable access to the data slice for `p`.
    ///
    /// Default implementation returns `None`, meaning the map is read-only.
    fn get_mut(&mut self, _p: PointId) -> Option<&mut [V]> {
        None
    }
}

/// Implement `Map` for `Section<V>`, allowing it to be used in data refinement.
impl<V: Clone + Default> Map<V> for Section<V> {
    fn get(&self, p: PointId) -> &[V] {
        // Use the restrict method to get an immutable slice.
        self.restrict(p)
    }

    fn get_mut(&mut self, p: PointId) -> Option<&mut [V]> {
        // Use the restrict_mut method to get a mutable slice, wrapped in Some.
        Some(self.restrict_mut(p))
    }
}

/// A one‐to‐one mapping from coarse→fine dof, carrying orientation.
pub type Sifter = Vec<(PointId, crate::topology::arrow::Orientation)>;

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

    #[cfg(feature = "data_refine")]
    /// A generic array of values indexed by mesh points, supporting refinement/assembly.
    ///
    /// This is a minimal version, to be extended with refine/assemble algorithms.
    #[derive(Clone, Debug)]
    pub struct SievedArray<P, V> {
        /// The underlying atlas (point -> offset, len).
        pub(crate) atlas: crate::data::atlas::Atlas,
        /// The data storage.
        pub(crate) data: Vec<V>,
        /// Phantom for point type.
        _phantom: std::marker::PhantomData<P>,
    }

    #[cfg(feature = "data_refine")]
    impl<P, V: Clone + Default> SievedArray<P, V>
    where
        P: Into<crate::topology::point::PointId> + Copy + Eq,
    {
        /// Create a new SievedArray with the given atlas, filled with default values.
        pub fn new(atlas: crate::data::atlas::Atlas) -> Self {
            let data = vec![V::default(); atlas.total_len()];
            Self { atlas, data, _phantom: std::marker::PhantomData }
        }

        /// Get an immutable slice for a point.
        pub fn get(&self, p: crate::topology::point::PointId) -> &[V] {
            let (off, len) = self.atlas.get(p).expect("point not in atlas");
            &self.data[off .. off + len]
        }

        /// Get a mutable slice for a point.
        pub fn get_mut(&mut self, p: crate::topology::point::PointId) -> &mut [V] {
            let (off, len) = self.atlas.get(p).expect("point not in atlas");
            &mut self.data[off .. off + len]
        }

        /// Set the values for a point.
        pub fn set(&mut self, p: crate::topology::point::PointId, val: &[V]) {
            let tgt = self.get_mut(p);
            assert_eq!(tgt.len(), val.len());
            tgt.clone_from_slice(val);
        }

        /// Iterate over (PointId, &[V]) pairs in atlas order.
        pub fn iter<'s>(&'s self) -> impl Iterator<Item=(crate::topology::point::PointId, &'s [V])> {
            self.atlas.points().map(move |p| (p, self.get(p)))
        }

        /// Refine the data from a coarse SievedArray into this (finer) SievedArray.
        ///
        /// For each (coarse_pt, fine_pts) in `refinement`, copies the data from the coarse point
        /// into each fine point, respecting orientation (if provided in the Sifter).
        ///
        /// # Arguments
        /// * `coarse` - The coarse SievedArray to refine from.
        /// * `refinement` - A slice of (coarse point, Vec<(fine point, Orientation)>) pairs.
        pub fn refine_with_sifter(
            &mut self,
            coarse: &SievedArray<P, V>,
            refinement: &[(P, Vec<(P, crate::topology::arrow::Orientation)>)],
        ) {
            for (coarse_pt, fine_pts) in refinement.iter() {
                let coarse_slice = coarse.get((*coarse_pt).into());
                for (fine_pt, orient) in fine_pts.iter() {
                    let fine_slice = self.get_mut((*fine_pt).into());
                    assert_eq!(coarse_slice.len(), fine_slice.len(), "dof mismatch in refinement");
                    match orient {
                        crate::topology::arrow::Orientation::Forward => {
                            fine_slice.clone_from_slice(coarse_slice);
                        }
                        crate::topology::arrow::Orientation::Reverse => {
                            for (dst, src) in fine_slice.iter_mut().zip(coarse_slice.iter().rev()) {
                                *dst = src.clone();
                            }
                        }
                    }
                }
            }
        }

        /// Refine the data from a coarse SievedArray into this (finer) SievedArray.
        ///
        /// This version expects a refinement map of (coarse point, Vec<fine point>),
        /// and assumes all orientations are positive.
        pub fn refine(
            &mut self,
            coarse: &SievedArray<P, V>,
            refinement: &[(P, Vec<P>)]
        ) {
            for (coarse_pt, fine_pts) in refinement.iter() {
                let coarse_slice = coarse.get((*coarse_pt).into());
                for fine_pt in fine_pts.iter() {
                    let fine_slice = self.get_mut((*fine_pt).into());
                    assert_eq!(coarse_slice.len(), fine_slice.len(), "dof mismatch in refinement");
                    fine_slice.clone_from_slice(coarse_slice);
                }
            }
        }

        /// Assemble data from this (finer) SievedArray into a coarser SievedArray.
        ///
        /// This is a placeholder for the full assembly algorithm as per Knepley & Karpeev (2009).
        /// The method will sum/average data from `self` into `coarse` according to the refinement map.
        pub fn assemble(&self, _coarse: &mut SievedArray<P, V>, _refinement: &[(P, Vec<P>)]) {
            // TODO: Implement full assembly logic
            // For now, this is a stub.
            unimplemented!("SievedArray::assemble is not yet implemented");
        }
    }

    #[cfg(feature = "data_refine")]
    impl<P, V: Clone + Default> crate::data::section::Map<V> for SievedArray<P, V>
        where
        P: Into<crate::topology::point::PointId> + Copy + Eq,
    {
        fn get(&self, p: crate::topology::point::PointId) -> &[V] {
            self.get(p)
        }
        fn get_mut(&mut self, p: crate::topology::point::PointId) -> Option<&mut [V]> {
            Some(self.get_mut(p))
        }
    }

    #[test]
    #[cfg(feature = "data_refine")]
    fn sieved_array_set_and_get() {
        let mut arr = make_sieved_array();
        arr.set(PointId::new(1), &[10, 20]);
        arr.set(PointId::new(2), &[30]);
        assert_eq!(arr.get(PointId::new(1)), &[10, 20]);
        assert_eq!(arr.get(PointId::new(2)), &[30]);
    }

    #[test]
    #[cfg(feature = "data_refine")]
    fn sieved_array_iter_order() {
        let mut arr = make_sieved_array();
        arr.set(PointId::new(1), &[1, 2]);
        arr.set(PointId::new(2), &[3]);
        let vals: Vec<_> = arr.iter().map(|(_, v)| v[0]).collect();
        assert_eq!(vals, vec![1, 3]);
    }

    #[test]
    #[cfg(feature = "data_refine")]
    fn sieved_array_map_trait() {
        use crate::data::section::Map;
        let mut arr = make_sieved_array();
        arr.set(PointId::new(1), &[5, 6]);
        assert_eq!(<SievedArray<PointId, i32> as Map<i32>>::get(&arr, PointId::new(1)), arr.get(PointId::new(1)));
        assert!(<SievedArray<PointId, i32> as Map<i32>>::get_mut(&mut arr, PointId::new(1)).is_some());
    }

    #[test]
    #[cfg(feature = "data_refine")]
    fn sieved_array_refine_panics() {
        use crate::topology::point::PointId;
        let mut fine = make_sieved_array();
        let coarse = make_sieved_array();
        let refinement: Vec<(PointId, Vec<PointId>)> = Vec::new();
        // Should not panic: empty refinement is a no-op
        SievedArray::<PointId, i32>::refine(&mut fine, &coarse, &refinement);
    }

    #[test]
    #[cfg(feature = "data_refine")]
    #[should_panic(expected = "not yet implemented")]
    fn sieved_array_assemble_panics() {
        use crate::topology::point::PointId;
        let fine = make_sieved_array();
        let mut coarse = make_sieved_array();
        let refinement: Vec<(PointId, Vec<PointId>)> = Vec::new();
        SievedArray::<PointId, i32>::assemble(&fine, &mut coarse, &refinement);
    }

    #[cfg(feature = "data_refine")]
    pub(super) fn make_sieved_array() -> SievedArray<PointId, i32> {
        let mut atlas = crate::data::atlas::Atlas::default();
        atlas.insert(PointId::new(1), 2); // 2 dof
        atlas.insert(PointId::new(2), 1);
        SievedArray::<PointId, i32>::new(atlas)
    }

    #[test]
    #[cfg(feature = "data_refine")]
    fn sieved_array_refine_simple() {
        use crate::topology::arrow::Orientation;
        use crate::topology::point::PointId;
        // Coarse: pt 1 (2 dof)
        let mut coarse_atlas = crate::data::atlas::Atlas::default();
        coarse_atlas.insert(PointId::new(1), 2); // 2 dof
        // Fine: pt 10 (2 dof), pt 11 (2 dof)
        let mut fine_atlas = crate::data::atlas::Atlas::default();
        fine_atlas.insert(PointId::new(10), 2);
        fine_atlas.insert(PointId::new(11), 2);
        let mut coarse = SievedArray::<PointId, i32>::new(coarse_atlas);
        let mut fine = SievedArray::<PointId, i32>::new(fine_atlas);
        coarse.set(PointId::new(1), &[7, 8]);
        // Refinement: 1 -> [10 (forward), 11 (reverse)]
        let refinement = vec![
            (PointId::new(1), vec![
                (PointId::new(10), Orientation::Forward),
                (PointId::new(11), Orientation::Reverse),
            ]),
        ];
        fine.refine_with_sifter(&coarse, &refinement);
        assert_eq!(fine.get(PointId::new(10)), &[7, 8]);
        assert_eq!(fine.get(PointId::new(11)), &[8, 7]);
    }

    #[test]
    #[cfg(feature = "data_refine")]
    fn sieved_array_refine_positive_only() {
        use crate::topology::point::PointId;
        // Coarse: pt 1 (2 dof)
        let mut coarse_atlas = crate::data::atlas::Atlas::default();
        coarse_atlas.insert(PointId::new(1), 2); // 2 dof
        // Fine: pt 10 (2 dof), pt 11 (2 dof)
        let mut fine_atlas = crate::data::atlas::Atlas::default();
        fine_atlas.insert(PointId::new(10), 2);
        fine_atlas.insert(PointId::new(11), 2);
        let mut coarse = SievedArray::<PointId, i32>::new(coarse_atlas);
        let mut fine = SievedArray::<PointId, i32>::new(fine_atlas);
        coarse.set(PointId::new(1), &[42, 99]);
        // Refinement: 1 -> [10, 11] (all forward)
        let refinement = vec![
            (PointId::new(1), vec![PointId::new(10), PointId::new(11)]),
        ];
        fine.refine(&coarse, &refinement);
        assert_eq!(fine.get(PointId::new(10)), &[42, 99]);
        assert_eq!(fine.get(PointId::new(11)), &[42, 99]);
    }
}

