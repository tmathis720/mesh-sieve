//! Section: Field data storage over a topology atlas.
//!
//! The `Section<V>` type couples an `Atlas` (mapping points to slices in a
//! contiguous array) with a `Vec<V>` to hold the actual data. It provides
//! methods for inserting, accessing, and iterating per-point data slices.

use crate::data::atlas::Atlas;
use crate::topology::point::PointId;
use crate::topology::stratum::InvalidateCache;

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
        let (offset, len) = self.atlas.get(p).expect("PointId not found in atlas");
        &self.data[offset..offset + len]
    }

    /// Mutable view of the data slice for a given point `p`.
    ///
    /// # Panics
    /// Panics if `p` is not registered in the atlas.
    #[inline]
    pub fn restrict_mut(&mut self, p: PointId) -> &mut [V] {
        let (offset, len) = self.atlas.get(p).expect("PointId not found in atlas");
        &mut self.data[offset..offset + len]
    }

    /// Overwrite the data slice at point `p` with the values in `val`.
    ///
    /// # Panics
    /// Panics if the length of `val` does not match the slice length for `p`.
    pub fn set(&mut self, p: PointId, val: &[V]) {
        let target = self.restrict_mut(p);
        assert_eq!(
            target.len(),
            val.len(),
            "Input slice length must match point's DOF count"
        );
        target.clone_from_slice(val);
        crate::topology::stratum::InvalidateCache::invalidate_cache(self);
    }

    /// Iterate over `(PointId, &[V])` for all points in atlas order.
    ///
    /// Useful for serializing or visiting all data in a deterministic order.
    pub fn iter(&self) -> impl Iterator<Item = (PointId, &[V])> {
        // Use the atlas's point order for deterministic iteration.
        self.atlas
            .points()
            .map(move |pid| (pid, self.restrict(pid)))
    }

    /// Add a new point to the section, resizing data as needed.
    pub fn add_point(&mut self, p: PointId, len: usize) {
        self.atlas.insert(p, len);
        self.data.resize(self.atlas.total_len(), V::default());
        crate::topology::stratum::InvalidateCache::invalidate_cache(self);
    }

    /// Remove a point from the section, rebuilding data to keep slices contiguous.
    pub fn remove_point(&mut self, p: PointId) {
        self.atlas.remove_point(p);
        // Rebuild data: allocate new vec, copy each remaining slice from old data
        let mut new_data = Vec::with_capacity(self.atlas.total_len());
        for pid in self.atlas.points() {
            let (offset, len) = self.atlas.get(pid).unwrap();
            let old_slice = &self.data[offset..offset+len];
            new_data.extend_from_slice(old_slice);
        }
        self.data = new_data;
        crate::topology::stratum::InvalidateCache::invalidate_cache(self);
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
    pub fn scatter_from(&mut self, other: &[V], atlas_map: &[(usize, usize)]) {
        let mut start = 0;
        for (offset, len) in atlas_map.iter() {
            let end = start + *len;
            let chunk = &other[start..end];
            self.data[*offset..offset + *len].clone_from_slice(chunk);
            start = end;
        }
        crate::topology::stratum::InvalidateCache::invalidate_cache(self);
    }
}

/// A **zero‐cost view** of per‐point data, supporting both read‐only and write mappings.
/// Commonly implemented by `Section<V>` or user‐supplied read‐only wrappers.
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

impl<V> InvalidateCache for Section<V> {
    fn invalidate_cache(&mut self) {
        // If you ever cache anything derived from atlas/data, clear it here.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::atlas::Atlas;
    use crate::topology::sieve::Sieve;
    use crate::topology::sieve::InMemorySieve;
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
        s.set(PointId::new(1), &[1.0, 2.0]);
        s.set(PointId::new(2), &[3.5]);

        assert_eq!(s.restrict(PointId::new(1)), &[1.0, 2.0]);
        assert_eq!(s.restrict(PointId::new(2)), &[3.5]);
    }

    #[test]
    fn iter_order() {
        let mut s = make_section();
        s.set(PointId::new(1), &[9.0, 8.0]);
        s.set(PointId::new(2), &[7.0]);

        let collected: Vec<_> = s.iter().map(|(_, sl)| sl[0]).collect();
        assert_eq!(collected, vec![9.0, 7.0]); // atlas order
    }

    #[test]
    fn map_trait_section_get_and_mut() {
        use super::Map;
        let mut s = make_section();
        s.set(PointId::new(1), &[1.0, 2.0]);
        s.set(PointId::new(2), &[3.5]);
        // get == restrict
        assert_eq!(
            <Section<f64> as Map<f64>>::get(&s, PointId::new(1)),
            s.restrict(PointId::new(1))
        );
        // get_mut returns Some for Section
        assert!(<Section<f64> as Map<f64>>::get_mut(&mut s, PointId::new(1)).is_some());
    }

    #[test]
    fn scatter_from() {
        let mut s = make_section();
        // Initial data: [0.0, 0.0, 0.0]
        assert_eq!(s.data, &[0.0, 0.0, 0.0]);

        // Scatter in two chunks: [1.0, 2.0] and [3.5]
        s.scatter_from(&[1.0, 2.0, 3.5], &[(0, 2), (2, 1)]);

        // Resulting data: [1.0, 2.0, 3.5]
        assert_eq!(s.data, &[1.0, 2.0, 3.5]);
    }

    #[test]
    fn section_round_trip_and_scatter() {
        let mut atlas = Atlas::default();
        atlas.insert(PointId::new(1), 2);
        atlas.insert(PointId::new(2), 1);
        let mut s = Section::<f64>::new(atlas.clone());
        // set and restrict
        s.set(PointId::new(1), &[1.1, 2.2]);
        s.set(PointId::new(2), &[3.3]);
        assert_eq!(s.restrict(PointId::new(1)), &[1.1, 2.2]);
        assert_eq!(s.restrict(PointId::new(2)), &[3.3]);
        // scatter_from
        let mut s2 = Section::<f64>::new(atlas);
        s2.scatter_from(&[1.1, 2.2, 3.3], &[(0, 2), (2, 1)]);
        assert_eq!(s2.restrict(PointId::new(1)), &[1.1, 2.2]);
        assert_eq!(s2.restrict(PointId::new(2)), &[3.3]);
    }

    #[test]
    fn section_map_trait_and_readonly() {
        use super::Map;
        let mut atlas = Atlas::default();
        atlas.insert(PointId::new(1), 1);
        let mut s = Section::<i32>::new(atlas);
        s.set(PointId::new(1), &[42]);
        // Map trait get
        assert_eq!(<Section<i32> as Map<i32>>::get(&s, PointId::new(1)), &[42]);
        // Map trait get_mut
        assert!(<Section<i32> as Map<i32>>::get_mut(&mut s, PointId::new(1)).is_some());
    }
}

pub use crate::data::refine::Sifter;
