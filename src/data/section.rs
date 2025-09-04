//! Section: Field data storage over a topology atlas.
//!
//! The `Section<V>` type couples an `Atlas` (mapping points to slices in a
//! contiguous array) with a `Vec<V>` to hold the actual data. It provides
//! methods for inserting, accessing, and iterating per-point data slices.

use crate::data::atlas::Atlas;
use crate::mesh_error::MeshSieveError;
use crate::topology::point::PointId;
use crate::topology::cache::InvalidateCache;
use crate::data::refine::delta::SliceDelta;

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
        let data = vec![V::default(); atlas.total_len()];
        Section { atlas, data }
    }

    /// Read-only view of the data slice for a given point `p`.
    ///
    /// # Errors
    /// Returns `Err(PointNotInAtlas(p))` if the point is not registered in the atlas,
    /// or `Err(MissingSectionPoint(p))` if the data buffer is inconsistent.
    pub fn try_restrict(&self, p: PointId) -> Result<&[V], MeshSieveError> {
        let (offset, len) = self.atlas.get(p).ok_or(MeshSieveError::PointNotInAtlas(p))?;
        self.data.get(offset..offset+len).ok_or(MeshSieveError::MissingSectionPoint(p))
    }
    #[deprecated(note = "Use try_restrict which returns Result instead of panicking")]
    pub fn restrict(&self, p: PointId) -> &[V] {
        self.try_restrict(p).unwrap()
    }

    /// Mutable view of the data slice for a given point `p`.
    ///
    /// # Errors
    /// Returns `Err(PointNotInAtlas(p))` if the point is not registered in the atlas,
    /// or `Err(MissingSectionPoint(p))` if the data buffer is inconsistent.
    pub fn try_restrict_mut(&mut self, p: PointId) -> Result<&mut [V], MeshSieveError> {
        let (offset, len) = self.atlas.get(p).ok_or(MeshSieveError::PointNotInAtlas(p))?;
        self.data.get_mut(offset..offset+len).ok_or(MeshSieveError::MissingSectionPoint(p))
    }
    #[deprecated(note = "Use try_restrict_mut which returns Result instead of panicking")]
    pub fn restrict_mut(&mut self, p: PointId) -> &mut [V] {
        self.try_restrict_mut(p).unwrap()
    }

    /// Overwrite the data slice at point `p` with the values in `val`.
    ///
    /// # Errors
    /// Returns `Err(PointNotInAtlas(p))` if the point is not registered in the atlas,
    /// or `Err(SliceLengthMismatch)` if the input slice length does not match the expected length.
    pub fn try_set(&mut self, p: PointId, val: &[V]) -> Result<(), MeshSieveError> {
        let target = self.try_restrict_mut(p)?;
        let expected = target.len();
        let found = val.len();
        if expected != found {
            return Err(MeshSieveError::SliceLengthMismatch { point: p, expected, found });
        }
        target.clone_from_slice(val);
        crate::topology::cache::InvalidateCache::invalidate_cache(self);
        Ok(())
    }
    #[deprecated(note = "Use try_set which returns Result instead of panicking")]
    pub fn set(&mut self, p: PointId, val: &[V]) {
        self.try_set(p, val).unwrap()
    }

    /// Iterate over `(PointId, &[V])` for all points in atlas order.
    pub fn iter(&self) -> impl Iterator<Item = (PointId, &[V])> + '_ {
        self.atlas.points().filter_map(move |pid| self.try_restrict(pid).ok().map(|sl| (pid, sl)))
    }

    /// Add a new point to the section, resizing data as needed.
    ///
    /// # Errors
    /// Returns `Err(AtlasInsertionFailed)` if the atlas insertion fails.
    pub fn try_add_point(&mut self, p: PointId, len: usize) -> Result<(), MeshSieveError> {
        self.atlas.try_insert(p, len).map_err(|e| MeshSieveError::AtlasInsertionFailed(p, Box::new(e)))?;
        self.data.resize(self.atlas.total_len(), V::default());
        crate::topology::cache::InvalidateCache::invalidate_cache(self);
        Ok(())
    }

    /// Remove a point from the section, rebuilding data to keep slices contiguous.
    ///
    /// # Errors
    /// Returns `Err(MissingSectionPoint)` if a point is missing from the old atlas or data.
    pub fn try_remove_point(&mut self, p: PointId) -> Result<(), MeshSieveError> {
        let old_atlas = self.atlas.clone();
        self.atlas.remove_point(p)?;
        let mut new_data = Vec::with_capacity(self.atlas.total_len());
        for pid in self.atlas.points() {
            let (old_offset, old_len) = old_atlas.get(pid).ok_or(MeshSieveError::MissingSectionPoint(pid))?;
            let old_slice = self.data.get(old_offset..old_offset+old_len).ok_or(MeshSieveError::MissingSectionPoint(pid))?;
            new_data.extend_from_slice(old_slice);
        }
        self.data = new_data;
        crate::topology::cache::InvalidateCache::invalidate_cache(self);
        Ok(())
    }

    /// Apply a delta from `src_point` → `dst_point` directly within the section buffer.
    ///
    /// If the underlying slices do not overlap, the delta is applied without any
    /// additional allocation. Otherwise, the source slice is first copied into a
    /// temporary buffer to maintain aliasing safety.
    ///
    /// # Errors
    /// - [`MeshSieveError::PointNotInAtlas`] if either point is missing.
    /// - [`MeshSieveError::SliceLengthMismatch`] if the slice lengths differ.
    pub fn try_apply_delta_between_points<D: SliceDelta<V>>(
        &mut self,
        src_point: PointId,
        dst_point: PointId,
        delta: &D,
    ) -> Result<(), MeshSieveError> {
        use MeshSieveError::*;

        let (soff, slen) = self
            .atlas
            .get(src_point)
            .ok_or(PointNotInAtlas(src_point))?;
        let (doff, dlen) = self
            .atlas
            .get(dst_point)
            .ok_or(PointNotInAtlas(dst_point))?;

        if slen != dlen {
            return Err(SliceLengthMismatch {
                point: dst_point,
                expected: slen,
                found: dlen,
            });
        }
        if slen == 0 {
            return Ok(());
        }

        let disjoint = soff + slen <= doff || doff + dlen <= soff;

        if disjoint {
            let data = &mut self.data;
            if soff < doff {
                let (a, b) = data.split_at_mut(doff);
                let src = &a[soff..soff + slen];
                let dst = &mut b[0..dlen];
                delta.apply(src, dst)?;
            } else {
                let (a, b) = data.split_at_mut(soff);
                let dst = &mut a[doff..doff + dlen];
                let src = &b[0..slen];
                delta.apply(src, dst)?;
            }
        } else {
            let src_copy: Vec<V> = self.data[soff..soff + slen].to_vec();
            let dst = &mut self.data[doff..doff + dlen];
            delta.apply(&src_copy, dst)?;
        }

        crate::topology::cache::InvalidateCache::invalidate_cache(self);
        #[cfg(debug_assertions)]
        self.debug_assert_invariants();
        Ok(())
    }
}

impl<V: Clone + Send> Section<V> {
    /// Scatter values from an external buffer `other` into this section.
    ///
    /// # Errors
    /// Returns `Err(ScatterLengthMismatch)` if the input length does not match expected,
    /// or `Err(ScatterChunkMismatch)` if a chunk is out of bounds.
    pub fn try_scatter_from(&mut self, other: &[V], atlas_map: &[(usize, usize)]) -> Result<(), MeshSieveError> {
        let total_expected: usize = atlas_map.iter().map(|&(_, l)| l).sum();
        let found = other.len();
        if total_expected != found {
            return Err(MeshSieveError::ScatterLengthMismatch { expected: total_expected, found });
        }
        let mut start = 0;
        for &(offset, len) in atlas_map {
            let end = start + len;
            let chunk = other.get(start..end).ok_or(MeshSieveError::ScatterChunkMismatch { offset: start, len })?;
            let dest = self.data.get_mut(offset..offset+len).ok_or(MeshSieveError::ScatterChunkMismatch { offset, len })?;
            dest.clone_from_slice(chunk);
            start = end;
        }
        crate::topology::cache::InvalidateCache::invalidate_cache(self);
        Ok(())
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
        self.try_restrict(p).unwrap_or_else(|e| panic!("Section::get: {:?}", e))
    }

    fn get_mut(&mut self, p: PointId) -> Option<&mut [V]> {
        // Use the try_restrict_mut method to get a mutable slice, wrapped in Some.
        Some(self.try_restrict_mut(p).unwrap_or_else(|e| panic!("Section::get_mut: {:?}", e)))
    }
}

impl<V> InvalidateCache for Section<V> {
    fn invalidate_cache(&mut self) {
        // If you ever cache anything derived from atlas/data, clear it here.
    }
}

#[cfg(any(debug_assertions, feature = "strict-invariants"))]
impl<V> Section<V> {
    pub(crate) fn debug_assert_invariants(&self) {
        debug_assert_eq!(
            self.atlas.total_len(),
            self.data.len(),
            "section data length does not match atlas total"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::atlas::Atlas;
    use crate::topology::point::PointId;

    fn make_section() -> Section<f64> {
        let mut atlas = Atlas::default();
        atlas.try_insert(PointId::new(1).unwrap(), 2).unwrap(); // 2 dof
        atlas.try_insert(PointId::new(2).unwrap(), 1).unwrap();
        Section::<f64>::new(atlas)
    }

    #[test]
    fn restrict_and_set() {
        let mut s = make_section();
        s.try_set(PointId::new(1).unwrap(), &[1.0, 2.0]).unwrap();
        s.try_set(PointId::new(2).unwrap(), &[3.5]).unwrap();

        assert_eq!(s.try_restrict(PointId::new(1).unwrap()).unwrap(), &[1.0, 2.0]);
        assert_eq!(s.try_restrict(PointId::new(2).unwrap()).unwrap(), &[3.5]);
    }

    #[test]
    fn iter_order() {
        let mut s = make_section();
        s.try_set(PointId::new(1).unwrap(), &[9.0, 8.0]).unwrap();
        s.try_set(PointId::new(2).unwrap(), &[7.0]).unwrap();

        let collected: Vec<_> = s.iter().map(|(_, sl)| sl[0]).collect();
        assert_eq!(collected, vec![9.0, 7.0]); // atlas order
    }

    #[test]
    fn map_trait_section_get_and_mut() {
        use super::Map;
        let mut s = make_section();
        s.try_set(PointId::new(1).unwrap(), &[1.0, 2.0]).unwrap();
        s.try_set(PointId::new(2).unwrap(), &[3.5]).unwrap();
        // get == restrict
        assert_eq!(
            <Section<f64> as Map<f64>>::get(&s, PointId::new(1).unwrap()),
            s.try_restrict(PointId::new(1).unwrap()).unwrap()
        );
        // get_mut returns Some for Section
        assert!(<Section<f64> as Map<f64>>::get_mut(&mut s, PointId::new(1).unwrap()).is_some());
    }

    #[test]
    fn scatter_from() {
        let mut s = make_section();
        // Initial data: [0.0, 0.0, 0.0]
        assert_eq!(s.data, &[0.0, 0.0, 0.0]);

        // Scatter in two chunks: [1.0, 2.0] and [3.5]
        let _ = s.try_scatter_from(&[1.0, 2.0, 3.5], &[(0, 2), (2, 1)]);

        // Resulting data: [1.0, 2.0, 3.5]
        assert_eq!(s.data, &[1.0, 2.0, 3.5]);
    }

    #[test]
    fn section_round_trip_and_scatter() {
        let mut atlas = Atlas::default();
        atlas.try_insert(PointId::new(1).unwrap(), 2).unwrap();
        atlas.try_insert(PointId::new(2).unwrap(), 1).unwrap();
        let mut s = Section::<f64>::new(atlas.clone());
        // set and restrict
        s.try_set(PointId::new(1).unwrap(), &[1.1, 2.2]).unwrap();
        s.try_set(PointId::new(2).unwrap(), &[3.3]).unwrap();
        assert_eq!(s.try_restrict(PointId::new(1).unwrap()).unwrap(), &[1.1, 2.2]);
        assert_eq!(s.try_restrict(PointId::new(2).unwrap()).unwrap(), &[3.3]);
        // scatter_from
        let mut s2 = Section::<f64>::new(atlas);
        s2.try_scatter_from(&[1.1, 2.2, 3.3], &[(0, 2), (2, 1)]).unwrap();
        assert_eq!(s2.try_restrict(PointId::new(1).unwrap()).unwrap(), &[1.1, 2.2]);
        assert_eq!(s2.try_restrict(PointId::new(2).unwrap()).unwrap(), &[3.3]);
    }

    #[test]
    fn section_map_trait_and_readonly() {
        use super::Map;
        let mut atlas = Atlas::default();
        atlas.try_insert(PointId::new(1).unwrap(), 1).unwrap();
        let mut s = Section::<i32>::new(atlas);
        s.try_set(PointId::new(1).unwrap(), &[42]).unwrap();
        // Map trait get
        assert_eq!(<Section<i32> as Map<i32>>::get(&s, PointId::new(1).unwrap()), &[42]);
        // Map trait get_mut
        assert!(<Section<i32> as Map<i32>>::get_mut(&mut s, PointId::new(1).unwrap()).is_some());
    }

    #[test]
    fn add_point_expands_and_defaults() {
        let mut atlas = Atlas::default();
        atlas.try_insert(PointId::new(1).unwrap(), 2).unwrap();
        let mut s = Section::<i32>::new(atlas.clone());
        // initial capacity = 2
        assert_eq!(s.iter().count(), 1);
        // add a new point of length 3
        s.try_add_point(PointId::new(2).unwrap(), 3).unwrap();
        // now iter() yields 2 points
        let mut pts: Vec<_> = s.iter().map(|(p, _)| p).collect();
        pts.sort_unstable();
        assert_eq!(pts, vec![PointId::new(1).unwrap(), PointId::new(2).unwrap()]);
        // its slice is all default (0)
        assert_eq!(s.try_restrict(PointId::new(2).unwrap()).unwrap(), &[0, 0, 0]);
        // setting and reading works
        s.try_set(PointId::new(2).unwrap(), &[7,8,9]).unwrap();
        assert_eq!(s.try_restrict(PointId::new(2).unwrap()).unwrap(), &[7,8,9]);
    }

    #[test]
    fn remove_point_compacts_and_forgets() {
        // build atlas with 3 points, lengths [2,1,2]
        let mut atlas = Atlas::default();
        atlas.try_insert(PointId::new(1).unwrap(), 2).unwrap();
        atlas.try_insert(PointId::new(2).unwrap(), 1).unwrap();
        atlas.try_insert(PointId::new(3).unwrap(), 2).unwrap();
        let mut s = Section::<i32>::new(atlas);
        // set some dummy values
        s.try_set(PointId::new(1).unwrap(), &[10,11]).unwrap();
        s.try_set(PointId::new(2).unwrap(), &[22]).unwrap();
        s.try_set(PointId::new(3).unwrap(), &[33,34]).unwrap();
        // remove the middle point
        let _ = s.try_remove_point(PointId::new(2).unwrap());
        // now only 1 and 3 remain, in order
        let pts: Vec<_> = s.iter().map(|(p, _)| p).collect();
        assert_eq!(pts, vec![PointId::new(1).unwrap(), PointId::new(3).unwrap()]);
        // data buffer should be [10,11,33,34]
        let all: Vec<_> = s.data.iter().copied().collect();
        assert_eq!(all, vec![10,11,33,34]);
        // restricting the removed point panics
        std::panic::catch_unwind(|| { let _ = s.try_restrict(PointId::new(2).unwrap()).unwrap(); }).expect_err("should panic");
    }

    #[test]
    #[should_panic(expected = "called `Result::unwrap()` on an `Err` value")]
    fn restrict_missing_panics() {
        let s = make_section();
        let _ = s.try_restrict(PointId::new(99).unwrap()).unwrap();
    }

    #[test]
    fn set_wrong_length_returns_err() {
        let mut s = make_section();
        let err = s.try_set(PointId::new(1).unwrap(), &[1.0]).unwrap_err();
        assert_eq!(err, MeshSieveError::SliceLengthMismatch {
            point: PointId::new(1).unwrap(),
            expected: 2,
            found: 1
        });
    }

    #[test]
    fn invalidate_cache_noop() {
        let mut s = make_section();
        // Just ensure this compiles and does nothing
        InvalidateCache::invalidate_cache(&mut s);
    }
}

pub use crate::data::refine::Sifter;
