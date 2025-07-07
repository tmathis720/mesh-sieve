//! Atlas: Mapping mesh points to contiguous slices in a global data array.
//!
//! The `Atlas` struct provides a bijective mapping between topological
//! points (`PointId`) and sub-slices of a flat data buffer. This is useful
//! for packing degrees‐of‐freedom (DOFs) or other per‐point data into a
//! single contiguous `Vec` for efficient storage and communication.

use crate::mesh_error::MeshSieveError;
use crate::topology::point::PointId;
use crate::topology::stratum::InvalidateCache;
use std::collections::HashMap;

/// `Atlas` maintains:
/// - a lookup `map` from each `PointId` to its `(offset, len)` in the
///   global data buffer,
/// - an `order` vector to preserve insertion order for deterministic I/O,
/// - and `total_len` to track the next free offset.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Atlas {
    /// Maps each point to its slice descriptor: (starting offset, length).
    map: HashMap<PointId, (usize, usize)>,
    /// Keeps track of insertion order of points for ordered iteration.
    order: Vec<PointId>,
    /// Total length of all slices; also next available offset.
    total_len: usize,
}

impl InvalidateCache for Atlas {
    fn invalidate_cache(&mut self) {
        // Atlas itself does not cache derived structures, but if you add any, clear them here.
    }
}

impl Atlas {
    /// Insert a brand-new point `p` with a slice of length `len`.
    ///
    /// Returns the starting `offset` of this point’s slice in the
    /// underlying data buffer.
    ///
    /// # Errors
    /// Returns `Err(ZeroLengthSlice)` if `len == 0`,
    /// or `Err(DuplicatePoint(p))` if `p` was already present.
    ///
    /// # Example
    /// ```rust
    /// # fn try_main() -> Result<(), mesh_sieve::mesh_error::MeshSieveError> {
    /// use mesh_sieve::data::atlas::Atlas;
    /// use mesh_sieve::topology::point::PointId;
    /// let mut atlas = Atlas::default();
    /// let p = PointId::new(7)?;
    /// let offset = atlas.try_insert(p, 3)?;
    /// assert_eq!(offset, 0);
    /// assert_eq!(atlas.total_len(), 3);
    /// # Ok(())
    /// # }
    /// ```
    pub fn try_insert(&mut self, p: PointId, len: usize) -> Result<usize, MeshSieveError> {
        if len == 0 {
            return Err(MeshSieveError::ZeroLengthSlice);
        }
        if self.map.contains_key(&p) {
            return Err(MeshSieveError::DuplicatePoint(p));
        }
        let offset = self.total_len;
        self.map.insert(p, (offset, len));
        self.order.push(p);
        self.total_len += len;
        InvalidateCache::invalidate_cache(self);
        Ok(offset)
    }
    #[deprecated(note = "Use `try_insert` which returns Result instead of panicking")]
    pub fn insert(&mut self, p: PointId, len: usize) -> usize {
        self.try_insert(p, len).expect("Atlas::insert panicked; use try_insert")
    }

    /// Look up the slice descriptor `(offset, len)` for point `p`.
    ///
    /// Returns `Some((offset,len))` if `p` was previously inserted,
    /// or `None` otherwise.
    #[inline]
    pub fn get(&self, p: PointId) -> Option<(usize, usize)> {
        self.map.get(&p).copied()
    }

    /// Total length of all registered slices.
    ///
    /// This is equal to the sum of lengths of each point’s slice,
    /// and is the size of the global data buffer needed.
    #[inline]
    pub fn total_len(&self) -> usize {
        self.total_len
    }

    /// Iterator over all registered points in insertion (deterministic) order.
    ///
    /// Useful for serializing or iterating through slices in a stable order.
    #[inline]
    pub fn points<'a>(&'a self) -> impl Iterator<Item = PointId> + 'a {
        self.order.iter().copied()
    }

    /// Remove a point and its slice from the atlas, recomputing all offsets.
    pub fn remove_point(&mut self, p: PointId) -> Result<(), MeshSieveError> {
        let _ = self.map.remove(&p);
        self.order.retain(|&x| x != p);
        // Compute new offsets for all remaining points
        let mut next_offset = 0;
        let mut new_offsets = Vec::with_capacity(self.order.len());
        for &pt in &self.order {
            let len = match self.map.get(&pt) {
                Some(&(_, len)) => len,
                None => {
                    return Err(MeshSieveError::MissingAtlasPoint(pt));
                }
            };
            new_offsets.push((pt, next_offset, len));
            next_offset += len;
        }
        // Update map with new offsets
        for (pt, offset, len) in new_offsets {
            self.map.insert(pt, (offset, len));
        }
        self.total_len = next_offset;
        InvalidateCache::invalidate_cache(self);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::point::PointId;

    #[test]
    fn insert_and_lookup() {
        let mut a = Atlas::default();
        let p1 = PointId::new(1).unwrap();
        let off1 = a.try_insert(p1, 3);
        assert_eq!(off1.unwrap(), 0);
        let p2 = PointId::new(2).unwrap();
        let off2 = a.try_insert(p2, 5);
        assert_eq!(off2.unwrap(), 3);

        assert_eq!(a.get(p1), Some((0, 3)));
        assert_eq!(a.get(p2), Some((3, 5)));
        assert_eq!(a.total_len(), 8);
        assert_eq!(a.points().collect::<Vec<_>>(), vec![p1, p2]);
    }

    #[test]
    fn zero_len_rejected() {
        let mut a = Atlas::default();
        assert_eq!(
            a.try_insert(PointId::new(7).unwrap(), 0).unwrap_err(),
            MeshSieveError::ZeroLengthSlice
        );
    }

    #[test]
    fn atlas_cache_cleared_on_insert() {
        use crate::topology::stratum::InvalidateCache;
        use crate::topology::point::PointId;
        let mut atlas = Atlas::default();
        let _ = atlas.try_insert(PointId::new(1).unwrap(), 2);
        InvalidateCache::invalidate_cache(&mut atlas); // Should be a no-op
        let _ = atlas.try_insert(PointId::new(2).unwrap(), 1);
        // No panic, and points are present
        assert_eq!(atlas.get(PointId::new(1).unwrap()), Some((0, 2)));
        assert_eq!(atlas.get(PointId::new(2).unwrap()), Some((2, 1)));
    }

    #[test]
    fn remove_point_recomputes_offsets() {
        let mut a = Atlas::default();
        let p1 = PointId::new(1).unwrap();
        let p2 = PointId::new(2).unwrap();
        let p3 = PointId::new(3).unwrap();
        let _ = a.try_insert(p1, 3);
        let _ = a.try_insert(p2, 5);
        let _ = a.try_insert(p3, 2);
        a.remove_point(p2).unwrap();
        // p1 and p3 remain, with p3's offset updated
        assert_eq!(a.get(p1), Some((0, 3)));
        assert_eq!(a.get(p3), Some((3, 2)));
        assert_eq!(a.total_len(), 5);
        assert_eq!(a.points().collect::<Vec<_>>(), vec![p1, p3]);
    }

    #[test]
    fn duplicate_insert_panics() {
        let mut a = Atlas::default();
        let p = PointId::new(42).unwrap();
        let _ = a.try_insert(p, 1);
        // inserting the same point again must panic
        assert_eq!(a.try_insert(p, 2), Err(MeshSieveError::DuplicatePoint(p)));
    }

    #[test]
    fn get_missing_point_returns_none() {
        let a = Atlas::default();
        let p = PointId::new(99).unwrap();
        assert_eq!(a.get(p), None);
        assert_eq!(a.total_len(), 0);
        assert!(a.points().next().is_none());
    }

    #[test]
    fn remove_nonexistent_point_is_noop() {
        let mut a = Atlas::default();
        let p1 = PointId::new(1).unwrap();
        let _ = a.try_insert(p1, 3);
        let pre_len = a.total_len();
        a.remove_point(PointId::new(999).unwrap()).unwrap(); // does not exist
        // should be unchanged
        assert_eq!(a.total_len(), pre_len);
        assert_eq!(a.get(p1), Some((0,3)));
    }

    #[test]
    fn remove_first_and_last_points() {
        let mut a = Atlas::default();
        let p1 = PointId::new(1).unwrap();
        let p2 = PointId::new(2).unwrap();
        let p3 = PointId::new(3).unwrap();
        a.try_insert(p1, 2).unwrap();
        a.try_insert(p2, 4).unwrap();
        a.try_insert(p3, 1).unwrap();
        // remove first
        a.remove_point(p1).unwrap();
        assert_eq!(a.points().collect::<Vec<_>>(), vec![p2, p3]);
        assert_eq!(a.get(p2), Some((0,4)));
        assert_eq!(a.get(p3), Some((4,1)));
        // remove last
        a.remove_point(p3).unwrap();
        assert_eq!(a.points().collect::<Vec<_>>(), vec![p2]);
        assert_eq!(a.get(p2), Some((0,4)));
        assert_eq!(a.total_len(), 4);
    }

    #[test]
    fn clear_all_then_reinsert() {
        let mut a = Atlas::default();
        let pts = [PointId::new(10).unwrap(), PointId::new(20).unwrap()];
        for &p in &pts {
            a.try_insert(p, 5).unwrap();
        }
        for &p in &pts {
            a.remove_point(p).unwrap();
        }
        // now empty
        assert!(a.points().next().is_none());
        assert_eq!(a.total_len(), 0);
        // reinsert anew
        let off = a.try_insert(PointId::new(30).unwrap(), 7).unwrap();
        assert_eq!(off, 0);
        assert_eq!(a.points().collect::<Vec<_>>(), vec![PointId::new(30).unwrap()]);
    }

    #[test]
    fn serde_roundtrip() {
        let mut a = Atlas::default();
        a.try_insert(PointId::new(5).unwrap(), 3).unwrap();
        a.try_insert(PointId::new(6).unwrap(), 2).unwrap();
        let ser = serde_json::to_string(&a).expect("serialize");
        let de: Atlas = serde_json::from_str(&ser).expect("deserialize");
        assert_eq!(de.get(PointId::new(5).unwrap()), Some((0,3)));
        assert_eq!(de.get(PointId::new(6).unwrap()), Some((3,2)));
        assert_eq!(de.points().collect::<Vec<_>>(), vec![PointId::new(5).unwrap(), PointId::new(6).unwrap()]);
    }
}
