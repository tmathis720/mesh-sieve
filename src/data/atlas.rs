//! Atlas: Mapping mesh points to contiguous slices in a global data array.
//!
//! The `Atlas` struct provides a bijective mapping between topological
//! points (`PointId`) and sub-slices of a flat data buffer. This is useful
//! for packing degrees‐of‐freedom (DOFs) or other per‐point data into a
//! single contiguous `Vec` for efficient storage and communication.

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
    /// # Panics
    /// - if `len == 0`, since zero‐length slices are reserved/invalid.
    /// - if `p` has already been inserted.
    ///
    /// # Example
    /// ```rust,ignore
    /// # use mesh_sieve::data::atlas::Atlas;
    /// # use mesh_sieve::topology::point::PointId;
    /// let mut atlas = Atlas::default();
    /// let p = PointId::new(7);
    /// let offset = atlas.insert(p, 3);
    /// assert_eq!(offset, 0);
    /// assert_eq!(atlas.total_len(), 3);
    /// ```
    pub fn insert(&mut self, p: PointId, len: usize) -> usize {
        // Reserve length must be positive.
        assert!(len > 0, "len==0 reserved");
        // Prevent inserting the same point twice.
        assert!(!self.map.contains_key(&p), "point already present");

        // The starting offset is the current total length.
        let offset = self.total_len;

        // Record the mapping and update insertion order.
        self.map.insert(p, (offset, len));
        self.order.push(p);

        // Advance total length by this slice’s length.
        self.total_len += len;

        // Invalidate caches in any structure built on this Atlas (e.g., Section, SievedArray, etc.)
        InvalidateCache::invalidate_cache(self);

        offset
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
    pub fn remove_point(&mut self, p: PointId) {
        let _ = self.map.remove(&p);
        self.order.retain(|&x| x != p);
        // Compute new offsets for all remaining points
        let mut next_offset = 0;
        let mut new_offsets = Vec::with_capacity(self.order.len());
        for &pt in &self.order {
            let len = self.map.get(&pt).map(|&(_, len)| len).unwrap();
            new_offsets.push((pt, next_offset, len));
            next_offset += len;
        }
        // Update map with new offsets
        for (pt, offset, len) in new_offsets {
            self.map.insert(pt, (offset, len));
        }
        self.total_len = next_offset;
        InvalidateCache::invalidate_cache(self);
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
        let off1 = a.insert(p1, 3);
        assert_eq!(off1, 0);
        let p2 = PointId::new(2).unwrap();
        let off2 = a.insert(p2, 5);
        assert_eq!(off2, 3);

        assert_eq!(a.get(p1), Some((0, 3)));
        assert_eq!(a.get(p2), Some((3, 5)));
        assert_eq!(a.total_len(), 8);
        assert_eq!(a.points().collect::<Vec<_>>(), vec![p1, p2]);
    }

    #[test]
    #[should_panic]
    fn zero_len_rejected() {
        let mut a = Atlas::default();
        a.insert(PointId::new(7).unwrap(), 0);
    }

    #[test]
    fn atlas_cache_cleared_on_insert() {
        use crate::topology::stratum::InvalidateCache;
        use crate::topology::point::PointId;
        let mut atlas = Atlas::default();
        atlas.insert(PointId::new(1).unwrap(), 2);
        InvalidateCache::invalidate_cache(&mut atlas); // Should be a no-op
        atlas.insert(PointId::new(2).unwrap(), 1);
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
        a.insert(p1, 3);
        a.insert(p2, 5);
        a.insert(p3, 2);
        a.remove_point(p2);
        // p1 and p3 remain, with p3's offset updated
        assert_eq!(a.get(p1), Some((0, 3)));
        assert_eq!(a.get(p3), Some((3, 2)));
        assert_eq!(a.total_len(), 5);
        assert_eq!(a.points().collect::<Vec<_>>(), vec![p1, p3]);
    }

    #[test]
    #[should_panic(expected = "point already present")]
    fn duplicate_insert_panics() {
        let mut a = Atlas::default();
        let p = PointId::new(42).unwrap();
        a.insert(p, 1);
        // inserting the same point again must panic
        a.insert(p, 2);
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
        a.insert(p1, 3);
        let pre_len = a.total_len();
        a.remove_point(PointId::new(999).unwrap()); // does not exist
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
        a.insert(p1, 2);
        a.insert(p2, 4);
        a.insert(p3, 1);
        // remove first
        a.remove_point(p1);
        assert_eq!(a.points().collect::<Vec<_>>(), vec![p2, p3]);
        assert_eq!(a.get(p2), Some((0,4)));
        assert_eq!(a.get(p3), Some((4,1)));
        // remove last
        a.remove_point(p3);
        assert_eq!(a.points().collect::<Vec<_>>(), vec![p2]);
        assert_eq!(a.get(p2), Some((0,4)));
        assert_eq!(a.total_len(), 4);
    }

    #[test]
    fn clear_all_then_reinsert() {
        let mut a = Atlas::default();
        let pts = [PointId::new(10).unwrap(), PointId::new(20).unwrap()];
        for &p in &pts {
            a.insert(p, 5);
        }
        for &p in &pts {
            a.remove_point(p);
        }
        // now empty
        assert!(a.points().next().is_none());
        assert_eq!(a.total_len(), 0);
        // reinsert anew
        let off = a.insert(PointId::new(30).unwrap(), 7);
        assert_eq!(off, 0);
        assert_eq!(a.points().collect::<Vec<_>>(), vec![PointId::new(30).unwrap()]);
    }

    #[test]
    fn serde_roundtrip() {
        let mut a = Atlas::default();
        a.insert(PointId::new(5).unwrap(), 3);
        a.insert(PointId::new(6).unwrap(), 2);
        let ser = serde_json::to_string(&a).expect("serialize");
        let de: Atlas = serde_json::from_str(&ser).expect("deserialize");
        assert_eq!(de.get(PointId::new(5).unwrap()), Some((0,3)));
        assert_eq!(de.get(PointId::new(6).unwrap()), Some((3,2)));
        assert_eq!(de.points().collect::<Vec<_>>(), vec![PointId::new(5).unwrap(), PointId::new(6).unwrap()]);
    }
}
