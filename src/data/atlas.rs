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
    /// ```rust
    /// # use sieve_rs::data::atlas::Atlas;
    /// # use sieve_rs::topology::point::PointId;
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::point::PointId;

    #[test]
    fn insert_and_lookup() {
        let mut a = Atlas::default();
        let p1 = PointId::new(1);
        let off1 = a.insert(p1, 3);
        assert_eq!(off1, 0);
        let p2 = PointId::new(2);
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
        a.insert(PointId::new(7), 0);
    }

    #[test]
    fn atlas_cache_cleared_on_insert() {
        use crate::topology::stratum::InvalidateCache;
        use crate::topology::point::PointId;
        let mut atlas = Atlas::default();
        atlas.insert(PointId::new(1), 2);
        InvalidateCache::invalidate_cache(&mut atlas); // Should be a no-op
        atlas.insert(PointId::new(2), 1);
        // No panic, and points are present
        assert_eq!(atlas.get(PointId::new(1)), Some((0, 2)));
        assert_eq!(atlas.get(PointId::new(2)), Some((2, 1)));
    }
}
