//! Atlas: Mapping mesh points to contiguous slices in a global data array.
//!
//! The `Atlas` struct provides a bijective mapping between topological
//! points (`PointId`) and sub-slices of a flat data buffer. This is useful
//! for packing degrees‐of‐freedom (DOFs) or other per‐point data into a
//! single contiguous `Vec` for efficient storage and communication.

use crate::data::DebugInvariants;
use crate::mesh_error::MeshSieveError;
use crate::topology::cache::InvalidateCache;
use crate::topology::point::PointId;
use std::collections::HashMap;

/// `Atlas` maintains:
/// - a lookup `map` from each `PointId` to its `(offset, len)` in the
///   global data buffer,
/// - an `order` vector to preserve insertion order for deterministic I/O,
/// - and `total_len` to track the next free offset.
///
/// # Invariants
///
/// - Each point appears exactly once in `order`.
/// - `map` contains precisely the keys listed in `order`.
/// - Every slice has `len > 0` and `offset + len` fits in `usize`.
/// - Offsets are contiguous in insertion order and `total_len` equals the sum
///   of all lengths.
///
/// These invariants are checked after mutations in debug builds and when the
/// `check-invariants` feature is enabled. They can also be verified manually via
/// [`validate_invariants`](Self::validate_invariants).
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Atlas {
    /// Maps each point to its slice descriptor: (starting offset, length).
    map: HashMap<PointId, (usize, usize)>,
    /// Keeps track of insertion order of points for ordered iteration.
    order: Vec<PointId>,
    /// Total length of all slices; also next available offset.
    total_len: usize,
    /// Monotonic version that changes on any structural modification.
    version: u64,
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
    ///
    /// # Complexity
    /// Amortized **O(1)** for insertion; subsequent `get` is **O(1)**.
    /// Preserves **insertion order** (`order`), and `total_len` increases monotonically.
    ///
    /// # Determinism
    /// Insertion order is stable; offsets are contiguous by insertion order.
    /// See [`Atlas::remove_point`] for reindexing behavior.
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
        self.version = self.version.wrapping_add(1);
        InvalidateCache::invalidate_cache(self);
        #[cfg(any(debug_assertions, feature = "check-invariants"))]
        self.debug_assert_invariants();
        Ok(offset)
    }

    /// Look up the slice descriptor `(offset, len)` for point `p`.
    ///
    /// Returns `Some((offset,len))` if `p` was previously inserted,
    /// or `None` otherwise.
    #[inline]
    pub fn get(&self, p: PointId) -> Option<(usize, usize)> {
        self.map.get(&p).copied()
    }

    /// Returns true iff `p` is registered in the atlas.
    ///
    /// # Complexity
    /// **O(1)**.
    ///
    /// # Determinism
    /// No side effects.
    #[inline]
    pub fn contains(&self, p: PointId) -> bool {
        self.map.contains_key(&p)
    }

    /// Number of registered points.
    ///
    /// # Notes
    /// `len()` counts points, not total DOFs; the total DOF count is
    /// [`total_len`](Self::total_len).
    ///
    /// # Complexity
    /// **O(1)**.
    ///
    /// # Determinism
    /// No side effects.
    #[inline]
    pub fn len(&self) -> usize {
        debug_assert_eq!(self.order.len(), self.map.len());
        self.order.len()
    }

    /// Whether the atlas has zero points.
    ///
    /// # Complexity
    /// **O(1)**.
    ///
    /// # Determinism
    /// No side effects.
    #[inline]
    pub fn is_empty(&self) -> bool {
        debug_assert_eq!(self.order.is_empty(), self.map.is_empty());
        self.order.is_empty()
    }

    /// Total length of all registered slices.
    ///
    /// This is equal to the sum of lengths of each point’s slice,
    /// and is the size of the global data buffer needed.
    #[inline]
    pub fn total_len(&self) -> usize {
        self.total_len
    }

    /// Monotonic version that changes whenever the atlas structure changes.
    #[inline]
    pub fn version(&self) -> u64 {
        self.version
    }

    /// (offset,len) for each point in insertion order.
    #[inline]
    pub fn atlas_map(&self) -> Vec<(usize, usize)> {
        self.order.iter().map(|&p| self.map[&p]).collect()
    }

    /// (PointId,(offset,len)) for each point in insertion order.
    #[inline]
    pub fn atlas_entries(&self) -> Vec<(PointId, (usize, usize))> {
        self.order.iter().map(|&p| (p, self.map[&p])).collect()
    }

    /// Borrowing iterator over (PointId,(offset,len)) in insertion order.
    pub fn iter_entries<'a>(&'a self) -> impl Iterator<Item = (PointId, (usize, usize))> + 'a {
        self.order.iter().map(move |&p| (p, self.map[&p]))
    }

    /// Borrowing iterator over (offset,len) in insertion order.
    pub fn iter_spans<'a>(&'a self) -> impl Iterator<Item = (usize, usize)> + 'a {
        self.order.iter().map(move |&p| self.map[&p])
    }

    /// Iterator over all registered points in insertion (deterministic) order.
    ///
    /// Useful for serializing or iterating through slices in a stable order.
    #[inline]
    pub fn points<'a>(&'a self) -> impl Iterator<Item = PointId> + 'a {
        self.order.iter().copied()
    }

    /// Build a reusable scatter plan reflecting the current atlas state.
    pub fn build_scatter_plan(&self) -> crate::data::section::ScatterPlan {
        crate::data::section::ScatterPlan {
            atlas_version: self.version,
            spans: self.atlas_map(),
        }
    }

    /// Remove a point and its slice from the atlas, recomputing all offsets.
    ///
    /// # Errors
    /// Returns `Err(MeshSieveError::MissingAtlasPoint(p))` if `p` was not present.
    ///
    /// # Complexity
    /// **O(n)** to rebuild offsets; **O(n)** temporary space for the new offsets.
    /// Preserves relative order of the remaining points.
    ///
    /// # Determinism
    /// Deterministic: the resulting offsets are contiguous in the same insertion order.
    pub fn remove_point(&mut self, p: PointId) -> Result<(), MeshSieveError> {
        // Enforce presence of `p` before proceeding.
        let existed = self.map.remove(&p).is_some();
        if !existed {
            return Err(MeshSieveError::MissingAtlasPoint(p));
        }

        self.order.retain(|&x| x != p);

        // Compute new offsets for all remaining points
        let mut next_offset = 0usize;
        let mut new_offsets = Vec::with_capacity(self.order.len());
        for &pt in &self.order {
            let len = match self.map.get(&pt) {
                Some(&(_, len)) => len,
                None => {
                    // Defensive: order contains a point missing from map
                    return Err(MeshSieveError::MissingAtlasPoint(pt));
                }
            };
            new_offsets.push((pt, next_offset, len));
            next_offset += len;
        }

        // Apply new offsets
        for (pt, off, len) in new_offsets {
            self.map.insert(pt, (off, len));
        }
        self.total_len = next_offset;
        self.version = self.version.wrapping_add(1);
        InvalidateCache::invalidate_cache(self);
        #[cfg(any(debug_assertions, feature = "check-invariants"))]
        self.debug_assert_invariants();
        Ok(())
    }
}

impl DebugInvariants for Atlas {
    fn debug_assert_invariants(&self) {
        crate::data_debug_assert_ok!(self.validate_invariants(), "Atlas invalid");
    }

    fn validate_invariants(&self) -> Result<(), MeshSieveError> {
        use std::collections::HashSet;

        // 1) order is unique
        let set: HashSet<_> = self.order.iter().copied().collect();
        if set.len() != self.order.len() {
            // find duplicate deterministically
            let mut seen = HashSet::new();
            let dup = self
                .order
                .iter()
                .copied()
                .find(|p| !seen.insert(*p))
                .unwrap();
            return Err(MeshSieveError::DuplicatePoint(dup));
        }

        // 2) map.keys == order (ALWAYS check both directions)
        if let Some(&p) = self.order.iter().find(|&&p| !self.map.contains_key(&p)) {
            return Err(MeshSieveError::MissingAtlasPoint(p));
        }
        if let Some(&p) = self.map.keys().find(|p| !set.contains(p)) {
            return Err(MeshSieveError::DuplicatePoint(p));
        }

        // 3) positive lengths and checked add
        for &p in &self.order {
            let (off, len) = self.map[&p];
            if len == 0 {
                return Err(MeshSieveError::ZeroLengthSlice);
            }
            let _end = off
                .checked_add(len)
                .ok_or_else(|| MeshSieveError::ScatterChunkMismatch { offset: off, len })?;
        }

        // 4) contiguity and total_len
        let mut expected_off = 0usize;
        let mut sum = 0usize;
        for &p in &self.order {
            let (off, len) = self.map[&p];
            if off != expected_off {
                return Err(MeshSieveError::ScatterChunkMismatch { offset: off, len });
            }
            expected_off = off + len; // safe after check above
            sum = sum
                .checked_add(len)
                .ok_or_else(|| MeshSieveError::ScatterLengthMismatch {
                    expected: usize::MAX,
                    found: 0,
                })?;
        }
        if sum != self.total_len {
            return Err(MeshSieveError::ScatterLengthMismatch {
                expected: sum,
                found: self.total_len,
            });
        }

        Ok(())
    }
}

#[cfg(test)]
impl Atlas {
    /// Test helper to force the offset of a point without adjusting others.
    pub fn force_offset(&mut self, p: PointId, new_off: usize) {
        if let Some((_, len)) = self.map.get(&p).copied() {
            self.map.insert(p, (new_off, len));
        }
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
        use crate::topology::cache::InvalidateCache;
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
        assert_eq!(a.get(p2), Some((0, 4)));
        assert_eq!(a.get(p3), Some((4, 1)));
        // remove last
        a.remove_point(p3).unwrap();
        assert_eq!(a.points().collect::<Vec<_>>(), vec![p2]);
        assert_eq!(a.get(p2), Some((0, 4)));
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
        assert_eq!(
            a.points().collect::<Vec<_>>(),
            vec![PointId::new(30).unwrap()]
        );
    }

    #[test]
    fn serde_roundtrip() {
        let mut a = Atlas::default();
        a.try_insert(PointId::new(5).unwrap(), 3).unwrap();
        a.try_insert(PointId::new(6).unwrap(), 2).unwrap();
        let ser = serde_json::to_string(&a).expect("serialize");
        let de: Atlas = serde_json::from_str(&ser).expect("deserialize");
        assert_eq!(de.get(PointId::new(5).unwrap()), Some((0, 3)));
        assert_eq!(de.get(PointId::new(6).unwrap()), Some((3, 2)));
        assert_eq!(
            de.points().collect::<Vec<_>>(),
            vec![PointId::new(5).unwrap(), PointId::new(6).unwrap()]
        );
    }

    fn pid(id: u64) -> PointId {
        PointId::new(id).unwrap()
    }

    #[test]
    fn validate_fails_when_order_missing_map_key() {
        let mut a = Atlas::default();
        let p1 = pid(1);
        let p2 = pid(2);
        a.try_insert(p1, 1).unwrap();
        a.try_insert(p2, 2).unwrap();

        // Corrupt: remove p2 from order but keep in map
        a.order.retain(|&x| x != p2);

        let e = a.validate_invariants().unwrap_err();
        assert!(matches!(e, MeshSieveError::DuplicatePoint(pp) if pp == p2));
    }

    #[test]
    fn validate_fails_when_map_missing_order_key() {
        let mut a = Atlas::default();
        let p1 = pid(1);
        a.try_insert(p1, 3).unwrap();

        // Corrupt: remove p1 from map but keep in order
        a.map.remove(&p1);

        let e = a.validate_invariants().unwrap_err();
        assert!(matches!(e, MeshSieveError::MissingAtlasPoint(pp) if pp == p1));
    }

    #[test]
    fn remove_point_errors_if_absent() {
        let mut a = Atlas::default();
        let p1 = pid(1);
        let p2 = pid(2);
        a.try_insert(p1, 1).unwrap();

        let e = a.remove_point(p2).unwrap_err();
        assert!(matches!(e, MeshSieveError::MissingAtlasPoint(pp) if pp == p2));

        // Removing existing still works and reindexes
        a.remove_point(p1).unwrap();
        assert!(a.is_empty());
        assert_eq!(a.total_len(), 0);
    }
}
