//! Section: Field data storage over a topology atlas.
//!
//! The `Section<V>` type couples an `Atlas` (mapping points to slices in a
//! contiguous array) with a `Vec<V>` to hold the actual data. It provides
//! methods for inserting, accessing, and iterating per-point data slices.
//!
//! A legacy, infallible adapter trait `Map<V>` is available behind the
//! `map-adapter` feature. Prefer [`FallibleMap`] and `try_*` APIs.

use crate::data::DebugInvariants;
use crate::data::atlas::Atlas;
use crate::data::refine::delta::SliceDelta;
use crate::mesh_error::MeshSieveError;
use crate::topology::cache::InvalidateCache;
use crate::topology::point::PointId;

/// Precomputed plan for scattering data into a section.
#[derive(Clone, Debug)]
pub struct ScatterPlan {
    pub(crate) atlas_version: u64,
    pub(crate) spans: Vec<(usize, usize)>,
}

/// Storage for per-point field data, backed by an `Atlas`.
///
/// # Invariants
/// - `data.len()` equals the atlas' [`total_len`](Atlas::total_len).
/// - Every `(offset,len)` in the atlas falls within `data`.
///
/// These checks run after mutations in debug builds and when the
/// `check-invariants` feature is enabled. They can also be verified manually via
/// [`validate_invariants`](Self::validate_invariants).
#[derive(Clone, Debug)]
pub struct Section<V> {
    /// Atlas mapping each `PointId` to (offset, length) in `data`.
    atlas: Atlas,
    /// Contiguous storage of values for all points.
    data: Vec<V>,
}

/// Policy for handling slice length changes during atlas mutations.
#[derive(Clone, Debug)]
pub enum ResizePolicy<V> {
    /// Fill the entire new slice with `V::default()` when length changes.
    ZeroInit,
    /// Copy `min(old, new)` elements from the start; pad the tail if growing.
    PreservePrefix,
    /// Copy `min(old, new)` elements from the end; pad the head if growing.
    PreserveSuffix,
    /// Fill the slice with a provided value when (re)initializing.
    PadWith(V),
}

impl<V> Section<V> {
    /// Read-only view of the data slice for a given point `p`.
    ///
    /// # Errors
    /// Returns `Err(PointNotInAtlas(p))` if the point is not registered in the atlas,
    /// or `Err(MissingSectionPoint(p))` if the data buffer is inconsistent.
    pub fn try_restrict(&self, p: PointId) -> Result<&[V], MeshSieveError> {
        let (offset, len) = self
            .atlas
            .get(p)
            .ok_or(MeshSieveError::PointNotInAtlas(p))?;
        self.data
            .get(offset..offset + len)
            .ok_or(MeshSieveError::MissingSectionPoint(p))
    }

    /// Mutable view of the data slice for a given point `p`.
    ///
    /// # Errors
    /// Returns `Err(PointNotInAtlas(p))` if the point is not registered in the atlas,
    /// or `Err(MissingSectionPoint(p))` if the data buffer is inconsistent.
    pub fn try_restrict_mut(&mut self, p: PointId) -> Result<&mut [V], MeshSieveError> {
        let (offset, len) = self
            .atlas
            .get(p)
            .ok_or(MeshSieveError::PointNotInAtlas(p))?;
        self.data
            .get_mut(offset..offset + len)
            .ok_or(MeshSieveError::MissingSectionPoint(p))
    }

    /// Read-only handle to the backing atlas.
    ///
    /// Mutations must go through [`with_atlas_mut`](Self::with_atlas_mut)
    /// to keep `Section` and its data buffer consistent.
    ///
    /// # Complexity
    /// **O(1)**.
    #[inline]
    pub fn atlas(&self) -> &Atlas {
        &self.atlas
    }

    /// Iterate over `(PointId, &[V])` for all points in atlas order.
    pub fn iter(&self) -> impl Iterator<Item = (PointId, &[V])> + '_ {
        self.atlas
            .points()
            .filter_map(move |pid| self.try_restrict(pid).ok().map(|sl| (pid, sl)))
    }

    /// Read-only view of the entire flat buffer in insertion order.
    pub fn as_flat_slice(&self) -> &[V] {
        &self.data
    }

    /// Apply a closure to every `(PointId, &mut [V])` in insertion order.
    pub fn for_each_in_order_mut<F>(&mut self, mut f: F)
    where
        F: FnMut(PointId, &mut [V]),
    {
        #[cfg(any(debug_assertions, feature = "check-invariants"))]
        self.debug_assert_invariants();
        for pid in self.atlas.points() {
            let (off, len) = self.atlas.get(pid).expect("invariants");
            f(pid, &mut self.data[off..off + len]);
        }
        crate::topology::cache::InvalidateCache::invalidate_cache(self);
        #[cfg(any(debug_assertions, feature = "check-invariants"))]
        self.debug_assert_invariants();
    }

    /// Read-only variant of [`for_each_in_order_mut`].
    pub fn for_each_in_order<F>(&self, mut f: F)
    where
        F: FnMut(PointId, &[V]),
    {
        #[cfg(any(debug_assertions, feature = "check-invariants"))]
        self.debug_assert_invariants();
        for pid in self.atlas.points() {
            let (off, len) = self.atlas.get(pid).expect("invariants");
            f(pid, &self.data[off..off + len]);
        }
    }

    #[inline]
    fn spans_cover_contiguously(spans: &[(usize, usize)], total: usize) -> bool {
        if spans.is_empty() {
            return total == 0;
        }
        let mut expected_off = 0usize;
        for &(off, len) in spans {
            if off != expected_off {
                return false;
            }
            match expected_off.checked_add(len) {
                Some(next) => expected_off = next,
                None => return false,
            }
        }
        expected_off == total
    }
}

impl<V: Clone> Section<V> {
    /// Gather the entire section into a flat buffer in insertion order.
    pub fn gather_in_order(&self) -> Vec<V> {
        self.data.clone()
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
            return Err(MeshSieveError::SliceLengthMismatch {
                point: p,
                expected,
                found,
            });
        }
        target.clone_from_slice(val);
        crate::topology::cache::InvalidateCache::invalidate_cache(self);
        #[cfg(any(debug_assertions, feature = "check-invariants"))]
        self.debug_assert_invariants();
        Ok(())
    }

    /// Remove a point from the section, rebuilding data to keep slices contiguous.
    ///
    /// # Errors
    /// Returns `Err(MissingSectionPoint)` if a point is missing from the old atlas or data.
    ///
    /// # Complexity
    /// **O(n)** atlas reindex + **O(total_len_new)** data rebuild.
    ///
    /// # Determinism
    /// Deterministic rebuild in insertion order with no gaps.
    pub fn try_remove_point(&mut self, p: PointId) -> Result<(), MeshSieveError> {
        let old_atlas = self.atlas.clone();
        self.atlas.remove_point(p)?;
        let mut new_data = Vec::with_capacity(self.atlas.total_len());
        for pid in self.atlas.points() {
            let (old_offset, old_len) = old_atlas
                .get(pid)
                .ok_or(MeshSieveError::MissingSectionPoint(pid))?;
            let old_slice = self
                .data
                .get(old_offset..old_offset + old_len)
                .ok_or(MeshSieveError::MissingSectionPoint(pid))?;
            new_data.extend_from_slice(old_slice);
        }
        self.data = new_data;
        crate::topology::cache::InvalidateCache::invalidate_cache(self);
        #[cfg(any(debug_assertions, feature = "check-invariants"))]
        self.debug_assert_invariants();
        Ok(())
    }

    /// Apply a delta from `src_point` → `dst_point` directly within the section buffer.
    ///
    /// In debug builds or when the `check-invariants` feature is enabled, section
    /// invariants are validated before and after applying the delta. Violations
    /// panic prior to any slicing.
    ///
    /// If the underlying slices do not overlap, the delta is applied without any
    /// additional allocation. Otherwise, the source slice is first copied into a
    /// temporary buffer to maintain aliasing safety.
    ///
    /// # Errors
    /// - [`MeshSieveError::PointNotInAtlas`] if either point is missing.
    /// - [`MeshSieveError::SliceLengthMismatch`] if the slice lengths differ.
    /// - [`MeshSieveError::MissingSectionPoint`] if either slice is out of bounds.
    /// - [`MeshSieveError::ScatterChunkMismatch`] if an offset/length overflows.
    pub fn try_apply_delta_between_points<D: SliceDelta<V>>(
        &mut self,
        src_point: PointId,
        dst_point: PointId,
        delta: &D,
    ) -> Result<(), MeshSieveError> {
        use MeshSieveError::*;

        #[cfg(any(debug_assertions, feature = "check-invariants"))]
        self.debug_assert_invariants();

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

        let src_end = soff.checked_add(slen).ok_or(ScatterChunkMismatch {
            offset: soff,
            len: slen,
        })?;
        let dst_end = doff.checked_add(dlen).ok_or(ScatterChunkMismatch {
            offset: doff,
            len: dlen,
        })?;

        if src_end > self.data.len() {
            return Err(MissingSectionPoint(src_point));
        }
        if dst_end > self.data.len() {
            return Err(MissingSectionPoint(dst_point));
        }

        if slen == 0 {
            crate::topology::cache::InvalidateCache::invalidate_cache(self);
            #[cfg(any(debug_assertions, feature = "check-invariants"))]
            self.debug_assert_invariants();
            return Ok(());
        }

        let disjoint = src_end <= doff || dst_end <= soff;

        if disjoint {
            let data = &mut self.data;
            if soff < doff {
                let (a, b) = data.split_at_mut(doff);
                let src = &a[soff..src_end];
                let dst = &mut b[0..dlen];
                delta.apply(src, dst)?;
            } else {
                let (a, b) = data.split_at_mut(soff);
                let dst = &mut a[doff..dst_end];
                let src = &b[0..slen];
                delta.apply(src, dst)?;
            }
        } else {
            let src_copy: Vec<V> = self.data[soff..src_end].to_vec();
            let dst = &mut self.data[doff..dst_end];
            delta.apply(&src_copy, dst)?;
        }

        crate::topology::cache::InvalidateCache::invalidate_cache(self);
        #[cfg(any(debug_assertions, feature = "check-invariants"))]
        self.debug_assert_invariants();
        Ok(())
    }

    /// Scatter values from an external buffer `other` into this section.
    ///
    /// # Errors
    /// Returns `Err(ScatterLengthMismatch)` if the input length does not match expected,
    /// or `Err(ScatterChunkMismatch)` if a chunk is out of bounds.
    ///
    /// # Complexity
    /// **O(total_len)** to copy slices; validates bounds in **O(n)** where `n` is points.
    ///
    /// # Determinism
    /// Deterministic copy; `*_with_plan` additionally validates the plan version for safety.
    pub fn try_scatter_from(
        &mut self,
        other: &[V],
        atlas_map: &[(usize, usize)],
    ) -> Result<(), MeshSieveError> {
        if other.len() == self.data.len()
            && Self::spans_cover_contiguously(atlas_map, self.data.len())
        {
            self.data.clone_from_slice(other);
            crate::topology::cache::InvalidateCache::invalidate_cache(self);
            #[cfg(any(debug_assertions, feature = "check-invariants"))]
            self.debug_assert_invariants();
            return Ok(());
        }

        let total_expected: usize = atlas_map.iter().map(|&(_, l)| l).sum();
        let found = other.len();
        if total_expected != found {
            return Err(MeshSieveError::ScatterLengthMismatch {
                expected: total_expected,
                found,
            });
        }

        for &(offset, len) in atlas_map {
            let end = offset
                .checked_add(len)
                .ok_or(MeshSieveError::ScatterChunkMismatch { offset, len })?;
            if end > self.data.len() {
                return Err(MeshSieveError::ScatterChunkMismatch { offset, len });
            }
        }

        let mut start = 0usize;
        for &(offset, len) in atlas_map {
            let end = start + len;
            let chunk = other
                .get(start..end)
                .ok_or(MeshSieveError::ScatterChunkMismatch { offset: start, len })?;
            let dest = &mut self.data[offset..offset + len];
            dest.clone_from_slice(chunk);
            start = end;
        }

        crate::topology::cache::InvalidateCache::invalidate_cache(self);
        #[cfg(any(debug_assertions, feature = "check-invariants"))]
        self.debug_assert_invariants();
        Ok(())
    }

    /// Scatter using a precomputed plan; fails if the atlas has changed.
    ///
    /// # Complexity
    /// **O(total_len)** to copy slices; validates bounds in **O(n)** where `n` is points.
    ///
    /// # Determinism
    /// Deterministic copy; `*_with_plan` additionally validates the plan version for safety.
    pub fn try_scatter_with_plan(
        &mut self,
        buf: &[V],
        plan: &ScatterPlan,
    ) -> Result<(), MeshSieveError> {
        let cur = self.atlas.version();
        if plan.atlas_version != cur {
            return Err(MeshSieveError::AtlasPlanStale {
                expected: plan.atlas_version,
                found: cur,
            });
        }
        self.try_scatter_from(buf, &plan.spans)
    }
}

impl<V: Clone + Default> Section<V> {
    /// Construct a new `Section` given an existing `Atlas`.
    ///
    /// Initializes the data buffer with `V::default()` repeated for each
    /// degree of freedom in the atlas.
    ///
    /// # Complexity
    /// **O(total_len)** to fill with `V::default()`.
    ///
    /// # Determinism
    /// Initial layout matches the atlas’ insertion order deterministically.
    pub fn new(atlas: Atlas) -> Self {
        let data = vec![V::default(); atlas.total_len()];
        Section { atlas, data }
    }

    /// Add a new point to the section, resizing data as needed.
    ///
    /// # Errors
    /// Returns `Err(AtlasInsertionFailed)` if the atlas insertion fails.
    ///
    /// # Complexity
    /// **O(n)** atlas insertion (reindex) + **O(total_len_new)** data resize.
    ///
    /// # Determinism
    /// Data remains contiguous in **insertion order**; the new point is appended.
    pub fn try_add_point(&mut self, p: PointId, len: usize) -> Result<(), MeshSieveError> {
        self.atlas
            .try_insert(p, len)
            .map_err(|e| MeshSieveError::AtlasInsertionFailed(p, Box::new(e)))?;
        self.data.resize(self.atlas.total_len(), V::default());
        crate::topology::cache::InvalidateCache::invalidate_cache(self);
        #[cfg(any(debug_assertions, feature = "check-invariants"))]
        self.debug_assert_invariants();
        Ok(())
    }

    /// Safely mutate the atlas and rebuild `data` to remain consistent.
    ///
    /// - New points get `len` default-initialized values.
    /// - Removed points drop data.
    /// - Reordered/retuned points keep their old values (by `PointId`),
    ///   copied into the new contiguous layout.
    /// - **Length changes for existing points are rejected.** Use
    ///   [`with_atlas_resize`](Self::with_atlas_resize) to allow slice length
    ///   changes with an explicit policy.
    ///
    /// # Errors
    /// Returns [`MeshSieveError::AtlasSliceLengthChanged`] if any existing
    /// point's slice length differs after mutation.
    ///
    /// # Complexity
    /// **O(n)** mapping + **O(total_len_new)** copy/initialize.
    ///
    /// # Determinism
    /// Rebuild order follows atlas insertion order deterministically.
    pub fn with_atlas_mut<F>(&mut self, f: F) -> Result<(), MeshSieveError>
    where
        F: FnOnce(&mut Atlas),
    {
        // Snapshot current atlas to pull old spans
        let before = self.atlas.clone();

        // Let the user mutate
        f(&mut self.atlas);

        // Validate new atlas
        #[cfg(any(debug_assertions, feature = "check-invariants"))]
        self.atlas.debug_assert_invariants();

        // Gather points to avoid borrowing issues
        let new_points: Vec<_> = self.atlas.points().collect();

        // STRICT check: existing points must retain slice lengths
        use MeshSieveError::AtlasSliceLengthChanged;
        for &pid in &new_points {
            if let Some((_, len_old)) = before.get(pid) {
                let (_, len_new) = self
                    .atlas
                    .get(pid)
                    .expect("pid was just iterated from new atlas");
                if len_old != len_new {
                    self.atlas = before;
                    return Err(AtlasSliceLengthChanged {
                        point: pid,
                        old: len_old,
                        new: len_new,
                    });
                }
            }
        }

        // Rebuild data following insertion order of the new atlas
        let mut new_data = Vec::with_capacity(self.atlas.total_len());
        for pid in new_points {
            match before.get(pid) {
                // Existing point: copy old slice
                Some((off, len)) => {
                    let end = off + len;
                    let src = self
                        .data
                        .get(off..end)
                        .ok_or(MeshSieveError::MissingSectionPoint(pid))?;
                    new_data.extend_from_slice(src);
                }
                // New point: fill with defaults
                None => {
                    let (_off_new, len_new) = self
                        .atlas
                        .get(pid)
                        .ok_or(MeshSieveError::MissingAtlasPoint(pid))?;
                    new_data.extend(std::iter::repeat_with(V::default).take(len_new));
                }
            }
        }

        self.data = new_data;
        crate::topology::cache::InvalidateCache::invalidate_cache(self);
        #[cfg(any(debug_assertions, feature = "check-invariants"))]
        self.debug_assert_invariants();
        Ok(())
    }

    /// Mutate the atlas, allowing slice length changes per `policy`.
    ///
    /// Existing points preserve or initialize values according to `policy`.
    /// New points are initialized per `policy` as well. Removed points drop
    /// their data. On error, the atlas and data are rolled back to their
    /// original state.
    pub fn with_atlas_resize<F>(
        &mut self,
        policy: ResizePolicy<V>,
        f: F,
    ) -> Result<(), MeshSieveError>
    where
        F: FnOnce(&mut Atlas),
    {
        let before_atlas = self.atlas.clone();
        let before_data = self.data.clone();

        f(&mut self.atlas);

        #[cfg(any(debug_assertions, feature = "check-invariants"))]
        self.atlas.debug_assert_invariants();

        let rebuild = (|| -> Result<Vec<V>, MeshSieveError> {
            let mut new_data = Vec::with_capacity(self.atlas.total_len());
            let extend_fill = |buf: &mut Vec<V>, n: usize| match &policy {
                ResizePolicy::ZeroInit
                | ResizePolicy::PreservePrefix
                | ResizePolicy::PreserveSuffix => {
                    buf.extend(std::iter::repeat_with(V::default).take(n));
                }
                ResizePolicy::PadWith(val) => {
                    buf.extend(std::iter::repeat(val.clone()).take(n));
                }
            };

            for pid in self.atlas.points() {
                let (_off_new, new_len) = self
                    .atlas
                    .get(pid)
                    .ok_or(MeshSieveError::MissingAtlasPoint(pid))?;
                if let Some((off_old, old_len)) = before_atlas.get(pid) {
                    let src = before_data
                        .get(off_old..off_old + old_len)
                        .ok_or(MeshSieveError::MissingSectionPoint(pid))?;
                    match &policy {
                        ResizePolicy::ZeroInit => {
                            extend_fill(&mut new_data, new_len);
                        }
                        ResizePolicy::PadWith(_) => {
                            extend_fill(&mut new_data, new_len);
                        }
                        ResizePolicy::PreservePrefix => {
                            let k = core::cmp::min(old_len, new_len);
                            new_data.extend_from_slice(&src[..k]);
                            if new_len > k {
                                extend_fill(&mut new_data, new_len - k);
                            }
                        }
                        ResizePolicy::PreserveSuffix => {
                            let k = core::cmp::min(old_len, new_len);
                            if new_len > k {
                                extend_fill(&mut new_data, new_len - k);
                            }
                            new_data.extend_from_slice(&src[old_len - k..]);
                        }
                    }
                } else {
                    extend_fill(&mut new_data, new_len);
                }
            }

            Ok(new_data)
        })();

        match rebuild {
            Ok(new_data) => {
                self.data = new_data;
                crate::topology::cache::InvalidateCache::invalidate_cache(self);
                #[cfg(any(debug_assertions, feature = "check-invariants"))]
                self.debug_assert_invariants();
                Ok(())
            }
            Err(e) => {
                self.atlas = before_atlas;
                self.data = before_data;
                Err(e)
            }
        }
    }

    /// Scatter a flat buffer into the section in atlas insertion order.
    ///
    /// # Complexity
    /// **O(total_len)** to copy slices; validates bounds in **O(n)** where `n` is points.
    ///
    /// # Determinism
    /// Deterministic copy; `*_with_plan` additionally validates the plan version for safety.
    pub fn try_scatter_in_order(&mut self, buf: &[V]) -> Result<(), MeshSieveError> {
        #[cfg(any(debug_assertions, feature = "check-invariants"))]
        self.debug_assert_invariants();
        let spans = self.atlas.atlas_map();
        self.try_scatter_from(buf, &spans)
    }
}

/// Fallible read/write access to per-point slices.
pub trait FallibleMap<V> {
    /// Immutable access to `p`'s slice.
    fn try_get(&self, p: PointId) -> Result<&[V], MeshSieveError>;
    /// Mutable access to `p`'s slice.
    fn try_get_mut(&mut self, p: PointId) -> Result<&mut [V], MeshSieveError>;
}

/// Implement `FallibleMap` for `Section<V>`.
impl<V> FallibleMap<V> for Section<V> {
    #[inline]
    fn try_get(&self, p: PointId) -> Result<&[V], MeshSieveError> {
        self.try_restrict(p)
    }

    #[inline]
    fn try_get_mut(&mut self, p: PointId) -> Result<&mut [V], MeshSieveError> {
        self.try_restrict_mut(p)
    }
}

#[cfg(feature = "map-adapter")]
/// Infallible adapter for read/write access; intended for legacy code.
/// Prefer [`FallibleMap`] in new code.
///
/// # Panics
#[cfg(feature = "map-adapter")]
mod sealed {
    pub trait Sealed {}
    impl<V> Sealed for super::Section<V> {}
    impl<'a, V> Sealed for crate::data::refine::helpers::ReadOnlyMap<'a, V> {}
}

/// Infallible adapter for read/write access; intended for legacy code.
/// Prefer [`FallibleMap`] in new code.
///
/// # Panics
/// Implementations may panic if `p` is unknown.
#[cfg(feature = "map-adapter")]
#[cfg_attr(docsrs, doc(cfg(feature = "map-adapter")))]
pub trait Map<V>: sealed::Sealed {
    /// Immutable access to the data slice for `p`.
    fn get(&self, p: PointId) -> &[V];

    /// Optional mutable access to the data slice for `p`.
    ///
    /// Default implementation returns `None`, meaning the map is read-only.
    fn get_mut(&mut self, _p: PointId) -> Option<&mut [V]> {
        None
    }
}

#[cfg(feature = "map-adapter")]
impl<V> Map<V> for Section<V> {
    #[inline]
    fn get(&self, p: PointId) -> &[V] {
        self.try_restrict(p)
            .unwrap_or_else(|e| panic!("Map::get({p:?}) failed: {e}"))
    }

    #[inline]
    fn get_mut(&mut self, p: PointId) -> Option<&mut [V]> {
        Some(
            self.try_restrict_mut(p)
                .unwrap_or_else(|e| panic!("Map::get_mut({p:?}) failed: {e}")),
        )
    }
}

impl<V> DebugInvariants for Section<V> {
    fn debug_assert_invariants(&self) {
        crate::data_debug_assert_ok!(self.validate_invariants(), "Section invalid");
    }

    fn validate_invariants(&self) -> Result<(), MeshSieveError> {
        // Validate atlas first
        self.atlas.validate_invariants()?;

        if self.data.len() != self.atlas.total_len() {
            return Err(MeshSieveError::ScatterLengthMismatch {
                expected: self.atlas.total_len(),
                found: self.data.len(),
            });
        }

        for (pid, (off, len)) in self.atlas.iter_entries() {
            let end = off
                .checked_add(len)
                .ok_or_else(|| MeshSieveError::ScatterChunkMismatch { offset: off, len })?;
            if end > self.data.len() {
                return Err(MeshSieveError::MissingSectionPoint(pid));
            }
        }

        Ok(())
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

        assert_eq!(
            s.try_restrict(PointId::new(1).unwrap()).unwrap(),
            &[1.0, 2.0]
        );
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
    fn fallible_map_on_section() {
        use super::FallibleMap;
        let mut s = make_section();
        s.try_set(PointId::new(1).unwrap(), &[1.0, 2.0]).unwrap();
        // successful access
        assert_eq!(
            <Section<f64> as FallibleMap<f64>>::try_get(&s, PointId::new(1).unwrap()).unwrap(),
            &[1.0, 2.0]
        );
        // missing point yields error
        assert!(
            <Section<f64> as FallibleMap<f64>>::try_get(&s, PointId::new(99).unwrap()).is_err()
        );
        // mutable access works
        assert!(
            <Section<f64> as FallibleMap<f64>>::try_get_mut(&mut s, PointId::new(1).unwrap())
                .is_ok()
        );
    }

    #[cfg(feature = "map-adapter")]
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
        assert_eq!(
            s.try_restrict(PointId::new(1).unwrap()).unwrap(),
            &[1.1, 2.2]
        );
        assert_eq!(s.try_restrict(PointId::new(2).unwrap()).unwrap(), &[3.3]);
        // scatter_from
        let mut s2 = Section::<f64>::new(atlas);
        s2.try_scatter_from(&[1.1, 2.2, 3.3], &[(0, 2), (2, 1)])
            .unwrap();
        assert_eq!(
            s2.try_restrict(PointId::new(1).unwrap()).unwrap(),
            &[1.1, 2.2]
        );
        assert_eq!(s2.try_restrict(PointId::new(2).unwrap()).unwrap(), &[3.3]);
    }

    #[cfg(feature = "map-adapter")]
    #[test]
    fn section_map_trait_and_readonly() {
        use super::Map;
        let mut atlas = Atlas::default();
        atlas.try_insert(PointId::new(1).unwrap(), 1).unwrap();
        let mut s = Section::<i32>::new(atlas);
        s.try_set(PointId::new(1).unwrap(), &[42]).unwrap();
        // Map trait get
        assert_eq!(
            <Section<i32> as Map<i32>>::get(&s, PointId::new(1).unwrap()),
            &[42]
        );
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
        assert_eq!(
            pts,
            vec![PointId::new(1).unwrap(), PointId::new(2).unwrap()]
        );
        // its slice is all default (0)
        assert_eq!(
            s.try_restrict(PointId::new(2).unwrap()).unwrap(),
            &[0, 0, 0]
        );
        // setting and reading works
        s.try_set(PointId::new(2).unwrap(), &[7, 8, 9]).unwrap();
        assert_eq!(
            s.try_restrict(PointId::new(2).unwrap()).unwrap(),
            &[7, 8, 9]
        );
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
        s.try_set(PointId::new(1).unwrap(), &[10, 11]).unwrap();
        s.try_set(PointId::new(2).unwrap(), &[22]).unwrap();
        s.try_set(PointId::new(3).unwrap(), &[33, 34]).unwrap();
        // remove the middle point
        let _ = s.try_remove_point(PointId::new(2).unwrap());
        // now only 1 and 3 remain, in order
        let pts: Vec<_> = s.iter().map(|(p, _)| p).collect();
        assert_eq!(
            pts,
            vec![PointId::new(1).unwrap(), PointId::new(3).unwrap()]
        );
        // data buffer should be [10,11,33,34]
        let all: Vec<_> = s.data.iter().copied().collect();
        assert_eq!(all, vec![10, 11, 33, 34]);
        // restricting the removed point panics
        std::panic::catch_unwind(|| {
            let _ = s.try_restrict(PointId::new(2).unwrap()).unwrap();
        })
        .expect_err("should panic");
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
        assert_eq!(
            err,
            MeshSieveError::SliceLengthMismatch {
                point: PointId::new(1).unwrap(),
                expected: 2,
                found: 1
            }
        );
    }

    #[test]
    fn invalidate_cache_noop() {
        let mut s = make_section();
        // Just ensure this compiles and does nothing
        InvalidateCache::invalidate_cache(&mut s);
    }
}

#[cfg(test)]
mod tests_resize {
    use super::*;
    use crate::topology::point::PointId;

    fn pid(i: u64) -> PointId {
        PointId::new(i).unwrap()
    }

    #[test]
    fn strict_with_atlas_mut_rejects_len_change() {
        let mut atlas = Atlas::default();
        let p = pid(1);
        atlas.try_insert(p, 3).unwrap();
        let mut s: Section<i32> = Section::new(atlas);

        s.try_set(p, &[1, 2, 3]).unwrap();

        let err = s
            .with_atlas_mut(|a| {
                a.remove_point(p).unwrap();
                a.try_insert(p, 5).unwrap();
            })
            .unwrap_err();

        match err {
            MeshSieveError::AtlasSliceLengthChanged { point, old, new } => {
                assert_eq!(point, p);
                assert_eq!(old, 3);
                assert_eq!(new, 5);
            }
            e => panic!("unexpected error: {e:?}"),
        }

        assert_eq!(s.atlas().get(p).unwrap().1, 3);
        assert_eq!(s.try_restrict(p).unwrap(), &[1, 2, 3]);
    }

    #[test]
    fn resize_preserve_prefix_and_zero_extend() {
        let mut atlas = Atlas::default();
        let p = pid(1);
        atlas.try_insert(p, 3).unwrap();
        let mut s: Section<i32> = Section::new(atlas);
        s.try_set(p, &[10, 20, 30]).unwrap();

        s.with_atlas_resize(ResizePolicy::PreservePrefix, |a| {
            a.remove_point(p).unwrap();
            a.try_insert(p, 5).unwrap();
        })
        .unwrap();

        assert_eq!(s.try_restrict(p).unwrap(), &[10, 20, 30, 0, 0]);
    }

    #[test]
    fn resize_preserve_suffix_and_zero_prepad() {
        let mut atlas = Atlas::default();
        let p = pid(1);
        atlas.try_insert(p, 3).unwrap();
        let mut s: Section<i32> = Section::new(atlas);
        s.try_set(p, &[10, 20, 30]).unwrap();

        s.with_atlas_resize(ResizePolicy::PreserveSuffix, |a| {
            a.remove_point(p).unwrap();
            a.try_insert(p, 5).unwrap();
        })
        .unwrap();

        assert_eq!(s.try_restrict(p).unwrap(), &[0, 0, 10, 20, 30]);
    }

    #[test]
    fn resize_pad_with_custom_value() {
        let mut atlas = Atlas::default();
        let p = pid(1);
        atlas.try_insert(p, 2).unwrap();
        let mut s: Section<i32> = Section::new(atlas);
        s.try_set(p, &[7, 9]).unwrap();

        s.with_atlas_resize(ResizePolicy::PadWith(-1), |a| {
            a.remove_point(p).unwrap();
            a.try_insert(p, 4).unwrap();
        })
        .unwrap();

        assert_eq!(s.try_restrict(p).unwrap(), &[-1, -1, -1, -1]);
    }

    #[test]
    fn resize_shrink_preserve_prefix() {
        let mut atlas = Atlas::default();
        let p = pid(1);
        atlas.try_insert(p, 4).unwrap();
        let mut s: Section<i32> = Section::new(atlas);
        s.try_set(p, &[1, 2, 3, 4]).unwrap();

        s.with_atlas_resize(ResizePolicy::PreservePrefix, |a| {
            a.remove_point(p).unwrap();
            a.try_insert(p, 2).unwrap();
        })
        .unwrap();

        assert_eq!(s.try_restrict(p).unwrap(), &[1, 2]);
    }

    #[test]
    fn resize_shrink_preserve_suffix() {
        let mut atlas = Atlas::default();
        let p = pid(1);
        atlas.try_insert(p, 4).unwrap();
        let mut s: Section<i32> = Section::new(atlas);
        s.try_set(p, &[1, 2, 3, 4]).unwrap();

        s.with_atlas_resize(ResizePolicy::PreserveSuffix, |a| {
            a.remove_point(p).unwrap();
            a.try_insert(p, 2).unwrap();
        })
        .unwrap();

        assert_eq!(s.try_restrict(p).unwrap(), &[3, 4]);
    }
}

#[cfg(test)]
mod tests_invariants_precheck {
    use super::*;
    use crate::data::atlas::Atlas;
    use crate::topology::point::PointId;

    #[cfg(not(debug_assertions))]
    use crate::{mesh_error::MeshSieveError, topology::arrow::Polarity};

    fn pid(i: u64) -> PointId {
        PointId::new(i).unwrap()
    }

    #[test]
    #[cfg(not(debug_assertions))]
    fn delta_returns_missing_point_in_release_paths() {
        let mut atlas = Atlas::default();
        let p0 = pid(1);
        let p1 = pid(2);
        atlas.try_insert(p0, 2).unwrap();
        atlas.try_insert(p1, 2).unwrap();
        let mut s: Section<i32> = Section::new(atlas);

        let mut bad = s.atlas.clone();
        bad.remove_point(p1).unwrap();
        let dummy = pid(99);
        bad.try_insert(dummy, s.as_flat_slice().len() + 8).unwrap();
        bad.try_insert(p1, 2).unwrap();
        s.atlas = bad;

        let e = s
            .try_apply_delta_between_points(p0, p1, &Polarity::Forward)
            .unwrap_err();

        match e {
            MeshSieveError::MissingSectionPoint(q) => assert_eq!(q, p1),
            _ => panic!("unexpected error: {e:?}"),
        }
    }

    #[test]
    fn for_each_in_order_prechecks_in_debug() {
        let mut atlas = Atlas::default();
        let p = pid(7);
        atlas.try_insert(p, 1).unwrap();
        let mut s: Section<i32> = Section::new(atlas);

        s.data.clear();

        #[cfg(any(debug_assertions, feature = "check-invariants"))]
        {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                s.for_each_in_order(|_, _| {});
            }));
            assert!(result.is_err(), "expected invariant pre-check to panic");
        }
    }
}

pub use crate::data::refine::Sifter;
