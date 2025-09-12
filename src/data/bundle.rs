//! Bundle: Combines mesh topology, DOF storage, and data transfer rules.
//!
//! A `Bundle` ties together:
//! 1. A **vertical stack** of mesh points → DOF points (`stack`),  
//! 2. A **field section** storing per-point data (`section`),  
//! 3. A **delta** strategy (`delta`) for refining/assembling data.
//!
//! This abstraction supports push (refine) and pull (assemble) of data
//! across mesh hierarchy levels, as described in Knepley & Karpeev (2009).

#[allow(unused_imports)]
use crate::data::refine::delta::SliceDelta;
use crate::data::section::Section;
use crate::data::storage::{Storage, VecStorage};
use crate::overlap::delta::CopyDelta;
use crate::topology::point::PointId;
use crate::topology::sieve::Sieve;
use crate::topology::stack::{InMemoryStack, Stack};
use core::marker::PhantomData;

/// Reducer combining multiple cap slices into an accumulator slice.
///
/// Implementors supply zero-initialization, per-slice accumulation, and an
/// optional finalize step (e.g. for averaging). Callers are expected to ensure
/// all slices share the same length before invoking [`SliceReducer::accumulate`].
pub trait SliceReducer<V>: Sync {
    /// Create a zero-initialized accumulator of length `len`.
    fn make_zero(&self, len: usize) -> Vec<V>;

    /// Accumulate `src` into `acc` element-wise.
    ///
    /// # Panics
    /// Implementations may assume `acc.len() == src.len()`. Callers are
    /// responsible for validating slice lengths before invoking this method.
    fn accumulate(&self, acc: &mut [V], src: &[V])
    -> Result<(), crate::mesh_error::MeshSieveError>;

    /// Optional finalize step once all sources have been accumulated.
    fn finalize(
        &self,
        _acc: &mut [V],
        _count: usize,
    ) -> Result<(), crate::mesh_error::MeshSieveError> {
        Ok(())
    }
}

/// Element-wise averaging reducer used by [`Bundle::assemble`].
///
/// # Preconditions
/// Callers must ensure input slices have identical lengths before invoking
/// [`SliceReducer::accumulate`]; see [`Bundle::assemble_with`].
#[derive(Copy, Clone, Debug, Default)]
pub struct AverageReducer;

impl<V> SliceReducer<V> for AverageReducer
where
    V: Clone
        + Default
        + num_traits::FromPrimitive
        + core::ops::AddAssign
        + core::ops::Div<Output = V>,
{
    fn make_zero(&self, len: usize) -> Vec<V> {
        vec![V::default(); len]
    }

    fn accumulate(
        &self,
        acc: &mut [V],
        src: &[V],
    ) -> Result<(), crate::mesh_error::MeshSieveError> {
        use crate::mesh_error::MeshSieveError;
        if acc.len() != src.len() {
            return Err(MeshSieveError::ReducerLengthMismatch {
                expected: acc.len(),
                found: src.len(),
            });
        }
        for (dst, s) in acc.iter_mut().zip(src.iter()) {
            *dst += s.clone();
        }
        Ok(())
    }

    fn finalize(
        &self,
        acc: &mut [V],
        count: usize,
    ) -> Result<(), crate::mesh_error::MeshSieveError> {
        use crate::mesh_error::MeshSieveError;
        if count == 0 {
            return Ok(());
        }
        let denom: V = num_traits::FromPrimitive::from_usize(count)
            .ok_or(MeshSieveError::SievedArrayPrimitiveConversionFailure(count))?;
        for v in acc.iter_mut() {
            *v = v.clone() / denom.clone();
        }
        Ok(())
    }
}

/// `Bundle<V, S, D>` packages a mesh‐to‐DOF stack, a data section, and a `ValueDelta`-type.
///
/// - `V`: underlying data type stored at each DOF (e.g., `f64`, `i32`, …).
/// - `S`: storage backend for the section (defaults to [`VecStorage`]).
/// - `D`: overlap [`ValueDelta`](crate::overlap::delta::ValueDelta)<V> implementation guiding how
///   values are reduced/merged across parts (defaults to [`CopyDelta`]).
///   For per-slice permutation/orientation, see [`crate::data::refine::delta::SliceDelta`].
///
/// # Fields
/// - `stack`: vertical arrows from base mesh points → cap (DOF) points,
///      carrying a `Polarity` payload if needed.
/// - `section`: contiguous storage of data `V` for each point in the atlas.
/// - `delta`: rules for extracting (`restrict`) and merging (`fuse`) values.
pub struct Bundle<V, S: Storage<V> = VecStorage<V>, D = CopyDelta> {
    /// Vertical connectivity: base points → cap (DOF) points.
    pub stack: InMemoryStack<PointId, PointId, crate::topology::arrow::Polarity>,
    /// Field data storage, indexed by `PointId`.
    pub section: Section<V, S>,
    /// Delta strategy for refine/assemble operations.
    pub delta: D,
    #[doc(hidden)]
    pub _marker: PhantomData<V>,
}

impl<V, S: Storage<V>, D> Bundle<V, S, D>
where
    V: Clone + Default,
    D: crate::overlap::delta::ValueDelta<V, Part = V>,
{
    /// **Refine**: push data *down* the stack (base → cap) using per-arrow orientation.
    ///
    /// For each base point in the sieve closure of `bases`, applies the
    /// orientation delta from the base slice to each cap slice. Disjoint source
    /// and destination slices are copied without allocation; overlapping slices
    /// are temporarily buffered for safety.
    ///
    /// # Complexity
    /// **O(Σ deg(base) · k)**, where `deg(base)` is the number of cap points per base
    /// and `k` is the per-point slice length. One pass; no intermediate allocations.
    ///
    /// # Determinism
    /// Deterministic **per cap point slice**, independent of traversal order, provided
    /// the vertical mapping `base -> {caps}` has no duplicates. Polarity handling
    /// is local to each write.
    ///
    /// # Errors
    /// Propagates errors from [`Section::try_apply_delta_between_points`].
    pub fn refine(
        &mut self,
        bases: impl IntoIterator<Item = PointId>,
    ) -> Result<(), crate::mesh_error::MeshSieveError> {
        let bases_vec: Vec<PointId> = bases.into_iter().collect();
        for &b in &bases_vec {
            self.section.try_restrict(b)?;
        }
        for b in self.stack.base().closure(bases_vec) {
            for (cap, orientation) in self.stack.lift(b) {
                self.section
                    .try_apply_delta_between_points(b, cap, &orientation)?;
            }
        }
        Ok(())
    }

    /// **Assemble**: pull data *up* the stack (cap → base) using element-wise averaging.
    ///
    /// For each base point, gathers slices from all cap points and replaces the base
    /// slice with the element-wise average. Cap slices must match the base slice
    /// length.
    ///
    /// # Complexity
    /// **O(Σ deg(base) · k)**; one pass.
    ///
    /// # Determinism
    /// Deterministic; each cap contributes at most once.
    ///
    /// # Errors
    /// Returns an error if any cap slice length differs from the base slice, in
    /// which case the [`MeshSieveError::SliceLengthMismatch`] reports the
    /// offending cap `PointId`. Also propagates reducer-specific errors such as
    /// primitive conversion failures.
    ///
    /// # Behavior
    /// Validates slice lengths using the base slice as ground truth, collects all
    /// cap slices once, and reduces them element-wise into a fresh accumulator
    /// before writing the result back to the base slice.
    pub fn assemble_with<R: SliceReducer<V>>(
        &mut self,
        bases: impl IntoIterator<Item = PointId>,
        reducer: &R,
    ) -> Result<(), crate::mesh_error::MeshSieveError> {
        use crate::mesh_error::MeshSieveError;

        for b in self.stack.base().closure(bases) {
            // Stream the cap ids; avoid per-base allocation.
            let mut caps_iter = self.stack.lift(b).map(|(cap, _)| cap);

            // Empty? nothing to assemble for this base.
            let first_cap = match caps_iter.next() {
                Some(c) => c,
                None => continue,
            };

            // Base slice defines the expected length
            let base_len = self.section.try_restrict(b)?.len();

            // Initialize accumulator using the first cap slice after length validation.
            let first_slice = self.section.try_restrict(first_cap)?;
            if first_slice.len() != base_len {
                return Err(MeshSieveError::SliceLengthMismatch {
                    point: first_cap,
                    expected: base_len,
                    found: first_slice.len(),
                });
            }

            let mut acc = reducer.make_zero(base_len);
            reducer.accumulate(&mut acc, first_slice)?;
            let mut count = 1usize;

            for cap in caps_iter {
                let sl = self.section.try_restrict(cap)?;
                if sl.len() != base_len {
                    return Err(MeshSieveError::SliceLengthMismatch {
                        point: cap,
                        expected: base_len,
                        found: sl.len(),
                    });
                }
                reducer.accumulate(&mut acc, sl)?;
                count += 1;
            }

            reducer.finalize(&mut acc, count)?;
            self.section.try_set(b, &acc)?;
        }
        Ok(())
    }

    /// Backward-compatible assemble: element-wise average of cap slices.
    ///
    /// # Migration
    /// Prefer [`Bundle::assemble_with`] for explicit reduction control.
    pub fn assemble(
        &mut self,
        bases: impl IntoIterator<Item = PointId>,
    ) -> Result<(), crate::mesh_error::MeshSieveError>
    where
        V: Clone
            + Default
            + num_traits::FromPrimitive
            + std::ops::AddAssign
            + std::ops::Div<Output = V>,
    {
        self.assemble_with(bases, &AverageReducer)
    }

    /// Iterate over `(cap_point, &[V])` pairs for all DOFs attached to base `p`.
    ///
    /// # Errors
    /// Returns an error for any cap point missing in the underlying Section.
    pub fn dofs<'a>(
        &'a self,
        p: PointId,
    ) -> impl Iterator<Item = Result<(PointId, &'a [V]), crate::mesh_error::MeshSieveError>> + 'a
    {
        self.stack
            .lift(p)
            .map(move |(cap, _)| self.section.try_restrict(cap).map(|sl| (cap, sl)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::atlas::Atlas;
    use crate::data::storage::VecStorage;
    use crate::overlap::delta::CopyDelta;
    use crate::topology::arrow::Polarity;
    use core::marker::PhantomData;
    #[test]
    fn bundle_basic_refine_and_assemble() {
        let mut atlas = Atlas::default();
        atlas.try_insert(PointId::new(1).unwrap(), 1).unwrap();
        atlas.try_insert(PointId::new(2).unwrap(), 1).unwrap();
        atlas.try_insert(PointId::new(101).unwrap(), 1).unwrap(); // cap DOF for 1
        atlas.try_insert(PointId::new(102).unwrap(), 1).unwrap(); // cap DOF for 2
        let mut section = Section::<i32, VecStorage<i32>>::new(atlas.clone());
        section.try_set(PointId::new(1).unwrap(), &[10]).unwrap();
        section.try_set(PointId::new(2).unwrap(), &[20]).unwrap();
        let mut stack = InMemoryStack::<PointId, PointId, Polarity>::new();
        stack
            .base_mut()
            .unwrap()
            .add_arrow(PointId::new(1).unwrap(), PointId::new(1).unwrap(), ());
        stack
            .base_mut()
            .unwrap()
            .add_arrow(PointId::new(2).unwrap(), PointId::new(2).unwrap(), ());
        stack.cap_mut().unwrap().add_arrow(
            PointId::new(101).unwrap(),
            PointId::new(101).unwrap(),
            (),
        );
        stack.cap_mut().unwrap().add_arrow(
            PointId::new(102).unwrap(),
            PointId::new(102).unwrap(),
            (),
        );
        let _ = stack.add_arrow(
            PointId::new(1).unwrap(),
            PointId::new(101).unwrap(),
            Polarity::Forward,
        );
        let _ = stack.add_arrow(
            PointId::new(2).unwrap(),
            PointId::new(102).unwrap(),
            Polarity::Forward,
        );
        let mut bundle = Bundle {
            stack,
            section,
            delta: CopyDelta,
            _marker: PhantomData,
        };
        // Refine: push base values to cap
        bundle
            .refine([PointId::new(1).unwrap(), PointId::new(2).unwrap()])
            .unwrap();
        assert_eq!(
            bundle
                .section
                .try_restrict(PointId::new(101).unwrap())
                .unwrap(),
            &[10]
        );
        assert_eq!(
            bundle
                .section
                .try_restrict(PointId::new(102).unwrap())
                .unwrap(),
            &[20]
        );
        // Assemble: pull cap values back to base
        bundle
            .section
            .try_set(PointId::new(101).unwrap(), &[30])
            .unwrap();
        bundle
            .section
            .try_set(PointId::new(102).unwrap(), &[40])
            .unwrap();
        bundle
            .assemble([PointId::new(1).unwrap(), PointId::new(2).unwrap()])
            .unwrap();
        assert_eq!(
            bundle
                .section
                .try_restrict(PointId::new(1).unwrap())
                .unwrap(),
            &[30]
        );
        assert_eq!(
            bundle
                .section
                .try_restrict(PointId::new(2).unwrap())
                .unwrap(),
            &[40]
        );
    }
    #[test]
    fn empty_bundle_noop() {
        let atlas = Atlas::default();
        let section = Section::<i32, VecStorage<i32>>::new(atlas.clone());
        let stack = InMemoryStack::<PointId, PointId, Polarity>::new();
        let mut bundle = Bundle {
            stack,
            section,
            delta: CopyDelta,
            _marker: PhantomData,
        };
        // Should not panic, nothing to do
        bundle.refine(std::iter::empty::<PointId>()).unwrap();
        bundle.assemble(std::iter::empty::<PointId>()).unwrap();
    }

    #[test]
    fn multiple_dofs_only_first_moved() {
        let mut atlas = Atlas::default();
        atlas.try_insert(PointId::new(1).unwrap(), 2).unwrap();
        atlas.try_insert(PointId::new(101).unwrap(), 2).unwrap();
        let mut section = Section::<i32, VecStorage<i32>>::new(atlas.clone());
        section
            .try_set(PointId::new(1).unwrap(), &[10, 20])
            .unwrap();
        let mut stack = InMemoryStack::<PointId, PointId, Polarity>::new();
        stack
            .base_mut()
            .unwrap()
            .add_arrow(PointId::new(1).unwrap(), PointId::new(1).unwrap(), ());
        stack.cap_mut().unwrap().add_arrow(
            PointId::new(101).unwrap(),
            PointId::new(101).unwrap(),
            (),
        );
        let _ = stack.add_arrow(
            PointId::new(1).unwrap(),
            PointId::new(101).unwrap(),
            Polarity::Forward,
        );
        let mut bundle = Bundle {
            stack,
            section,
            delta: CopyDelta,
            _marker: PhantomData,
        };
        bundle.refine([PointId::new(1).unwrap()]).unwrap();
        let vals = bundle
            .section
            .try_restrict(PointId::new(101).unwrap())
            .unwrap();
        // Both slots should be copied
        assert_eq!(vals, &[10, 20]);
    }

    #[test]
    fn reverse_orientation_refine() {
        let mut atlas = Atlas::default();
        atlas.try_insert(PointId::new(1).unwrap(), 2).unwrap();
        atlas.try_insert(PointId::new(101).unwrap(), 2).unwrap();
        let mut section = Section::<i32, VecStorage<i32>>::new(atlas.clone());
        section.try_set(PointId::new(1).unwrap(), &[1, 2]).unwrap();
        let mut stack = InMemoryStack::<PointId, PointId, Polarity>::new();
        stack
            .base_mut()
            .unwrap()
            .add_arrow(PointId::new(1).unwrap(), PointId::new(1).unwrap(), ());
        stack.cap_mut().unwrap().add_arrow(
            PointId::new(101).unwrap(),
            PointId::new(101).unwrap(),
            (),
        );
        let _ = stack.add_arrow(
            PointId::new(1).unwrap(),
            PointId::new(101).unwrap(),
            Polarity::Reverse,
        );
        let mut bundle = Bundle {
            stack,
            section,
            delta: CopyDelta,
            _marker: PhantomData,
        };
        bundle.refine([PointId::new(1).unwrap()]).unwrap();
        // Should get reversed [2,1]
        assert_eq!(
            bundle
                .section
                .try_restrict(PointId::new(101).unwrap())
                .unwrap(),
            &[2, 1]
        );
    }

    #[test]
    fn assemble_with_add_delta() {
        use crate::overlap::delta::AddDelta;
        let mut atlas = Atlas::default();
        atlas.try_insert(PointId::new(1).unwrap(), 1).unwrap();
        atlas.try_insert(PointId::new(101).unwrap(), 1).unwrap();
        atlas.try_insert(PointId::new(102).unwrap(), 1).unwrap();
        let mut section = Section::<i32, VecStorage<i32>>::new(atlas.clone());
        section.try_set(PointId::new(101).unwrap(), &[5]).unwrap();
        section.try_set(PointId::new(102).unwrap(), &[7]).unwrap();
        let mut stack = InMemoryStack::<PointId, PointId, Polarity>::new();
        stack
            .base_mut()
            .unwrap()
            .add_arrow(PointId::new(1).unwrap(), PointId::new(1).unwrap(), ());
        stack.cap_mut().unwrap().add_arrow(
            PointId::new(101).unwrap(),
            PointId::new(101).unwrap(),
            (),
        );
        stack.cap_mut().unwrap().add_arrow(
            PointId::new(102).unwrap(),
            PointId::new(102).unwrap(),
            (),
        );
        let _ = stack.add_arrow(
            PointId::new(1).unwrap(),
            PointId::new(101).unwrap(),
            Polarity::Forward,
        );
        let _ = stack.add_arrow(
            PointId::new(1).unwrap(),
            PointId::new(102).unwrap(),
            Polarity::Forward,
        );
        let mut bundle = Bundle {
            stack,
            section,
            delta: AddDelta,
            _marker: PhantomData,
        };
        bundle.assemble([PointId::new(1).unwrap()]).unwrap();
        // base receives average (5+7)/2
        assert_eq!(
            bundle
                .section
                .try_restrict(PointId::new(1).unwrap())
                .unwrap(),
            &[6]
        );
    }

    #[test]
    fn dofs_iterator() {
        let mut atlas = Atlas::default();
        atlas.try_insert(PointId::new(1).unwrap(), 1).unwrap();
        atlas.try_insert(PointId::new(101).unwrap(), 1).unwrap();
        atlas.try_insert(PointId::new(102).unwrap(), 1).unwrap();
        let mut section = Section::<i32, VecStorage<i32>>::new(atlas.clone());
        section.try_set(PointId::new(101).unwrap(), &[8]).unwrap();
        section.try_set(PointId::new(102).unwrap(), &[9]).unwrap();
        let mut stack = InMemoryStack::<PointId, PointId, Polarity>::new();
        stack
            .base_mut()
            .unwrap()
            .add_arrow(PointId::new(1).unwrap(), PointId::new(1).unwrap(), ());
        stack.cap_mut().unwrap().add_arrow(
            PointId::new(101).unwrap(),
            PointId::new(101).unwrap(),
            (),
        );
        stack.cap_mut().unwrap().add_arrow(
            PointId::new(102).unwrap(),
            PointId::new(102).unwrap(),
            (),
        );
        let _ = stack.add_arrow(
            PointId::new(1).unwrap(),
            PointId::new(101).unwrap(),
            Polarity::Forward,
        );
        let _ = stack.add_arrow(
            PointId::new(1).unwrap(),
            PointId::new(102).unwrap(),
            Polarity::Forward,
        );
        let bundle = Bundle {
            stack,
            section,
            delta: CopyDelta,
            _marker: PhantomData,
        };
        let vec: Vec<_> = bundle.dofs(PointId::new(1).unwrap()).collect();
        let mut vec = vec.into_iter().collect::<Result<Vec<_>, _>>().unwrap();
        vec.sort_by_key(|(cap, _)| cap.get());
        assert_eq!(
            vec,
            vec![
                (PointId::new(101).unwrap(), &[8][..]),
                (PointId::new(102).unwrap(), &[9][..]),
            ]
        );
    }

    #[test]
    fn refine_unknown_base_errors() {
        let atlas = Atlas::default();
        let section = Section::<i32, VecStorage<i32>>::new(atlas.clone());
        let stack = InMemoryStack::<PointId, PointId, Polarity>::new();
        let mut bundle = Bundle {
            stack,
            section,
            delta: CopyDelta,
            _marker: PhantomData,
        };
        let err = bundle.refine([PointId::new(999).unwrap()]).unwrap_err();
        assert!(
            matches!(err, crate::mesh_error::MeshSieveError::PointNotInAtlas(pid) if pid.get() == 999)
        );
    }
}
