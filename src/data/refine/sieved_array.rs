//! A generic array of values indexed by mesh points, supporting refine/assemble.
//! (Extracted from section.rs)
//!
//! This module provides [`SievedArray`], a flexible structure for storing and
//! manipulating per-point data in mesh refinement and assembly operations.

use crate::data::atlas::Atlas;
use crate::data::refine::delta::SliceDelta;
use crate::topology::arrow::Orientation;
use crate::topology::point::PointId;

/// A generic array of values indexed by mesh points, supporting refinement and assembly.
///
/// # Type Parameters
/// - `P`: Point identifier type (must convert to [`PointId`]).
/// - `V`: Value type stored for each point.
#[derive(Clone, Debug)]
pub struct SievedArray<P, V> {
    pub(crate) atlas: Atlas,
    pub(crate) data: Vec<V>,
    _phantom: std::marker::PhantomData<P>,
}

impl<P, V: Clone + Default> SievedArray<P, V>
where
    P: Into<PointId> + Copy + Eq,
{
    /// Create a new `SievedArray` with the given atlas.
    pub fn new(atlas: Atlas) -> Self {
        let data = vec![V::default(); atlas.total_len()];
        Self {
            atlas,
            data,
            _phantom: std::marker::PhantomData,
        }
    }
    /// Get a read-only slice for the given point, or an error if not present.
    pub fn try_get(&self, p: PointId) -> Result<&[V], crate::mesh_error::MeshSieveError> {
        let (off, len) = self
            .atlas
            .get(p)
            .ok_or(crate::mesh_error::MeshSieveError::SievedArrayPointNotInAtlas(p))?;
        Ok(&self.data[off..off + len])
    }
    /// Get a mutable slice for the given point, or an error if not present.
    pub fn try_get_mut(
        &mut self,
        p: PointId,
    ) -> Result<&mut [V], crate::mesh_error::MeshSieveError> {
        let (off, len) = self
            .atlas
            .get(p)
            .ok_or(crate::mesh_error::MeshSieveError::SievedArrayPointNotInAtlas(p))?;
        Ok(&mut self.data[off..off + len])
    }
    /// Set the values for the given point from a slice, or return an error if lengths mismatch or point not found.
    pub fn try_set(
        &mut self,
        p: PointId,
        val: &[V],
    ) -> Result<(), crate::mesh_error::MeshSieveError> {
        let tgt = self.try_get_mut(p)?;
        if tgt.len() != val.len() {
            return Err(
                crate::mesh_error::MeshSieveError::SievedArraySliceLengthMismatch {
                    point: p,
                    expected: tgt.len(),
                    found: val.len(),
                },
            );
        }
        tgt.clone_from_slice(val);
        Ok(())
    }
    /// Iterate over all points and their associated slices, returning Results.
    pub fn try_iter<'s>(
        &'s self,
    ) -> impl Iterator<Item = Result<(PointId, &'s [V]), crate::mesh_error::MeshSieveError>> + 's
    {
        self.atlas
            .points()
            .map(move |p| self.try_get(p).map(|slice| (p, slice)))
    }
    /// Refine this array from a coarse array using a sifter (with orientations).
    ///
    /// The `refinement` mapping must be a partial function from fine points to
    /// coarse points; any fine point appearing more than once results in
    /// [`MeshSieveError::DuplicateRefinementTarget`].
    pub fn try_refine_with_sifter(
        &mut self,
        coarse: &SievedArray<P, V>,
        refinement: &[(P, Vec<(P, Orientation)>)],
    ) -> Result<(), crate::mesh_error::MeshSieveError> {
        use crate::mesh_error::MeshSieveError;

        let mut updates = Vec::<(P, Vec<V>)>::new();
        for (coarse_pt, fine_pts) in refinement.iter() {
            let cpid = (*coarse_pt).into();
            let coarse_slice = coarse.try_get(cpid)?;
            for (fine_pt, orient) in fine_pts.iter() {
                let fpid = (*fine_pt).into();
                let (_off, len) = self
                    .atlas
                    .get(fpid)
                    .ok_or(MeshSieveError::SievedArrayPointNotInAtlas(fpid))?;
                if coarse_slice.len() != len {
                    return Err(MeshSieveError::SievedArraySliceLengthMismatch {
                        point: fpid,
                        expected: coarse_slice.len(),
                        found: len,
                    });
                }
                let mut data = vec![V::default(); len];
                orient.apply(coarse_slice, &mut data)?;
                updates.push((*fine_pt, data));
            }
        }

        updates.sort_unstable_by_key(|(f, _)| (*f).into());
        for w in updates.windows(2) {
            let f0: PointId = (w[0].0).into();
            let f1: PointId = (w[1].0).into();
            if f0 == f1 {
                return Err(MeshSieveError::DuplicateRefinementTarget { fine: f0 });
            }
        }

        for (fine_pt, data) in updates {
            let dst = self.try_get_mut(fine_pt.into())?;
            debug_assert_eq!(dst.len(), data.len());
            dst.clone_from_slice(&data);
        }
        Ok(())
    }
    /// Refine this array from a coarse array using a simple mapping (all forward), propagating errors.
    pub fn try_refine(
        &mut self,
        coarse: &SievedArray<P, V>,
        refinement: &[(P, Vec<P>)],
    ) -> Result<(), crate::mesh_error::MeshSieveError> {
        let sifter: Vec<_> = refinement
            .iter()
            .map(|(c, fs)| (*c, fs.iter().map(|f| (*f, Orientation::Forward)).collect()))
            .collect();
        self.try_refine_with_sifter(coarse, &sifter)
    }
    /// Assemble fine data into coarse by averaging over refinement, propagating errors.
    pub fn try_assemble(
        &self,
        coarse: &mut SievedArray<P, V>,
        refinement: &[(P, Vec<P>)],
    ) -> Result<(), crate::mesh_error::MeshSieveError>
    where
        V: num_traits::FromPrimitive
            + std::ops::AddAssign
            + std::ops::Div<Output = V>
            + Clone
            + Default,
    {
        for (coarse_pt, fine_pts) in refinement.iter() {
            let mut accum = {
                let coarse_slice = coarse.try_get((*coarse_pt).into())?;
                vec![V::default(); coarse_slice.len()]
            };
            let mut count = 0;
            for fine_pt in fine_pts {
                let slice = self.try_get((*fine_pt).into())?;
                if slice.len() != accum.len() {
                    return Err(
                        crate::mesh_error::MeshSieveError::SievedArraySliceLengthMismatch {
                            point: (*fine_pt).into(),
                            expected: accum.len(),
                            found: slice.len(),
                        },
                    );
                }
                for (a, v) in accum.iter_mut().zip(slice.iter()) {
                    *a += v.clone();
                }
                count += 1;
            }
            if count > 0 {
                let divisor: V = num_traits::FromPrimitive::from_usize(count).ok_or(
                    crate::mesh_error::MeshSieveError::SievedArrayPrimitiveConversionFailure(count),
                )?;
                for a in accum.iter_mut() {
                    *a = a.clone() / divisor.clone();
                }
                coarse.try_set((*coarse_pt).into(), &accum)?;
            }
        }
        Ok(())
    }
    /// Backwards-compatible panicking versions (deprecated)
    #[deprecated(note = "Use try_get instead")]
    pub fn get(&self, p: PointId) -> &[V] {
        self.try_get(p).unwrap()
    }
    #[deprecated(note = "Use try_get_mut instead")]
    pub fn get_mut(&mut self, p: PointId) -> &mut [V] {
        self.try_get_mut(p).unwrap()
    }
    #[deprecated(note = "Use try_set instead")]
    pub fn set(&mut self, p: PointId, val: &[V]) {
        self.try_set(p, val).unwrap()
    }
    #[deprecated(note = "Use try_refine_with_sifter instead")]
    pub fn refine_with_sifter(
        &mut self,
        coarse: &SievedArray<P, V>,
        refinement: &[(P, Vec<(P, Orientation)>)],
    ) {
        self.try_refine_with_sifter(coarse, refinement).unwrap()
    }
    #[deprecated(note = "Use try_refine instead")]
    pub fn refine(&mut self, coarse: &SievedArray<P, V>, refinement: &[(P, Vec<P>)]) {
        self.try_refine(coarse, refinement).unwrap()
    }
    #[deprecated(note = "Use try_assemble instead")]
    pub fn assemble(&self, coarse: &mut SievedArray<P, V>, refinement: &[(P, Vec<P>)])
    where
        V: num_traits::FromPrimitive
            + std::ops::AddAssign
            + std::ops::Div<Output = V>
            + Clone
            + Default,
    {
        self.try_assemble(coarse, refinement).unwrap()
    }
}

#[cfg(feature = "rayon")]
use rayon::prelude::*;

impl<P, V: Clone + Default + Send + Sync> SievedArray<P, V>
where
    P: Into<PointId> + Copy + Eq + Send + Sync,
{
    /// Parallel refinement using a sifter, enabled with the `rayon` feature.
    ///
    /// Computes slice updates in parallel, short-circuiting on the first error
    /// and rejecting duplicate fine targets deterministically.
    #[cfg(feature = "rayon")]
    pub fn try_refine_with_sifter_parallel(
        &mut self,
        coarse: &Self,
        refinement: &[(P, Vec<(P, Orientation)>)],
    ) -> Result<(), crate::mesh_error::MeshSieveError> {
        use crate::mesh_error::MeshSieveError;
        use std::collections::HashMap;

        let fine_spans: HashMap<PointId, (usize, usize)> = self
            .atlas
            .iter_entries()
            .map(|(pid, span)| (pid, span))
            .collect();

        let updates: Vec<(P, Vec<V>)> = refinement
            .par_iter()
            .try_fold(
                || Vec::<(P, Vec<V>)>::new(),
                |mut local, (coarse_pt, fine_pts)| -> Result<_, MeshSieveError> {
                    let cpid = (*coarse_pt).into();
                    let coarse_slice = coarse.try_get(cpid)?;
                    for (fine_pt, orient) in fine_pts {
                        let fpid = (*fine_pt).into();
                        let (_off, len) = fine_spans
                            .get(&fpid)
                            .copied()
                            .ok_or(MeshSieveError::SievedArrayPointNotInAtlas(fpid))?;
                        if coarse_slice.len() != len {
                            return Err(MeshSieveError::SievedArraySliceLengthMismatch {
                                point: fpid,
                                expected: coarse_slice.len(),
                                found: len,
                            });
                        }
                        let mut data = vec![V::default(); len];
                        orient.apply(coarse_slice, &mut data)?;
                        local.push((*fine_pt, data));
                    }
                    Ok(local)
                },
            )
            .try_reduce(
                || Vec::<(P, Vec<V>)>::new(),
                |mut a, mut b| -> Result<_, MeshSieveError> {
                    a.append(&mut b);
                    Ok(a)
                },
            )?;

        let mut updates = updates;
        updates.sort_unstable_by_key(|(f, _)| (*f).into());
        for w in updates.windows(2) {
            let f0: PointId = (w[0].0).into();
            let f1: PointId = (w[1].0).into();
            if f0 == f1 {
                return Err(MeshSieveError::DuplicateRefinementTarget { fine: f0 });
            }
        }

        for (fine_pt, data) in updates {
            let dst = self.try_get_mut(fine_pt.into())?;
            debug_assert_eq!(dst.len(), data.len());
            dst.clone_from_slice(&data);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::data::atlas::Atlas;
    use crate::data::refine::sieved_array::SievedArray;
    use crate::mesh_error::MeshSieveError;
    use crate::topology::arrow::Orientation;
    use crate::topology::point::PointId;

    fn pt(i: u64) -> PointId {
        PointId::new(i).unwrap()
    }
    fn make_sieved() -> SievedArray<PointId, i32> {
        let mut atlas = Atlas::default();
        atlas.try_insert(pt(1), 2).unwrap();
        atlas.try_insert(pt(2), 2).unwrap();
        atlas.try_insert(pt(3), 2).unwrap();
        SievedArray::new(atlas)
    }

    #[test]
    fn sieved_array_basic_get_set_iter() {
        let mut atlas = Atlas::default();
        atlas.try_insert(pt(1), 2).unwrap();
        atlas.try_insert(pt(2), 1).unwrap();
        let mut arr = SievedArray::<PointId, i32>::new(atlas);
        arr.try_set(pt(1), &[1, 2]).unwrap();
        arr.try_set(pt(2), &[3]).unwrap();
        assert_eq!(arr.try_get(pt(1)).unwrap(), &[1, 2]);
        assert_eq!(arr.try_get(pt(2)).unwrap(), &[3]);
        let vals: Vec<_> = arr.try_iter().map(|r| r.unwrap().1[0]).collect();
        assert_eq!(vals, vec![1, 3]);
    }

    #[test]
    fn sieved_array_refine_with_sifter_forward_and_reverse() {
        let mut cat = Atlas::default();
        cat.try_insert(pt(1), 2).unwrap();
        let mut fat = Atlas::default();
        fat.try_insert(pt(2), 2).unwrap();
        fat.try_insert(pt(3), 2).unwrap();
        let mut coarse = SievedArray::new(cat);
        let mut fine = SievedArray::new(fat);
        coarse.try_set(pt(1), &[10, 20]).unwrap();
        let refinement = vec![(
            pt(1),
            vec![(pt(2), Orientation::Forward), (pt(3), Orientation::Reverse)],
        )];
        fine.try_refine_with_sifter(&coarse, &refinement).unwrap();
        assert_eq!(fine.try_get(pt(2)).unwrap(), &[10, 20]);
        assert_eq!(fine.try_get(pt(3)).unwrap(), &[20, 10]);
    }

    #[test]
    fn sieved_array_refine_forward_only() {
        let mut coarse = make_sieved();
        let mut fine = make_sieved();
        coarse.try_set(pt(1), &[5, 6]).unwrap();
        fine.try_refine(&coarse, &[(pt(1), vec![pt(2), pt(3)])])
            .unwrap();
        assert_eq!(fine.try_get(pt(2)).unwrap(), &[5, 6]);
        assert_eq!(fine.try_get(pt(3)).unwrap(), &[5, 6]);
    }

    #[test]
    fn sieved_array_assemble_average() {
        let mut coarse = make_sieved();
        let mut fine = make_sieved();
        fine.try_set(pt(1), &[2, 4]).unwrap();
        fine.try_set(pt(2), &[6, 8]).unwrap();
        fine.try_assemble(&mut coarse, &[(pt(3), vec![pt(1), pt(2)])])
            .unwrap();
        assert_eq!(coarse.try_get(pt(3)).unwrap(), &[4, 6]);
    }

    #[test]
    fn sieved_array_set_wrong_length_error() {
        let mut arr = make_sieved();
        let err = arr.try_set(pt(1), &[1]).unwrap_err();
        match err {
            MeshSieveError::SievedArraySliceLengthMismatch {
                point,
                expected,
                found,
            } => {
                assert_eq!(point, pt(1));
                assert_eq!(expected, 2);
                assert_eq!(found, 1);
            }
            _ => panic!("wrong error variant: {err:?}"),
        }
    }

    #[test]
    fn sieved_array_assemble_mismatch_error() {
        use crate::data::atlas::Atlas;
        let mut coarse_atlas = Atlas::default();
        let mut fine_atlas = Atlas::default();
        // pt(1) has length 2 in coarse, 1 in fine
        coarse_atlas.try_insert(pt(1), 2).unwrap();
        fine_atlas.try_insert(pt(1), 1).unwrap();
        let mut coarse = SievedArray::<PointId, i32>::new(coarse_atlas);
        let fine = SievedArray::new(fine_atlas);
        let err = fine
            .try_assemble(&mut coarse, &[(pt(1), vec![pt(1)])])
            .unwrap_err();
        match err {
            MeshSieveError::SievedArraySliceLengthMismatch {
                point,
                expected,
                found,
            } => {
                assert_eq!(point, pt(1));
                assert_eq!(expected, 2);
                assert_eq!(found, 1);
            }
            _ => panic!("wrong error variant: {err:?}"),
        }
    }

    #[test]
    fn sieved_array_point_not_in_atlas_error() {
        let arr = make_sieved();
        let missing = pt(99);
        let err = arr.try_get(missing).unwrap_err();
        match err {
            MeshSieveError::SievedArrayPointNotInAtlas(p) => assert_eq!(p, missing),
            _ => panic!("wrong error variant: {err:?}"),
        }
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn sieved_array_refine_with_sifter_parallel_works() {
        let mut coarse = make_sieved();
        let mut fine = make_sieved();
        coarse.try_set(pt(1), &[2, 3]).unwrap();
        let refinement = vec![(pt(1), vec![(pt(2), Orientation::Forward)])];
        fine.try_refine_with_sifter_parallel(&coarse, &refinement)
            .expect("parallel refinement failed");
        assert_eq!(fine.try_get(pt(2)).unwrap(), &[2, 3]);
    }
}
