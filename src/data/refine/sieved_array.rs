//! A generic array of values indexed by mesh points, supporting refine/assemble.
//! (Extracted from section.rs)

use crate::data::atlas::Atlas;
use crate::data::refine::delta::Delta;
use crate::topology::arrow::Orientation;
use crate::topology::point::PointId;

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
    pub fn new(atlas: Atlas) -> Self {
        let data = vec![V::default(); atlas.total_len()];
        Self {
            atlas,
            data,
            _phantom: std::marker::PhantomData,
        }
    }
    pub fn get(&self, p: PointId) -> &[V] {
        let (off, len) = self.atlas.get(p).expect("point not in atlas");
        &self.data[off..off + len]
    }
    pub fn get_mut(&mut self, p: PointId) -> &mut [V] {
        let (off, len) = self.atlas.get(p).expect("point not in atlas");
        &mut self.data[off..off + len]
    }
    pub fn set(&mut self, p: PointId, val: &[V]) {
        let tgt = self.get_mut(p);
        assert_eq!(tgt.len(), val.len());
        tgt.clone_from_slice(val);
    }
    pub fn iter<'s>(&'s self) -> impl Iterator<Item = (PointId, &'s [V])> {
        self.atlas.points().map(move |p| (p, self.get(p)))
    }
    pub fn refine_with_sifter(
        &mut self,
        coarse: &SievedArray<P, V>,
        refinement: &[(P, Vec<(P, Orientation)>)],
    ) {
        for (coarse_pt, fine_pts) in refinement.iter() {
            let coarse_slice = coarse.get((*coarse_pt).into());
            for (fine_pt, orient) in fine_pts.iter() {
                let fine_slice = self.get_mut((*fine_pt).into());
                assert_eq!(
                    coarse_slice.len(),
                    fine_slice.len(),
                    "dof mismatch in refinement"
                );
                orient.apply(coarse_slice, fine_slice);
            }
        }
    }
    pub fn refine(&mut self, coarse: &SievedArray<P, V>, refinement: &[(P, Vec<P>)]) {
        let sifter: Vec<_> = refinement
            .iter()
            .map(|(c, fs)| (*c, fs.iter().map(|f| (*f, Orientation::Forward)).collect()))
            .collect();
        self.refine_with_sifter(coarse, &sifter);
    }
    /// Assemble fine data into coarse by averaging over refinement.
    ///
    /// # Example: average (pseudo-code)
    ///
    /// ```text
    /// for (coarse_pt, fine_pts) in refinement.iter() {
    ///   let mut accum = vec![V::default(); coarse.get((*coarse_pt).into()).len()];
    ///   let mut count = 0;
    ///   for fine_pt in fine_pts {
    ///     let slice = self.get((*fine_pt).into());
    ///     for (a, v) in accum.iter_mut().zip(slice.iter()) {
    ///       *a += v.clone();
    ///     }
    ///     count += 1;
    ///   }
    ///   if count>0 {
    ///     for a in accum.iter_mut() { *a = a.clone() / num_traits::FromPrimitive::from_usize(count).unwrap(); }
    ///     coarse.set((*coarse_pt).into(), &accum);
    ///   }
    /// }
    /// ```
    ///
    /// # Example: max (pseudo-code)
    ///
    /// ```text
    /// for (coarse_pt, fine_pts) in refinement.iter() {
    ///   let mut accum = vec![V::default(); coarse.get((*coarse_pt).into()).len()];
    ///   for fine_pt in fine_pts {
    ///     let slice = self.get((*fine_pt).into());
    ///     for (a, v) in accum.iter_mut().zip(slice.iter()) {
    ///       *a = std::cmp::max(a.clone(), v.clone());
    ///     }
    ///   }
    ///   coarse.set((*coarse_pt).into(), &accum);
    /// }
    /// ```
    ///
    /// # Example: min (pseudo-code)
    ///
    /// ```text
    /// for (coarse_pt, fine_pts) in refinement.iter() {
    ///   let mut accum = vec![V::default(); coarse.get((*coarse_pt).into()).len()];
    ///   for fine_pt in fine_pts {
    ///     let slice = self.get((*fine_pt).into());
    ///     for (a, v) in accum.iter_mut().zip(slice.iter()) {
    ///       *a = std::cmp::min(a.clone(), v.clone());
    ///     }
    ///   }
    ///   coarse.set((*coarse_pt).into(), &accum);
    /// }
    /// ```
    pub fn assemble(&self, coarse: &mut SievedArray<P, V>, refinement: &[(P, Vec<P>)])
    where
        V: num_traits::FromPrimitive + std::ops::AddAssign + std::ops::Div<Output=V> + Clone + Default,
    {
        for (coarse_pt, fine_pts) in refinement.iter() {
            let mut accum = vec![V::default(); coarse.get((*coarse_pt).into()).len()];
            let mut count = 0;
            for fine_pt in fine_pts {
                let slice = self.get((*fine_pt).into());
                for (a, v) in accum.iter_mut().zip(slice.iter()) {
                    *a += v.clone();
                }
                count += 1;
            }
            if count > 0 {
                for a in accum.iter_mut() {
                    *a = a.clone() / num_traits::FromPrimitive::from_usize(count).unwrap();
                }
                coarse.set((*coarse_pt).into(), &accum);
            }
        }
    }
}

#[cfg(feature = "rayon")]
impl<P, V: Clone + Default + Send + Sync> SievedArray<P, V>
where
    P: Into<PointId> + Copy + Eq + Send + Sync,
{
    pub fn refine_with_sifter_parallel(
        &mut self,
        coarse: &Self,
        refinement: &[(P, Vec<(P, Orientation)>)]
    ) {
        use rayon::prelude::*;
        refinement.par_iter().for_each(|(c, fine_pts)| {
            let coarse_slice = coarse.get((*c).into());
            fine_pts.par_iter().for_each(|(f, o)| {
                let dst = self.get_mut((*f).into());
                o.apply(coarse_slice, dst);
            });
        });
    }
}

// #[cfg(test)] module here for all the SievedArray tests
