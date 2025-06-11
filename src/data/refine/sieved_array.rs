//! A generic array of values indexed by mesh points, supporting refine/assemble.
//! (Extracted from section.rs)

use crate::data::atlas::Atlas;
use crate::data::refine::delta::Delta;
use crate::topology::arrow::Orientation;
use crate::topology::point::PointId;
use rayon::prelude::*;

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
use rayon::prelude::*;

impl<P, V: Clone + Default + Send + Sync> SievedArray<P, V>
where
    P: Into<PointId> + Copy + Eq + Send + Sync,
{
    #[cfg(feature = "rayon")]
    pub fn refine_with_sifter_parallel(
        &mut self,
        coarse: &Self,
        refinement: &[(P, Vec<(P, Orientation)>)]
    ) {
        // Collect updates in parallel: (fine_pt, data)
        let updates: Vec<(P, Vec<V>)> = refinement.par_iter().flat_map(|(c, fine_pts)| {
            let coarse_slice = coarse.get((*c).into());
            fine_pts.par_iter().map(|(f, o)| {
                let mut data = vec![V::default(); coarse_slice.len()];
                o.apply(coarse_slice, &mut data);
                (*f, data)
            }).collect::<Vec<_>>()
        }).collect();

        // Apply updates sequentially
        for (f, data) in updates {
            let dst = self.get_mut(f.into());
            assert_eq!(dst.len(), data.len(), "dof mismatch in refinement");
            dst.clone_from_slice(&data);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::data::atlas::Atlas;
    use crate::topology::point::PointId;
    use crate::topology::arrow::Orientation;
    use crate::data::refine::sieved_array::SievedArray;

    fn pt(i: u64) -> PointId { PointId::new(i) }
    fn make_sieved() -> SievedArray<PointId, i32> {
        let mut atlas = Atlas::default();
        atlas.insert(pt(1), 2);
        atlas.insert(pt(2), 2);
        atlas.insert(pt(3), 2);
        SievedArray::new(atlas)
    }

    #[test]
    fn sieved_array_basic_get_set_iter() {
        let mut atlas = Atlas::default();
        atlas.insert(pt(1),2);
        atlas.insert(pt(2),1);
        let mut arr = SievedArray::<PointId,i32>::new(atlas);
        arr.set(pt(1), &[1,2]);
        arr.set(pt(2), &[3]);
        assert_eq!(arr.get(pt(1)), &[1,2]);
        assert_eq!(arr.get(pt(2)), &[3]);
        let vals: Vec<_> = arr.iter().map(|(_,v)| v[0]).collect();
        assert_eq!(vals, vec![1,3]);
    }

    #[test]
    fn sieved_array_refine_with_sifter_forward_and_reverse() {
        // coarse pt 1 len=2: data [10,20]
        // fine pts 2,3 len=2
        let mut cat = Atlas::default(); cat.insert(pt(1),2);
        let mut fat = Atlas::default(); fat.insert(pt(2),2); fat.insert(pt(3),2);
        let mut coarse = SievedArray::new(cat);
        let mut fine   = SievedArray::new(fat);
        coarse.set(pt(1), &[10,20]);
        let refinement = vec![
          (pt(1), vec![(pt(2), Orientation::Forward),(pt(3),Orientation::Reverse)])
        ];
        fine.refine_with_sifter(&coarse, &refinement);
        assert_eq!(fine.get(pt(2)), &[10,20]);
        assert_eq!(fine.get(pt(3)), &[20,10]);
    }

    #[test]
    fn sieved_array_refine_forward_only() {
        let mut coarse = make_sieved();
        let mut fine   = make_sieved();
        coarse.set(pt(1), &[5,6]);
        fine.refine(&coarse, &[(pt(1), vec![pt(2),pt(3)])]);
        assert_eq!(fine.get(pt(2)), &[5,6]);
        assert_eq!(fine.get(pt(3)), &[5,6]);
    }

    #[test]
    fn sieved_array_assemble_average() {
        let mut coarse = make_sieved();
        let mut fine   = make_sieved();
        // coarse unset, fine carries two slices
        fine.set(pt(1), &[2,4]);
        fine.set(pt(2), &[6,8]);
        fine.assemble(&mut coarse, &[(pt(3), vec![pt(1),pt(2)])]);
        // point 3 of coarse should be avg of [2,4] & [6,8] => [4,6]
        assert_eq!(coarse.get(pt(3)), &[4,6]);
    }

    #[test]
    #[should_panic]
    fn sieved_array_set_wrong_length_panics() {
        let mut arr = make_sieved();
        arr.set(pt(1), &[1]); // length mismatch
    }

    #[test]
    #[should_panic]
    fn sieved_array_assemble_mismatch_panics() {
        let mut coarse = make_sieved();
        let mut fine   = make_sieved();
        // coarse len=2, fine len=1
        fine.set(pt(1), &[9]);
        fine.assemble(&mut coarse, &[(pt(1), vec![pt(1)])]);
    }

    #[cfg(feature="rayon")]
    #[test]
    fn sieved_array_refine_with_sifter_parallel_works() {
        let mut coarse = make_sieved();
        let mut fine   = make_sieved();
        coarse.set(pt(1), &[2,3]);
        let refinement = vec![(pt(1), vec![(pt(2),Orientation::Forward)])];
        fine.refine_with_sifter_parallel(&coarse, &refinement);
        assert_eq!(fine.get(pt(2)), &[2,3]);
    }
}
