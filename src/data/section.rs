//! Field data over a topology atlas.

use crate::data::atlas::Atlas;
use crate::topology::point::PointId;

#[derive(Clone, Debug)]
pub struct Section<V> {
    atlas: Atlas,
    data: Vec<V>,
}

impl<V: Clone + Default> Section<V> {
    /// Creates an empty section with an existing atlas.
    pub fn new(atlas: Atlas) -> Self {
        Self {
            data: vec![V::default(); atlas.total_len()],
            atlas,
        }
    }

    /// Immutable restriction (read-only slice).
    #[inline] pub fn restrict(&self, p: PointId) -> &[V] {
        let (off, len) = self.atlas.get(p)
            .expect("point not in atlas");
        &self.data[off .. off + len]
    }

    /// Mutable restriction (write-capable slice).
    #[inline] pub fn restrict_mut(&mut self, p: PointId) -> &mut [V] {
        let (off, len) = self.atlas.get(p)
            .expect("point not in atlas");
        &mut self.data[off .. off + len]
    }

    /// Copies values into the storage for point `p`.
    pub fn set(&mut self, p: PointId, val: &[V]) {
        let tgt = self.restrict_mut(p);
        assert_eq!(tgt.len(), val.len());
        tgt.clone_from_slice(val);
    }

    /// Iterate `(PointId, &[V])` pairs in atlas order.
    pub fn iter<'s>(&'s self) -> impl Iterator<Item=(PointId, &'s [V])> {
        self.atlas.points()
            .map(move |p| (p, self.restrict(p)))
    }
}

impl<V: Clone + Send> Section<V> {
    /// Place `other` into slice of *this* determined by `atlas_map`.
    /// (General building block for MPI receive or threaded gather.)
    pub fn scatter_from(&mut self,
                        other: &[V],
                        atlas_map: &[(usize, usize)])
    {
        for ((off,len), src_chunk) in
            atlas_map.iter().zip(other.chunks_exact(atlas_map[0].1)) {
            self.data[*off .. off+*len].clone_from_slice(src_chunk);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::atlas::Atlas;
    use crate::topology::point::PointId;

    fn make_section() -> Section<f64> {
        let mut atlas = Atlas::default();
        atlas.insert(PointId::new(1), 2); // 2 dof
        atlas.insert(PointId::new(2), 1);
        Section::<f64>::new(atlas)
    }

    #[test]
    fn restrict_and_set() {
        let mut s = make_section();
        s.set(PointId::new(1), &[1.0,2.0]);
        s.set(PointId::new(2), &[3.5]);

        assert_eq!(s.restrict(PointId::new(1)), &[1.0,2.0]);
        assert_eq!(s.restrict(PointId::new(2)), &[3.5]);
    }

    #[test]
    fn iter_order() {
        let mut s = make_section();
        s.set(PointId::new(1), &[9.0,8.0]);
        s.set(PointId::new(2), &[7.0]);

        let collected: Vec<_> = s.iter()
                                 .map(|(_,sl)| sl[0])
                                 .collect();
        assert_eq!(collected, vec![9.0,7.0]); // atlas order
    }
}
