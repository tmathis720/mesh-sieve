//! Point → slice descriptor lookup

use std::collections::HashMap;
use crate::topology::point::PointId;

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Atlas {
    map: HashMap<PointId, (usize, usize)>, // (offset, len)
    order: Vec<PointId>,                   // deterministic I/O
    total_len: usize,                      // running sum – fast append
}

impl Atlas {
    /// Inserts a brand-new point with its required slice length.
    /// Returns starting offset.
    pub fn insert(&mut self, p: PointId, len: usize) -> usize {
        assert!(len > 0, "len==0 reserved");
        assert!(self.map.get(&p).is_none(), "point already present");

        let offset = self.total_len;
        self.map.insert(p, (offset, len));
        self.order.push(p);
        self.total_len += len;
        offset
    }

    /// Returns (offset,len) if the point is registered.
    #[inline] pub fn get(&self, p: PointId) -> Option<(usize, usize)> {
        self.map.get(&p).copied()
    }
    #[inline] pub fn total_len(&self) -> usize { self.total_len }
    #[inline] pub fn points<'a>(&'a self) -> impl Iterator<Item=PointId> + 'a {
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

        assert_eq!(a.get(p1), Some((0,3)));
        assert_eq!(a.get(p2), Some((3,5)));
        assert_eq!(a.total_len(), 8);
        assert_eq!(a.points().collect::<Vec<_>>(), vec![p1,p2]);
    }

    #[test]
    #[should_panic]
    fn zero_len_rejected() {
        let mut a = Atlas::default();
        a.insert(PointId::new(7), 0);
    }
}
