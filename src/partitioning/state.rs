//! Cluster ID management for partitioning algorithms.
//!
//! Provides [`ClusterIds`], a union–find (disjoint-set) structure used to track
//! vertex clusters.  When the `mpi-support` feature is enabled the structure is
//! lock‑free and safe for concurrent use via atomics.  Without the feature a
//! lightweight serial implementation is provided.

use thiserror::Error;

/// Errors returned by `ClusterIds` methods.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ClusterError {
    /// The requested vertex index was beyond the size of the union-find.
    #[error("vertex index {0} out of bounds (0..{1})")]
    IndexOutOfBounds(usize, usize),
}

// ---------------------------------------------------------------------------
// Concurrent version
// ---------------------------------------------------------------------------

#[cfg(feature = "mpi-support")]
use std::sync::atomic::{
    AtomicU32, AtomicU8,
    Ordering::{AcqRel, Acquire, Release},
};

#[cfg(feature = "mpi-support")]
pub struct ClusterIds {
    parent: Vec<AtomicU32>,
    rank: Vec<AtomicU8>,
    len: usize,
}

#[cfg(feature = "mpi-support")]
impl ClusterIds {
    pub fn new(size: usize) -> Self {
        assert!(size <= u32::MAX as usize, "ClusterIds: size exceeds u32");
        let parent = (0..size).map(|i| AtomicU32::new(i as u32)).collect();
        let rank = (0..size).map(|_| AtomicU8::new(0)).collect();
        Self {
            parent,
            rank,
            len: size,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    fn load_parent(&self, i: u32) -> u32 {
        self.parent[i as usize].load(Acquire)
    }

    pub fn get(&self, idx: usize) -> Result<u32, ClusterError> {
        if idx >= self.len {
            return Err(ClusterError::IndexOutOfBounds(idx, self.len));
        }
        Ok(self.load_parent(idx as u32))
    }

    pub fn set(&self, idx: usize, val: u32) -> Result<(), ClusterError> {
        if idx >= self.len {
            return Err(ClusterError::IndexOutOfBounds(idx, self.len));
        }
        self.parent[idx].store(val, Release);
        Ok(())
    }

    pub fn find(&self, i: usize) -> Result<u32, ClusterError> {
        if i >= self.len {
            return Err(ClusterError::IndexOutOfBounds(i, self.len));
        }
        let mut x = i as u32;
        loop {
            let p = self.load_parent(x);
            let gp = self.load_parent(p);
            if p == x {
                return Ok(x);
            }
            let _ = self.parent[x as usize].compare_exchange(p, gp, AcqRel, Acquire);
            x = p;
        }
    }

    pub fn union(&self, a: usize, b: usize) -> Result<u32, ClusterError> {
        if a >= self.len || b >= self.len {
            return Err(ClusterError::IndexOutOfBounds(a.max(b), self.len));
        }
        let mut ra = self.find(a)?;
        loop {
            let mut rb = self.find(b)?;
            if ra == rb {
                return Ok(ra);
            }

            let rank_a = self.rank[ra as usize].load(Acquire);
            let rank_b = self.rank[rb as usize].load(Acquire);
            let (child, parent) = match rank_a.cmp(&rank_b) {
                std::cmp::Ordering::Less => (ra, rb),
                std::cmp::Ordering::Greater => (rb, ra),
                std::cmp::Ordering::Equal => {
                    if rb > ra {
                        (rb, ra)
                    } else {
                        (ra, rb)
                    }
                }
            };

            let node = &self.parent[child as usize];
            match node.compare_exchange(child, parent, AcqRel, Acquire) {
                Ok(_) => {
                    if rank_a == rank_b {
                        self.rank[parent as usize].fetch_add(1, AcqRel);
                    }
                    return Ok(parent);
                }
                Err(cur) => {
                    if cur != child {
                        ra = self.find(a)?;
                        continue;
                    }
                }
            }
        }
    }

    pub fn compress_all(&self) -> Result<(), ClusterError> {
        for i in 0..self.len {
            let root = self.find(i)?;
            self.parent[i].store(root, Release);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Serial fallback
// ---------------------------------------------------------------------------

#[cfg(not(feature = "mpi-support"))]
use std::cell::Cell;

#[cfg(not(feature = "mpi-support"))]
pub struct ClusterIds {
    parent: Vec<Cell<u32>>,
    rank: Vec<Cell<u8>>,
    len: usize,
}

#[cfg(not(feature = "mpi-support"))]
impl ClusterIds {
    pub fn new(size: usize) -> Self {
        assert!(size <= u32::MAX as usize, "ClusterIds: size exceeds u32");
        let parent = (0..size).map(|i| Cell::new(i as u32)).collect();
        let rank = (0..size).map(|_| Cell::new(0u8)).collect();
        Self {
            parent,
            rank,
            len: size,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn get(&self, idx: usize) -> Result<u32, ClusterError> {
        if idx >= self.len {
            return Err(ClusterError::IndexOutOfBounds(idx, self.len));
        }
        Ok(self.parent[idx].get())
    }

    pub fn set(&self, idx: usize, val: u32) -> Result<(), ClusterError> {
        if idx >= self.len {
            return Err(ClusterError::IndexOutOfBounds(idx, self.len));
        }
        self.parent[idx].set(val);
        Ok(())
    }

    pub fn find(&self, i: usize) -> Result<u32, ClusterError> {
        if i >= self.len {
            return Err(ClusterError::IndexOutOfBounds(i, self.len));
        }
        let mut x = i as u32;
        loop {
            let p = self.parent[x as usize].get();
            let gp = self.parent[p as usize].get();
            if p == x {
                return Ok(x);
            }
            self.parent[x as usize].set(gp);
            x = p;
        }
    }

    pub fn union(&self, a: usize, b: usize) -> Result<u32, ClusterError> {
        if a >= self.len || b >= self.len {
            return Err(ClusterError::IndexOutOfBounds(a.max(b), self.len));
        }
        let mut ra = self.find(a)?;
        let mut rb = self.find(b)?;
        if ra == rb {
            return Ok(ra);
        }

        let rank_a = self.rank[ra as usize].get();
        let rank_b = self.rank[rb as usize].get();
        let (child, parent) = match rank_a.cmp(&rank_b) {
            std::cmp::Ordering::Less => (ra, rb),
            std::cmp::Ordering::Greater => (rb, ra),
            std::cmp::Ordering::Equal => {
                if rb > ra {
                    (rb, ra)
                } else {
                    (ra, rb)
                }
            }
        };
        self.parent[child as usize].set(parent);
        if rank_a == rank_b {
            let r = self.rank[parent as usize].get();
            self.rank[parent as usize].set(r + 1);
        }
        Ok(parent)
    }

    pub fn compress_all(&self) -> Result<(), ClusterError> {
        for i in 0..self.len {
            let root = self.find(i)?;
            self.parent[i].set(root);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn union_find_basic() -> Result<(), ClusterError> {
        let uf = ClusterIds::new(5);
        for i in 0..5 {
            assert_eq!(uf.find(i)?, i as u32);
        }
        let r = uf.union(1, 2)?;
        assert_eq!(uf.find(1)?, r);
        assert_eq!(uf.find(2)?, r);
        let r2 = uf.union(2, 3)?;
        assert_eq!(uf.find(3)?, r2);
        assert_eq!(uf.find(1)?, r2);
        uf.compress_all()?;
        let root = uf.find(1)?;
        for i in 1..=3 {
            assert_eq!(uf.get(i)?, root);
        }
        Ok(())
    }

    #[test]
    fn out_of_bounds_errors() {
        let uf = ClusterIds::new(3);
        assert!(matches!(
            uf.get(10),
            Err(ClusterError::IndexOutOfBounds(10, 3))
        ));
        assert!(matches!(
            uf.set(10, 0),
            Err(ClusterError::IndexOutOfBounds(10, 3))
        ));
        assert!(matches!(
            uf.find(10),
            Err(ClusterError::IndexOutOfBounds(10, 3))
        ));
        assert!(matches!(
            uf.union(10, 1),
            Err(ClusterError::IndexOutOfBounds(10, 3))
        ));
    }
}
