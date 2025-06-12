use std::sync::atomic::AtomicU32;

/// Stores cluster IDs for each vertex (1-to-1 with vertex index).
#[cfg(feature = "mpi-support")]
pub struct ClusterIds {
    pub ids: Vec<AtomicU32>,
}

#[cfg(feature = "mpi-support")]
impl ClusterIds {
    pub fn new(size: usize) -> Self {
        let ids = (0..size).map(|_| AtomicU32::new(0)).collect();
        Self { ids }
    }
    pub fn get(&self, idx: usize) -> u32 {
        self.ids[idx].load(std::sync::atomic::Ordering::Relaxed)
    }
    pub fn set(&self, idx: usize, val: u32) {
        self.ids[idx].store(val, std::sync::atomic::Ordering::Relaxed)
    }
    /// Find the root of the set for idx, with path compression.
    pub fn find(&self, idx: usize) -> u32 {
        let mut root = self.get(idx);
        while root != self.get(root as usize) {
            root = self.get(root as usize);
        }
        // Path compression
        let mut cur = idx as u32;
        while cur != root {
            let parent = self.get(cur as usize);
            self.set(cur as usize, root);
            cur = parent;
        }
        root
    }
    /// Union two sets, returns the new root.
    pub fn union(&self, a: usize, b: usize) -> u32 {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return ra;
        }
        let (small, big) = if ra < rb { (ra, rb) } else { (rb, ra) };
        self.set(small as usize, big);
        big
    }
    /// Compress all paths so every node points directly to its root.
    pub fn compress_all(&self) {
        for u in 0..self.ids.len() {
            self.find(u);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn union_find_basic() {
        let uf = ClusterIds::new(5);
        // Initially, each is its own root
        for i in 0..5 {
            assert_eq!(uf.find(i), i as u32);
        }
        // Union 1 and 2
        let r = uf.union(1, 2);
        assert_eq!(uf.find(1), r);
        assert_eq!(uf.find(2), r);
        // Union 2 and 3
        let r2 = uf.union(2, 3);
        assert_eq!(uf.find(3), r2);
        assert_eq!(uf.find(1), r2);
        // Compress all
        uf.compress_all();
        let root = uf.find(1);
        for i in 1..=3 {
            assert_eq!(uf.get(i), root);
        }
    }
}

#[cfg(feature = "mpi-support")]
impl ClusterIds {
    /// Additional methods or overrides for onizuka partitioning can be added here.
}
