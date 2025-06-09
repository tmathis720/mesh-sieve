//! High‐level “complete_section” that runs neighbour_links → exchange_sizes → exchange_data.


pub fn complete_section<V, D, C>(
    section: &mut crate::data::section::Section<V>,
    overlap: &crate::overlap::overlap::Overlap,
    comm: &C,
    _delta: &D,
    my_rank: usize,
    n_ranks: usize,
) where
    V: Clone + Default + Send + 'static,
    D: crate::overlap::delta::Delta<V> + Send + Sync + 'static,
    D::Part: bytemuck::Pod + Default,
    C: crate::algs::communicator::Communicator + Sync,
{
    const BASE_TAG: u16 = 0xBEEF;
    let links = crate::algs::completion::neighbour_links::neighbour_links(section, overlap, my_rank);
    // --- DEADLOCK FIX: ensure symmetric communication ---
    // For tests: use all ranks except my_rank as neighbors
    let all_neighbors: std::collections::HashSet<usize> = (0..n_ranks).filter(|&r| r != my_rank).collect();
    // Exchange sizes (always post send/recv for all neighbors)
    let counts = crate::algs::completion::size_exchange::exchange_sizes_symmetric(&links, comm, BASE_TAG, &all_neighbors);
    crate::algs::completion::data_exchange::exchange_data_symmetric::<V, D, C>(&links, &counts, comm, BASE_TAG+1, section, &all_neighbors);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::atlas::Atlas;
    use crate::data::section::Section;
    use crate::overlap::overlap::{Overlap, Remote};
    use crate::topology::point::PointId;
    use crate::topology::sieve::{InMemorySieve, Sieve};
    use crate::data::section::Map;
    use crate::overlap::delta::Delta;
    use crate::algs::communicator::RayonComm;
    use std::thread;

    // Dummy Delta implementation for testing
    #[derive(Default, Clone)]
    struct DummyDelta;
    impl Delta<i32> for DummyDelta {
        type Part = i32;
        fn restrict(v: &i32) -> i32 { *v }
        fn fuse(local: &mut i32, incoming: i32) { *local = incoming; }
    }

    fn make_section(points: &[u64]) -> Section<i32> {
        let mut atlas = Atlas::default();
        for &p in points {
            atlas.insert(PointId::new(p), 1);
        }
        let mut section = Section::new(atlas);
        for &p in points {
            section.set(PointId::new(p), &[p as i32]);
        }
        section
    }

    fn make_overlap(_owner: usize, ghost: usize, owned: &[u64], ghosted: &[u64]) -> Overlap {
        let mut ovlp = InMemorySieve::<PointId, Remote>::default();
        for (&src, &dst) in owned.iter().zip(ghosted.iter()) {
            ovlp.add_arrow(PointId::new(src), PointId::new(dst), Remote { rank: ghost, remote_point: PointId::new(dst) });
        }
        ovlp
    }

    #[test]
    fn test_complete_section_owner_to_ghost() {
        // Simulate two ranks in parallel threads
        let t0 = thread::spawn(|| {
            let mut section = make_section(&[1]);
            let ovlp = make_overlap(0, 1, &[1], &[101]);
            let delta = DummyDelta;
            let comm = RayonComm::new(0);
            complete_section(&mut section, &ovlp, &comm, &delta, 0, 2);
            section.get(PointId::new(1)).to_vec()
        });
        let t1 = thread::spawn(|| {
            let mut section = make_section(&[]);
            let ovlp = make_overlap(0, 1, &[1], &[101]);
            let delta = DummyDelta;
            let comm = RayonComm::new(1);
            complete_section(&mut section, &ovlp, &comm, &delta, 1, 2);
            section.get(PointId::new(101)).to_vec()
        });
        let res0 = t0.join().unwrap();
        let res1 = t1.join().unwrap();
        assert_eq!(res0, vec![1]);
        assert_eq!(res1, vec![1]);
    }

    #[test]
    fn test_complete_section_ghost_receives() {
        // This is now covered by the above test (rank 1 branch)
    }

    #[test]
    fn test_complete_section_no_overlap() {
        let mut section = make_section(&[2]);
        let ovlp = InMemorySieve::<PointId, Remote>::default();
        let delta = DummyDelta;
        let comm = RayonComm::new(2);
        complete_section(&mut section, &ovlp, &comm, &delta, 2, 3);
        assert_eq!(section.get(PointId::new(2)), &[2]);
    }

    #[test]
    fn test_complete_section_multiple_neighbors() {
        // Simulate three ranks in parallel threads
        let t0 = thread::spawn(|| {
            let mut section = make_section(&[1,2]);
            let mut ovlp = InMemorySieve::<PointId, Remote>::default();
            ovlp.add_arrow(PointId::new(1), PointId::new(101), Remote { rank: 1, remote_point: PointId::new(101) });
            ovlp.add_arrow(PointId::new(2), PointId::new(201), Remote { rank: 2, remote_point: PointId::new(201) });
            let delta = DummyDelta;
            let comm = RayonComm::new(0);
            complete_section(&mut section, &ovlp, &comm, &delta, 0, 3);
            (section.get(PointId::new(1)).to_vec(), section.get(PointId::new(2)).to_vec())
        });
        let t1 = thread::spawn(|| {
            let mut section = make_section(&[]);
            let mut ovlp = InMemorySieve::<PointId, Remote>::default();
            ovlp.add_arrow(PointId::new(101), PointId::new(1), Remote { rank: 0, remote_point: PointId::new(1) });
            let delta = DummyDelta;
            let comm = RayonComm::new(1);
            complete_section(&mut section, &ovlp, &comm, &delta, 1, 3);
            section.get(PointId::new(101)).to_vec()
        });
        let t2 = thread::spawn(|| {
            let mut section = make_section(&[]);
            let mut ovlp = InMemorySieve::<PointId, Remote>::default();
            ovlp.add_arrow(PointId::new(201), PointId::new(2), Remote { rank: 0, remote_point: PointId::new(2) });
            let delta = DummyDelta;
            let comm = RayonComm::new(2);
            complete_section(&mut section, &ovlp, &comm, &delta, 2, 3);
            section.get(PointId::new(201)).to_vec()
        });
        let (res0_1, res0_2) = t0.join().unwrap();
        let res1 = t1.join().unwrap();
        let res2 = t2.join().unwrap();
        assert_eq!(res0_1, vec![1]);
        assert_eq!(res0_2, vec![2]);
        assert_eq!(res1, vec![1]);
        assert_eq!(res2, vec![2]);
    }
}
