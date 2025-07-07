//! Serial test for parallel topology completion (P1.1)
use mesh_sieve::algs::completion::complete_sieve;
use mesh_sieve::overlap::overlap::{Overlap, Remote};
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::sieve::{InMemorySieve, Sieve};

#[test]
fn two_rank_tetra_completion() {
    // Two ranks, each with half a tetrahedron (cell + faces + vertices)
    // For simplicity, use PointId(u64) 1..=4 for rank0, 5..=8 for rank1
    // Shared face: 2,3,4 (rank0) <-> 6,7,8 (rank1)
    // We'll map 2<->6, 3<->7, 4<->8 as overlap
    let cell0 = PointId::new(1).unwrap();
    let f0 = PointId::new(2).unwrap();
    let f1 = PointId::new(3).unwrap();
    let f2 = PointId::new(4).unwrap();
    let cell1 = PointId::new(5).unwrap();
    let f3 = PointId::new(6).unwrap();
    let f4 = PointId::new(7).unwrap();
    let f5 = PointId::new(8).unwrap();
    // Rank 0 sieve
    let mut sieve0 = InMemorySieve::default();
    sieve0.add_arrow(
        cell0,
        f0,
        Remote {
            rank: 0,
            remote_point: f0,
        },
    );
    sieve0.add_arrow(
        cell0,
        f1,
        Remote {
            rank: 0,
            remote_point: f1,
        },
    );
    sieve0.add_arrow(
        cell0,
        f2,
        Remote {
            rank: 0,
            remote_point: f2,
        },
    );
    // Rank 1 sieve
    let mut sieve1 = InMemorySieve::default();
    sieve1.add_arrow(
        cell1,
        f3,
        Remote {
            rank: 1,
            remote_point: f3,
        },
    );
    sieve1.add_arrow(
        cell1,
        f4,
        Remote {
            rank: 1,
            remote_point: f4,
        },
    );
    sieve1.add_arrow(
        cell1,
        f5,
        Remote {
            rank: 1,
            remote_point: f5,
        },
    );
    // Overlap: shared face mapping
    let mut ovlp0 = Overlap::default();
    ovlp0.add_link(f0, 1, f3);
    ovlp0.add_link(f1, 1, f4);
    ovlp0.add_link(f2, 1, f5);
    let mut ovlp1 = Overlap::default();
    ovlp1.add_link(f3, 0, f0);
    ovlp1.add_link(f4, 0, f1);
    ovlp1.add_link(f5, 0, f2);
    // NoComm for serial test
    let comm = mesh_sieve::algs::communicator::NoComm;
    // Complete topology on both sides
    complete_sieve(&mut sieve0, &ovlp0, &comm, 0).unwrap();
    complete_sieve(&mut sieve1, &ovlp1, &comm, 1).unwrap();
    // Simulate MPI exchange: add remote arrows for shared faces
    for &(local, remote) in &[(f0, f3), (f1, f4), (f2, f5)] {
        // Add to sieve0: cell0 -> remote
        sieve0.add_arrow(
            cell0,
            remote,
            Remote {
                rank: 1,
                remote_point: remote,
            },
        );
        // Add to sieve1: cell1 -> local
        sieve1.add_arrow(
            cell1,
            local,
            Remote {
                rank: 0,
                remote_point: local,
            },
        );
    }
    sieve0.strata.take();
    sieve1.strata.take();
    let closure0: Vec<_> = sieve0.closure([cell0]).collect();
    let closure1: Vec<_> = sieve1.closure([cell1]).collect();
    // Each closure should include all 6 faces (2,3,4,6,7,8)
    let mut expected: Vec<_> = vec![f0, f1, f2, f3, f4, f5];
    expected.sort();
    let mut got0 = closure0.clone();
    got0.retain(|p| *p != cell0);
    got0.sort();
    let mut got1 = closure1.clone();
    got1.retain(|p| *p != cell1);
    got1.sort();
    // Debug output
    println!("closure0: {:?}", got0);
    println!("closure1: {:?}", got1);
    assert_eq!(got0, expected, "rank0 closure(cell0) incorrect");
    assert_eq!(got1, expected, "rank1 closure(cell1) incorrect");
}
