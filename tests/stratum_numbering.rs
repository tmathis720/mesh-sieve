use mesh_sieve::algs::communicator::RayonComm;
use mesh_sieve::overlap::overlap::Overlap;
use mesh_sieve::topology::ownership::PointOwnership;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::sieve::InMemorySieve;
use mesh_sieve::topology::sieve::Sieve;
use mesh_sieve::topology::sieve::strata::{
    StratumAxis, stratum_numbering_global, stratum_numbering_local,
};

fn make_sieve() -> InMemorySieve<PointId, ()> {
    let mut sieve = InMemorySieve::default();
    sieve.add_arrow(PointId::new(1).unwrap(), PointId::new(3).unwrap(), ());
    sieve.add_arrow(PointId::new(2).unwrap(), PointId::new(3).unwrap(), ());
    sieve
}

#[test]
fn stratum_numbering_local_stable() {
    let sieve = make_sieve();
    let height = stratum_numbering_local(&sieve, StratumAxis::Height).unwrap();
    let depth = stratum_numbering_local(&sieve, StratumAxis::Depth).unwrap();

    let p1 = PointId::new(1).unwrap();
    let p2 = PointId::new(2).unwrap();
    let p3 = PointId::new(3).unwrap();

    assert_eq!(height.stratum_of(p1), Some(0));
    assert_eq!(height.stratum_of(p2), Some(0));
    assert_eq!(height.stratum_of(p3), Some(1));
    assert_eq!(height.id_of(p1), Some(0));
    assert_eq!(height.id_of(p2), Some(1));
    assert_eq!(height.id_of(p3), Some(0));
    assert_eq!(height.stratum_size(0), Some(2));
    assert_eq!(height.stratum_size(1), Some(1));

    assert_eq!(depth.stratum_of(p3), Some(0));
    assert_eq!(depth.stratum_of(p1), Some(1));
    assert_eq!(depth.stratum_of(p2), Some(1));
    assert_eq!(depth.id_of(p3), Some(0));
    assert_eq!(depth.id_of(p1), Some(0));
    assert_eq!(depth.id_of(p2), Some(1));
    assert_eq!(depth.stratum_size(0), Some(1));
    assert_eq!(depth.stratum_size(1), Some(2));
}

#[test]
fn stratum_numbering_global_consistent() {
    let comm0 = RayonComm::new(0, 2);
    let comm1 = RayonComm::new(1, 2);

    let handle0 = std::thread::spawn(move || {
        let sieve = make_sieve();
        let mut ownership = PointOwnership::with_capacity(3);
        ownership.set(PointId::new(1).unwrap(), 0, false).unwrap();
        ownership.set(PointId::new(2).unwrap(), 1, true).unwrap();
        ownership.set(PointId::new(3).unwrap(), 1, true).unwrap();

        let mut overlap = Overlap::new();
        overlap
            .try_add_link(PointId::new(1).unwrap(), 1, PointId::new(1).unwrap())
            .unwrap();
        overlap
            .try_add_link(PointId::new(2).unwrap(), 1, PointId::new(2).unwrap())
            .unwrap();
        overlap
            .try_add_link(PointId::new(3).unwrap(), 1, PointId::new(3).unwrap())
            .unwrap();

        stratum_numbering_global(&sieve, StratumAxis::Height, &overlap, &ownership, &comm0, 0)
            .unwrap()
    });

    let handle1 = std::thread::spawn(move || {
        let sieve = make_sieve();
        let mut ownership = PointOwnership::with_capacity(3);
        ownership.set(PointId::new(1).unwrap(), 0, true).unwrap();
        ownership.set(PointId::new(2).unwrap(), 1, false).unwrap();
        ownership.set(PointId::new(3).unwrap(), 1, false).unwrap();

        let mut overlap = Overlap::new();
        overlap
            .try_add_link(PointId::new(1).unwrap(), 0, PointId::new(1).unwrap())
            .unwrap();
        overlap
            .try_add_link(PointId::new(2).unwrap(), 0, PointId::new(2).unwrap())
            .unwrap();
        overlap
            .try_add_link(PointId::new(3).unwrap(), 0, PointId::new(3).unwrap())
            .unwrap();

        stratum_numbering_global(&sieve, StratumAxis::Height, &overlap, &ownership, &comm1, 1)
            .unwrap()
    });

    let numbering0 = handle0.join().expect("rank0 thread failed");
    let numbering1 = handle1.join().expect("rank1 thread failed");

    let p1 = PointId::new(1).unwrap();
    let p2 = PointId::new(2).unwrap();
    let p3 = PointId::new(3).unwrap();

    assert_eq!(numbering0.stratum_size(0), Some(2));
    assert_eq!(numbering0.stratum_size(1), Some(1));
    assert_eq!(numbering1.stratum_size(0), Some(2));
    assert_eq!(numbering1.stratum_size(1), Some(1));

    assert_eq!(numbering0.id_of(p1), Some(0));
    assert_eq!(numbering0.id_of(p2), Some(1));
    assert_eq!(numbering0.id_of(p3), Some(0));

    assert_eq!(numbering1.id_of(p1), Some(0));
    assert_eq!(numbering1.id_of(p2), Some(1));
    assert_eq!(numbering1.id_of(p3), Some(0));
}
