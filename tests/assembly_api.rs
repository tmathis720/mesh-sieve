use mesh_sieve::algs::assembly::{AssemblyCommTags, assemble_section_with_tags_and_ownership};
use mesh_sieve::algs::communicator::{CommTag, RayonComm};
use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::constrained_section::DofConstraint;
use mesh_sieve::data::section::Section;
use mesh_sieve::data::storage::VecStorage;
use mesh_sieve::overlap::delta::AddDelta;
use mesh_sieve::overlap::overlap::Overlap;
use mesh_sieve::topology::ownership::PointOwnership;
use mesh_sieve::topology::point::PointId;
use std::collections::BTreeMap;

fn pid(id: u64) -> PointId {
    PointId::new(id).unwrap()
}

fn make_section(points: &[u64]) -> Section<f64, VecStorage<f64>> {
    let mut atlas = Atlas::default();
    for &p in points {
        atlas.try_insert(pid(p), 1).unwrap();
    }
    Section::new(atlas)
}

#[test]
fn assemble_with_overlap_ownership_and_constraints() {
    let comm0 = RayonComm::new(0, 2);
    let comm1 = RayonComm::new(1, 2);
    let tags = AssemblyCommTags::from_base(CommTag::new(0xCAFE));

    let handle1 = std::thread::spawn(move || {
        let mut ownership = PointOwnership::with_capacity(2);
        ownership.set(pid(1), 0, true).unwrap();
        ownership.set(pid(2), 1, false).unwrap();

        let mut overlap = Overlap::new();
        overlap.try_add_link(pid(1), 0, pid(1)).unwrap();
        overlap.try_add_link(pid(2), 0, pid(2)).unwrap();

        let mut section = make_section(&[1, 2]);
        section.try_set(pid(1), &[3.0]).unwrap();
        section.try_set(pid(2), &[4.0]).unwrap();

        let mut constraints = BTreeMap::new();
        constraints.insert(pid(1), vec![DofConstraint::new(0, 9.0)]);

        assemble_section_with_tags_and_ownership::<_, _, AddDelta, _, _>(
            &mut section,
            &overlap,
            &ownership,
            &comm1,
            1,
            tags,
            &constraints,
        )
        .unwrap();

        (section, ownership)
    });

    let mut ownership = PointOwnership::with_capacity(2);
    ownership.set(pid(1), 0, false).unwrap();
    ownership.set(pid(2), 1, true).unwrap();

    let mut overlap = Overlap::new();
    overlap.try_add_link(pid(1), 1, pid(1)).unwrap();
    overlap.try_add_link(pid(2), 1, pid(2)).unwrap();

    let mut section = make_section(&[1, 2]);
    section.try_set(pid(1), &[1.0]).unwrap();
    section.try_set(pid(2), &[2.0]).unwrap();

    let mut constraints = BTreeMap::new();
    constraints.insert(pid(1), vec![DofConstraint::new(0, 9.0)]);

    assemble_section_with_tags_and_ownership::<_, _, AddDelta, _, _>(
        &mut section,
        &overlap,
        &ownership,
        &comm0,
        0,
        tags,
        &constraints,
    )
    .unwrap();

    let (section1, ownership1) = handle1.join().expect("rank1 thread failed");

    assert_eq!(ownership.is_ghost(pid(1)), Some(false));
    assert_eq!(ownership.is_ghost(pid(2)), Some(true));
    assert_eq!(ownership1.is_ghost(pid(1)), Some(true));
    assert_eq!(ownership1.is_ghost(pid(2)), Some(false));

    let val0_p1 = section.try_restrict(pid(1)).unwrap()[0];
    let val0_p2 = section.try_restrict(pid(2)).unwrap()[0];
    let val1_p1 = section1.try_restrict(pid(1)).unwrap()[0];
    let val1_p2 = section1.try_restrict(pid(2)).unwrap()[0];

    assert_eq!(val0_p1, 9.0);
    assert_eq!(val1_p1, 9.0);
    assert_eq!(val0_p2, 6.0);
    assert_eq!(val1_p2, 6.0);
}
