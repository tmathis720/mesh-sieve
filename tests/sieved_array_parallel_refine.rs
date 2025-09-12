#[cfg(feature = "rayon")]
use mesh_sieve::data::atlas::Atlas;
#[cfg(feature = "rayon")]
use mesh_sieve::data::refine::sieved_array::SievedArray;
#[cfg(feature = "rayon")]
use mesh_sieve::mesh_error::MeshSieveError;
#[cfg(feature = "rayon")]
use mesh_sieve::topology::arrow::Polarity;
#[cfg(feature = "rayon")]
use mesh_sieve::topology::point::PointId;

#[cfg(feature = "rayon")]
fn pt(i: u64) -> PointId {
    PointId::new(i).unwrap()
}

#[cfg(feature = "rayon")]
fn make_sieved(points: &[(PointId, usize)]) -> SievedArray<PointId, i32> {
    let mut atlas = Atlas::default();
    for (p, len) in points {
        atlas.try_insert(*p, *len).unwrap();
    }
    SievedArray::new(atlas)
}

#[cfg(feature = "rayon")]
#[test]
fn parallel_refine_matches_serial() {
    let c = pt(1);
    let f1 = pt(2);
    let f2 = pt(3);
    let mut coarse = make_sieved(&[(c, 2)]);
    coarse.try_set(c, &[10, 20]).unwrap();
    let mut fine_serial = make_sieved(&[(f1, 2), (f2, 2)]);
    let mut fine_parallel = fine_serial.clone();

    let refinement = vec![(c, vec![(f1, Polarity::Forward), (f2, Polarity::Reverse)])];
    fine_serial
        .try_refine_with_sifter(&coarse, &refinement)
        .unwrap();
    fine_parallel
        .try_refine_with_sifter_parallel(&coarse, &refinement)
        .unwrap();
    assert_eq!(fine_serial.try_get(f1).unwrap(), &[10, 20]);
    assert_eq!(fine_serial.try_get(f2).unwrap(), &[20, 10]);
    assert_eq!(
        fine_serial.try_get(f1).unwrap(),
        fine_parallel.try_get(f1).unwrap()
    );
    assert_eq!(
        fine_serial.try_get(f2).unwrap(),
        fine_parallel.try_get(f2).unwrap()
    );
}

#[cfg(feature = "rayon")]
#[test]
fn parallel_refine_short_circuits_on_error() {
    let c = pt(1);
    let f_ok = pt(2);
    let f_bad = pt(3);
    let mut coarse = make_sieved(&[(c, 2)]);
    coarse.try_set(c, &[1, 2]).unwrap();
    let mut fine = make_sieved(&[(f_ok, 2), (f_bad, 1)]); // f_bad has mismatched len
    fine.try_set(f_ok, &[7, 7]).unwrap();
    fine.try_set(f_bad, &[9]).unwrap();
    let refinement = vec![(
        c,
        vec![(f_ok, Polarity::Forward), (f_bad, Polarity::Forward)],
    )];
    let err = fine
        .try_refine_with_sifter_parallel(&coarse, &refinement)
        .unwrap_err();
    assert!(matches!(
        err,
        MeshSieveError::SievedArraySliceLengthMismatch { .. }
    ));
    // ensure no partial writes occurred
    assert_eq!(fine.try_get(f_ok).unwrap(), &[7, 7]);
    assert_eq!(fine.try_get(f_bad).unwrap(), &[9]);
}

#[cfg(feature = "rayon")]
#[test]
fn parallel_refine_detects_duplicates() {
    let c1 = pt(1);
    let c2 = pt(2);
    let f = pt(3);
    let mut coarse = make_sieved(&[(c1, 2), (c2, 2)]);
    coarse.try_set(c1, &[1, 2]).unwrap();
    coarse.try_set(c2, &[3, 4]).unwrap();
    let mut fine = make_sieved(&[(f, 2)]);
    let refinement = vec![
        (c1, vec![(f, Polarity::Forward)]),
        (c2, vec![(f, Polarity::Forward)]),
    ];
    let err = fine
        .try_refine_with_sifter_parallel(&coarse, &refinement)
        .unwrap_err();
    match err {
        MeshSieveError::DuplicateRefinementTarget { fine: p } => assert_eq!(p, f),
        other => panic!("unexpected error: {other:?}"),
    }
}

#[cfg(feature = "rayon")]
#[test]
fn parallel_refine_is_deterministic() {
    use rand::seq::SliceRandom;
    let c1 = pt(1);
    let c2 = pt(2);
    let f1 = pt(3);
    let f2 = pt(4);
    let mut coarse = make_sieved(&[(c1, 2), (c2, 2)]);
    coarse.try_set(c1, &[5, 6]).unwrap();
    coarse.try_set(c2, &[7, 8]).unwrap();
    let refinement = vec![
        (c1, vec![(f1, Polarity::Forward)]),
        (c2, vec![(f2, Polarity::Reverse)]),
    ];
    // serial reference
    let mut reference = make_sieved(&[(f1, 2), (f2, 2)]);
    reference
        .try_refine_with_sifter(&coarse, &refinement)
        .unwrap();
    let expected_f1 = reference.try_get(f1).unwrap().to_vec();
    let expected_f2 = reference.try_get(f2).unwrap().to_vec();

    for _ in 0..10 {
        let mut shuffled = refinement.clone();
        shuffled.shuffle(&mut rand::thread_rng());
        let mut fine = make_sieved(&[(f1, 2), (f2, 2)]);
        fine.try_refine_with_sifter_parallel(&coarse, &shuffled)
            .unwrap();
        assert_eq!(fine.try_get(f1).unwrap(), expected_f1.as_slice());
        assert_eq!(fine.try_get(f2).unwrap(), expected_f2.as_slice());
    }
}
