use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::sieve::{InMemorySieve, Sieve};
use mesh_sieve::topology::utils::dimension;

fn pid(id: u64) -> PointId {
    PointId::new(id).unwrap()
}

#[test]
fn dimension_for_1d_mesh() {
    let mut sieve = InMemorySieve::<PointId, ()>::default();
    let edge = pid(10);
    let v0 = pid(1);
    let v1 = pid(2);
    sieve.add_arrow(edge, v0, ());
    sieve.add_arrow(edge, v1, ());

    assert_eq!(dimension(&mut sieve).unwrap(), 1);
}

#[test]
fn dimension_for_2d_mesh() {
    let mut sieve = InMemorySieve::<PointId, ()>::default();
    let cell = pid(100);
    let v0 = pid(1);
    let v1 = pid(2);
    let v2 = pid(3);

    let e01 = pid(11);
    let e12 = pid(12);
    let e20 = pid(13);

    sieve.add_arrow(e01, v0, ());
    sieve.add_arrow(e01, v1, ());
    sieve.add_arrow(e12, v1, ());
    sieve.add_arrow(e12, v2, ());
    sieve.add_arrow(e20, v2, ());
    sieve.add_arrow(e20, v0, ());

    sieve.add_arrow(cell, e01, ());
    sieve.add_arrow(cell, e12, ());
    sieve.add_arrow(cell, e20, ());

    assert_eq!(dimension(&mut sieve).unwrap(), 2);
}

#[test]
fn dimension_for_3d_mesh() {
    let mut sieve = InMemorySieve::<PointId, ()>::default();
    let cell = pid(1000);

    let v0 = pid(1);
    let v1 = pid(2);
    let v2 = pid(3);
    let v3 = pid(4);

    let e01 = pid(11);
    let e12 = pid(12);
    let e20 = pid(13);
    let e03 = pid(14);
    let e13 = pid(15);
    let e23 = pid(16);

    sieve.add_arrow(e01, v0, ());
    sieve.add_arrow(e01, v1, ());
    sieve.add_arrow(e12, v1, ());
    sieve.add_arrow(e12, v2, ());
    sieve.add_arrow(e20, v2, ());
    sieve.add_arrow(e20, v0, ());
    sieve.add_arrow(e03, v0, ());
    sieve.add_arrow(e03, v3, ());
    sieve.add_arrow(e13, v1, ());
    sieve.add_arrow(e13, v3, ());
    sieve.add_arrow(e23, v2, ());
    sieve.add_arrow(e23, v3, ());

    let f012 = pid(21);
    let f013 = pid(22);
    let f123 = pid(23);
    let f023 = pid(24);

    sieve.add_arrow(f012, e01, ());
    sieve.add_arrow(f012, e12, ());
    sieve.add_arrow(f012, e20, ());
    sieve.add_arrow(f013, e01, ());
    sieve.add_arrow(f013, e13, ());
    sieve.add_arrow(f013, e03, ());
    sieve.add_arrow(f123, e12, ());
    sieve.add_arrow(f123, e23, ());
    sieve.add_arrow(f123, e13, ());
    sieve.add_arrow(f023, e20, ());
    sieve.add_arrow(f023, e03, ());
    sieve.add_arrow(f023, e23, ());

    sieve.add_arrow(cell, f012, ());
    sieve.add_arrow(cell, f013, ());
    sieve.add_arrow(cell, f123, ());
    sieve.add_arrow(cell, f023, ());

    assert_eq!(dimension(&mut sieve).unwrap(), 3);
}
