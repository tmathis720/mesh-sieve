use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::refine::helpers::{restrict_closure_vec, try_restrict_closure_vec};
use mesh_sieve::data::section::Section;
use mesh_sieve::mesh_error::MeshSieveError;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::sieve::in_memory::InMemorySieve;

fn v(i: u64) -> PointId {
    PointId::new(i).unwrap()
}

#[test]
#[should_panic]
fn restrict_closure_panics_on_missing_point() {
    let sieve = InMemorySieve::<PointId, ()>::default();
    let atlas = Atlas::default();
    let section = Section::<i32>::new(atlas);
    // legacy helper panics when point is missing
    let _ = restrict_closure_vec(&sieve, &section, [v(1)]);
}

#[test]
fn try_restrict_closure_err_on_missing_point() {
    let sieve = InMemorySieve::<PointId, ()>::default();
    let atlas = Atlas::default();
    let section = Section::<i32>::new(atlas);
    let err = try_restrict_closure_vec(&sieve, &section, [v(1)]).unwrap_err();
    assert!(matches!(err, MeshSieveError::PointNotInAtlas(pid) if pid == v(1)));
}
