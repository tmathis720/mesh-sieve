use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::section::Section;
use mesh_sieve::mesh_error::MeshSieveError;
use mesh_sieve::topology::point::PointId;

fn pid(n: u64) -> PointId {
    PointId::new(n).unwrap()
}

#[test]
fn scatter_fast_path_identical_layout() {
    let mut atlas = Atlas::default();
    atlas.try_insert(pid(1), 3).unwrap();
    atlas.try_insert(pid(2), 2).unwrap();
    let mut sec = Section::<i32>::new(atlas.clone());

    let buf = vec![10, 11, 12, 20, 21];
    let spans = atlas.atlas_map();
    sec.try_scatter_from(&buf, &spans).unwrap();

    assert_eq!(sec.as_flat_slice(), &buf[..]);
}

#[test]
fn scatter_generic_noncontiguous_layout() {
    let mut atlas = Atlas::default();
    atlas.try_insert(pid(1), 2).unwrap();
    atlas.try_insert(pid(2), 2).unwrap();
    let mut sec = Section::<i32>::new(atlas.clone());

    let spans = vec![(2usize, 2usize), (0usize, 2usize)];
    let buf = vec![100, 101, 200, 201];

    sec.try_scatter_from(&buf, &spans).unwrap();

    assert_eq!(sec.as_flat_slice(), &[200, 201, 100, 101]);
}

#[test]
fn scatter_mismatched_total_len_is_error() {
    let mut atlas = Atlas::default();
    atlas.try_insert(pid(1), 3).unwrap();
    let mut sec = Section::<i32>::new(atlas.clone());

    let spans = atlas.atlas_map();
    let buf = vec![1, 2];

    let err = sec.try_scatter_from(&buf, &spans).unwrap_err();
    assert!(matches!(err, MeshSieveError::ScatterLengthMismatch { .. }));
}
