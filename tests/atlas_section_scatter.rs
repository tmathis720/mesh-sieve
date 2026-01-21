use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::section::Section;
use mesh_sieve::data::storage::VecStorage;
use mesh_sieve::mesh_error::MeshSieveError;
use mesh_sieve::topology::point::PointId;

fn build_section() -> Section<i32, VecStorage<i32>> {
    let mut atlas = Atlas::default();
    atlas.try_insert(PointId::new(1).unwrap(), 2).unwrap();
    atlas.try_insert(PointId::new(2).unwrap(), 1).unwrap();
    atlas.try_insert(PointId::new(3).unwrap(), 2).unwrap();
    Section::<i32, VecStorage<i32>>::new(atlas)
}

#[test]
fn round_trip_in_order() {
    let mut sec = build_section();
    let buf = vec![1, 2, 3, 4, 5];
    sec.try_scatter_in_order(&buf).unwrap();
    assert_eq!(sec.as_flat_slice(), buf.as_slice());
    assert_eq!(sec.gather_in_order(), buf);
}

#[test]
fn remove_point_round_trip() {
    let mut sec = build_section();
    let buf = vec![1, 2, 3, 4, 5];
    sec.try_scatter_in_order(&buf).unwrap();
    let p2 = PointId::new(2).unwrap();
    sec.try_remove_point(p2).unwrap();
    let expected = vec![1, 2, 4, 5];
    assert_eq!(sec.gather_in_order(), expected);
    let new_buf = vec![9, 8, 7, 6];
    sec.try_scatter_in_order(&new_buf).unwrap();
    assert_eq!(sec.gather_in_order(), new_buf);
}

#[test]
fn plan_staleness() {
    let mut atlas = Atlas::default();
    let p1 = PointId::new(1).unwrap();
    let p2 = PointId::new(2).unwrap();
    atlas.try_insert(p1, 1).unwrap();
    atlas.try_insert(p2, 1).unwrap();
    let plan = atlas.build_scatter_plan();
    let mut sec = Section::<i32, VecStorage<i32>>::new(atlas);
    let buf = vec![10, 20];
    let p3 = PointId::new(3).unwrap();
    sec.try_add_point(p3, 1).unwrap();
    assert!(matches!(
        sec.try_scatter_with_plan(&buf, &plan),
        Err(MeshSieveError::AtlasPlanStale { .. })
    ));
}

#[test]
fn scatter_bounds_validation() {
    let mut sec = build_section();
    assert!(matches!(
        sec.try_scatter_in_order(&[1, 2, 3, 4]),
        Err(MeshSieveError::ScatterLengthMismatch { .. })
    ));
    assert!(matches!(
        sec.try_scatter_from(&vec![1, 2, 3, 4, 5, 6], &[(0, 6)]),
        Err(MeshSieveError::ScatterChunkMismatch { .. })
    ));
}

#[test]
fn closure_mappers() {
    let mut sec = build_section();
    let mut val = 0;
    sec.for_each_in_order_mut(|_, sl| {
        for v in sl {
            *v = val;
            val += 1;
        }
    });
    let expected: Vec<i32> = (0..sec.as_flat_slice().len() as i32).collect();
    assert_eq!(sec.as_flat_slice(), expected.as_slice());
    let mut sum = 0;
    sec.for_each_in_order(|_, sl| {
        sum += sl.iter().sum::<i32>();
    });
    assert_eq!(sum, expected.iter().sum::<i32>());
}
