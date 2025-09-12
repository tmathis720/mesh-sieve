use mesh_sieve::data::{atlas::Atlas, section::Section, storage::VecStorage};
use mesh_sieve::topology::point::PointId;

#[test]
fn section_with_atlas_mut_rebuilds_data() -> Result<(), Box<dyn std::error::Error>> {
    let mut a = Atlas::default();
    let p = PointId::new(10)?;
    a.try_insert(p, 2)?;
    let mut s = Section::<i32, VecStorage<i32>>::new(a);
    s.try_set(p, &[7, 8])?;

    // Add a new point and ensure it's defaulted
    s.with_atlas_mut(|atlas| {
        atlas.try_insert(PointId::new(11).unwrap(), 3).unwrap();
    })?;
    assert_eq!(s.atlas().total_len(), 5);
    assert_eq!(s.try_restrict(p)?, &[7, 8]);
    assert_eq!(s.try_restrict(PointId::new(11)?)?, &[0, 0, 0]);
    Ok(())
}

#[test]
fn section_with_atlas_mut_rejects_len_change() -> Result<(), Box<dyn std::error::Error>> {
    use mesh_sieve::mesh_error::MeshSieveError;
    let mut a = Atlas::default();
    let p = PointId::new(10)?;
    a.try_insert(p, 3)?;
    let mut s = Section::<i32, VecStorage<i32>>::new(a);
    let err = s
        .with_atlas_mut(|atlas| {
            atlas.remove_point(p).unwrap();
            atlas.try_insert(p, 4).unwrap();
        })
        .unwrap_err();
    assert!(matches!(
        err,
        MeshSieveError::AtlasPointLengthChanged { expected: 3, found: 4, .. }
    ));
    Ok(())
}
