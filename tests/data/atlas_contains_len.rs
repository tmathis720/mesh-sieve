use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::topology::point::PointId;

#[test]
fn atlas_contains_and_len_work() -> Result<(), Box<dyn std::error::Error>> {
    let mut a = Atlas::default();
    let p1 = PointId::new(1)?;
    let p2 = PointId::new(2)?;
    assert_eq!(a.len(), 0);
    assert!(!a.contains(p1));
    a.try_insert(p1, 2)?;
    a.try_insert(p2, 1)?;
    assert!(a.contains(p1) && a.contains(p2));
    assert_eq!(a.len(), 2);
    assert!(!a.is_empty());
    Ok(())
}
