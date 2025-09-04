use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::refine::sieved_array::SievedArray;
use mesh_sieve::topology::point::PointId;

#[test]
fn iter_in_order_matches_slices() -> Result<(), Box<dyn std::error::Error>> {
    let mut a = Atlas::default();
    let p1 = PointId::new(1)?;
    a.try_insert(p1, 2)?;
    let p2 = PointId::new(2)?;
    a.try_insert(p2, 3)?;
    let mut arr = SievedArray::<PointId, i32>::new(a);
    arr.try_set(p1, &[1, 2])?;
    arr.try_set(p2, &[3, 4, 5])?;

    let got: Vec<_> = arr.iter_in_order().collect();
    assert_eq!(got.len(), 2);
    assert_eq!(got[0], (p1, &[1, 2][..]));
    assert_eq!(got[1], (p2, &[3, 4, 5][..]));
    Ok(())
}
