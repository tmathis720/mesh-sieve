use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::section::Section;
use mesh_sieve::data::DebugInvariants;
use mesh_sieve::topology::point::PointId;

#[test]
fn atlas_basic_ok() -> Result<(), Box<dyn std::error::Error>> {
    let mut a = Atlas::default();
    let p1 = PointId::new(1)?;
    let p2 = PointId::new(2)?;
    a.try_insert(p1, 3)?;
    a.try_insert(p2, 2)?;
    assert!(a.validate_invariants().is_ok());
    Ok(())
}

#[test]
fn section_size_matches_total_len() -> Result<(), Box<dyn std::error::Error>> {
    let mut a = Atlas::default();
    let p = PointId::new(10)?;
    a.try_insert(p, 4)?;
    let s = Section::<i32>::new(a);
    assert!(s.validate_invariants().is_ok());
    Ok(())
}

#[test]
fn atlas_detects_contiguity_violation() -> Result<(), Box<dyn std::error::Error>> {
    let mut a = Atlas::default();
    let p1 = PointId::new(1)?; let p2 = PointId::new(2)?;
    a.try_insert(p1, 2)?;
    a.try_insert(p2, 2)?;
    // Introduce a gap
    a.force_offset(p2, 3);
    assert!(a.validate_invariants().is_err());
    Ok(())
}
