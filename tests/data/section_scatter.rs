use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::section::Section;
use mesh_sieve::topology::point::PointId;

#[test]
fn section_scatter_in_order_matches_expected_slices() -> Result<(), Box<dyn std::error::Error>> {
    let mut a = Atlas::default();
    let p1 = PointId::new(1)?; a.try_insert(p1, 2)?;
    let p2 = PointId::new(2)?; a.try_insert(p2, 3)?;
    let p3 = PointId::new(3)?; a.try_insert(p3, 1)?;

    let mut s = Section::<i32>::new(a.clone());
    let total = a.total_len();
    let flat: Vec<i32> = (0..total as i32).collect();
    s.try_scatter_in_order(&flat)?;

    for (pid, (off, len)) in a.iter_entries() {
        let exp = &flat[off..off+len];
        assert_eq!(s.try_restrict(pid)?, exp, "mismatch at point {pid}");
    }
    Ok(())
}

#[test]
fn section_scatter_in_order_length_mismatch_errors() -> Result<(), Box<dyn std::error::Error>> {
    let mut a = Atlas::default();
    a.try_insert(PointId::new(1)?, 3)?;
    a.try_insert(PointId::new(2)?, 2)?;
    let mut s = Section::<u32>::new(a);

    let bad = vec![0u32; 4];
    let err = s.try_scatter_in_order(&bad).unwrap_err();
    assert!(format!("{err}").contains("scatter source length mismatch"));
    Ok(())
}

#[test]
fn section_scatter_with_plan_stale_rejected() -> Result<(), Box<dyn std::error::Error>> {
    let mut a = Atlas::default();
    let p = PointId::new(7)?; a.try_insert(p, 2)?;
    let plan = a.build_scatter_plan();

    let mut s = Section::<i32>::new(a);
    s.try_add_point(PointId::new(8)?, 1)?;

    let buf = vec![0i32; s.as_flat_slice().len()];
    let err = s.try_scatter_with_plan(&buf, &plan).unwrap_err();
    assert!(format!("{err}").contains("Atlas plan is stale"));
    Ok(())
}
