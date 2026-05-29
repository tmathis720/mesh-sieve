use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::coordinates::Coordinates;
use mesh_sieve::data::section::Section;
use mesh_sieve::data::storage::VecStorage;
use mesh_sieve::diagnostics::{
    FvReadinessOptions, FvReadinessThresholds, MeshCheckOptions, fv_readiness_report,
    fv_readiness_report_json, fv_readiness_report_text, mesh_dot_graph, mesh_json_debug_dump,
    run_mesh_checks,
};
use mesh_sieve::mesh_error::MeshSieveError;
use mesh_sieve::topology::cell_type::CellType;
use mesh_sieve::topology::ownership::PointOwnership;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::sieve::{MeshSieve, MutableSieve, Sieve};

fn pid(id: u64) -> PointId {
    PointId::new(id).unwrap()
}

fn tiny_mesh() -> MeshSieve {
    let mut mesh = MeshSieve::default();
    mesh.add_arrow(pid(3), pid(1), ());
    mesh.add_arrow(pid(3), pid(2), ());
    mesh
}

fn triangle_pair_fixture() -> Result<
    (
        MeshSieve,
        Section<CellType, VecStorage<CellType>>,
        Coordinates<f64, VecStorage<f64>>,
    ),
    MeshSieveError,
> {
    let mut s = MeshSieve::default();
    for id in 1..=11 {
        MutableSieve::add_point(&mut s, pid(id));
    }
    let (v1, v2, v3, v4, e12, e23, e31, e24, e43, c1, c2) = (
        pid(1),
        pid(2),
        pid(3),
        pid(4),
        pid(5),
        pid(6),
        pid(7),
        pid(8),
        pid(9),
        pid(10),
        pid(11),
    );
    for (e, a, b) in [
        (e12, v1, v2),
        (e23, v2, v3),
        (e31, v3, v1),
        (e24, v2, v4),
        (e43, v4, v3),
    ] {
        s.add_arrow(e, a, ());
        s.add_arrow(e, b, ());
    }
    for e in [e12, e23, e31] {
        s.add_arrow(c1, e, ());
    }
    for e in [e23, e24, e43] {
        s.add_arrow(c2, e, ());
    }
    let mut a = Atlas::default();
    for v in [v1, v2, v3, v4] {
        a.try_insert(v, 2)?;
    }
    let mut coords = Coordinates::<f64, VecStorage<f64>>::try_new(2, 2, a)?;
    coords.section_mut().try_set(v1, &[0.0, 0.0])?;
    coords.section_mut().try_set(v2, &[1.0, 0.0])?;
    coords.section_mut().try_set(v3, &[0.0, 1.0])?;
    coords.section_mut().try_set(v4, &[1.0, 1.0])?;
    let mut ta = Atlas::default();
    for q in [c1, c2, e12, e23, e31, e24, e43] {
        ta.try_insert(q, 1)?;
    }
    let mut types = Section::<CellType, VecStorage<CellType>>::new(ta);
    types.try_set(c1, &[CellType::Triangle])?;
    types.try_set(c2, &[CellType::Triangle])?;
    for e in [e12, e23, e31, e24, e43] {
        types.try_set(e, &[CellType::Segment])?;
    }
    Ok((s, types, coords))
}

#[test]
fn check_all_detects_missing_ownership_precisely() {
    let mut mesh = tiny_mesh();
    let mut ownership = PointOwnership::default();
    ownership.set(pid(1), 0, false).unwrap();

    let err = run_mesh_checks::<
        f64,
        VecStorage<f64>,
        VecStorage<mesh_sieve::topology::cell_type::CellType>,
    >(
        &mut mesh,
        None,
        None,
        Some(&ownership),
        None,
        std::iter::empty(),
        MeshCheckOptions::all(),
    )
    .unwrap_err();

    assert!(matches!(
        err,
        MeshSieveError::TopologyPointMissingOwnership { .. }
    ));
}

#[test]
fn viewers_emit_expected_formats() {
    let mesh = tiny_mesh();
    let dot = mesh_dot_graph(&mesh);
    assert!(dot.starts_with("digraph MeshSieve"));
    let json = mesh_json_debug_dump(&mesh);
    assert!(json.contains("\"edges\""));

    let mut atlas = Atlas::default();
    atlas.try_insert(pid(1), 1).unwrap();
    let _section: Section<f64, VecStorage<f64>> = Section::new(atlas);
}

#[test]
fn fv_readiness_pass_and_output_formats_are_deterministic() -> Result<(), MeshSieveError> {
    let (sieve, cell_types, coords) = triangle_pair_fixture()?;
    let thresholds = FvReadinessThresholds {
        max_non_orthogonality_deg: 180.0,
        max_skewness: 10.0,
        min_cell_char_length: 0.0,
    };
    let report = fv_readiness_report(
        &sieve,
        &cell_types,
        &coords,
        thresholds,
        FvReadinessOptions::default(),
    )?;
    assert!(report.passed);
    assert_eq!(report.violations.len(), 0);
    let json = fv_readiness_report_json(&report)?;
    assert_eq!(
        json,
        "{\"passed\":true,\"thresholds\":{\"max_non_orthogonality_deg\":180.0,\"max_skewness\":10.0,\"min_cell_char_length\":0.0},\"non_orthogonality\":{\"failed\":0,\"total\":5},\"skewness\":{\"failed\":0,\"total\":5},\"cell_char_length\":{\"failed\":0,\"total\":2},\"violations\":[]}"
    );
    let text = fv_readiness_report_text(&report);
    assert!(text.contains("fv_readiness_passed=true"));
    assert!(text.contains("non_orthogonality_failed=0/5"));
    Ok(())
}

#[test]
fn fv_readiness_violation_and_strict_mode_error() -> Result<(), MeshSieveError> {
    let (sieve, cell_types, coords) = triangle_pair_fixture()?;
    let thresholds = FvReadinessThresholds {
        max_non_orthogonality_deg: 0.0,
        max_skewness: 0.0,
        min_cell_char_length: 1.0,
    };
    let report = fv_readiness_report(
        &sieve,
        &cell_types,
        &coords,
        thresholds,
        FvReadinessOptions::default(),
    )?;
    assert!(!report.passed);
    assert!(!report.violations.is_empty());
    assert!(
        report
            .violations
            .iter()
            .any(|v| v.metric == "non_orthogonality_deg")
    );
    assert!(report.violations.iter().any(|v| v.metric == "skewness"));
    assert!(
        report
            .violations
            .iter()
            .any(|v| v.metric == "cell_char_length")
    );
    let err = fv_readiness_report(
        &sieve,
        &cell_types,
        &coords,
        thresholds,
        FvReadinessOptions { strict: true },
    )
    .unwrap_err();
    assert!(matches!(err, MeshSieveError::InvalidGeometry(_)));
    Ok(())
}
