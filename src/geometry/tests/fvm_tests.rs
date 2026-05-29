use crate::data::atlas::Atlas;
use crate::data::coordinates::Coordinates;
use crate::data::section::Section;
use crate::data::storage::VecStorage;
use crate::geometry::fvm::build_fvm_face_metrics;
use crate::mesh_error::MeshSieveError;
use crate::topology::cell_type::CellType;
use crate::topology::point::PointId;
use crate::topology::sieve::{InMemorySieve, MutableSieve, Sieve};

fn p(id: u64) -> PointId {
    PointId::new(id).unwrap()
}

#[test]
fn fvm_internal_and_boundary_faces_triangle_pair() -> Result<(), MeshSieveError> {
    let mut s = InMemorySieve::<PointId, ()>::default();
    for id in 1..=11 {
        MutableSieve::add_point(&mut s, p(id));
    }
    let (v1, v2, v3, v4, e12, e23, e31, e24, e43, c1, c2) = (
        p(1),
        p(2),
        p(3),
        p(4),
        p(5),
        p(6),
        p(7),
        p(8),
        p(9),
        p(10),
        p(11),
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

    let metrics = build_fvm_face_metrics(&s, &types, &coords)?;
    assert_eq!(metrics.len(), 5);
    let internal = metrics.iter().find(|m| m.face == e23).unwrap();
    assert_eq!(internal.owner, c1);
    assert_eq!(internal.neighbor, Some(c2));
    assert!(internal.non_orthogonality_deg.unwrap() >= 0.0);
    let boundary = metrics.iter().find(|m| m.face == e12).unwrap();
    assert_eq!(boundary.neighbor, None);
    assert!(boundary.orthogonal_distance > 0.0);
    Ok(())
}

#[test]
fn fvm_owner_neighbor_orientation_and_skewness_for_warped_quad() -> Result<(), MeshSieveError> {
    let mut s = InMemorySieve::<PointId, ()>::default();
    for id in 1..=13 {
        MutableSieve::add_point(&mut s, p(id));
    }
    let (v1, v2, v3, v4, fint, fb1, fb2, fb3, fb4, c1, c2, b1, b2) = (
        p(1),
        p(2),
        p(3),
        p(4),
        p(5),
        p(6),
        p(7),
        p(8),
        p(9),
        p(10),
        p(11),
        p(12),
        p(13),
    );
    for (f, vs) in [
        (fint, vec![v1, v2, v3, v4]),
        (fb1, vec![v1, v2]),
        (fb2, vec![v2, v3]),
        (fb3, vec![v3, v4]),
        (fb4, vec![v4, v1]),
    ] {
        for v in vs {
            s.add_arrow(f, v, ());
        }
    }
    for f in [fint, fb1, fb2] {
        s.add_arrow(c1, f, ());
    }
    for f in [fint, fb3, fb4] {
        s.add_arrow(c2, f, ());
    }
    s.add_arrow(b1, fb1, ());
    s.add_arrow(b2, fb3, ());

    let mut a = Atlas::default();
    for v in [v1, v2, v3, v4] {
        a.try_insert(v, 3)?;
    }
    let mut coords = Coordinates::<f64, VecStorage<f64>>::try_new(3, 3, a)?;
    coords.section_mut().try_set(v1, &[0.0, 0.0, 0.0])?;
    coords.section_mut().try_set(v2, &[1.0, 0.0, 0.2])?;
    coords.section_mut().try_set(v3, &[1.0, 1.0, 0.1])?;
    coords.section_mut().try_set(v4, &[0.0, 1.0, -0.1])?;

    let mut ta = Atlas::default();
    for q in [c1, c2, b1, b2, fint, fb1, fb2, fb3, fb4] {
        ta.try_insert(q, 1)?;
    }
    let mut types = Section::<CellType, VecStorage<CellType>>::new(ta);
    for c in [c1, c2, b1, b2] {
        types.try_set(c, &[CellType::Polyhedron])?;
    }
    types.try_set(fint, &[CellType::Quadrilateral])?;
    for f in [fb1, fb2, fb3, fb4] {
        types.try_set(f, &[CellType::Segment])?;
    }

    let metrics = build_fvm_face_metrics(&s, &types, &coords)?;
    let m = metrics.iter().find(|m| m.face == fint).unwrap();
    assert_eq!(m.owner, c1);
    assert_eq!(m.neighbor, Some(c2));
    let d = m.owner_to_neighbor.unwrap();
    let alignment = m.area_vector[0] * d[0] + m.area_vector[1] * d[1] + m.area_vector[2] * d[2];
    assert!(alignment >= -1.0e-12);
    assert!(m.non_orthogonality_deg.unwrap_or(0.0) <= 90.0);
    assert!(m.skewness_vector.iter().map(|x| x * x).sum::<f64>() > 0.0);
    Ok(())
}

#[test]
fn fvm_layered_prism_like_mesh_invariants() -> Result<(), MeshSieveError> {
    let mut s = InMemorySieve::<PointId, ()>::default();
    for id in 1..=31 {
        MutableSieve::add_point(&mut s, p(id));
    }
    let (v1, v2, v3, v4, v5, v6, v7, v8, v9) =
        (p(1), p(2), p(3), p(4), p(5), p(6), p(7), p(8), p(9));
    let (fbot, ftop1, ftop2, f12, f23, f31, f45, f56, f64, f17, f28, f39) = (
        p(10),
        p(11),
        p(12),
        p(13),
        p(14),
        p(15),
        p(16),
        p(17),
        p(18),
        p(6),
        p(7),
        p(8),
    );
    for (f, vs) in [
        (fbot, vec![v1, v2, v3]),
        (ftop1, vec![v4, v5, v6]),
        (ftop2, vec![v7, v8, v9]),
        (f12, vec![v1, v2, v5, v4]),
        (f23, vec![v2, v3, v6, v5]),
        (f31, vec![v3, v1, v4, v6]),
        (f45, vec![v4, v5, v8, v7]),
        (f56, vec![v5, v6, v9, v8]),
        (f64, vec![v6, v4, v7, v9]),
        (f17, vec![v1, v2]),
        (f28, vec![v2, v3]),
        (f39, vec![v3, v1]),
    ] {
        for v in vs {
            s.add_arrow(f, v, ());
        }
    }
    let c1 = p(30);
    let c2 = p(31);
    for f in [fbot, ftop1, f12, f23, f31] {
        s.add_arrow(c1, f, ());
    }
    for f in [ftop1, ftop2, f45, f56, f64] {
        s.add_arrow(c2, f, ());
    }

    let mut a = Atlas::default();
    for v in [v1, v2, v3, v4, v5, v6, v7, v8, v9] {
        a.try_insert(v, 3)?;
    }
    let mut coords = Coordinates::<f64, VecStorage<f64>>::try_new(3, 3, a)?;
    for (v, c) in [
        (v1, [0.0, 0.0, 0.0]),
        (v2, [1.0, 0.0, 0.0]),
        (v3, [0.0, 1.0, 0.1]),
        (v4, [0.0, 0.0, 1.0]),
        (v5, [1.0, 0.0, 1.1]),
        (v6, [0.0, 1.0, 1.0]),
        (v7, [0.0, 0.0, 2.0]),
        (v8, [1.0, 0.0, 2.1]),
        (v9, [0.0, 1.0, 2.0]),
    ] {
        coords.section_mut().try_set(v, &c)?;
    }

    let mut ta = Atlas::default();
    for q in [
        c1, c2, fbot, ftop1, ftop2, f12, f23, f31, f45, f56, f64, f17, f28, f39,
    ] {
        ta.try_insert(q, 1)?;
    }
    let mut types = Section::<CellType, VecStorage<CellType>>::new(ta);
    types.try_set(c1, &[CellType::Prism])?;
    types.try_set(c2, &[CellType::Prism])?;
    for f in [fbot, ftop1, ftop2] {
        types.try_set(f, &[CellType::Triangle])?;
    }
    for f in [f12, f23, f31, f45, f56, f64] {
        types.try_set(f, &[CellType::Quadrilateral])?;
    }
    for f in [f17, f28, f39] {
        types.try_set(f, &[CellType::Segment])?;
    }

    let metrics = build_fvm_face_metrics(&s, &types, &coords)?;
    if let Some(shared) = metrics.iter().find(|m| m.neighbor.is_some()) {
        assert_eq!(shared.owner, c1);
        assert_eq!(shared.neighbor, Some(c2));
        assert!(shared.orthogonal_distance > 0.0);
        assert!(shared.non_orthogonality_deg.unwrap_or(0.0) >= 0.0);
    }
    assert!(
        metrics
            .iter()
            .filter(|m| m.neighbor.is_none())
            .all(|m| m.area_magnitude >= 0.0)
    );
    Ok(())
}
