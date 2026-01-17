use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::section::Section;
use mesh_sieve::data::storage::VecStorage;
use mesh_sieve::topology::cell_type::CellType;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::refine::refine_mesh;
use mesh_sieve::topology::sieve::{InMemorySieve, Sieve};

fn pt(id: u64) -> PointId {
    PointId::new(id).unwrap()
}

fn cell_types_section(cells: &[(PointId, CellType)]) -> Section<CellType, VecStorage<CellType>> {
    let mut atlas = Atlas::default();
    for (cell, _) in cells {
        atlas.try_insert(*cell, 1).unwrap();
    }
    let mut section = Section::<CellType, VecStorage<CellType>>::new(atlas);
    for (cell, cell_type) in cells {
        section.try_set(*cell, &[*cell_type]).unwrap();
    }
    section
}

#[test]
fn refine_tetrahedron_cells() {
    let mut sieve = InMemorySieve::<PointId, ()>::default();
    let cell = pt(10);
    let vertices = [pt(1), pt(2), pt(3), pt(4)];
    for v in vertices {
        sieve.add_arrow(cell, v, ());
    }

    let cells = cell_types_section(&[(cell, CellType::Tetrahedron)]);
    let refined = refine_mesh(&mut sieve, &cells).expect("tet refinement should succeed");

    assert_eq!(refined.cell_refinement.len(), 1);
    let fine_cells = &refined.cell_refinement[0].1;
    assert_eq!(fine_cells.len(), 8);
    for (fine_cell, _) in fine_cells {
        let cone: Vec<_> = refined.sieve.cone_points(*fine_cell).collect();
        assert_eq!(cone.len(), 4);
    }
}

#[test]
fn refine_tetrahedron_with_face_and_edge_strata() {
    let mut sieve = InMemorySieve::<PointId, ()>::default();
    let cell = pt(100);
    let v0 = pt(1);
    let v1 = pt(2);
    let v2 = pt(3);
    let v3 = pt(4);

    let e01 = pt(11);
    let e12 = pt(12);
    let e20 = pt(13);
    let e03 = pt(14);
    let e13 = pt(15);
    let e23 = pt(16);

    sieve.add_arrow(e01, v0, ());
    sieve.add_arrow(e01, v1, ());
    sieve.add_arrow(e12, v1, ());
    sieve.add_arrow(e12, v2, ());
    sieve.add_arrow(e20, v2, ());
    sieve.add_arrow(e20, v0, ());
    sieve.add_arrow(e03, v0, ());
    sieve.add_arrow(e03, v3, ());
    sieve.add_arrow(e13, v1, ());
    sieve.add_arrow(e13, v3, ());
    sieve.add_arrow(e23, v2, ());
    sieve.add_arrow(e23, v3, ());

    let f012 = pt(21);
    let f013 = pt(22);
    let f123 = pt(23);
    let f023 = pt(24);

    sieve.add_arrow(f012, e01, ());
    sieve.add_arrow(f012, e12, ());
    sieve.add_arrow(f012, e20, ());
    sieve.add_arrow(f013, e01, ());
    sieve.add_arrow(f013, e13, ());
    sieve.add_arrow(f013, e03, ());
    sieve.add_arrow(f123, e12, ());
    sieve.add_arrow(f123, e23, ());
    sieve.add_arrow(f123, e13, ());
    sieve.add_arrow(f023, e20, ());
    sieve.add_arrow(f023, e03, ());
    sieve.add_arrow(f023, e23, ());

    sieve.add_arrow(cell, f012, ());
    sieve.add_arrow(cell, f013, ());
    sieve.add_arrow(cell, f123, ());
    sieve.add_arrow(cell, f023, ());

    let cells = cell_types_section(&[(cell, CellType::Tetrahedron)]);
    let refined = refine_mesh(&mut sieve, &cells)
        .expect("tet refinement should accept intermediate face/edge strata");

    assert_eq!(refined.cell_refinement.len(), 1);
    let fine_cells = &refined.cell_refinement[0].1;
    assert_eq!(fine_cells.len(), 8);
    for (fine_cell, _) in fine_cells {
        let cone: Vec<_> = refined.sieve.cone_points(*fine_cell).collect();
        assert_eq!(cone.len(), 4);
    }
}
