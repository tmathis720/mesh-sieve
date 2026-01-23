use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::constrained_section::ConstrainedSection;
use mesh_sieve::data::hanging_node_constraints::LinearConstraintTerm;
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
fn hanging_constraints_apply_on_nonconforming_patch() {
    let mut sieve = InMemorySieve::<PointId, ()>::default();
    let cell = pt(10);
    let v0 = pt(1);
    let v1 = pt(2);
    let v2 = pt(3);
    for v in [v0, v1, v2] {
        sieve.add_arrow(cell, v, ());
    }

    let cells = cell_types_section(&[(cell, CellType::Triangle)]);
    let refined = refine_mesh(&mut sieve, &cells).expect("triangle refinement should succeed");

    let midpoint = pt(11);
    let mut atlas = Atlas::default();
    for p in refined.sieve.points() {
        atlas.try_insert(p, 1).unwrap();
    }
    let mut section = Section::<f64, VecStorage<f64>>::new(atlas);
    for p in refined.sieve.points() {
        section.try_set(p, &[0.0]).unwrap();
    }
    section.try_set(v0, &[1.0]).unwrap();
    section.try_set(v1, &[3.0]).unwrap();
    section.try_set(v2, &[5.0]).unwrap();

    let mut constrained = ConstrainedSection::new(section);
    constrained.hanging_constraints_mut().insert_constraint(
        midpoint,
        0,
        vec![
            LinearConstraintTerm::new(v0, 0, 0.5),
            LinearConstraintTerm::new(v1, 0, 0.5),
        ],
    );

    constrained
        .apply_all_constraints()
        .expect("hanging constraints should apply");

    let value = constrained
        .section()
        .try_restrict(midpoint)
        .expect("midpoint slice exists")[0];
    assert_eq!(value, 2.0);
}
