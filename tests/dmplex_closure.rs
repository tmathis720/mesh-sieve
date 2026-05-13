use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::{
    ClosureOrder, IdentitySectionSym, Section, TableSectionSym, VecStorage, add_closure,
    build_closure_index, build_closure_index_unoriented, get_closure, set_closure,
};
use mesh_sieve::discretization::runtime::dof_map_from_closure_index;
use mesh_sieve::topology::orientation::Sign;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::sieve::oriented::OrientedSieve;
use mesh_sieve::topology::sieve::{InMemoryOrientedSieve, InMemorySieve, Sieve};

fn p(id: u64) -> PointId {
    PointId::new(id).unwrap()
}

#[test]
fn dmplex_closure_get_set_add_follow_breadth_first_indices() {
    let mut topo = InMemorySieve::<PointId, ()>::default();
    topo.add_arrow(p(1), p(2), ());
    topo.add_arrow(p(1), p(3), ());
    topo.add_arrow(p(2), p(4), ());

    let mut atlas = Atlas::default();
    for (point, len) in [(p(1), 1), (p(2), 2), (p(3), 1), (p(4), 1)] {
        atlas.try_insert(point, len).unwrap();
    }
    let mut section = Section::<i32, VecStorage<i32>>::new(atlas);
    section.try_scatter_in_order(&[10, 20, 21, 30, 40]).unwrap();

    let index = build_closure_index_unoriented(
        &topo,
        &section,
        p(1),
        0,
        &ClosureOrder::BreadthFirstDmpLex,
        &IdentitySectionSym,
    )
    .unwrap();

    assert_eq!(
        index.point_order().collect::<Vec<_>>(),
        vec![p(1), p(2), p(3), p(4)]
    );
    assert_eq!(
        get_closure(&section, &index).unwrap(),
        vec![10, 20, 21, 30, 40]
    );

    set_closure(&mut section, &index, &[1, 2, 3, 4, 5]).unwrap();
    add_closure(&mut section, &index, &[10, 20, 30, 40, 50]).unwrap();

    assert_eq!(
        get_closure(&section, &index).unwrap(),
        vec![11, 22, 33, 44, 55]
    );
}

#[test]
fn orientation_symmetry_controls_closure_values_and_dof_map_slots() {
    let mut topo = InMemoryOrientedSieve::<PointId, (), Sign>::default();
    topo.add_arrow_o(p(1), p(2), (), Sign(true));

    let mut atlas = Atlas::default();
    atlas.try_insert(p(2), 3).unwrap();
    let mut section = Section::<i32, VecStorage<i32>>::new(atlas);
    section.try_scatter_in_order(&[7, 8, 9]).unwrap();

    let mut sym = TableSectionSym::new();
    sym.insert_orientation(Sign(true), vec![2, 1, 0]);

    let index = build_closure_index(
        &topo,
        &section,
        p(1),
        42,
        &ClosureOrder::BreadthFirstDmpLex,
        &sym,
    )
    .unwrap();

    assert_eq!(get_closure(&section, &index).unwrap(), vec![9, 8, 7]);

    let dof_map = dof_map_from_closure_index(&index);
    let slots: Vec<_> = dof_map
        .closure_dofs()
        .iter()
        .map(|dof| (dof.point, dof.local_dof))
        .collect();
    assert_eq!(slots, vec![(p(2), 2), (p(2), 1), (p(2), 0)]);
    assert_eq!(dof_map.slot_index(p(2), 2), Some(0));
    assert_eq!(dof_map.slot_index(p(2), 0), Some(2));
}

#[test]
fn lexicographic_and_custom_orders_are_predictable() {
    let mut topo = InMemorySieve::<PointId, ()>::default();
    topo.add_arrow(p(10), p(30), ());
    topo.add_arrow(p(10), p(20), ());
    topo.add_arrow(p(30), p(40), ());

    let mut atlas = Atlas::default();
    for point in [p(10), p(20), p(30), p(40)] {
        atlas.try_insert(point, 1).unwrap();
    }
    let section = Section::<i32, VecStorage<i32>>::new(atlas);

    let lex = build_closure_index_unoriented(
        &topo,
        &section,
        p(10),
        0,
        &ClosureOrder::LexicographicTensor { dims: vec![2, 2] },
        &IdentitySectionSym,
    )
    .unwrap();
    assert_eq!(
        lex.point_order().collect::<Vec<_>>(),
        vec![p(10), p(20), p(30), p(40)]
    );

    let custom = build_closure_index_unoriented(
        &topo,
        &section,
        p(10),
        0,
        &ClosureOrder::Custom(vec![p(40), p(20)]),
        &IdentitySectionSym,
    )
    .unwrap();
    assert_eq!(
        custom.point_order().collect::<Vec<_>>(),
        vec![p(40), p(20), p(10), p(30)]
    );
}
