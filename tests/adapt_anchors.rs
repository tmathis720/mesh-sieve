use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::coordinates::Coordinates;
use mesh_sieve::data::section::Section;
use mesh_sieve::data::storage::VecStorage;
use mesh_sieve::topology::anchors::AnchorKind;
use mesh_sieve::topology::cell_type::CellType;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::refine::{
    AnisotropicSplitHints, RefineOptions, refine_mesh_with_options,
};
use mesh_sieve::topology::sieve::{InMemorySieve, Sieve};
use std::collections::HashMap;

fn pt(id: u64) -> PointId {
    PointId::new(id).unwrap()
}

#[test]
fn anisotropic_triangle_edge_split_creates_hanging_anchor_and_constraints() {
    let mut sieve = InMemorySieve::<PointId, ()>::default();
    let cell = pt(10);
    for vertex in [pt(1), pt(2), pt(3)] {
        sieve.add_arrow(cell, vertex, ());
    }

    let mut atlas = Atlas::default();
    atlas.try_insert(cell, 1).unwrap();
    let mut cell_types = Section::<CellType, VecStorage<CellType>>::new(atlas);
    cell_types.try_set(cell, &[CellType::Triangle]).unwrap();

    let mut coord_atlas = Atlas::default();
    for vertex in [pt(1), pt(2), pt(3)] {
        coord_atlas.try_insert(vertex, 2).unwrap();
    }
    let mut coord_section = Section::<f64, VecStorage<f64>>::new(coord_atlas);
    coord_section.try_set(pt(1), &[0.0, 0.0]).unwrap();
    coord_section.try_set(pt(2), &[2.0, 0.0]).unwrap();
    coord_section.try_set(pt(3), &[0.0, 1.0]).unwrap();
    let coordinates = Coordinates::from_section(2, 2, coord_section).unwrap();

    let mut edge_splits = HashMap::new();
    edge_splits.insert(cell, vec![[pt(1), pt(2)]]);
    let refined = refine_mesh_with_options(
        &mut sieve,
        &cell_types,
        Some(&coordinates),
        RefineOptions {
            check_geometry: false,
            anisotropic_splits: Some(AnisotropicSplitHints {
                edge_splits,
                face_splits: HashMap::new(),
            }),
        },
    )
    .unwrap();

    let fine_cells = &refined.cell_refinement[0].1;
    assert_eq!(fine_cells.len(), 2);

    let hanging = refined
        .anchors
        .iter()
        .find(|(_, anchor)| anchor.kind == AnchorKind::Hanging)
        .expect("expected a hanging midpoint anchor");
    assert_eq!(hanging.1.parents, vec![pt(1), pt(2)]);
    assert!(
        refined
            .hanging_constraints
            .constraints()
            .contains_key(&hanging.0)
    );

    let closure = refined
        .anchors
        .anchor_aware_closure(&refined.sieve, [fine_cells[0].0]);
    assert!(closure.contains(&pt(1)));
    assert!(closure.contains(&pt(2)));
}
