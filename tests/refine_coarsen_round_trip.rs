use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::refine::sieved_array::SievedArray;
use mesh_sieve::data::section::Section;
use mesh_sieve::data::storage::VecStorage;
use mesh_sieve::topology::adapt::{AdaptivityAction, AdaptivityOptions, adapt_topology};
use mesh_sieve::topology::cell_type::CellType;
use mesh_sieve::topology::coarsen::{CoarsenEntity, CoarsenedTopology};
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

fn atlas_for(points: &[PointId]) -> Atlas {
    let mut atlas = Atlas::default();
    for p in points {
        atlas.try_insert(*p, 1).unwrap();
    }
    atlas
}

fn coarsen_from_refinement(
    cell: PointId,
    refined: &mesh_sieve::topology::refine::RefinedMesh,
    coarse_vertices: &[PointId],
) -> CoarsenedTopology {
    let fine_points = refined
        .cell_refinement
        .iter()
        .find(|(c, _)| *c == cell)
        .expect("cell refinement missing");
    let entity = CoarsenEntity {
        coarse_point: cell,
        fine_points: fine_points.1.clone(),
        cone: coarse_vertices.to_vec(),
    };
    mesh_sieve::topology::coarsen::coarsen_topology(&refined.sieve, &[entity])
        .expect("coarsening should succeed")
}

#[test]
fn refine_then_coarsen_round_trip_topology_and_data() {
    let mut sieve = InMemorySieve::<PointId, ()>::default();
    let cell = pt(10);
    let vertices = [pt(1), pt(2), pt(3)];
    for v in vertices {
        sieve.add_arrow(cell, v, ());
    }

    let cells = cell_types_section(&[(cell, CellType::Triangle)]);
    let refined = refine_mesh(&mut sieve, &cells).expect("triangle refinement should succeed");

    let coarsened = coarsen_from_refinement(cell, &refined, &vertices);

    let mut coarsened_cone: Vec<_> = coarsened.sieve.cone_points(cell).collect();
    coarsened_cone.sort_unstable();
    let mut coarse_vertices: Vec<_> = vertices.to_vec();
    coarse_vertices.sort_unstable();
    assert_eq!(coarsened_cone, coarse_vertices);

    let coarse_atlas = atlas_for(&[cell]);
    let mut coarse_array = SievedArray::<PointId, f64>::new(coarse_atlas.clone());
    coarse_array.try_set(cell, &[2.0]).unwrap();

    let fine_points: Vec<PointId> = refined
        .cell_refinement
        .iter()
        .find(|(c, _)| *c == cell)
        .unwrap()
        .1
        .iter()
        .map(|(f, _)| *f)
        .collect();
    let fine_atlas = atlas_for(&fine_points);
    let mut fine_array = SievedArray::<PointId, f64>::new(fine_atlas);
    fine_array
        .try_refine_with_sifter(&coarse_array, &refined.cell_refinement)
        .unwrap();

    let mut coarse_round_trip = SievedArray::<PointId, f64>::new(coarse_atlas);
    let assemble_map = vec![(cell, fine_points)];
    fine_array
        .try_assemble(&mut coarse_round_trip, &assemble_map)
        .unwrap();
    let round_trip_value = coarse_round_trip.try_get(cell).unwrap()[0];
    assert!((round_trip_value - 2.0).abs() < f64::EPSILON);
}

#[test]
fn adaptivity_driver_refine_or_coarsen() {
    let mut sieve = InMemorySieve::<PointId, ()>::default();
    let cell = pt(20);
    let vertices = [pt(11), pt(12), pt(13)];
    for v in vertices {
        sieve.add_arrow(cell, v, ());
    }

    let cells = cell_types_section(&[(cell, CellType::Triangle)]);
    let refine_result = adapt_topology(
        &mut sieve,
        &cells,
        None::<&mesh_sieve::data::coordinates::Coordinates<f64, VecStorage<f64>>>,
        |_cell, _cell_type| 1.0,
        |_cells| Vec::new(),
        AdaptivityOptions::default(),
    )
    .unwrap();
    assert!(matches!(refine_result.action, AdaptivityAction::Refined(_)));

    let refined = match refine_result.action {
        AdaptivityAction::Refined(refined) => refined,
        _ => panic!("expected refinement"),
    };

    let mut refined_sieve = refined.sieve.clone();
    let coarsen_result = adapt_topology(
        &mut refined_sieve,
        &cells,
        None::<&mesh_sieve::data::coordinates::Coordinates<f64, VecStorage<f64>>>,
        |_cell, _cell_type| 0.0,
        |cells_to_coarsen| {
            let fine_points = refined.cell_refinement[0].1.clone();
            vec![CoarsenEntity {
                coarse_point: cells_to_coarsen[0],
                fine_points,
                cone: vertices.to_vec(),
            }]
        },
        AdaptivityOptions::default(),
    )
    .unwrap();
    assert!(matches!(
        coarsen_result.action,
        AdaptivityAction::Coarsened(_)
    ));
}
