use mesh_sieve::adapt::{
    adapt_with_quality_and_transfer, evaluate_quality_metrics, AdaptationAction, QualityThresholds,
};
use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::coordinates::Coordinates;
use mesh_sieve::data::refine::sieved_array::SievedArray;
use mesh_sieve::data::section::Section;
use mesh_sieve::data::storage::VecStorage;
use mesh_sieve::topology::cell_type::CellType;
use mesh_sieve::topology::coarsen::CoarsenEntity;
use mesh_sieve::topology::point::PointId;
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
fn refine_low_quality_cells_by_size_metric() {
    let mut sieve = InMemorySieve::<PointId, ()>::default();
    let cell = pt(10);
    let vertices = [pt(1), pt(2), pt(3)];
    for v in vertices {
        sieve.add_arrow(cell, v, ());
    }

    let cell_types = cell_types_section(&[(cell, CellType::Triangle)]);

    let mut coord_atlas = Atlas::default();
    for v in vertices {
        coord_atlas.try_insert(v, 2).unwrap();
    }
    let mut coords = Coordinates::<f64, VecStorage<f64>>::try_new(2, coord_atlas).unwrap();
    coords
        .try_restrict_mut(vertices[0])
        .unwrap()
        .copy_from_slice(&[0.0, 0.0]);
    coords
        .try_restrict_mut(vertices[1])
        .unwrap()
        .copy_from_slice(&[4.0, 0.0]);
    coords
        .try_restrict_mut(vertices[2])
        .unwrap()
        .copy_from_slice(&[0.0, 1.0]);

    let mut data_atlas = Atlas::default();
    data_atlas.try_insert(cell, 1).unwrap();
    let mut cell_data = SievedArray::<PointId, f64>::new(data_atlas);
    cell_data.try_set(cell, &[1.0]).unwrap();

    let metrics = evaluate_quality_metrics(&mut sieve, &cell_types, &coords).unwrap();
    let coarse_size = metrics[0].1.cell_size;
    assert!(coarse_size > 1.0);

    let thresholds = QualityThresholds {
        refine_min_angle_deg: 0.0,
        refine_max_aspect_ratio: f64::INFINITY,
        refine_max_size: 1.0,
        coarsen_min_angle_deg: 0.0,
        coarsen_max_aspect_ratio: f64::INFINITY,
        coarsen_min_size: 0.0,
        check_geometry: false,
    };

    let result = adapt_with_quality_and_transfer(
        &mut sieve,
        &cell_types,
        &coords,
        &cell_data,
        |_cells| Vec::<CoarsenEntity>::new(),
        thresholds,
    )
    .unwrap();

    assert_eq!(result.refine_cells, vec![cell]);

    let (refined, refined_data) = match result.action {
        AdaptationAction::Refined { mesh, data } => (mesh, data),
        _ => panic!("expected refinement"),
    };

    for (fine_cell, _) in refined
        .cell_refinement
        .iter()
        .flat_map(|(_, fine_cells)| fine_cells.iter())
    {
        let fine_slice = refined_data.try_get(*fine_cell).unwrap();
        assert_eq!(fine_slice, &[1.0]);
    }

    let refined_cells: Vec<_> = refined
        .cell_refinement
        .iter()
        .flat_map(|(_, fine_cells)| fine_cells.iter().map(|(cell_id, _)| *cell_id))
        .collect();
    let refined_types: Vec<_> = refined_cells
        .iter()
        .map(|cell_id| (*cell_id, CellType::Triangle))
        .collect();
    let refined_cell_types = cell_types_section(&refined_types);
    let refined_coords = refined
        .coordinates
        .as_ref()
        .expect("refinement should create coordinates");
    let mut refined_sieve = refined.sieve.clone();
    let refined_metrics = evaluate_quality_metrics(
        &mut refined_sieve,
        &refined_cell_types,
        refined_coords,
    )
    .unwrap();
    let max_refined_size = refined_metrics
        .iter()
        .map(|(_, metric)| metric.cell_size)
        .fold(0.0, f64::max);

    assert!(max_refined_size < coarse_size);
    assert!(max_refined_size <= thresholds.refine_max_size);
}
