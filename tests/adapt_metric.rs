use mesh_sieve::adapt::{
    adapt_with_metric_and_transfer, MetricAdaptationAction, MetricThresholds, MetricTensor,
};
use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::coordinates::Coordinates;
use mesh_sieve::data::refine::sieved_array::SievedArray;
use mesh_sieve::data::section::Section;
use mesh_sieve::data::storage::VecStorage;
use mesh_sieve::topology::cell_type::CellType;
use mesh_sieve::topology::coarsen::CoarsenEntity;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::sieve::InMemorySieve;

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
fn refine_cells_from_metric_tensor() {
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
    let mut coords = Coordinates::<f64, VecStorage<f64>>::try_new(2, 2, coord_atlas).unwrap();
    coords
        .try_restrict_mut(vertices[0])
        .unwrap()
        .copy_from_slice(&[0.0, 0.0]);
    coords
        .try_restrict_mut(vertices[1])
        .unwrap()
        .copy_from_slice(&[3.0, 0.0]);
    coords
        .try_restrict_mut(vertices[2])
        .unwrap()
        .copy_from_slice(&[0.0, 1.0]);

    let mut metric_atlas = Atlas::default();
    metric_atlas.try_insert(cell, 1).unwrap();
    let mut metrics = Section::<MetricTensor, VecStorage<MetricTensor>>::new(metric_atlas);
    metrics
        .try_set(cell, &[MetricTensor::new_2d(1.0, 1.0, 0.0)])
        .unwrap();

    let mut data_atlas = Atlas::default();
    data_atlas.try_insert(cell, 1).unwrap();
    let mut cell_data = SievedArray::<PointId, f64>::new(data_atlas);
    cell_data.try_set(cell, &[2.0]).unwrap();

    let thresholds = MetricThresholds {
        refine_max_edge_length: 2.0,
        coarsen_max_edge_length: 0.5,
        split_edge_length: 2.0,
        split_face_length: 2.0,
        check_geometry: false,
    };

    let result = adapt_with_metric_and_transfer(
        &mut sieve,
        &cell_types,
        &coords,
        &metrics,
        &cell_data,
        |_hints| Vec::<CoarsenEntity>::new(),
        thresholds,
    )
    .unwrap();

    assert_eq!(result.refine_cells, vec![cell]);
    assert!(result
        .split_hints
        .iter()
        .any(|hint| hint.cell == cell && !hint.split_edges.is_empty()));

    let refined = match result.action {
        MetricAdaptationAction::Refined { mesh } => mesh,
        _ => panic!("expected refinement"),
    };

    let refined_data = result.data.expect("expected refined data");
    for (fine_cell, _) in refined
        .cell_refinement
        .iter()
        .flat_map(|(_, fine_cells)| fine_cells.iter())
    {
        let fine_slice = refined_data.try_get(*fine_cell).unwrap();
        assert_eq!(fine_slice, &[2.0]);
    }
}
