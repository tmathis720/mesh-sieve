use mesh_sieve::adapt::{MetricAdaptationAction, MetricTensor, MetricThresholds};
use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::coordinates::Coordinates;
use mesh_sieve::data::section::Section;
use mesh_sieve::data::storage::VecStorage;
use mesh_sieve::dm::{MeshDM, MeshDMMetricAdaptOptions, MeshDMTransferStrategy};
use mesh_sieve::io::MeshData;
use mesh_sieve::topology::Sieve;
use mesh_sieve::topology::cell_type::CellType;
use mesh_sieve::topology::labels::LabelSet;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::sieve::MeshSieve;

fn pt(id: u64) -> PointId {
    PointId::new(id).unwrap()
}

fn single_triangle_dm() -> (
    MeshDM<f64>,
    Section<MetricTensor, VecStorage<MetricTensor>>,
    PointId,
) {
    let mut sieve = MeshSieve::default();
    let cell = pt(10);
    let vertices = [pt(1), pt(2), pt(3)];
    for v in vertices {
        sieve.add_arrow(cell, v, ());
    }

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

    let mut cell_atlas = Atlas::default();
    cell_atlas.try_insert(cell, 1).unwrap();
    let mut cell_types = Section::<CellType, VecStorage<CellType>>::new(cell_atlas.clone());
    cell_types.try_set(cell, &[CellType::Triangle]).unwrap();

    let mut solution = Section::<f64, VecStorage<f64>>::new(cell_atlas.clone());
    solution.try_set(cell, &[42.0]).unwrap();

    let mut labels = LabelSet::new();
    labels.set_label(cell, "material", 7);

    let mesh_data = MeshData {
        sieve,
        coordinates: Some(coords),
        sections: [("solution".to_string(), solution)].into_iter().collect(),
        mixed_sections: Default::default(),
        labels: Some(labels),
        cell_types: Some(cell_types),
        discretization: None,
    };

    let mut metric = Section::<MetricTensor, VecStorage<MetricTensor>>::new(cell_atlas);
    metric
        .try_set(cell, &[MetricTensor::new_2d(1.0, 1.0, 0.0)])
        .unwrap();

    (MeshDM::<f64>::from_mesh_data(mesh_data), metric, cell)
}

#[test]
fn dm_metric_refinement_transfers_metadata_with_provenance() {
    let (mut dm, metric, cell) = single_triangle_dm();
    let mut options = MeshDMMetricAdaptOptions::default();
    options.transfer = MeshDMTransferStrategy::PreserveAll;
    options.thresholds = MetricThresholds {
        refine_max_edge_length: 2.0,
        coarsen_max_edge_length: 0.1,
        split_edge_length: 2.0,
        split_face_length: 2.0,
        check_geometry: false,
    };

    let result = dm
        .adapt_with_attached_metric("metric", &metric, |_| Vec::new(), options)
        .unwrap();

    assert!(matches!(
        result.action,
        MetricAdaptationAction::Refined { .. }
    ));
    assert_eq!(result.provenance.old_to_new.len(), 1);
    assert_eq!(result.provenance.old_to_new[0].0, cell);
    assert!(result.diagnostics.provenance_edges >= 4);
    assert_eq!(result.diagnostics.transferred_sections, 1);

    let fine_cells = &result.provenance.old_to_new[0].1;
    let solution = dm.section("solution").expect("transferred section");
    let labels = dm.labels().expect("transferred labels");
    let cell_types = dm.cell_types().expect("transferred cell types");
    for fine in fine_cells {
        assert_eq!(solution.try_restrict(*fine).unwrap(), &[42.0]);
        assert_eq!(labels.get_label(*fine, "material"), Some(7));
        assert_eq!(
            cell_types.try_restrict(*fine).unwrap(),
            &[CellType::Triangle]
        );
    }
    assert!(dm.coordinates().unwrap().section().iter().count() > 3);
}

#[test]
fn dm_metric_iterative_nochange_converges_and_keeps_metadata_stable() {
    let (mut dm, metric, cell) = single_triangle_dm();
    let mut options = MeshDMMetricAdaptOptions::default();
    options.transfer = MeshDMTransferStrategy::PreserveAll;
    options.thresholds = MetricThresholds {
        refine_max_edge_length: 10.0,
        coarsen_max_edge_length: 0.0,
        ..MetricThresholds::default()
    };

    let history = dm
        .adapt_with_attached_metric_iterative(
            "metric",
            &metric,
            |_| Vec::new(),
            options,
            3,
            |_| false,
        )
        .unwrap();

    assert_eq!(history.len(), 1);
    assert!(matches!(
        history[0].action,
        MetricAdaptationAction::NoChange
    ));
    assert_eq!(dm.labels().unwrap().get_label(cell, "material"), Some(7));
    assert_eq!(
        dm.section("solution").unwrap().try_restrict(cell).unwrap(),
        &[42.0]
    );
}
