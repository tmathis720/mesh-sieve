// cargo run --example adapt_with_metric
use mesh_sieve::adapt::{
    MetricAdaptationAction, MetricTensor, MetricThresholds, adapt_with_metric,
};
use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::coordinates::Coordinates;
use mesh_sieve::data::section::Section;
use mesh_sieve::data::storage::VecStorage;
use mesh_sieve::topology::Sieve;
use mesh_sieve::topology::cell_type::CellType;
use mesh_sieve::topology::coarsen::CoarsenEntity;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::sieve::InMemorySieve;

fn pt(id: u64) -> PointId {
    PointId::new(id).unwrap()
}

fn main() {
    let mut sieve = InMemorySieve::<PointId, ()>::default();
    let cell = pt(10);
    let vertices = [pt(1), pt(2), pt(3)];
    for v in vertices {
        sieve.add_arrow(cell, v, ());
    }

    let mut cell_atlas = Atlas::default();
    cell_atlas.try_insert(cell, 1).unwrap();
    let mut cell_types = Section::<CellType, VecStorage<CellType>>::new(cell_atlas);
    cell_types.try_set(cell, &[CellType::Triangle]).unwrap();

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

    let thresholds = MetricThresholds::default();
    let result = adapt_with_metric(
        &mut sieve,
        &cell_types,
        &coords,
        &metrics,
        |_hints| Vec::<CoarsenEntity>::new(),
        thresholds,
    )
    .unwrap();

    match result.action {
        MetricAdaptationAction::Refined { mesh } => {
            println!("refined {} cells", mesh.cell_refinement.len());
        }
        MetricAdaptationAction::Coarsened { mesh } => {
            println!("coarsened {} entities", mesh.transfer_map.len());
        }
        MetricAdaptationAction::NoChange => println!("no adaptation needed"),
    }
}
