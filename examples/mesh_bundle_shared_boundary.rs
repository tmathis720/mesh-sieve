// cargo run --example mesh_bundle_shared_boundary
use mesh_sieve::data::{Coordinates, VecStorage};
use mesh_sieve::io::{MeshBundle, MeshData};
use mesh_sieve::mesh_error::MeshSieveError;
use mesh_sieve::prelude::Atlas;
use mesh_sieve::topology::cell_type::CellType;
use mesh_sieve::topology::labels::LabelSet;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::sieve::{InMemorySieve, MutableSieve, Sieve};

fn build_sieve(points: &[PointId], arrows: &[(PointId, PointId)]) -> InMemorySieve<PointId, ()> {
    let mut sieve = InMemorySieve::default();
    for point in points {
        MutableSieve::add_point(&mut sieve, *point);
    }
    for (source, target) in arrows {
        Sieve::add_arrow(&mut sieve, *source, *target, ());
    }
    sieve
}

fn build_coordinates(
    points: &[PointId],
    values: &[(PointId, [f64; 2])],
) -> Result<Coordinates<f64, VecStorage<f64>>, MeshSieveError> {
    let mut atlas = Atlas::default();
    for point in points {
        atlas.try_insert(*point, 2)?;
    }
    let mut coords = Coordinates::try_new(2, 2, atlas)?;
    for &(point, value) in values {
        coords.section_mut().try_set(point, &value)?;
    }
    Ok(coords)
}

fn main() -> Result<(), MeshSieveError> {
    let v0 = PointId::new(1)?;
    let shared = PointId::new(2)?;
    let v1 = PointId::new(3)?;
    let v2 = PointId::new(4)?;
    let cell_a = PointId::new(10)?;
    let cell_b = PointId::new(20)?;

    let sieve_a = build_sieve(&[v0, shared, v1, cell_a], &[(cell_a, v0), (cell_a, shared)]);
    let sieve_b = build_sieve(&[shared, v2, cell_b], &[(cell_b, shared), (cell_b, v2)]);

    let coords_a = build_coordinates(
        &[v0, shared, v1],
        &[(v0, [0.0, 0.0]), (shared, [1.0, 0.0]), (v1, [0.5, 1.0])],
    )?;
    let coords_b = build_coordinates(&[shared, v2], &[(shared, [1.0, 0.0]), (v2, [2.0, 0.0])])?;

    let mut labels = LabelSet::new();
    labels.set_label(shared, "boundary", 7);

    let mut mesh_a = MeshData::<_, f64, VecStorage<f64>, VecStorage<CellType>>::new(sieve_a);
    mesh_a.coordinates = Some(coords_a);
    mesh_a.labels = Some(labels);

    let mut mesh_b = MeshData::<_, f64, VecStorage<f64>, VecStorage<CellType>>::new(sieve_b);
    mesh_b.coordinates = Some(coords_b);

    let mut bundle = MeshBundle::new(vec![mesh_a, mesh_b]);
    bundle.sync_labels();
    bundle.sync_coordinates()?;

    let shared_label = bundle.meshes[1]
        .labels
        .as_ref()
        .and_then(|labels| labels.get_label(shared, "boundary"));
    println!("shared boundary label on mesh B: {shared_label:?}");

    Ok(())
}
