use mesh_sieve::algs::transform::{CoordinateTransform, TransformHooks, transform_mesh};
use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::coordinates::Coordinates;
use mesh_sieve::data::section::Section;
use mesh_sieve::data::storage::VecStorage;
use mesh_sieve::geometry::quality::cell_quality_from_section;
use mesh_sieve::io::MeshData;
use mesh_sieve::topology::cell_type::CellType;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::sieve::{InMemorySieve, Sieve};
use std::collections::BTreeSet;

fn build_triangle_mesh() -> (
    MeshData<InMemorySieve<PointId, ()>, f64, VecStorage<f64>, VecStorage<CellType>>,
    [PointId; 3],
) {
    let cell = PointId::new(10).unwrap();
    let v0 = PointId::new(1).unwrap();
    let v1 = PointId::new(2).unwrap();
    let v2 = PointId::new(3).unwrap();

    let mut sieve = InMemorySieve::<PointId, ()>::default();
    Sieve::add_arrow(&mut sieve, cell, v0, ());
    Sieve::add_arrow(&mut sieve, cell, v1, ());
    Sieve::add_arrow(&mut sieve, cell, v2, ());

    let mut coord_atlas = Atlas::default();
    coord_atlas.try_insert(v0, 2).unwrap();
    coord_atlas.try_insert(v1, 2).unwrap();
    coord_atlas.try_insert(v2, 2).unwrap();

    let mut coords = Coordinates::<f64, VecStorage<f64>>::try_new(2, 2, coord_atlas).unwrap();
    coords
        .try_restrict_mut(v0)
        .unwrap()
        .copy_from_slice(&[0.0, 0.0]);
    coords
        .try_restrict_mut(v1)
        .unwrap()
        .copy_from_slice(&[1.0, 0.0]);
    coords
        .try_restrict_mut(v2)
        .unwrap()
        .copy_from_slice(&[0.0, 1.0]);

    let mut cell_atlas = Atlas::default();
    cell_atlas.try_insert(cell, 1).unwrap();
    let mut cell_types = Section::<CellType, VecStorage<CellType>>::new(cell_atlas);
    cell_types
        .try_restrict_mut(cell)
        .unwrap()
        .copy_from_slice(&[CellType::Triangle]);

    let mut mesh = MeshData::new(sieve);
    mesh.coordinates = Some(coords);
    mesh.cell_types = Some(cell_types);

    (mesh, [v0, v1, v2])
}

#[test]
fn transform_mesh_displacement_updates_quality() {
    let (mut mesh, vertices) = build_triangle_mesh();
    let points_before: BTreeSet<_> = mesh.sieve.points().collect();
    let quality_before = cell_quality_from_section(
        CellType::Triangle,
        &vertices,
        mesh.coordinates.as_ref().unwrap(),
    )
    .unwrap();

    let atlas = mesh.coordinates.as_ref().unwrap().section().atlas().clone();
    let mut displacement = Section::<f64, VecStorage<f64>>::new(atlas);
    displacement
        .try_restrict_mut(vertices[0])
        .unwrap()
        .copy_from_slice(&[0.0, 0.0]);
    displacement
        .try_restrict_mut(vertices[1])
        .unwrap()
        .copy_from_slice(&[0.5, 0.0]);
    displacement
        .try_restrict_mut(vertices[2])
        .unwrap()
        .copy_from_slice(&[0.0, 0.25]);

    let mut quality_after = None;
    let hooks = TransformHooks {
        after_update: Some(&mut |mesh| {
            let coords = mesh.coordinates.as_ref().unwrap();
            quality_after = Some(cell_quality_from_section(
                CellType::Triangle,
                &vertices,
                coords,
            )?);
            Ok(())
        }),
    };

    transform_mesh(
        &mut mesh,
        CoordinateTransform::Displacement(&displacement),
        hooks,
    )
    .unwrap();

    let points_after: BTreeSet<_> = mesh.sieve.points().collect();
    assert_eq!(points_before, points_after);

    let coords = mesh.coordinates.as_ref().unwrap();
    assert_eq!(coords.try_restrict(vertices[1]).unwrap(), &[1.5, 0.0]);
    assert_eq!(coords.try_restrict(vertices[2]).unwrap(), &[0.0, 1.25]);

    let quality_after = quality_after.expect("hook should populate quality");
    assert_ne!(quality_before.aspect_ratio, quality_after.aspect_ratio);
    assert!(quality_after.jacobian_sign > 0.0);
}

#[test]
fn transform_mesh_function_applies_update() {
    let (mut mesh, vertices) = build_triangle_mesh();
    let mut translate = |_: PointId, coords: &mut [f64]| {
        coords[0] += 2.0;
        coords[1] -= 1.0;
        Ok(())
    };

    let hooks = TransformHooks { after_update: None };
    transform_mesh(
        &mut mesh,
        CoordinateTransform::Function(&mut translate),
        hooks,
    )
    .unwrap();

    let coords = mesh.coordinates.as_ref().unwrap();
    assert_eq!(coords.try_restrict(vertices[0]).unwrap(), &[2.0, -1.0]);
    assert_eq!(coords.try_restrict(vertices[1]).unwrap(), &[3.0, -1.0]);
    assert_eq!(coords.try_restrict(vertices[2]).unwrap(), &[2.0, 0.0]);
}
