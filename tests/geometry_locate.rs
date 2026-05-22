use mesh_sieve::algs::meshgen::{MeshGenOptions, StructuredCellType, structured_box_2d};
use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::coordinates::Coordinates;
use mesh_sieve::data::section::Section;
use mesh_sieve::data::storage::VecStorage;
use mesh_sieve::geometry::locate::{
    PeriodicDomain, PointLocator, locate_point, project_section_to_vertices,
};
use mesh_sieve::mesh_error::MeshSieveError;
use mesh_sieve::topology::cell_type::CellType;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::sieve::{InMemorySieve, MutableSieve, Sieve};

fn point(id: u64) -> PointId {
    PointId::new(id).unwrap()
}

#[test]
fn structured_quad_grid_locates_interior_and_boundary_points() -> Result<(), MeshSieveError> {
    let mesh = structured_box_2d(
        2,
        2,
        [0.0, 0.0],
        [1.0, 1.0],
        StructuredCellType::Quadrilateral,
        MeshGenOptions::default(),
    )?;
    let coords = mesh.coordinates.as_ref().unwrap();
    let cell_types = mesh.cell_types.as_ref().unwrap();
    let locator = PointLocator::grid_hash(&mesh.sieve, cell_types, coords)?;

    let interior = locator.locate_point(&[0.25, 0.25])?.expect("interior");
    assert_eq!(interior.cell_type, CellType::Quadrilateral);
    assert!(
        interior
            .reference_coordinates
            .iter()
            .all(|r| *r >= -1.0e-10 && *r <= 1.0 + 1.0e-10)
    );

    let boundary = locator.locate_point(&[1.0, 0.5])?.expect("boundary");
    assert_eq!(boundary.cell_type, CellType::Quadrilateral);
    assert!(
        boundary
            .reference_coordinates
            .iter()
            .any(|r| (*r - 1.0).abs() <= 1.0e-10)
    );

    assert!(locator.locate_point(&[1.2, 0.5])?.is_none());
    Ok(())
}

#[test]
fn bvh_locator_handles_structured_triangle_grid() -> Result<(), MeshSieveError> {
    let mesh = structured_box_2d(
        1,
        1,
        [0.0, 0.0],
        [1.0, 1.0],
        StructuredCellType::Triangle,
        MeshGenOptions::default(),
    )?;
    let locator = PointLocator::bvh(
        &mesh.sieve,
        mesh.cell_types.as_ref().unwrap(),
        mesh.coordinates.as_ref().unwrap(),
    )?;
    let location = locator
        .locate_point(&[0.75, 0.25])?
        .expect("point in a triangle");
    assert_eq!(location.cell_type, CellType::Triangle);
    assert!(location.reference_coordinates[0] + location.reference_coordinates[1] <= 1.0 + 1.0e-10);
    Ok(())
}

#[test]
fn periodic_domain_wraps_queries_and_localizes_cell_coordinates() -> Result<(), MeshSieveError> {
    let mesh = structured_box_2d(
        1,
        1,
        [0.0, 0.0],
        [1.0, 1.0],
        StructuredCellType::Quadrilateral,
        MeshGenOptions::default(),
    )?;
    let locator = PointLocator::grid_hash(
        &mesh.sieve,
        mesh.cell_types.as_ref().unwrap(),
        mesh.coordinates.as_ref().unwrap(),
    )?;
    let domain = PeriodicDomain::new(&[0.0, 0.0], &[1.0, 1.0], &[true, false])?;

    let wrapped = locator
        .locate_periodic_point(&[1.25, 0.25], &domain)?
        .expect("wrapped point");
    let direct = locator.locate_point(&[0.25, 0.25])?.expect("direct point");
    assert_eq!(wrapped.cell, direct.cell);

    let localized = locator.localized_cell_coordinates(direct.cell, &[1.01, 0.5], Some(&domain))?;
    assert!(
        localized
            .coordinates
            .iter()
            .filter(|coord| (coord[0] - 1.0).abs() <= 1.0e-10)
            .count()
            >= 2
    );
    assert_eq!(localized.vertices.len(), 4);
    Ok(())
}

fn mixed_mesh() -> Result<
    (
        InMemorySieve<PointId, ()>,
        Coordinates<f64, VecStorage<f64>>,
        Section<CellType, VecStorage<CellType>>,
    ),
    MeshSieveError,
> {
    let mut sieve = InMemorySieve::<PointId, ()>::default();
    let vertices = [point(1), point(2), point(3), point(4), point(5)];
    let tri = point(6);
    let quad = point(7);
    for p in vertices.into_iter().chain([tri, quad]) {
        MutableSieve::add_point(&mut sieve, p);
    }
    for v in [point(1), point(2), point(3)] {
        sieve.add_arrow(tri, v, ());
    }
    for v in [point(2), point(4), point(5), point(3)] {
        sieve.add_arrow(quad, v, ());
    }

    let mut coord_atlas = Atlas::default();
    for v in vertices {
        coord_atlas.try_insert(v, 2)?;
    }
    let mut coords = Coordinates::<f64, VecStorage<f64>>::try_new(2, 2, coord_atlas)?;
    coords.section_mut().try_set(point(1), &[0.0, 0.0])?;
    coords.section_mut().try_set(point(2), &[1.0, 0.0])?;
    coords.section_mut().try_set(point(3), &[0.0, 1.0])?;
    coords.section_mut().try_set(point(4), &[2.0, 0.0])?;
    coords.section_mut().try_set(point(5), &[2.0, 1.0])?;

    let mut type_atlas = Atlas::default();
    for p in vertices.into_iter().chain([tri, quad]) {
        type_atlas.try_insert(p, 1)?;
    }
    let mut cell_types = Section::<CellType, VecStorage<CellType>>::new(type_atlas);
    for v in vertices {
        cell_types.try_set(v, &[CellType::Vertex])?;
    }
    cell_types.try_set(tri, &[CellType::Triangle])?;
    cell_types.try_set(quad, &[CellType::Quadrilateral])?;
    Ok((sieve, coords, cell_types))
}

#[test]
fn mixed_triangle_quad_mesh_locates_supported_cell_types() -> Result<(), MeshSieveError> {
    let (sieve, coords, cell_types) = mixed_mesh()?;
    let tri_hit = locate_point(&sieve, &cell_types, &coords, &[0.2, 0.2])?.expect("triangle");
    assert_eq!(tri_hit.cell_type, CellType::Triangle);
    let quad_hit = locate_point(&sieve, &cell_types, &coords, &[1.5, 0.5])?.expect("quad");
    assert_eq!(quad_hit.cell_type, CellType::Quadrilateral);
    assert!(locate_point(&sieve, &cell_types, &coords, &[3.0, 0.5])?.is_none());
    Ok(())
}

#[test]
fn projection_interpolates_linear_vertex_section_between_meshes() -> Result<(), MeshSieveError> {
    let source = structured_box_2d(
        1,
        1,
        [0.0, 0.0],
        [1.0, 1.0],
        StructuredCellType::Quadrilateral,
        MeshGenOptions::default(),
    )?;
    let target = structured_box_2d(
        2,
        2,
        [0.0, 0.0],
        [1.0, 1.0],
        StructuredCellType::Quadrilateral,
        MeshGenOptions::default(),
    )?;
    let source_coords = source.coordinates.as_ref().unwrap();
    let mut atlas = Atlas::default();
    for (vertex, _) in source_coords.section().iter() {
        atlas.try_insert(vertex, 1)?;
    }
    let mut field = Section::<f64, VecStorage<f64>>::new(atlas);
    for (vertex, coord) in source_coords.section().iter() {
        field.try_set(vertex, &[coord[0] + 2.0 * coord[1]])?;
    }

    let projected = project_section_to_vertices(
        &source.sieve,
        source.cell_types.as_ref().unwrap(),
        source_coords,
        &field,
        &target.sieve,
        target.cell_types.as_ref().unwrap(),
        target.coordinates.as_ref().unwrap(),
    )?;

    for (vertex, coord) in target.coordinates.as_ref().unwrap().section().iter() {
        let expected = coord[0] + 2.0 * coord[1];
        let value = projected.try_restrict(vertex)?[0];
        assert!((value - expected).abs() <= 1.0e-10);
    }
    Ok(())
}
