// cargo run --example extrude_boundary_constraints
use mesh_sieve::algs::boundary::classify_boundary_points;
use mesh_sieve::algs::extrude::extrude_surface_layers;
use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::constrained_section::ConstrainedSection;
use mesh_sieve::data::coordinates::Coordinates;
use mesh_sieve::data::section::Section;
use mesh_sieve::data::storage::VecStorage;
use mesh_sieve::mesh_error::MeshSieveError;
use mesh_sieve::topology::cell_type::CellType;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::sieve::{InMemorySieve, MutableSieve, Sieve};

fn main() -> Result<(), MeshSieveError> {
    let mut surface = InMemorySieve::<PointId, ()>::default();
    let cell = PointId::new(10)?;
    let vertices = [PointId::new(1)?, PointId::new(2)?, PointId::new(3)?, PointId::new(4)?];

    MutableSieve::add_point(&mut surface, cell);
    for v in vertices {
        MutableSieve::add_point(&mut surface, v);
        surface.add_arrow(cell, v, ());
    }

    let mut cell_atlas = Atlas::default();
    cell_atlas.try_insert(cell, 1)?;
    for v in vertices {
        cell_atlas.try_insert(v, 1)?;
    }
    let mut cell_types = Section::<CellType, VecStorage<CellType>>::new(cell_atlas);
    cell_types.try_set(cell, &[CellType::Quadrilateral])?;
    for v in vertices {
        cell_types.try_set(v, &[CellType::Vertex])?;
    }

    let mut coord_atlas = Atlas::default();
    for v in vertices {
        coord_atlas.try_insert(v, 2)?;
    }
    let mut coords = Coordinates::<f64, VecStorage<f64>>::try_new(2, 2, coord_atlas)?;
    coords.section_mut().try_set(vertices[0], &[0.0, 0.0])?;
    coords.section_mut().try_set(vertices[1], &[1.0, 0.0])?;
    coords.section_mut().try_set(vertices[2], &[1.0, 1.0])?;
    coords.section_mut().try_set(vertices[3], &[0.0, 1.0])?;

    let layer_offsets = [0.0, 1.0, 2.0];
    let extruded = extrude_surface_layers(&surface, &cell_types, &coords, &layer_offsets)?;

    let vertex_points: Vec<_> = extruded
        .sieve
        .points()
        .filter(|p| extruded.sieve.cone_points(*p).next().is_none())
        .collect();
    assert!(!vertex_points.is_empty());

    let classification = classify_boundary_points(&extruded.sieve, vertex_points.iter().copied())?;
    let expected_interior = vertices.len() * layer_offsets.len().saturating_sub(2);
    assert_eq!(classification.interior.len(), expected_interior);
    assert_eq!(
        classification.boundary.len() + classification.interior.len(),
        vertex_points.len()
    );
    assert!(classification.boundary.len() >= 3);

    let mut field_atlas = Atlas::default();
    for p in extruded.sieve.points() {
        field_atlas.try_insert(p, 1)?;
    }
    let mut section = Section::<f64, VecStorage<f64>>::new(field_atlas);
    for p in extruded.sieve.points() {
        section.try_set(p, &[p.get() as f64])?;
    }

    let mut constrained = ConstrainedSection::new(section);
    let boundary_value = 123.0;
    for p in classification.boundary.iter().take(3) {
        constrained.insert_constraint(*p, 0, boundary_value)?;
    }
    constrained.apply_constraints()?;

    for p in classification.boundary.iter().take(3) {
        assert_eq!(constrained.section().try_restrict(*p)?[0], boundary_value);
    }

    Ok(())
}
