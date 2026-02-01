use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::coordinates::Coordinates;
use mesh_sieve::data::section::Section;
use mesh_sieve::data::storage::VecStorage;
use mesh_sieve::topology::cell_type::CellType;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::refine::{RefineOptions, refine_mesh_with_options};
use mesh_sieve::topology::sieve::{InMemorySieve, Sieve};
use std::collections::HashSet;

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

fn find_points_by_coords(
    coords: &Coordinates<f64, VecStorage<f64>>,
    target: &[f64],
) -> Vec<PointId> {
    coords
        .section()
        .iter()
        .filter_map(|(point, values)| {
            if values == target {
                Some(point)
            } else {
                None
            }
        })
        .collect()
}

fn refined_cell_vertices(
    refined: &InMemorySieve<PointId, ()>,
    fine_cells: &[PointId],
) -> HashSet<PointId> {
    let mut vertices = HashSet::new();
    for cell in fine_cells {
        for v in refined.cone_points(*cell) {
            vertices.insert(v);
        }
    }
    vertices
}

#[test]
fn refine_mixed_polygon_and_quad_reuses_shared_edge_midpoint() {
    let mut sieve = InMemorySieve::<PointId, ()>::default();
    let quad = pt(100);
    let polygon = pt(200);
    let vertices = [
        (pt(1), [0.0, 0.0]),
        (pt(2), [1.0, 0.0]),
        (pt(3), [1.0, 1.0]),
        (pt(4), [0.0, 1.0]),
        (pt(5), [2.0, 1.0]),
        (pt(6), [2.0, 0.0]),
        (pt(7), [1.5, -0.5]),
    ];
    for v in [pt(1), pt(2), pt(3), pt(4)] {
        sieve.add_arrow(quad, v, ());
    }
    for v in [pt(2), pt(3), pt(5), pt(6), pt(7)] {
        sieve.add_arrow(polygon, v, ());
    }

    let cell_types = cell_types_section(&[
        (quad, CellType::Quadrilateral),
        (polygon, CellType::Polygon(5)),
    ]);

    let mut coord_atlas = Atlas::default();
    for (vertex, _) in vertices.iter() {
        coord_atlas.try_insert(*vertex, 2).unwrap();
    }
    let mut coords = Coordinates::<f64, VecStorage<f64>>::try_new(2, 2, coord_atlas)
        .expect("coordinate atlas should be valid");
    for (vertex, values) in vertices.iter() {
        coords
            .try_restrict_mut(*vertex)
            .unwrap()
            .copy_from_slice(values);
    }

    let refined = refine_mesh_with_options(
        &mut sieve,
        &cell_types,
        Some(&coords),
        RefineOptions::default(),
    )
    .expect("mixed polygon/quad refinement should succeed");
    let refined_coords = refined
        .coordinates
        .as_ref()
        .expect("refined coordinates should be available");

    let midpoint = find_points_by_coords(refined_coords, &[1.0, 0.5]);
    assert_eq!(midpoint.len(), 1, "shared midpoint should be unique");
    let midpoint = midpoint[0];

    let quad_fine: Vec<_> = refined
        .cell_refinement
        .iter()
        .find(|(cell, _)| *cell == quad)
        .unwrap()
        .1
        .iter()
        .map(|(cell, _)| *cell)
        .collect();
    let polygon_fine: Vec<_> = refined
        .cell_refinement
        .iter()
        .find(|(cell, _)| *cell == polygon)
        .unwrap()
        .1
        .iter()
        .map(|(cell, _)| *cell)
        .collect();

    let quad_vertices = refined_cell_vertices(&refined.sieve, &quad_fine);
    let polygon_vertices = refined_cell_vertices(&refined.sieve, &polygon_fine);
    assert!(quad_vertices.contains(&midpoint));
    assert!(polygon_vertices.contains(&midpoint));
}

#[test]
fn refine_mixed_hex_and_polyhedron_reuses_shared_face_center() {
    let mut sieve = InMemorySieve::<PointId, ()>::default();
    let hex = pt(300);
    let poly = pt(400);

    let vertices = [
        (pt(1), [0.0, 0.0, 0.0]),
        (pt(2), [1.0, 0.0, 0.0]),
        (pt(3), [1.0, 1.0, 0.0]),
        (pt(4), [0.0, 1.0, 0.0]),
        (pt(5), [0.0, 0.0, 1.0]),
        (pt(6), [1.0, 0.0, 1.0]),
        (pt(7), [1.0, 1.0, 1.0]),
        (pt(8), [0.0, 1.0, 1.0]),
        (pt(9), [2.0, 0.0, 0.0]),
        (pt(10), [2.0, 1.0, 0.0]),
        (pt(11), [2.0, 1.0, 1.0]),
        (pt(12), [2.0, 0.0, 1.0]),
    ];

    for v in [pt(1), pt(2), pt(3), pt(4), pt(5), pt(6), pt(7), pt(8)] {
        sieve.add_arrow(hex, v, ());
    }

    let face_ids = [pt(501), pt(502), pt(503), pt(504), pt(505), pt(506)];
    for face in face_ids {
        sieve.add_arrow(poly, face, ());
    }

    let poly_faces = [
        [pt(2), pt(9), pt(10), pt(3)],  // bottom
        [pt(6), pt(12), pt(11), pt(7)], // top
        [pt(2), pt(9), pt(12), pt(6)],  // front
        [pt(9), pt(10), pt(11), pt(12)], // right
        [pt(10), pt(3), pt(7), pt(11)], // back
        [pt(3), pt(2), pt(6), pt(7)],   // shared left face
    ];
    for (face, vertices) in face_ids.iter().zip(poly_faces.iter()) {
        for v in vertices {
            sieve.add_arrow(*face, *v, ());
        }
    }

    let cell_types = cell_types_section(&[(hex, CellType::Hexahedron), (poly, CellType::Polyhedron)]);

    let mut coord_atlas = Atlas::default();
    for (vertex, _) in vertices.iter() {
        coord_atlas.try_insert(*vertex, 3).unwrap();
    }
    let mut coords = Coordinates::<f64, VecStorage<f64>>::try_new(3, 3, coord_atlas)
        .expect("coordinate atlas should be valid");
    for (vertex, values) in vertices.iter() {
        coords
            .try_restrict_mut(*vertex)
            .unwrap()
            .copy_from_slice(values);
    }

    let refined = refine_mesh_with_options(
        &mut sieve,
        &cell_types,
        Some(&coords),
        RefineOptions::default(),
    )
    .expect("mixed hex/polyhedron refinement should succeed");
    let refined_coords = refined
        .coordinates
        .as_ref()
        .expect("refined coordinates should be available");

    let face_center = find_points_by_coords(refined_coords, &[1.0, 0.5, 0.5]);
    assert_eq!(face_center.len(), 1, "shared face center should be unique");
    let face_center = face_center[0];

    let hex_fine: Vec<_> = refined
        .cell_refinement
        .iter()
        .find(|(cell, _)| *cell == hex)
        .unwrap()
        .1
        .iter()
        .map(|(cell, _)| *cell)
        .collect();
    let poly_fine: Vec<_> = refined
        .cell_refinement
        .iter()
        .find(|(cell, _)| *cell == poly)
        .unwrap()
        .1
        .iter()
        .map(|(cell, _)| *cell)
        .collect();

    let hex_vertices = refined_cell_vertices(&refined.sieve, &hex_fine);
    let poly_vertices = refined_cell_vertices(&refined.sieve, &poly_fine);
    assert!(hex_vertices.contains(&face_center));
    assert!(poly_vertices.contains(&face_center));
}
