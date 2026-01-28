use mesh_sieve::data::section::Section;
use mesh_sieve::data::storage::VecStorage;
use mesh_sieve::mesh_generation::{
    BOUNDARY_X_MAX, BOUNDARY_X_MIN, BOUNDARY_Y_MAX, BOUNDARY_Y_MIN, BOUNDARY_Z_MAX, BOUNDARY_Z_MIN,
    MeshGenerationOptions, Periodicity, hex_mesh, interval_mesh, quad_mesh,
};
use mesh_sieve::topology::cell_type::CellType;
use mesh_sieve::topology::labels::LabelSet;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::sieve::Sieve;

type CellTypeSection = Section<CellType, VecStorage<CellType>>;

fn vertex_points(cell_types: &CellTypeSection) -> Vec<PointId> {
    let mut points: Vec<_> = cell_types
        .iter()
        .filter_map(|(p, slice)| (slice == [CellType::Vertex]).then_some(p))
        .collect();
    points.sort_unstable();
    points
}

fn cell_points(cell_types: &CellTypeSection, cell_type: CellType) -> Vec<PointId> {
    let mut points: Vec<_> = cell_types
        .iter()
        .filter_map(|(p, slice)| (slice == [cell_type]).then_some(p))
        .collect();
    points.sort_unstable();
    points
}

fn labels(mesh_labels: &Option<LabelSet>) -> &LabelSet {
    mesh_labels
        .as_ref()
        .expect("expected labels to be populated")
}

#[test]
fn interval_mesh_counts_labels_and_coords() {
    let options = MeshGenerationOptions {
        periodic: Periodicity {
            x: true,
            y: false,
            z: false,
        },
    };
    let generated = interval_mesh(2, 0.0, 2.0, options).unwrap();
    let mesh = generated.mesh;
    let coords = mesh.coordinates.as_ref().unwrap();
    let cell_types = mesh.cell_types.as_ref().unwrap();

    assert_eq!(mesh.sieve.points().count(), 5);

    let vertices = vertex_points(cell_types);
    assert_eq!(vertices.len(), 3);
    let cells = cell_points(cell_types, CellType::Segment);
    assert_eq!(cells.len(), 2);
    for cell in cells {
        assert_eq!(mesh.sieve.cone(cell).count(), 2);
    }

    let labels = labels(&mesh.labels);
    assert_eq!(labels.stratum_size(BOUNDARY_X_MIN, 1), 1);
    assert_eq!(labels.stratum_size(BOUNDARY_X_MAX, 1), 1);

    let c0 = coords.section().try_restrict(vertices[0]).unwrap();
    let c1 = coords.section().try_restrict(vertices[1]).unwrap();
    let c2 = coords.section().try_restrict(vertices[2]).unwrap();
    assert_eq!(c0, &[0.0]);
    assert_eq!(c1, &[1.0]);
    assert_eq!(c2, &[2.0]);

    let mut eq = generated.periodic.expect("expected periodic equivalence");
    assert!(eq.are_equivalent(vertices[0], vertices[2]));
}

#[test]
fn quad_mesh_counts_labels_and_coords() {
    let options = MeshGenerationOptions {
        periodic: Periodicity {
            x: true,
            y: true,
            z: false,
        },
    };
    let generated = quad_mesh(1, 1, [0.0, 0.0], [1.0, 1.0], options).unwrap();
    let mesh = generated.mesh;
    let coords = mesh.coordinates.as_ref().unwrap();
    let cell_types = mesh.cell_types.as_ref().unwrap();

    assert_eq!(mesh.sieve.points().count(), 5);

    let vertices = vertex_points(cell_types);
    assert_eq!(vertices.len(), 4);
    let cells = cell_points(cell_types, CellType::Quadrilateral);
    assert_eq!(cells.len(), 1);
    assert_eq!(mesh.sieve.cone(cells[0]).count(), 4);

    let labels = labels(&mesh.labels);
    assert_eq!(labels.stratum_size(BOUNDARY_X_MIN, 1), 2);
    assert_eq!(labels.stratum_size(BOUNDARY_X_MAX, 1), 2);
    assert_eq!(labels.stratum_size(BOUNDARY_Y_MIN, 1), 2);
    assert_eq!(labels.stratum_size(BOUNDARY_Y_MAX, 1), 2);

    let c00 = coords.section().try_restrict(vertices[0]).unwrap();
    let c10 = coords.section().try_restrict(vertices[1]).unwrap();
    let c01 = coords.section().try_restrict(vertices[2]).unwrap();
    let c11 = coords.section().try_restrict(vertices[3]).unwrap();
    assert_eq!(c00, &[0.0, 0.0]);
    assert_eq!(c10, &[1.0, 0.0]);
    assert_eq!(c01, &[0.0, 1.0]);
    assert_eq!(c11, &[1.0, 1.0]);

    let mut eq = generated.periodic.expect("expected periodic equivalence");
    assert!(eq.are_equivalent(vertices[0], vertices[1]));
    assert!(eq.are_equivalent(vertices[0], vertices[2]));
}

#[test]
fn hex_mesh_counts_labels_and_coords() {
    let options = MeshGenerationOptions {
        periodic: Periodicity {
            x: true,
            y: false,
            z: true,
        },
    };
    let generated = hex_mesh(1, 1, 1, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], options).unwrap();
    let mesh = generated.mesh;
    let coords = mesh.coordinates.as_ref().unwrap();
    let cell_types = mesh.cell_types.as_ref().unwrap();

    assert_eq!(mesh.sieve.points().count(), 9);

    let vertices = vertex_points(cell_types);
    assert_eq!(vertices.len(), 8);
    let cells = cell_points(cell_types, CellType::Hexahedron);
    assert_eq!(cells.len(), 1);
    assert_eq!(mesh.sieve.cone(cells[0]).count(), 8);

    let labels = labels(&mesh.labels);
    assert_eq!(labels.stratum_size(BOUNDARY_X_MIN, 1), 4);
    assert_eq!(labels.stratum_size(BOUNDARY_X_MAX, 1), 4);
    assert_eq!(labels.stratum_size(BOUNDARY_Y_MIN, 1), 4);
    assert_eq!(labels.stratum_size(BOUNDARY_Y_MAX, 1), 4);
    assert_eq!(labels.stratum_size(BOUNDARY_Z_MIN, 1), 4);
    assert_eq!(labels.stratum_size(BOUNDARY_Z_MAX, 1), 4);

    let c000 = coords.section().try_restrict(vertices[0]).unwrap();
    let c100 = coords.section().try_restrict(vertices[1]).unwrap();
    let c010 = coords.section().try_restrict(vertices[2]).unwrap();
    let c110 = coords.section().try_restrict(vertices[3]).unwrap();
    let c001 = coords.section().try_restrict(vertices[4]).unwrap();
    let c101 = coords.section().try_restrict(vertices[5]).unwrap();
    let c011 = coords.section().try_restrict(vertices[6]).unwrap();
    let c111 = coords.section().try_restrict(vertices[7]).unwrap();
    assert_eq!(c000, &[0.0, 0.0, 0.0]);
    assert_eq!(c100, &[1.0, 0.0, 0.0]);
    assert_eq!(c010, &[0.0, 1.0, 0.0]);
    assert_eq!(c110, &[1.0, 1.0, 0.0]);
    assert_eq!(c001, &[0.0, 0.0, 1.0]);
    assert_eq!(c101, &[1.0, 0.0, 1.0]);
    assert_eq!(c011, &[0.0, 1.0, 1.0]);
    assert_eq!(c111, &[1.0, 1.0, 1.0]);

    let mut eq = generated.periodic.expect("expected periodic equivalence");
    assert!(eq.are_equivalent(vertices[0], vertices[1]));
    assert!(eq.are_equivalent(vertices[0], vertices[4]));
}
