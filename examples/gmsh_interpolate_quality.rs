// cargo run --example gmsh_interpolate_quality
use mesh_sieve::algs::interpolate::interpolate_edges_faces;
use mesh_sieve::geometry::quality::cell_quality_from_section;
use mesh_sieve::io::gmsh::GmshReader;
use mesh_sieve::io::SieveSectionReader;
use mesh_sieve::mesh_error::MeshSieveError;
use mesh_sieve::topology::cell_type::CellType;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::sieve::Sieve;

fn main() -> Result<(), MeshSieveError> {
    let msh = r#"
$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
4
1 0 0 0
2 1 0 0
3 1 1 0
4 0 1 0
$EndNodes
$Elements
2
1 2 2 7 42 1 2 3
2 2 2 8 42 1 3 4
$EndElements
"#;

    let reader = GmshReader::default();
    let mut mesh = reader.read(msh.as_bytes())?;

    let labels = mesh.labels.as_ref().expect("Gmsh reader adds labels");
    let cell_1 = PointId::new(1)?;
    let cell_2 = PointId::new(2)?;
    assert_eq!(labels.get_label(cell_1, "gmsh:physical"), Some(7));
    assert_eq!(labels.get_label(cell_2, "gmsh:physical"), Some(8));

    let cell_types = mesh.cell_types.as_ref().expect("Gmsh reader sets cell types");
    assert_eq!(cell_types.try_restrict(cell_1)?, &[CellType::Triangle]);
    assert_eq!(cell_types.try_restrict(cell_2)?, &[CellType::Triangle]);

    let mut cell_vertices = Vec::new();
    for cell in [cell_1, cell_2] {
        let verts: Vec<_> = mesh.sieve.cone_points(cell).collect();
        assert_eq!(verts.len(), 3);
        cell_vertices.push((cell, verts));
    }

    let interp = interpolate_edges_faces(
        &mut mesh.sieve,
        mesh.cell_types.as_mut().expect("cell types mutable"),
    )?;
    assert_eq!(interp.edge_points.len(), 5);
    assert!(interp.face_points.is_empty());

    let coordinates = mesh
        .coordinates
        .as_ref()
        .expect("Gmsh reader provides coordinates");
    for (cell, verts) in cell_vertices {
        let quality = cell_quality_from_section(CellType::Triangle, &verts, coordinates)?;
        assert!(quality.jacobian_sign > 0.0);
        assert!(quality.min_angle_deg >= 40.0);
        assert!(quality.aspect_ratio <= 3.0);
        println!("cell {cell:?} quality={quality:?}");
    }

    Ok(())
}
