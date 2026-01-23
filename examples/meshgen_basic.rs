use mesh_sieve::algs::meshgen::{
    MeshGenOptions, StructuredCellType, cylinder_shell, sphere_shell, structured_box_2d,
    structured_box_3d,
};
use mesh_sieve::data::section::Section;
use mesh_sieve::data::storage::VecStorage;
use mesh_sieve::mesh_error::MeshSieveError;
use mesh_sieve::topology::cell_type::CellType;

fn count_type(section: &Section<CellType, VecStorage<CellType>>, cell_type: CellType) -> usize {
    section
        .iter()
        .filter(|(_, slice)| slice[0] == cell_type)
        .count()
}

fn main() -> Result<(), MeshSieveError> {
    let tri_mesh = structured_box_2d(
        1,
        1,
        [0.0, 0.0],
        [1.0, 1.0],
        StructuredCellType::Triangle,
        MeshGenOptions::default(),
    )?;
    let tri_types = tri_mesh.cell_types.as_ref().unwrap();
    assert_eq!(count_type(tri_types, CellType::Vertex), 4);
    assert_eq!(count_type(tri_types, CellType::Triangle), 2);

    let hex_mesh = structured_box_3d(
        1,
        1,
        1,
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        StructuredCellType::Hexahedron,
        MeshGenOptions::default(),
    )?;
    let hex_types = hex_mesh.cell_types.as_ref().unwrap();
    assert_eq!(count_type(hex_types, CellType::Vertex), 8);
    assert_eq!(count_type(hex_types, CellType::Hexahedron), 1);

    let sphere = sphere_shell(1.0, 2, 4, MeshGenOptions::default())?;
    let sphere_types = sphere.cell_types.as_ref().unwrap();
    assert_eq!(count_type(sphere_types, CellType::Vertex), 6);
    assert_eq!(count_type(sphere_types, CellType::Triangle), 8);

    let cylinder = cylinder_shell(1.0, 1.0, 4, 1, MeshGenOptions::default())?;
    let cylinder_types = cylinder.cell_types.as_ref().unwrap();
    assert_eq!(count_type(cylinder_types, CellType::Vertex), 8);
    assert_eq!(count_type(cylinder_types, CellType::Quadrilateral), 4);

    Ok(())
}
