// cargo run --example poisson_segment
use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::discretization::{Discretization, DiscretizationMetadata, RegionKey};
use mesh_sieve::data::section::Section;
use mesh_sieve::data::storage::VecStorage;
use mesh_sieve::discretization::runtime::{
    DofMap, assemble_local_matrix, assemble_local_vector, cell_vertices, local_load_vector,
    local_stiffness_matrix, runtime_from_metadata, tabulate_element,
};
use mesh_sieve::mesh_error::MeshSieveError;
use mesh_sieve::mesh_generation::{MeshGenerationOptions, interval_mesh};
use mesh_sieve::topology::cell_type::CellType;

fn main() -> Result<(), MeshSieveError> {
    let mesh = interval_mesh(2, 0.0, 1.0, MeshGenerationOptions::default())?.mesh;
    let coordinates = mesh
        .coordinates
        .as_ref()
        .ok_or_else(|| MeshSieveError::InvalidGeometry("missing coordinates".to_string()))?;
    let cell_types = mesh
        .cell_types
        .as_ref()
        .ok_or_else(|| MeshSieveError::InvalidGeometry("missing cell types".to_string()))?;

    let mut discretization = Discretization::new();
    discretization.field_mut("u").set_cell_type_metadata(
        CellType::Segment,
        DiscretizationMetadata::new("lagrange_p1", "gauss1"),
    );
    let field = discretization
        .field("u")
        .ok_or_else(|| MeshSieveError::InvalidGeometry("missing field metadata".to_string()))?;
    let metadata = field
        .metadata_for(&RegionKey::cell_type(CellType::Segment))
        .ok_or_else(|| MeshSieveError::InvalidGeometry("missing segment metadata".to_string()))?;
    let runtime = runtime_from_metadata(metadata, CellType::Segment)?;

    let mut vertices = Vec::new();
    for (point, slice) in cell_types.iter() {
        if slice.first() == Some(&CellType::Vertex) {
            vertices.push(point);
        }
    }
    let dof_map = DofMap::new(vertices);

    let mut rhs_atlas = Atlas::default();
    let mut mat_atlas = Atlas::default();
    for &dof in dof_map.dofs() {
        rhs_atlas.try_insert(dof, 1)?;
        mat_atlas.try_insert(dof, dof_map.len())?;
    }
    let mut rhs = Section::<f64, VecStorage<f64>>::new(rhs_atlas);
    let mut matrix = Section::<f64, VecStorage<f64>>::new(mat_atlas);

    for (cell, slice) in cell_types.iter() {
        if slice.first() != Some(&CellType::Segment) {
            continue;
        }
        let element_vertices = cell_vertices(&mesh.sieve, cell, runtime.basis.num_nodes())?;
        let node_coords = element_vertices
            .iter()
            .map(|p| coordinates.try_restrict(*p).map(|vals| vals.to_vec()))
            .collect::<Result<Vec<_>, _>>()?;
        let tabulation = tabulate_element(&runtime, &node_coords)?;
        let local_k = local_stiffness_matrix(&tabulation);
        let local_f = local_load_vector(&tabulation, |_| 1.0);
        assemble_local_matrix(&mut matrix, &element_vertices, &dof_map, &local_k)?;
        assemble_local_vector(&mut rhs, &element_vertices, &local_f)?;
    }

    let expected_matrix = vec![
        vec![2.0, -2.0, 0.0],
        vec![-2.0, 4.0, -2.0],
        vec![0.0, -2.0, 2.0],
    ];
    let expected_rhs = vec![0.25, 0.5, 0.25];

    let mut rhs_values = Vec::new();
    let mut matrix_rows = Vec::new();
    for &dof in dof_map.dofs() {
        rhs_values.push(rhs.try_restrict(dof)?[0]);
        matrix_rows.push(matrix.try_restrict(dof)?.to_vec());
    }

    for i in 0..expected_rhs.len() {
        assert!(
            (rhs_values[i] - expected_rhs[i]).abs() < 1.0e-12,
            "rhs mismatch at {i}: expected {}, found {}",
            expected_rhs[i],
            rhs_values[i]
        );
    }
    for i in 0..expected_matrix.len() {
        for j in 0..expected_matrix.len() {
            assert!(
                (matrix_rows[i][j] - expected_matrix[i][j]).abs() < 1.0e-12,
                "matrix mismatch at ({i},{j})"
            );
        }
    }
    for row in &matrix_rows {
        let row_sum: f64 = row.iter().sum();
        assert!(row_sum.abs() < 1.0e-12, "row sum mismatch");
    }
    let total_rhs: f64 = rhs_values.iter().sum();
    assert!((total_rhs - 1.0).abs() < 1.0e-12, "rhs sum mismatch");

    Ok(())
}
