use mesh_sieve::algs::assembly::preallocation_csr_from_closure;
use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::closure::ClosureOrder;
use mesh_sieve::data::discretization::DiscretizationMetadata;
use mesh_sieve::data::section::Section;
use mesh_sieve::data::storage::VecStorage;
use mesh_sieve::discretization::runtime::{
    Basis, CellGeometry, FiniteVolumeMetadata, cell_geometry_from_vertices, flux_stencils,
    runtime_from_metadata,
};
use mesh_sieve::topology::cell_type::CellType;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::sieve::{InMemorySieve, Sieve};

fn p(id: u64) -> PointId {
    PointId::new(id).unwrap()
}

#[test]
fn configurable_lagrange_bases_cover_supported_cells() {
    let cases = [
        (CellType::Triangle, 6),
        (CellType::Tetrahedron, 10),
        (CellType::Quadrilateral, 9),
        (CellType::Hexahedron, 27),
        (CellType::Prism, 18),
        (CellType::Pyramid, 14),
    ];
    for (cell, nodes) in cases {
        let basis = Basis::lagrange(cell, 2).unwrap();
        assert_eq!(basis.num_nodes(), nodes, "{cell:?}");
        let metadata = DiscretizationMetadata::new("lagrange", "gauss2")
            .with_basis_metadata(2, [] as [&str; 0]);
        let runtime = runtime_from_metadata(&metadata, cell).unwrap();
        assert_eq!(runtime.basis.num_nodes(), nodes, "metadata {cell:?}");
        assert_eq!(runtime.quadrature.dimension(), cell.dimension() as usize);
    }
}

#[test]
fn triangle_poisson_element_tabulates_partition_of_unity() {
    let basis = Basis::lagrange(CellType::Triangle, 2).unwrap();
    let quad = runtime_from_metadata(
        &DiscretizationMetadata::new("lagrange", "gauss2").with_basis_metadata(2, [] as [&str; 0]),
        CellType::Triangle,
    )
    .unwrap()
    .quadrature;
    let tab = basis.tabulate(&quad.points).unwrap();
    for values in tab.values {
        assert!((values.iter().sum::<f64>() - 1.0).abs() < 1e-10);
    }
}

#[test]
fn finite_volume_geometry_and_flux_stencils_are_topology_driven() {
    let fv = FiniteVolumeMetadata::new(3)
        .with_reconstruction_order(2)
        .with_limiter("minmod");
    assert_eq!(fv.components, 3);
    let geom: CellGeometry = cell_geometry_from_vertices(&[
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
        vec![0.0, 1.0],
    ])
    .unwrap();
    assert_eq!(geom.centroid, vec![0.5, 0.5]);
    assert!((geom.volume - 1.0).abs() < 1e-12);

    let mut sieve = InMemorySieve::<PointId, ()>::default();
    sieve.add_arrow(p(10), p(1), ());
    sieve.add_arrow(p(11), p(1), ());
    let stencils = flux_stencils(&sieve, [p(1)]);
    assert_eq!(stencils[0].left, p(10));
    assert_eq!(stencils[0].right, Some(p(11)));
}

#[test]
fn closure_preallocation_builds_csr_from_section_dofs() {
    let mut sieve = InMemorySieve::<PointId, ()>::default();
    sieve.add_arrow(p(10), p(1), ());
    sieve.add_arrow(p(10), p(2), ());
    sieve.add_arrow(p(11), p(2), ());
    sieve.add_arrow(p(11), p(3), ());

    let mut atlas = Atlas::default();
    atlas.try_insert(p(1), 1).unwrap();
    atlas.try_insert(p(2), 2).unwrap();
    atlas.try_insert(p(3), 1).unwrap();
    let section = Section::<f64, VecStorage<f64>>::new(atlas);

    let csr = preallocation_csr_from_closure(
        &sieve,
        &section,
        [p(10), p(11)],
        0,
        &ClosureOrder::BreadthFirstDmpLex,
    )
    .unwrap();
    assert_eq!(csr.rows.len(), 4);
    assert_eq!(csr.xadj.len(), 5);
    assert!(csr.adjncy.len() >= 8);
}

#[test]
fn explicit_discretization_matrix_covers_common_fe_fv_cells() {
    use mesh_sieve::discretization::runtime::{
        BasisFamily, QuadratureFamily, supported_discretizations,
    };
    use mesh_sieve::physics::fvm::validate_fv_cell_type;

    let rows = supported_discretizations();
    for cell in [
        CellType::Segment,
        CellType::Triangle,
        CellType::Quadrilateral,
        CellType::Tetrahedron,
        CellType::Hexahedron,
        CellType::Prism,
        CellType::Pyramid,
    ] {
        let row = rows
            .iter()
            .find(|row| row.cell_type == cell)
            .expect("capability row");
        assert_eq!(row.dimension, cell.dimension() as usize);
        assert!(row.finite_volume);
        validate_fv_cell_type(cell).unwrap();
    }

    let triangle = rows
        .iter()
        .find(|row| row.cell_type == CellType::Triangle)
        .unwrap();
    assert!(triangle.basis_families.contains(&BasisFamily::Simplex));
    assert!(
        triangle
            .quadrature_families
            .contains(&QuadratureFamily::Simplex)
    );

    let hex = rows
        .iter()
        .find(|row| row.cell_type == CellType::Hexahedron)
        .unwrap();
    assert!(hex.basis_families.contains(&BasisFamily::TensorProduct));
    assert!(
        hex.quadrature_families
            .contains(&QuadratureFamily::GaussLegendre)
    );
}

#[test]
fn unsupported_discretizations_are_policy_errors() {
    use mesh_sieve::discretization::runtime::{BasisFamily, QuadratureRule};
    use mesh_sieve::physics::fvm::validate_fv_cell_type;

    let bad_basis =
        Basis::lagrange_with_family(CellType::Quadrilateral, 2, BasisFamily::Simplex).unwrap_err();
    assert!(format!("{bad_basis}").contains("unsupported Lagrange"));

    let bad_quadrature =
        QuadratureRule::from_metadata("newton_cotes2", CellType::Triangle).unwrap_err();
    assert!(format!("{bad_quadrature}").contains("unsupported quadrature"));

    let bad_fv = validate_fv_cell_type(CellType::Simplex(4)).unwrap_err();
    assert!(format!("{bad_fv}").contains("unsupported finite-volume"));
}

#[test]
fn high_order_coordinates_drive_quadratic_closure_assembly() {
    use mesh_sieve::data::coordinates::{Coordinates, HighOrderCoordinates};
    use mesh_sieve::physics::fe::assemble_element_matrices_from_closure;

    let cell = p(10);
    let vertices = [p(1), p(2), p(3)];
    let mut sieve = InMemorySieve::<PointId, ()>::default();
    for vertex in vertices {
        sieve.add_arrow(cell, vertex, ());
    }

    let mut coord_atlas = Atlas::default();
    for vertex in vertices {
        coord_atlas.try_insert(vertex, 2).unwrap();
    }
    let mut coords = Coordinates::<f64, VecStorage<f64>>::try_new(2, 2, coord_atlas).unwrap();
    coords.section_mut().try_set(p(1), &[0.0, 0.0]).unwrap();
    coords.section_mut().try_set(p(2), &[1.0, 0.0]).unwrap();
    coords.section_mut().try_set(p(3), &[0.0, 1.0]).unwrap();

    let mut ho_atlas = Atlas::default();
    ho_atlas.try_insert(cell, 12).unwrap();
    let mut ho = HighOrderCoordinates::<f64, VecStorage<f64>>::try_new(2, ho_atlas).unwrap();
    ho.section_mut()
        .try_set(
            cell,
            &[
                0.0, 0.0, // (0,0)
                0.0, 0.5, // (0,1/2)
                0.0, 1.0, // (0,1)
                0.5, 0.0, // (1/2,0)
                0.5, 0.5, // (1/2,1/2)
                1.0, 0.0, // (1,0)
            ],
        )
        .unwrap();
    coords.set_high_order(ho).unwrap();

    let metadata =
        DiscretizationMetadata::new("lagrange", "gauss2").with_basis_metadata(2, [] as [&str; 0]);
    let matrices = assemble_element_matrices_from_closure(
        &sieve,
        &coords,
        CellType::Triangle,
        cell,
        0,
        &ClosureOrder::BreadthFirstDmpLex,
        &metadata,
        |_| 1.0,
    )
    .unwrap();
    assert_eq!(matrices.load.len(), 6);
    assert_eq!(matrices.stiffness.len(), 36);
}

#[test]
fn fv_stencils_are_deterministic_for_supported_cell_families() {
    use mesh_sieve::discretization::runtime::supported_discretizations;
    use mesh_sieve::physics::fvm::validate_fv_cell_type;

    for (idx, row) in supported_discretizations()
        .iter()
        .filter(|row| row.finite_volume)
        .enumerate()
    {
        validate_fv_cell_type(row.cell_type).unwrap();
        let face = p(1000 + idx as u64);
        let left = p(2000 + idx as u64 * 2);
        let right = p(2001 + idx as u64 * 2);
        let mut sieve = InMemorySieve::<PointId, ()>::default();
        sieve.add_arrow(right, face, ());
        sieve.add_arrow(left, face, ());
        let stencils = flux_stencils(&sieve, [face]);
        assert_eq!(stencils.len(), 1, "{:?}", row.cell_type);
        assert_eq!(stencils[0].left, left, "{:?}", row.cell_type);
        assert_eq!(stencils[0].right, Some(right), "{:?}", row.cell_type);
    }
}
