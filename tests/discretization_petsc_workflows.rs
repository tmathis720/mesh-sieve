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
