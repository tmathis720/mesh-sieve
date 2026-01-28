use crate::data::atlas::Atlas;
use crate::data::section::Section;
use crate::data::storage::VecStorage;
use crate::mesh_error::MeshSieveError;
use crate::topology::cell_type::CellType;
use crate::topology::point::PointId;
use crate::topology::sieve::{InMemorySieve, Sieve};
use crate::topology::validation::{
    NonManifoldHandling, TopologyValidationOptions, validate_sieve_topology,
};

fn pid(id: u64) -> PointId {
    PointId::new(id).expect("valid PointId")
}

fn non_manifold_triangle_mesh() -> Result<
    (
        InMemorySieve<PointId, ()>,
        Section<CellType, VecStorage<CellType>>,
    ),
    MeshSieveError,
> {
    let mut sieve = InMemorySieve::<PointId, ()>::default();

    let v1 = pid(1);
    let v2 = pid(2);
    let v3 = pid(3);
    let v4 = pid(4);
    let v5 = pid(5);

    let e12 = pid(10);
    let e23 = pid(11);
    let e31 = pid(12);
    let e24 = pid(13);
    let e41 = pid(14);
    let e25 = pid(15);
    let e51 = pid(16);

    let c1 = pid(100);
    let c2 = pid(101);
    let c3 = pid(102);

    sieve.add_arrow(e12, v1, ());
    sieve.add_arrow(e12, v2, ());
    sieve.add_arrow(e23, v2, ());
    sieve.add_arrow(e23, v3, ());
    sieve.add_arrow(e31, v3, ());
    sieve.add_arrow(e31, v1, ());
    sieve.add_arrow(e24, v2, ());
    sieve.add_arrow(e24, v4, ());
    sieve.add_arrow(e41, v4, ());
    sieve.add_arrow(e41, v1, ());
    sieve.add_arrow(e25, v2, ());
    sieve.add_arrow(e25, v5, ());
    sieve.add_arrow(e51, v5, ());
    sieve.add_arrow(e51, v1, ());

    sieve.add_arrow(c1, e12, ());
    sieve.add_arrow(c1, e23, ());
    sieve.add_arrow(c1, e31, ());
    sieve.add_arrow(c2, e12, ());
    sieve.add_arrow(c2, e24, ());
    sieve.add_arrow(c2, e41, ());
    sieve.add_arrow(c3, e12, ());
    sieve.add_arrow(c3, e25, ());
    sieve.add_arrow(c3, e51, ());

    let mut atlas = Atlas::default();
    for &cell in &[c1, c2, c3] {
        atlas.try_insert(cell, 1)?;
    }
    let mut cell_types = Section::<CellType, VecStorage<CellType>>::new(atlas);
    for &cell in &[c1, c2, c3] {
        cell_types.try_set(cell, &[CellType::Triangle])?;
    }

    Ok((sieve, cell_types))
}

#[test]
fn non_manifold_edges_error() -> Result<(), MeshSieveError> {
    let (sieve, cell_types) = non_manifold_triangle_mesh()?;
    let options = TopologyValidationOptions {
        check_cone_sizes: true,
        check_duplicate_arrows: true,
        check_closure_consistency: true,
        non_manifold: NonManifoldHandling::Error,
    };
    let err = validate_sieve_topology(&sieve, &cell_types, options).unwrap_err();
    assert!(matches!(
        err,
        MeshSieveError::NonManifoldIncidentCells { .. }
    ));
    Ok(())
}

#[test]
fn non_manifold_edges_warns() -> Result<(), MeshSieveError> {
    let (sieve, cell_types) = non_manifold_triangle_mesh()?;
    let options = TopologyValidationOptions {
        check_cone_sizes: true,
        check_duplicate_arrows: true,
        check_closure_consistency: true,
        non_manifold: NonManifoldHandling::Warn,
    };
    assert!(validate_sieve_topology(&sieve, &cell_types, options).is_ok());
    Ok(())
}
