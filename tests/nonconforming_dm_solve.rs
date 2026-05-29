use mesh_sieve::algs::communicator::NoComm;
use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::closure::ClosureOrder;
use mesh_sieve::data::hanging_node_constraints::{HangingNodeConstraints, LinearConstraintTerm};
use mesh_sieve::data::section::Section;
use mesh_sieve::data::storage::VecStorage;
use mesh_sieve::diagnostics::PrepareForSolveOptions;
use mesh_sieve::dm::MeshDM;
use mesh_sieve::physics::fe::insert_element_residual_with_hanging_constraints;
use mesh_sieve::topology::anchors::{AnchorKind, TopologicalAnchors};
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::sieve::{MeshSieve, Sieve};

fn p(id: u64) -> PointId {
    PointId::new(id).unwrap()
}

fn nonconforming_mesh() -> MeshSieve {
    let mut mesh = MeshSieve::default();
    // One fine/nonconforming cell whose third vertex is a hanging midpoint on
    // the coarse edge between points 1 and 2.
    for point in [p(1), p(2), p(4)] {
        mesh.add_arrow(p(10), point, ());
    }
    mesh
}

fn scalar_section() -> Section<f64, VecStorage<f64>> {
    let mut atlas = Atlas::default();
    for point in [p(1), p(2), p(4), p(10)] {
        atlas.try_insert(point, 1).unwrap();
    }
    let mut section = Section::<f64, VecStorage<f64>>::new(atlas);
    section.try_set(p(1), &[2.0]).unwrap();
    section.try_set(p(2), &[6.0]).unwrap();
    section.try_set(p(4), &[0.0]).unwrap();
    section.try_set(p(10), &[0.0]).unwrap();
    section
}

fn hanging_constraints() -> HangingNodeConstraints<f64> {
    let mut constraints = HangingNodeConstraints::default();
    constraints.insert_constraint(
        p(4),
        0,
        vec![
            LinearConstraintTerm::new(p(1), 0, 0.5),
            LinearConstraintTerm::new(p(2), 0, 0.5),
        ],
    );
    constraints
}

#[test]
fn nonconforming_constraints_flow_through_dm_numbering_closure_and_residual() {
    let mut anchors = TopologicalAnchors::default();
    anchors.insert(p(4), [p(1), p(2)], AnchorKind::Hanging);
    let constraints = hanging_constraints();

    let mut dm = MeshDM::<f64>::builder(nonconforming_mesh())
        .section("u", scalar_section())
        .build()
        .unwrap();
    dm.set_topological_anchors(anchors);
    dm.set_hanging_constraints_for_section("u", constraints.clone());

    let diagnostics = dm
        .prepare_for_solve(
            &NoComm,
            PrepareForSolveOptions {
                require_coordinates: false,
                require_cell_types: false,
                synchronize_ghost_sections: false,
                ..PrepareForSolveOptions::default()
            },
        )
        .unwrap();
    assert!(diagnostics.ready);

    let global = dm.global_section("u").unwrap();
    assert_eq!(global.total_dofs(), 3, "hanging point is eliminated");
    assert_eq!(global.global_index(p(1), 0).unwrap(), 0);
    assert_eq!(global.global_index(p(2), 0).unwrap(), 1);
    assert_eq!(global.global_index(p(10), 0).unwrap(), 2);
    assert!(global.global_index(p(4), 0).is_err());

    let closure = dm
        .get_closure_values("u", p(10), &ClosureOrder::BreadthFirstDmpLex)
        .unwrap();
    assert_eq!(closure.values, vec![0.0, 2.0, 6.0, 4.0]);
    assert_eq!(
        closure
            .points
            .iter()
            .map(|entry| entry.point)
            .collect::<Vec<_>>(),
        vec![p(10), p(1), p(2), p(4)]
    );

    let mut residual = dm.create_global_vector("u").unwrap().values;
    insert_element_residual_with_hanging_constraints(
        &closure,
        &[0.0, 10.0, 20.0, 8.0],
        global,
        &constraints,
        &mut residual,
    )
    .unwrap();
    assert_eq!(residual, vec![14.0, 24.0, 0.0]);
}
