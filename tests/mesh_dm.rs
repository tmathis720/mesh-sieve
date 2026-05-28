use mesh_sieve::algs::communicator::NoComm;
use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::section::Section;
use mesh_sieve::data::storage::VecStorage;
use mesh_sieve::dm::{MeshDM, MeshDMOptions};
use mesh_sieve::prelude::{PointId, Sieve};
use mesh_sieve::topology::sieve::MeshSieve;

fn pid(id: u64) -> PointId {
    PointId::new(id).unwrap()
}

fn tiny_mesh() -> MeshSieve {
    let mut mesh = MeshSieve::default();
    mesh.add_arrow(pid(3), pid(1), ());
    mesh.add_arrow(pid(3), pid(2), ());
    mesh.add_arrow(pid(4), pid(2), ());
    mesh.add_arrow(pid(4), pid(5), ());
    mesh
}

fn scalar_section() -> Section<f64, VecStorage<f64>> {
    let mut atlas = Atlas::default();
    atlas.try_insert(pid(1), 1).unwrap();
    atlas.try_insert(pid(2), 1).unwrap();
    atlas.try_insert(pid(3), 1).unwrap();
    atlas.try_insert(pid(4), 1).unwrap();
    atlas.try_insert(pid(5), 1).unwrap();
    let mut section = Section::<f64, VecStorage<f64>>::new(atlas);
    for point in [pid(1), pid(2), pid(3), pid(4), pid(5)] {
        section.try_set(point, &[point.get() as f64]).unwrap();
    }
    section
}

#[test]
fn mesh_dm_builds_vectors_and_global_sections() {
    let dm = MeshDM::<f64>::builder(tiny_mesh())
        .section("u", scalar_section())
        .build()
        .unwrap();

    assert_eq!(dm.height_stratum(0).unwrap(), vec![pid(3), pid(4)]);
    assert_eq!(dm.depth_stratum(0).unwrap(), vec![pid(1), pid(2), pid(5)]);
    assert_eq!(dm.create_local_vector("u").unwrap().values.len(), 5);

    let mut dm = dm;
    dm.build_global_sections(&NoComm).unwrap();
    let global = dm.create_global_vector("u").unwrap();
    assert_eq!(global.values.len(), 5);
}

#[test]
fn mesh_dm_reorders_sections_and_builds_preallocation_graph() {
    let options = MeshDMOptions {
        reorder_section: Some(mesh_sieve::prelude::StratifiedOrdering::CellFirst),
        check_skeleton: true,
        check_faces: true,
        ..MeshDMOptions::default()
    };

    let dm = MeshDM::<f64>::builder(tiny_mesh())
        .section("u", scalar_section())
        .options(options)
        .build()
        .unwrap();

    let section_points: Vec<_> = dm.section("u").unwrap().atlas().points().collect();
    assert_eq!(&section_points[0..2], &[pid(3), pid(4)]);

    let graph = dm.matrix_preallocation_graph(dm.height_stratum(0).unwrap(), Default::default());
    assert_eq!(graph.order, vec![pid(3), pid(4)]);
    assert_eq!(graph.row_nnz.len(), 2);
}

#[test]
fn mesh_dm_closure_and_star_wrappers_use_fe_ordering() {
    let dm = MeshDM::<f64>::builder(tiny_mesh())
        .section("u", scalar_section())
        .build()
        .unwrap();

    let order = mesh_sieve::data::closure::ClosureOrder::BreadthFirstDmpLex;
    assert_eq!(
        dm.closure_points(pid(3), &order).unwrap(),
        vec![pid(3), pid(1), pid(2)]
    );
    assert_eq!(
        dm.transitive_closure_points(pid(3), &order).unwrap(),
        vec![pid(3), pid(1), pid(2)]
    );
    assert_eq!(dm.star_points(pid(2)), vec![pid(2), pid(3), pid(4)]);

    let closure = dm.get_closure_values("u", pid(3), &order).unwrap();
    assert_eq!(closure.cell, pid(3));
    assert_eq!(closure.values, vec![3.0, 1.0, 2.0]);
}

#[test]
fn mesh_dm_label_helpers_filter_sections_and_build_subdm() {
    let mut labels = mesh_sieve::topology::labels::LabelSet::new();
    labels.set_label(pid(3), "marker", 7);

    let dm = MeshDM::<f64>::builder(tiny_mesh())
        .labels(labels)
        .section("u", scalar_section())
        .build()
        .unwrap();

    let points = dm
        .points_by_label_in_section(
            "marker",
            7,
            "u",
            mesh_sieve::algs::submesh::SubmeshSelection::FullClosure,
        )
        .unwrap();
    assert_eq!(points, vec![pid(1), pid(2), pid(3)]);

    let constrained = dm
        .create_constrained_view_from_labels(
            "u",
            &[mesh_sieve::data::LabelConstraintSpec::new(
                "marker",
                7,
                vec![0],
            )],
        )
        .unwrap();
    assert!(constrained.constraints().contains_key(&pid(3)));

    let sub = dm
        .sub_dm_by_label_with_sections(
            "marker",
            7,
            mesh_sieve::algs::submesh::SubmeshSelection::FullClosure,
            Some(&["u"]),
        )
        .unwrap();
    assert_eq!(sub.maps.sub_to_parent, vec![pid(1), pid(2), pid(3)]);
    assert!(sub.dm.section("u").is_some());
}
