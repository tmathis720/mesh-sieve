use mesh_sieve::algs::communicator::NoComm;
use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::section::Section;
use mesh_sieve::data::storage::VecStorage;
use mesh_sieve::dm::{MeshDM, MeshDMOptions, MeshVectorInsertMode};
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
fn mesh_dm_exposes_chart_cones_and_supports() {
    let dm = MeshDM::<f64>::builder(tiny_mesh()).build().unwrap();

    let chart = dm.point_chart().unwrap();
    assert_eq!((chart.start, chart.end_inclusive), (1, 5));
    assert_eq!(chart.point_count, 5);
    assert!(chart.is_dense);
    assert_eq!(dm.cone_points(pid(3)), vec![pid(1), pid(2)]);
    assert_eq!(dm.oriented_cone(pid(3)), vec![(pid(1), 0), (pid(2), 0)]);
    assert_eq!(dm.cone_size(pid(3)), 2);
    assert_eq!(dm.support_points(pid(2)), vec![pid(3), pid(4)]);
    assert_eq!(dm.oriented_support(pid(2)), vec![(pid(3), 0), (pid(4), 0)]);
    assert_eq!(dm.support_size(pid(2)), 2);
    assert_eq!(
        dm.closure_points_at_depth(
            pid(3),
            0,
            &mesh_sieve::data::closure::ClosureOrder::BreadthFirstDmpLex,
        )
        .unwrap(),
        vec![pid(1), pid(2)]
    );
}

#[test]
fn mesh_dm_creates_sections_from_depth_layouts() {
    let mut labels = mesh_sieve::topology::labels::LabelSet::new();
    labels.set_label(pid(1), "boundary", 1);
    labels.set_label(pid(3), "boundary", 1);
    let mut dm = MeshDM::<f64>::builder(tiny_mesh())
        .labels(labels)
        .build()
        .unwrap();

    dm.create_section_from_depth("all", &[2, 1]).unwrap();
    let all = dm.section("all").unwrap();
    assert_eq!(all.atlas().total_len(), 8);
    assert_eq!(all.atlas().get(pid(1)).unwrap().1, 2);
    assert_eq!(all.atlas().get(pid(3)).unwrap().1, 1);

    dm.create_section_from_depth_on_label("boundary", &[2, 1], "boundary", 1)
        .unwrap();
    let boundary = dm.section("boundary").unwrap();
    assert_eq!(
        boundary.atlas().points().collect::<Vec<_>>(),
        vec![pid(1), pid(3)]
    );
    assert_eq!(boundary.atlas().total_len(), 3);
}

#[test]
fn mesh_dm_gathers_inserts_and_adds_local_vector_closures() {
    let dm = MeshDM::<f64>::builder(tiny_mesh())
        .section("u", scalar_section())
        .build()
        .unwrap();
    let order = mesh_sieve::data::closure::ClosureOrder::BreadthFirstDmpLex;
    let mut vector = dm.create_local_vector("u").unwrap();

    dm.set_local_vector_closure(
        "u",
        &mut vector,
        pid(3),
        &order,
        &[30.0, 10.0, 20.0],
        MeshVectorInsertMode::InsertValues,
    )
    .unwrap();
    assert_eq!(
        dm.get_local_vector_closure("u", &vector, pid(3), &order)
            .unwrap(),
        vec![30.0, 10.0, 20.0]
    );
    assert_eq!(
        dm.get_local_vector_closure_at_depth("u", &vector, pid(3), 0, &order)
            .unwrap(),
        vec![10.0, 20.0]
    );

    dm.set_local_vector_closure(
        "u",
        &mut vector,
        pid(4),
        &order,
        &[4.0, 2.0, 5.0],
        MeshVectorInsertMode::AddValues,
    )
    .unwrap();
    assert_eq!(
        dm.get_local_vector_closure("u", &vector, pid(4), &order)
            .unwrap(),
        vec![4.0, 22.0, 5.0]
    );
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

fn tiny_coordinates() -> mesh_sieve::data::coordinates::Coordinates<f64, VecStorage<f64>> {
    let mut atlas = Atlas::default();
    for point in [pid(1), pid(2), pid(5)] {
        atlas.try_insert(point, 1).unwrap();
    }
    let mut section = Section::<f64, VecStorage<f64>>::new(atlas);
    section.try_set(pid(1), &[0.0]).unwrap();
    section.try_set(pid(2), &[1.0]).unwrap();
    section.try_set(pid(5), &[2.0]).unwrap();
    mesh_sieve::data::coordinates::Coordinates::from_section(1, 1, section).unwrap()
}

fn tiny_cell_types() -> Section<
    mesh_sieve::topology::cell_type::CellType,
    VecStorage<mesh_sieve::topology::cell_type::CellType>,
> {
    let mut atlas = Atlas::default();
    atlas.try_insert(pid(3), 1).unwrap();
    atlas.try_insert(pid(4), 1).unwrap();
    let mut section = Section::<
        mesh_sieve::topology::cell_type::CellType,
        VecStorage<mesh_sieve::topology::cell_type::CellType>,
    >::new(atlas);
    section
        .try_set(
            pid(3),
            &[mesh_sieve::topology::cell_type::CellType::Segment],
        )
        .unwrap();
    section
        .try_set(
            pid(4),
            &[mesh_sieve::topology::cell_type::CellType::Segment],
        )
        .unwrap();
    section
}

fn solve_ready_dm() -> MeshDM<f64> {
    MeshDM::<f64>::builder(tiny_mesh())
        .coordinates(tiny_coordinates())
        .cell_types(tiny_cell_types())
        .section("u", scalar_section())
        .build()
        .unwrap()
}

#[test]
fn mesh_dm_prepare_for_solve_reports_missing_prerequisites() {
    let mut dm = MeshDM::<f64>::builder(tiny_mesh())
        .section("u", scalar_section())
        .build()
        .unwrap();

    let report = dm
        .prepare_for_solve(
            &NoComm,
            mesh_sieve::diagnostics::PrepareForSolveOptions {
                create_serial_ownership: false,
                ..Default::default()
            },
        )
        .unwrap();

    assert!(!report.ready);
    assert!(
        report
            .prerequisites
            .iter()
            .any(|p| p.name == "coordinates" && p.required && !p.present)
    );
    assert!(
        report
            .prerequisites
            .iter()
            .any(|p| p.name == "cell_types" && p.required && !p.present)
    );
    assert!(
        report
            .prerequisites
            .iter()
            .any(|p| p.name == "ownership" && p.required && !p.present)
    );
    assert!(report.steps.iter().all(|s| s.status == "skipped"));
}

#[test]
fn mesh_dm_prepare_for_solve_is_stable_for_identical_topology_and_partitioning() {
    let mut left = solve_ready_dm();
    let mut right = solve_ready_dm();

    let left_report = left.prepare_for_solve(&NoComm, Default::default()).unwrap();
    let right_report = right
        .prepare_for_solve(&NoComm, Default::default())
        .unwrap();

    let left_json =
        mesh_sieve::diagnostics::prepare_for_solve_diagnostics_json(&left_report).unwrap();
    let right_json =
        mesh_sieve::diagnostics::prepare_for_solve_diagnostics_json(&right_report).unwrap();
    assert_eq!(left_json, right_json);
    assert_eq!(
        left_json,
        "{\"ready\":true,\"prerequisites\":[{\"name\":\"coordinates\",\"required\":true,\"present\":true,\"complete\":true,\"detail\":\"vertex_points=3, missing_vertex_coordinates=0\"},{\"name\":\"cell_types\",\"required\":true,\"present\":true,\"complete\":true,\"detail\":\"cells=2, missing_cell_types=0\"},{\"name\":\"ownership\",\"required\":true,\"present\":true,\"complete\":true,\"detail\":\"topology_points=5, missing_ownership=0, ghost_points=0\"},{\"name\":\"overlap\",\"required\":false,\"present\":false,\"complete\":true,\"detail\":\"overlap graph is not attached; ghost_points=0, overlap_required=false\"}],\"steps\":[{\"name\":\"section_global_numbering\",\"status\":\"completed\",\"detail\":\"numbered_sections=1\"},{\"name\":\"matrix_preallocation_graph\",\"status\":\"completed\",\"detail\":\"rows=2, edges=2\"},{\"name\":\"ownership_overlap_checks\",\"status\":\"completed\",\"detail\":\"ownership and overlap topology are consistent\"},{\"name\":\"section_synchronization\",\"status\":\"skipped\",\"detail\":\"no overlap/ownership state available for ghost synchronization\"}],\"global_sections\":[\"u\"],\"preallocation\":{\"rows\":2,\"edges\":2,\"order\":[3,4],\"row_nnz\":[1,1]},\"synchronized_sections\":[]}"
    );
}
