use mesh_sieve::algs::dual_graph::{DualGraphOpts, build_dual_with_opts};
use mesh_sieve::algs::lattice::AdjacencyOpts;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::sieve::Sieve;
use mesh_sieve::topology::sieve::in_memory::InMemorySieve;

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------

fn v(i: u64) -> PointId {
    PointId::new(i).unwrap()
}

// Three cells: c0 and c1 share a face; c1 and c2 only share a vertex
fn mesh_face_vertex() -> (InMemorySieve<PointId, ()>, Vec<PointId>) {
    let (c0, c1, c2) = (v(10), v(11), v(12));
    let f_shared = v(20);
    let f0 = v(21);
    let f1 = v(22);
    let f1v = v(23);
    let f2v = v(24);
    let f2 = v(25);
    let v_shared = v(30);

    let mut s = InMemorySieve::<PointId, ()>::new();
    // cell0 faces
    for &f in &[f_shared, f0] {
        s.add_arrow(c0, f, ());
    }
    // cell1 faces
    for &f in &[f_shared, f1, f1v] {
        s.add_arrow(c1, f, ());
    }
    // cell2 faces
    for &f in &[f2, f2v] {
        s.add_arrow(c2, f, ());
    }
    // connect faces that expose the shared vertex
    s.add_arrow(f1v, v_shared, ());
    s.add_arrow(f2v, v_shared, ());

    (s, vec![c0, c1, c2])
}

// Three cells all sharing the same face -> clique
fn mesh_nonmanifold() -> (InMemorySieve<PointId, ()>, Vec<PointId>) {
    let (c0, c1, c2) = (v(100), v(101), v(102));
    let f_shared = v(200);
    let mut s = InMemorySieve::<PointId, ()>::new();
    for &c in &[c0, c1, c2] {
        s.add_arrow(c, f_shared, ());
    }
    (s, vec![c0, c1, c2])
}

// ----------------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------------

#[test]
fn face_only_adjacency_default() {
    let (s, cells) = mesh_face_vertex();
    let dg = build_dual_with_opts(&s, cells.clone(), DualGraphOpts::default(), None);
    // only c0<->c1 via shared face
    assert_eq!(dg.xadj, vec![0, 1, 2, 2]);
    assert_eq!(dg.adjncy, vec![1, 0]);
}

#[test]
fn widened_boundary_includes_vertices() {
    let (s, cells) = mesh_face_vertex();
    let opts = DualGraphOpts {
        boundary: AdjacencyOpts {
            max_down_depth: Some(2),
        },
        symmetrize: true,
    };
    let dg = build_dual_with_opts(&s, cells.clone(), opts, None);
    // c0<->c1 via face, c1<->c2 via vertex
    assert_eq!(dg.xadj, vec![0, 1, 3, 4]);
    assert_eq!(dg.adjncy, vec![1, 0, 2, 1]);
}

#[test]
fn non_manifold_face_forms_clique() {
    let (s, cells) = mesh_nonmanifold();
    let dg1 = build_dual_with_opts(&s, cells.clone(), DualGraphOpts::default(), None);
    let dg2 = build_dual_with_opts(&s, cells.clone(), DualGraphOpts::default(), None);
    // triangle clique
    assert_eq!(dg1.xadj, vec![0, 2, 4, 6]);
    assert_eq!(dg1.adjncy, vec![1, 2, 0, 2, 0, 1]);
    // deterministic across runs
    assert_eq!(dg1.xadj, dg2.xadj);
    assert_eq!(dg1.adjncy, dg2.adjncy);
}

#[test]
fn weights_callback_applied() {
    use std::collections::HashMap;
    use std::sync::OnceLock;

    static WEIGHTS: OnceLock<HashMap<PointId, i32>> = OnceLock::new();
    fn w(p: PointId) -> i32 {
        *WEIGHTS.get().unwrap().get(&p).unwrap()
    }

    let (s, cells) = mesh_face_vertex();
    let map: HashMap<_, _> = cells
        .iter()
        .map(|&c| (c, s.cone_points(c).count() as i32))
        .collect();
    WEIGHTS.set(map).unwrap();
    let dg = build_dual_with_opts(&s, cells.clone(), DualGraphOpts::default(), Some(w));
    assert_eq!(dg.vwgt, vec![2, 3, 2]);
}

#[test]
fn symmetric_and_self_free() {
    let (s, cells) = mesh_nonmanifold();
    let dg = build_dual_with_opts(&s, cells.clone(), DualGraphOpts::default(), None);
    for i in 0..cells.len() {
        let nbrs = &dg.adjncy[dg.xadj[i]..dg.xadj[i + 1]];
        assert!(!nbrs.contains(&i));
        for &j in nbrs {
            let nbrs_j = &dg.adjncy[dg.xadj[j]..dg.xadj[j + 1]];
            assert!(nbrs_j.contains(&i));
        }
    }
}
