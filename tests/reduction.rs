mod util;
use mesh_sieve::algs::reduction::{transitive_closure_edges, transitive_reduction_dag};
use mesh_sieve::topology::sieve::Sieve;
use util::*;

#[test]
fn transitive_reduction_removes_implied_edge() {
    let mut s = sieve_from(&[(1, 2), (2, 3), (1, 3)]);
    let removed = transitive_reduction_dag(&mut s).expect("DAG");
    assert_eq!(removed, 1);
    let cone1: Vec<_> = s.cone_points(pid(1)).collect();
    assert!(!cone1.contains(&pid(3)));
    assert!(cone1.contains(&pid(2)));
}

#[test]
fn transitive_closure_edges_reports_missing_edges() {
    let mut s = sieve_from(&[(1, 2), (2, 3)]);
    let edges = transitive_closure_edges(&mut s).expect("ok");
    assert!(edges.contains(&(pid(1), pid(3))));
}

#[test]
fn cycle_detection_bubbles_error() {
    let mut s = sieve_from(&[(1, 2), (2, 1)]);
    let err = transitive_reduction_dag(&mut s)
        .err()
        .expect("should error");
    let msg = format!("{:?}", err);
    assert!(msg.contains("Cycle") || msg.contains("chart"), "{msg}");
}
