use mesh_sieve::algs::reduction::{transitive_closure_edges, transitive_reduction_dag};
use mesh_sieve::mesh_error::MeshSieveError;
use mesh_sieve::topology::sieve::{InMemorySieve, InMemorySieveDeterministic, Sieve};

#[test]
fn diamond_reduction_removes_direct_edge() {
    let (u, a, b, v) = (1u32, 2u32, 3u32, 4u32);
    let mut s = InMemorySieve::<u32, ()>::default();
    s.add_arrow(u, a, ());
    s.add_arrow(u, b, ());
    s.add_arrow(a, v, ());
    s.add_arrow(b, v, ());
    s.add_arrow(u, v, ());
    let removed = transitive_reduction_dag(&mut s).unwrap();
    assert_eq!(removed, 1);
    assert!(s.has_arrow(u, a) && s.has_arrow(u, b));
    assert!(s.has_arrow(a, v) && s.has_arrow(b, v));
    assert!(!s.has_arrow(u, v));
}

#[test]
fn diamond_closure_finds_edge() {
    let (u, a, b, v) = (1u32, 2u32, 3u32, 4u32);
    let mut s = InMemorySieve::<u32, ()>::default();
    s.add_arrow(u, a, ());
    s.add_arrow(u, b, ());
    s.add_arrow(a, v, ());
    s.add_arrow(b, v, ());
    let edges = transitive_closure_edges(&mut s).unwrap();
    assert_eq!(edges, vec![(u, v)]);
}

#[test]
fn chain_reduction_and_closure() {
    let mut s = InMemorySieve::<u32, ()>::default();
    for i in 1..5 {
        s.add_arrow(i, i + 1, ());
    }
    s.add_arrow(1, 5, ());
    let removed = transitive_reduction_dag(&mut s).unwrap();
    assert_eq!(removed, 1);
    for i in 1..5 {
        assert!(s.has_arrow(i, i + 1));
    }
    assert!(!s.has_arrow(1, 5));

    // Closure on pure chain
    let mut t = InMemorySieve::<u32, ()>::default();
    for i in 1..5 {
        t.add_arrow(i, i + 1, ());
    }
    let edges = transitive_closure_edges(&mut t).unwrap();
    let expected = vec![(1, 3), (1, 4), (1, 5), (2, 4), (2, 5), (3, 5)];
    assert_eq!(edges, expected);
}

#[test]
fn cycle_detection_errors() {
    let mut s = InMemorySieve::<u32, ()>::default();
    s.add_arrow(1, 2, ());
    s.add_arrow(2, 3, ());
    s.add_arrow(3, 1, ());
    assert!(matches!(
        transitive_reduction_dag(&mut s),
        Err(MeshSieveError::CycleDetected)
    ));
    assert!(matches!(
        transitive_closure_edges(&mut s),
        Err(MeshSieveError::CycleDetected)
    ));
}

#[test]
fn deterministic_results() {
    let mut s1 = InMemorySieve::<u32, ()>::default();
    s1.add_arrow(1, 3, ());
    s1.add_arrow(1, 2, ());
    s1.add_arrow(2, 4, ());
    s1.add_arrow(3, 4, ());
    s1.add_arrow(1, 4, ());
    let mut s2 = s1.clone();
    let r1 = transitive_reduction_dag(&mut s1).unwrap();
    let r2 = transitive_reduction_dag(&mut s2).unwrap();
    assert_eq!(r1, r2);
    for (a, b) in &[(1, 2), (2, 4), (3, 4)] {
        assert!(s1.has_arrow(*a, *b) && s2.has_arrow(*a, *b));
    }
    assert!(!s1.has_arrow(1, 4) && !s2.has_arrow(1, 4));

    // Closure determinism
    let mut det = InMemorySieve::<u32, ()>::default();
    det.add_arrow(1, 2, ());
    det.add_arrow(2, 3, ());
    let e1 = transitive_closure_edges(&mut det).unwrap();
    let e2 = transitive_closure_edges(&mut det).unwrap();
    assert_eq!(e1, e2);
}
