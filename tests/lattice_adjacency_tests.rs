use mesh_sieve::algs::lattice::{AdjacencyOpts, adjacent, adjacent_with};
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::sieve::Sieve;
use mesh_sieve::topology::sieve::in_memory::InMemorySieve;

fn v(i: u64) -> PointId {
    PointId::new(i).unwrap()
}

#[test]
fn determinism() {
    let mut s = InMemorySieve::<PointId, ()>::new();
    // two triangles sharing an edge: 1->(3,4,5), 2->(5,6,7); edge5->verts
    s.add_arrow(v(1), v(3), ());
    s.add_arrow(v(1), v(4), ());
    s.add_arrow(v(1), v(5), ());
    s.add_arrow(v(2), v(5), ());
    s.add_arrow(v(2), v(6), ());
    s.add_arrow(v(2), v(7), ());
    s.add_arrow(v(5), v(8), ());
    s.add_arrow(v(5), v(9), ());
    let first = adjacent(&s, v(1));
    assert_eq!(first, vec![v(2)]);
    for _ in 0..100 {
        let n = adjacent(&s, v(1));
        assert_eq!(n, first);
    }
}

#[test]
fn depth_semantics() {
    let mut s = InMemorySieve::<PointId, ()>::new();
    // cell1 shares edge5 with cell2 and vertex9 with cell3
    s.add_arrow(v(1), v(3), ());
    s.add_arrow(v(1), v(4), ());
    s.add_arrow(v(1), v(5), ());
    s.add_arrow(v(2), v(5), ());
    s.add_arrow(v(2), v(6), ());
    s.add_arrow(v(2), v(7), ());
    s.add_arrow(v(5), v(8), ());
    s.add_arrow(v(5), v(9), ());
    s.add_arrow(v(10), v(9), ());

    let none = adjacent_with(
        &s,
        v(1),
        AdjacencyOpts {
            max_down_depth: Some(0),
            same_stratum_only: true,
        },
    );
    assert!(none.is_empty());
    let faces = adjacent_with(
        &s,
        v(1),
        AdjacencyOpts {
            max_down_depth: Some(1),
            same_stratum_only: true,
        },
    );
    assert_eq!(faces, vec![v(2)]);
    let verts = adjacent_with(
        &s,
        v(1),
        AdjacencyOpts {
            max_down_depth: Some(2),
            same_stratum_only: true,
        },
    );
    assert_eq!(verts, vec![v(2), v(10)]);
    let full = adjacent_with(
        &s,
        v(1),
        AdjacencyOpts {
            max_down_depth: None,
            same_stratum_only: true,
        },
    );
    for n in &verts {
        assert!(full.contains(n));
    }
    assert!(full.len() >= verts.len());
}

#[test]
fn cycle_safe() {
    let mut s = InMemorySieve::<PointId, ()>::new();
    // 1 <-> 2 cycle
    s.add_arrow(v(1), v(2), ());
    s.add_arrow(v(2), v(1), ());
    let neigh = adjacent_with(
        &s,
        v(1),
        AdjacencyOpts {
            max_down_depth: None,
            same_stratum_only: false,
        },
    );
    assert_eq!(neigh, vec![v(2)]);
}

#[test]
fn same_stratum_filter() {
    let mut s = InMemorySieve::<PointId, ()>::new();
    // cells 1 and 2 share edge5; vertex9 is boundary of edge5
    s.add_arrow(v(1), v(5), ());
    s.add_arrow(v(2), v(5), ());
    s.add_arrow(v(5), v(9), ());
    let cells_only = adjacent_with(
        &s,
        v(1),
        AdjacencyOpts {
            max_down_depth: Some(2),
            same_stratum_only: true,
        },
    );
    assert_eq!(cells_only, vec![v(2)]);
    let cross = adjacent_with(
        &s,
        v(1),
        AdjacencyOpts {
            max_down_depth: Some(2),
            same_stratum_only: false,
        },
    );
    assert_eq!(cross, vec![v(2), v(5)]);
}
