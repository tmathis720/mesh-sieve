// Trait-level and minimal separator tests for Sieve
use crate::topology::sieve::{InMemorySieve, Sieve};
use crate::topology::point::PointId;

fn v(i: u64) -> PointId {
    PointId::new(i)
}

#[test]
fn meet_two_triangles_shared_vertices() {
    // Two triangles: 10 (1,2,3), 11 (2,3,4), sharing vertices 2 and 3
    let mut s = InMemorySieve::<PointId, ()>::default();
    // triangle 10
    s.add_arrow(v(10), v(1), ());
    s.add_arrow(v(10), v(2), ());
    s.add_arrow(v(10), v(3), ());
    // triangle 11
    s.add_arrow(v(11), v(2), ());
    s.add_arrow(v(11), v(3), ());
    s.add_arrow(v(11), v(4), ());
    // support edges
    for (src, dsts) in [(v(10), [v(1), v(2), v(3)]), (v(11), [v(2), v(3), v(4)])] {
        for d in dsts {
            s.add_arrow(d, src, ());
        }
    }
    let sep: Vec<_> = s.meet(v(10), v(11)).collect();
    assert!(sep.is_empty());
}

#[test]
fn meet_disjoint_cells() {
    let mut s = InMemorySieve::<PointId, ()>::default();
    s.add_arrow(v(10), v(1), ());
    s.add_arrow(v(10), v(2), ());
    s.add_arrow(v(11), v(3), ());
    s.add_arrow(v(11), v(4), ());
    for (src, dsts) in [(v(10), [v(1), v(2)]), (v(11), [v(3), v(4)])] {
        for d in dsts {
            s.add_arrow(d, src, ());
        }
    }
    let sep: Vec<_> = s.meet(v(10), v(11)).collect();
    assert!(sep.is_empty());
}

#[test]
fn meet_same_cell() {
    let mut s = InMemorySieve::<PointId, ()>::default();
    s.add_arrow(v(10), v(1), ());
    s.add_arrow(v(10), v(2), ());
    for d in [v(1), v(2)] {
        s.add_arrow(d, v(10), ());
    }
    let sep: Vec<_> = s.meet(v(10), v(10)).collect();
    assert!(sep.is_empty());
}

#[test]
fn meet_two_triangles_shared_edge_entity() {
    // Model: triangles 10 (1,2,3), 11 (2,3,4), sharing edge 20 (2,3)
    // Sieve: triangle -> edge(s), edge -> vertex(s)
    let mut s = InMemorySieve::<PointId, ()>::default();
    // triangle 10
    s.add_arrow(v(10), v(21), ()); // edge (1,2)
    s.add_arrow(v(10), v(20), ()); // edge (2,3)
    s.add_arrow(v(10), v(22), ()); // edge (3,1)
    // triangle 11
    s.add_arrow(v(11), v(20), ()); // edge (2,3)
    s.add_arrow(v(11), v(23), ()); // edge (3,4)
    s.add_arrow(v(11), v(24), ()); // edge (4,2)
    // edge to vertices
    s.add_arrow(v(20), v(2), ());
    s.add_arrow(v(20), v(3), ());
    s.add_arrow(v(21), v(1), ());
    s.add_arrow(v(21), v(2), ());
    s.add_arrow(v(22), v(3), ());
    s.add_arrow(v(22), v(1), ());
    s.add_arrow(v(23), v(3), ());
    s.add_arrow(v(23), v(4), ());
    s.add_arrow(v(24), v(4), ());
    s.add_arrow(v(24), v(2), ());
    // support arrows (optional, for bidirectionality)
    for (src, dsts) in [
        (v(10), vec![v(21), v(20), v(22)]),
        (v(11), vec![v(20), v(23), v(24)]),
        (v(20), vec![v(2), v(3)]),
        (v(21), vec![v(1), v(2)]),
        (v(22), vec![v(3), v(1)]),
        (v(23), vec![v(3), v(4)]),
        (v(24), vec![v(4), v(2)]),
    ] {
        for d in dsts {
            s.add_arrow(d, src, ());
        }
    }
    let sep: Vec<_> = s.meet(v(10), v(11)).collect();
    assert_eq!(sep, vec![]);
}

#[test]
fn meet_two_triangles_shared_edge_entity_refined() {
    // Triangles 10 (1,2,3), 11 (2,3,4), sharing edge 20 (2,3)
    // Only triangles point to edges, and edges point to vertices. No support arrows.
    let mut s = InMemorySieve::<PointId, ()>::default();
    // triangle 10
    s.add_arrow(v(10), v(21), ()); // edge (1,2)
    s.add_arrow(v(10), v(20), ()); // edge (2,3)
    s.add_arrow(v(10), v(22), ()); // edge (3,1)
    // triangle 11
    s.add_arrow(v(11), v(20), ()); // edge (2,3)
    s.add_arrow(v(11), v(23), ()); // edge (3,4)
    s.add_arrow(v(11), v(24), ()); // edge (4,2)
    // edge to vertices
    s.add_arrow(v(20), v(2), ());
    s.add_arrow(v(20), v(3), ());
    s.add_arrow(v(21), v(1), ());
    s.add_arrow(v(21), v(2), ());
    s.add_arrow(v(22), v(3), ());
    s.add_arrow(v(22), v(1), ());
    s.add_arrow(v(23), v(3), ());
    s.add_arrow(v(23), v(4), ());
    s.add_arrow(v(24), v(4), ());
    s.add_arrow(v(24), v(2), ());
    let sep: Vec<_> = s.meet(v(10), v(11)).collect();
    assert_eq!(sep, vec![]);
}

#[test]
fn join_on_star_graph() {
    // Star: 1→2, 1→3, 1→4
    let mut s = InMemorySieve::<u32, ()>::default();
    s.add_arrow(1, 2,());
    s.add_arrow(1, 3,());
    s.add_arrow(1, 4,());
    let join: Vec<_> = s.join(2, 3).collect();
    // join(2,3) should yield [1,2,3] (closure_both)
    let mut expected = vec![1,2,3];
    join.iter().for_each(|x| assert!(expected.contains(x)));
    assert_eq!(join.len(), 3);
}

#[test]
fn meet_and_join_combined() {
    // Two triangles sharing edge 20 (2,3)
    let mut s = InMemorySieve::<u32, ()>::default();
    // triangle 10: 10→21,10→20,10→22; triangle 11: 11→20,11→23,11→24
    s.add_arrow(10, 21, ()); s.add_arrow(10, 20, ()); s.add_arrow(10, 22,());
    s.add_arrow(11, 20, ()); s.add_arrow(11, 23, ()); s.add_arrow(11, 24,());
    // edge to vertices
    s.add_arrow(20, 2, ()); s.add_arrow(20, 3,());
    s.add_arrow(21, 1, ()); s.add_arrow(21, 2,());
    s.add_arrow(22, 3, ()); s.add_arrow(22, 1,());
    s.add_arrow(23, 3, ()); s.add_arrow(23, 4,());
    s.add_arrow(24, 4, ()); s.add_arrow(24, 2,());
    // meet(10,11) should be empty (no minimal separator)
    let sep: Vec<_> = s.meet(10, 11).collect();
    assert_eq!(sep, vec![]);
    // join(21,23) should yield [10,11,21,23] (closure_both)
    let mut join: Vec<_> = s.join(21, 23).collect();
    join.sort();
    assert_eq!(join, vec![10,11,21,23]);
}
