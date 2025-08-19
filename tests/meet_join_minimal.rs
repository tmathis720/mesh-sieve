use mesh_sieve::topology::sieve::in_memory::InMemorySieve;
use mesh_sieve::topology::sieve::Sieve;

// Build a tiny simplicial-ish mesh: two triangles (0,1) sharing an edge (4).
// 0 -> edges {2,3,4}; 1 -> edges {4,5,6}; edges -> vertices.
#[test]
fn meet_and_join_match_sieve_defs() {
    let mut s = InMemorySieve::<u32, ()>::new();

    // cells to edges
    s.add_arrow(0, 2, ());
    s.add_arrow(0, 3, ());
    s.add_arrow(0, 4, ());
    s.add_arrow(1, 4, ());
    s.add_arrow(1, 5, ());
    s.add_arrow(1, 6, ());
    // edges to vertices (pick anything consistent; 4 is shared)
    s.add_arrow(2, 7, ());
    s.add_arrow(2, 8, ());
    s.add_arrow(3, 8, ());
    s.add_arrow(3, 9, ());
    s.add_arrow(4, 8, ());
    s.add_arrow(4, 10, ());
    s.add_arrow(5, 10, ());
    s.add_arrow(5, 11, ());
    s.add_arrow(6, 11, ());
    s.add_arrow(6, 12, ());

    // meet(0,1) should be the minimal separator of closures: the shared edge {4}
    let meet: Vec<_> = s.meet(0, 1).collect();
    assert_eq!(meet, vec![4]);

    // join(2,4) should be the minimal common coface: the cell {0}
    let join: Vec<_> = s.join(2, 4).collect();
    assert_eq!(join, vec![0]);

    // join(2,5) has no common coface in this toy mesh
    let join_empty: Vec<_> = s.join(2, 5).collect();
    assert!(join_empty.is_empty());
}
