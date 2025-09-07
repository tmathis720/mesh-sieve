#![allow(clippy::needless_collect)]

use mesh_sieve::algs::traversal as trv;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::sieve::Sieve;
use mesh_sieve::topology::sieve::in_memory::InMemorySieve;

fn v(i: u32) -> PointId {
    PointId::new((i + 1) as u64).unwrap()
}

/// Build the toy mesh from the Sieve paper (Fig.1) consistent with Table 2 results:
/// - Cells: 0, 1
/// - Edges: 2,3,4 (cell 0), 4,5,6 (cell 1) with 4 shared
/// - Vertices: 7,8,9,10 (we only need {7,8,10} to match closure(1), star(8))
///
/// The arrows realize the covering (downward) relation:
/// cell -> edges, edge -> vertices.
/// See: cone/closure/support/star definitions and Table 2 examples.
///   cone(0) {2,3,4}; support(4) {0,1};
///   closure(1) {1,4,5,6,7,10,8}; star(8) {2,4,6,8,0,1};
///   meet(0,1) {4}; join(2,4) {0}; join(2,5) {}.
fn build_fig1_sieve() -> InMemorySieve<PointId, ()> {
    let mut s = InMemorySieve::<PointId, ()>::new();

    // Cells to edges (downward)
    s.add_arrow(v(0), v(2), ());
    s.add_arrow(v(0), v(3), ());
    s.add_arrow(v(0), v(4), ());
    s.add_arrow(v(1), v(4), ());
    s.add_arrow(v(1), v(5), ());
    s.add_arrow(v(1), v(6), ());

    // Edges to vertices (downward)
    // Choose assignments to satisfy Table 2: closure(1) and star(8)
    // - edge 4 must touch {8,10} (shared by cells 0 and 1)
    // - edge 5 must touch {10,7}
    // - edge 6 must touch {7,8}
    // - edge 2 must touch {8,9} so that vertex 8's star includes edge 2 and cell 0
    // - edge 3 touches {9,7} (no vertex 8 here)
    s.add_arrow(v(2), v(8), ());
    s.add_arrow(v(2), v(9), ());
    s.add_arrow(v(3), v(9), ());
    s.add_arrow(v(3), v(11), ());
    s.add_arrow(v(4), v(8), ());
    s.add_arrow(v(4), v(10), ());
    s.add_arrow(v(5), v(10), ());
    s.add_arrow(v(5), v(7), ());
    s.add_arrow(v(6), v(7), ());
    s.add_arrow(v(6), v(8), ());
    s
}

#[test]
fn cone_and_support_match_table2() {
    let s = build_fig1_sieve();
    let mut cone0: Vec<_> = s.cone(v(0)).map(|(q, _)| q).collect();
    cone0.sort_unstable();
    assert_eq!(cone0, vec![v(2), v(3), v(4)]); // Table 2 cone(0) {2,3,4}  [oai_citation:4‡Knepley and Karpeev - 2009 - Mesh Algorithms for PDE with Sieve I Mesh Distribution.pdf](file-service://file-Ev5kaM59Ed2eJRxmNcEP9F)

    let mut sup4: Vec<_> = s.support(v(4)).map(|(p, _)| p).collect();
    sup4.sort_unstable();
    assert_eq!(sup4, vec![v(0), v(1)]); // Table 2 support(4) {0,1}  [oai_citation:5‡Knepley and Karpeev - 2009 - Mesh Algorithms for PDE with Sieve I Mesh Distribution.pdf](file-service://file-Ev5kaM59Ed2eJRxmNcEP9F)
}

#[test]
fn closure_and_star_match_table2() {
    let s = build_fig1_sieve();

    let mut cl1: Vec<_> = s.closure([v(1)]).collect();
    cl1.sort_unstable();
    assert_eq!(cl1, vec![v(1), v(4), v(5), v(6), v(7), v(8), v(10)]); // Table 2 closure(1)  [oai_citation:6‡Knepley and Karpeev - 2009 - Mesh Algorithms for PDE with Sieve I Mesh Distribution.pdf](file-service://file-Ev5kaM59Ed2eJRxmNcEP9F)

    let mut st8: Vec<_> = s.star([v(8)]).collect();
    st8.sort_unstable();
    assert_eq!(st8, vec![v(0), v(1), v(2), v(4), v(6), v(8)]); // Table 2 star(8)  [oai_citation:7‡Knepley and Karpeev - 2009 - Mesh Algorithms for PDE with Sieve I Mesh Distribution.pdf](file-service://file-Ev5kaM59Ed2eJRxmNcEP9F)
}

#[test]
fn meet_and_join_match_table2() {
    let s = build_fig1_sieve();

    let mut meet01: Vec<_> = s.meet(v(0), v(1)).collect();
    meet01.sort_unstable();
    assert_eq!(meet01, vec![v(4)]); // Table 2 meet(0,1) {4}  [oai_citation:8‡Knepley and Karpeev - 2009 - Mesh Algorithms for PDE with Sieve I Mesh Distribution.pdf](file-service://file-Ev5kaM59Ed2eJRxmNcEP9F)

    let mut join24: Vec<_> = s.join(v(2), v(4)).collect();
    join24.sort_unstable();
    assert_eq!(join24, vec![v(0)]); // Table 2 join(2,4) {0}  [oai_citation:9‡Knepley and Karpeev - 2009 - Mesh Algorithms for PDE with Sieve I Mesh Distribution.pdf](file-service://file-Ev5kaM59Ed2eJRxmNcEP9F)

    let join25: Vec<_> = s.join(v(2), v(5)).collect();
    assert!(join25.is_empty()); // Table 2 join(2,5) {}  [oai_citation:10‡Knepley and Karpeev - 2009 - Mesh Algorithms for PDE with Sieve I Mesh Distribution.pdf](file-service://file-Ev5kaM59Ed2eJRxmNcEP9F)
}

/// Golden tests for `link()` on the canonical toy mesh.
///
/// Your current definition:
///     link(p) = star(p) ∩ closure(p) minus {p} ∪ cone(p) ∪ support(p)
/// In a proper mesh-incidence DAG (arrows from higher to lower dimension),
/// star(p) contains cofaces (higher dimension), and closure(p) contains faces (lower dimension).
/// Thus their intersection is {p} only, so subtracting p and its immediate neighbors yields ∅.
/// This is consistent with the Sieve view where closure/star correspond exactly
/// to topological closure/star in a cell complex.  [oai_citation:11‡P1295.pdf](file-service://file-57etMKmLgiwKdu4SCHJWvj)
#[test]
fn link_is_empty_for_cell_edge_vertex_in_fig1() {
    let s = build_fig1_sieve();

    // Cell 0
    let mut l0 = trv::link(&s, v(0));
    l0.sort_unstable();
    assert!(l0.is_empty());

    // Shared edge 4
    let mut l4 = trv::link(&s, v(4));
    l4.sort_unstable();
    assert!(l4.is_empty());

    // Vertex 8
    let mut l8 = trv::link(&s, v(8));
    l8.sort_unstable();
    assert!(l8.is_empty());
}

/// Optional: sanity check—`link(p)` is empty for all points in this incidence DAG.
#[test]
fn link_is_empty_for_all_points_in_fig1() {
    let s = build_fig1_sieve();
    for p in s.points() {
        let mut l = trv::link(&s, p);
        l.sort_unstable();
        assert!(l.is_empty(), "link({p}) should be empty on incidence DAG");
    }
}
