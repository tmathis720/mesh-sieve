use mesh_sieve::topology::arrow::Polarity;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::sieve::{InMemorySieve, Sieve};

/// Build a tiny oriented primal mesh: two quads with one shared edge.
/// One shared edge between `Q0` and `Q1` has reversed orientation to
/// exercise delta logic.
pub fn tiny_oriented_mesh() -> InMemorySieve<PointId, Polarity> {
    let mut s: InMemorySieve<PointId, Polarity> = InMemorySieve::default();
    let e = |i: u64| PointId::new(100 + i).unwrap();
    let q = |i: u64| PointId::new(200 + i).unwrap();

    // Q0 edges (Forward, Forward, Reverse, Forward)
    Sieve::add_arrow(&mut s, q(0), e(0), Polarity::Forward);
    Sieve::add_arrow(&mut s, q(0), e(1), Polarity::Forward);
    Sieve::add_arrow(&mut s, q(0), e(2), Polarity::Reverse); // reversed shared edge
    Sieve::add_arrow(&mut s, q(0), e(3), Polarity::Forward);

    // Q1 edges (all Forward) sharing edge e(2)
    Sieve::add_arrow(&mut s, q(1), e(2), Polarity::Forward);
    Sieve::add_arrow(&mut s, q(1), e(4), Polarity::Forward);
    Sieve::add_arrow(&mut s, q(1), e(5), Polarity::Forward);
    Sieve::add_arrow(&mut s, q(1), e(6), Polarity::Forward);

    s
}

/// Dual graph: two cells `Q0` and `Q1` with one undirected edge between them.
pub fn tiny_dual_graph_edges() -> Vec<(usize, usize)> {
    vec![(0, 1)]
}
