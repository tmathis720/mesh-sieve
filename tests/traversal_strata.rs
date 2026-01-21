use mesh_sieve::algs::traversal::{TraversalOrder, closure_to_depth, closure_to_height};
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::sieve::Sieve;
use mesh_sieve::topology::sieve::in_memory::InMemorySieve;
use mesh_sieve::topology::sieve::strata::compute_strata;

#[test]
fn closure_depth_and_height_respect_strata_on_mixed_mesh() {
    let mut s = InMemorySieve::<PointId, ()>::new();
    let p = |i: u64| PointId::new(i).unwrap();

    let c0 = p(1);
    let e0 = p(2);
    let v0 = p(3);
    let c1 = p(4);
    let v1 = p(5);

    // Mixed-depth: c0 -> e0 -> v0 (depth 2), c1 -> v1 (depth 1).
    s.add_arrow(c0, e0, ());
    s.add_arrow(e0, v0, ());
    s.add_arrow(c1, v1, ());

    let sorted_depth = closure_to_depth(&s, [c0, c1], 1, TraversalOrder::Sorted).unwrap();
    let expected_depth = vec![e0, v0, c1, v1];
    assert_eq!(sorted_depth, expected_depth);

    let chart_depth = closure_to_depth(&s, [c0, c1], 1, TraversalOrder::Chart).unwrap();
    let strata = compute_strata(&s).unwrap();
    let idx = |p: &PointId| strata.chart_index.get(p).copied().unwrap();
    assert!(chart_depth.windows(2).all(|w| idx(&w[0]) <= idx(&w[1])));
    let mut chart_sorted = chart_depth.clone();
    chart_sorted.sort_unstable();
    assert_eq!(chart_sorted, expected_depth);

    let sorted_height = closure_to_height(&s, [c0, c1], 1, TraversalOrder::Sorted).unwrap();
    let expected_height = vec![c0, e0, c1, v1];
    assert_eq!(sorted_height, expected_height);
}
