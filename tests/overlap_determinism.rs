use mesh_sieve::overlap::overlap::{Overlap, part};
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::sieve::{InMemorySieve, Sieve};

fn pid(n: u64) -> PointId {
    PointId::new(n).unwrap()
}

#[test]
fn bulk_insertion_is_deterministic() {
    let mut ov = Overlap::new();

    // Scrambled order with duplicates
    let edges = vec![
        (pid(5), 2),
        (pid(1), 0),
        (pid(3), 1),
        (pid(1), 0),
        (pid(2), 0),
        (pid(3), 1),
        (pid(4), 2),
        (pid(2), 0),
    ];

    let n1 = ov.add_links_structural_bulk(edges.clone());
    let n2 = ov.add_links_structural_bulk(edges);
    assert!(n1 > 0);
    assert_eq!(n2, 0);

    for r in ov.neighbor_ranks() {
        let locals_in_order: Vec<_> = ov
            .support(part(r))
            .map(|(src, _rem)| src.as_local().expect("support yields Local ids"))
            .collect();

        let mut expected = locals_in_order.clone();
        expected.sort_unstable();
        assert_eq!(
            locals_in_order, expected,
            "adjacency for Part({r}) not sorted"
        );
    }
}

#[test]
fn ensure_closure_of_support_order_is_deterministic() {
    // Minimal mesh where closure includes multiple points
    let mut mesh = InMemorySieve::<PointId, ()>::default();
    // 10 -> 11, 10 -> 12, 11 -> 13, 12 -> 14
    mesh.add_arrow(pid(10), pid(11), ());
    mesh.add_arrow(pid(10), pid(12), ());
    mesh.add_arrow(pid(11), pid(13), ());
    mesh.add_arrow(pid(12), pid(14), ());
    let mut ov = Overlap::new();

    // Seed edge
    ov.add_links_structural_bulk(vec![(pid(10), 7)]);

    mesh_sieve::overlap::overlap::ensure_closure_of_support(&mut ov, &mesh);

    let locals_in_order: Vec<_> = ov
        .support(part(7))
        .map(|(src, _rem)| src.as_local().expect("support yields Local ids"))
        .collect();
    let mut expected = locals_in_order.clone();
    expected.sort_unstable();
    assert_eq!(locals_in_order, expected);
}
