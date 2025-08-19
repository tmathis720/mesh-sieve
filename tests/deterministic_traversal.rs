use mesh_sieve::algs::traversal::{closure_ordered, star_ordered};
use mesh_sieve::topology::sieve::Sieve;
use mesh_sieve::topology::sieve::in_memory::InMemorySieve;

#[test]
fn closure_is_deterministic_in_chart_order() {
    let mut s = InMemorySieve::<u32, ()>::new();
    // 0 -> 2,1; 1 -> 3; 2 -> 3
    s.add_arrow(0, 2, ());
    s.add_arrow(0, 1, ());
    s.add_arrow(1, 3, ());
    s.add_arrow(2, 3, ());

    let cl1 = closure_ordered(&mut s, [0]).unwrap();
    let cl2 = closure_ordered(&mut s, [0]).unwrap();
    assert_eq!(cl1, cl2);
    assert_eq!(cl1, vec![0, 1, 2, 3]);
}

#[test]
fn star_is_deterministic_in_chart_order() {
    let mut s = InMemorySieve::<u32, ()>::new();
    // 0 -> 1, 2 -> 1, 3 -> 2
    s.add_arrow(0, 1, ());
    s.add_arrow(2, 1, ());
    s.add_arrow(3, 2, ());

    let st1 = star_ordered(&mut s, [1]).unwrap();
    let st2 = star_ordered(&mut s, [1]).unwrap();
    assert_eq!(st1, st2);
}
