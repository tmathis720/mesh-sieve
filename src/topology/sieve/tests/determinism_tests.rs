use crate::topology::sieve::{traversal_iter::ClosureIter, InMemorySieve, Sieve};

#[test]
fn points_sorted_is_stable() {
    let mut s = InMemorySieve::<u32, ()>::default();
    s.add_arrow(10, 20, ());
    s.add_arrow(30, 20, ());
    let a = s.points_sorted();
    let b = s.points_sorted();
    assert_eq!(a, b);
    let mut expected = vec![10, 20, 30];
    expected.sort();
    assert_eq!(a, expected);
}

#[test]
fn closure_iter_sorted_is_stable() {
    let mut s = InMemorySieve::<u32, ()>::default();
    s.add_arrow(1, 2, ());
    s.add_arrow(1, 3, ());
    s.add_arrow(2, 4, ());
    s.add_arrow(3, 4, ());

    let run = || -> Vec<u32> { s.closure_iter_sorted([3, 1, 2]).collect() };
    let a = run();
    let b = run();
    assert_eq!(a, b);
}

#[test]
fn closure_iter_sorted_neighbors_is_lexicographic() {
    let mut s = InMemorySieve::<u32, ()>::default();
    // insert edges in non-lexicographic order
    s.add_arrow(1, 3, ());
    s.add_arrow(1, 2, ());
    s.add_arrow(2, 5, ());
    s.add_arrow(2, 4, ());

    let v: Vec<_> = ClosureIter::new_sorted_neighbors(&s, [1]).collect();
    assert_eq!(v, vec![1, 2, 4, 5, 3]);
}

#[test]
fn points_chart_order_is_height_major() {
    let mut s = InMemorySieve::<u32, ()>::default();
    s.add_arrow(1, 2, ());
    s.add_arrow(2, 4, ());
    s.add_arrow(1, 3, ());
    s.add_arrow(3, 5, ());
    let chart = s.points_chart_order().unwrap();
    assert_eq!(chart, vec![1, 2, 3, 4, 5]);
}
