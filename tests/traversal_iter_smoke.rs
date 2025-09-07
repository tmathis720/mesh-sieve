use mesh_sieve::topology::sieve::Sieve;
use mesh_sieve::topology::sieve::in_memory::InMemorySieve;

#[test]
fn closure_iter_behaves_like_boxed_closure() {
    let mut s = InMemorySieve::<u32, ()>::new();
    // 1 -> 2 -> 3
    s.add_arrow(1, 2, ());
    s.add_arrow(2, 3, ());

    let mut v1: Vec<_> = s.closure([1]).collect();
    v1.sort_unstable();

    let mut v2: Vec<_> = s.closure_iter([1]).collect();
    v2.sort_unstable();

    assert_eq!(v1, v2);
    assert_eq!(v2, vec![1, 2, 3]);
}

#[test]
fn star_iter_behaves_like_boxed_star() {
    let mut s = InMemorySieve::<u32, ()>::new();
    // 1 -> 2 <- 3
    s.add_arrow(1, 2, ());
    s.add_arrow(3, 2, ());

    let mut v1: Vec<_> = s.star([2]).collect();
    v1.sort_unstable();

    let mut v2: Vec<_> = s.star_iter([2]).collect();
    v2.sort_unstable();

    assert_eq!(v1, v2);
    assert_eq!(v2, vec![1, 2, 3]);
}

#[test]
fn closure_both_iter_visits_both_directions() {
    let mut s = InMemorySieve::<u32, ()>::new();
    // 1 -> 2 -> 3 and 2 <- 4
    s.add_arrow(1, 2, ());
    s.add_arrow(2, 3, ());
    s.add_arrow(4, 2, ());

    let mut v: Vec<_> = s.closure_both_iter([2]).collect();
    v.sort_unstable();

    assert_eq!(v, vec![1, 2, 3, 4]);
}
