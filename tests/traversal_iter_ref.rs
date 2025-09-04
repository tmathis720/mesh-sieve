use mesh_sieve::topology::sieve::Sieve;
use mesh_sieve::topology::sieve::in_memory::InMemorySieve;
use mesh_sieve::topology::sieve::traversal_iter::{ClosureIter, ClosureIterRef};

#[test]
fn ref_and_value_closure_visit_same_set() {
    let mut s = InMemorySieve::<u32, ()>::default();
    s.add_arrow(1, 2, ());
    s.add_arrow(1, 3, ());
    s.add_arrow(2, 4, ());
    s.add_arrow(3, 4, ());

    let a: std::collections::BTreeSet<_> = ClosureIter::new(&s, [1]).collect();
    let b: std::collections::BTreeSet<_> = ClosureIterRef::new_ref(&s, [1]).collect();
    assert_eq!(a, b);
}

#[test]
fn ref_sorted_neighbors_is_deterministic() {
    let mut s = InMemorySieve::<u32, ()>::default();
    s.add_arrow(1, 3, ());
    s.add_arrow(1, 2, ());
    s.add_arrow(2, 5, ());
    s.add_arrow(2, 4, ());

    let v1: Vec<_> = ClosureIterRef::new_ref_sorted_neighbors(&s, [1]).collect();
    let v2: Vec<_> = ClosureIterRef::new_ref_sorted_neighbors(&s, [1]).collect();
    assert_eq!(v1, v2);
}
