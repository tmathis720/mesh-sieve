use mesh_sieve::topology::sieve::frozen_csr::FrozenSieveCsr;
use mesh_sieve::topology::sieve::{InMemorySieve, Sieve, SieveRef};

#[test]
fn frozen_matches_inmemory_neighbors() {
    let mut s = InMemorySieve::<u32, i32>::default();
    s.add_arrow(1, 2, 7);
    s.add_arrow(1, 3, 8);
    s.add_arrow(3, 4, 9);

    let f = FrozenSieveCsr::from_sieve(s.clone());

    let mut a: Vec<_> = s.cone_ref(1).map(|(q, w)| (q, *w)).collect();
    let mut b: Vec<_> = f.cone_ref(1).map(|(q, w)| (q, *w)).collect();
    a.sort();
    b.sort();
    assert_eq!(a, b);
}

#[test]
fn frozen_is_deterministic() {
    let mut s = InMemorySieve::<u32, ()>::default();
    s.add_arrow(1, 3, ());
    s.add_arrow(1, 2, ());
    s.add_arrow(2, 5, ());
    s.add_arrow(2, 4, ());

    let f = FrozenSieveCsr::from_sieve(s);
    let v1: Vec<_> = f.closure_iter([1]).collect();
    let v2: Vec<_> = f.closure_iter([1]).collect();
    assert_eq!(v1, v2);
    assert_eq!(v1, vec![1, 3, 2, 5, 4]);
}
