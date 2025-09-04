use std::sync::Arc;

use mesh_sieve::topology::sieve::{
    InMemoryOrientedSieveArc, InMemorySieveArc, InMemoryStackArc, Sieve,
};
use mesh_sieve::topology::stack::Stack;

#[test]
fn no_extra_allocations_on_traversal() {
    let mut s = InMemorySieveArc::<u32, String>::default();
    let p = Arc::new("abc".to_owned());
    s.add_arrow(1, 2, p.clone());

    let (dst1, a1) = s.cone(1).next().unwrap();
    let (dst2, a2) = s.cone(1).next().unwrap();
    assert_eq!(dst1, 2);
    assert_eq!(dst2, 2);
    assert!(Arc::ptr_eq(&a1, &p));
    assert!(Arc::ptr_eq(&a2, &p));
}

#[test]
fn add_arrow_val_wraps_once() {
    let mut s = InMemorySieveArc::<u32, i32>::default();
    s.add_arrow_val(1, 2, 7);
    let (_, a1) = s.cone(1).next().unwrap();
    let (_, a2) = s.cone(1).next().unwrap();
    assert_eq!(*a1, 7);
    assert!(Arc::ptr_eq(&a1, &a2));
}

#[test]
fn oriented_upsert_replaces_payload() {
    let mut s = InMemoryOrientedSieveArc::<u32, i32, i32>::default();
    s.add_arrow_o_val(1, 2, 10, 5);
    let (_, a1) = s.cone(1).next().unwrap();
    s.add_arrow_o_val(1, 2, 20, -3);
    let (_, a2) = s.cone(1).next().unwrap();
    assert_eq!((*a1, *a2), (10, 20));
}

#[test]
fn stack_shares_payload_across_directions() {
    let mut st = InMemoryStackArc::<u32, u32, i32>::default();
    st.add_arrow_val(1, 100, 9).unwrap();
    let (cap, a) = st.lift(1).next().unwrap();
    let (base, b) = st.drop(100).next().unwrap();
    assert_eq!((cap, base, *a, *b), (100, 1, 9, 9));
    assert!(Arc::ptr_eq(&a, &b));
}
