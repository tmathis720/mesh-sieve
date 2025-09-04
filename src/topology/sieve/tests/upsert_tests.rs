use crate::topology::orientation::Sign;
use crate::topology::sieve::in_memory_oriented::InMemoryOrientedSieve;
use crate::topology::sieve::{InMemorySieve, Sieve};
use crate::topology::stack::{InMemoryStack, Stack};

#[test]
fn add_arrow_upserts_payload() {
    let mut s = InMemorySieve::<u32, i32>::default();
    s.add_arrow(1, 2, 10);
    s.add_arrow(1, 2, 20); // upsert
    let outs: Vec<_> = s.cone(1).collect();
    assert_eq!(outs.len(), 1);
    assert_eq!(outs[0], (2, 20));
    let ins: Vec<_> = s.support(2).collect();
    assert_eq!(ins, vec![(1, 20)]);
}

#[test]
fn add_cone_dedups() {
    let mut s = InMemorySieve::<u32, i32>::default();
    s.add_cone(1, vec![(2, 1), (2, 2), (3, 3)]);
    let mut cone: Vec<_> = s.cone(1).collect();
    cone.sort_by_key(|(d, _)| *d);
    assert_eq!(cone, vec![(2, 2), (3, 3)]);
}

#[test]
fn set_cone_last_wins_and_mirrors() {
    let mut s = InMemorySieve::<u32, i32>::default();
    s.set_cone(1, vec![(2, 1), (2, 9), (3, 3)]);
    let outs: Vec<_> = s.cone(1).collect();
    assert_eq!(outs.len(), 2);
    assert!(outs.contains(&(2, 9)));
    assert!(outs.contains(&(3, 3)));
    let ins: Vec<_> = s.support(2).collect();
    assert_eq!(ins, vec![(1, 9)]);
}

#[test]
fn oriented_add_upserts_payload_and_orientation() {
    let mut s = InMemoryOrientedSieve::<u32, i32>::default();
    s.add_arrow_o(1, 2, 10, Sign(true));
    s.add_arrow_o(1, 2, 20, Sign(false));
    let outs: Vec<_> = s.cone(1).collect();
    assert_eq!(outs, vec![(2, 20)]);
    let outs_o: Vec<_> = s.cone_o(1).collect();
    assert_eq!(outs_o, vec![(2, Sign(false))]);
}

#[test]
fn stack_add_upserts() {
    let mut st = InMemoryStack::<u32, u32, i32>::default();
    st.add_arrow(1, 100, 7).unwrap();
    st.add_arrow(1, 100, 9).unwrap();
    let ups: Vec<_> = st.lift(1).collect();
    assert_eq!(ups, vec![(100, 9)]);
    let downs: Vec<_> = st.drop(100).collect();
    assert_eq!(downs, vec![(1, 9)]);
}
