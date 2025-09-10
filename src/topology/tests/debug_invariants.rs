#![cfg(any(debug_assertions, feature = "strict-invariants"))]

use crate::topology::orientation::BitFlip;
use crate::topology::sieve::{InMemoryOrientedSieve, InMemorySieve, OrientedSieve, Sieve};
use crate::topology::stack::{InMemoryStack, Stack};

#[test]
#[should_panic]
fn duplicate_edge_panics_in_debug() {
    let mut s = InMemorySieve::<u32, ()>::default();
    s.add_arrow(1, 2, ());
    s.adjacency_out.get_mut(&1).unwrap().push((2, ())); // create duplicate
    s.debug_assert_invariants();
}

#[test]
#[should_panic]
fn orientation_mismatch_panics_in_debug() {
    let mut s = InMemoryOrientedSieve::<u32, (), BitFlip>::default();
    s.add_arrow_o(1, 2, (), BitFlip(true));
    s.adjacency_in.get_mut(&2).unwrap()[0].2 = BitFlip(false);
    s.debug_assert_invariants();
}

#[test]
#[should_panic]
fn stack_mirror_missing_panics_in_debug() {
    let mut st = InMemoryStack::<u32, u32, ()>::new();
    st.base.add_arrow(1, 1, ());
    st.cap.add_arrow(2, 2, ());
    st.add_arrow(1, 2, ()).unwrap();
    st.down.get_mut(&2).unwrap().clear();
    st.debug_assert_invariants();
}

#[test]
#[should_panic]
fn stack_membership_panics_in_debug() {
    let mut st = InMemoryStack::<u32, u32, ()>::new();
    st.add_arrow(1, 2, ()).unwrap();
    st.debug_assert_invariants();
}
