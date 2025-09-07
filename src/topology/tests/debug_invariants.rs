use crate::topology::orientation::BitFlip;
use crate::topology::sieve::{InMemoryOrientedSieve, InMemorySieve, OrientedSieve, Sieve};
use crate::topology::stack::{InMemoryStack, Stack};

#[test]
#[should_panic]
fn duplicate_edge_panics_in_debug() {
    let mut s = InMemorySieve::<u32, ()>::default();
    s.add_arrow(1, 2, ());
    s.adjacency_out.get_mut(&1).unwrap().push((2, ())); // create duplicate
    #[cfg(any(debug_assertions, feature = "strict-invariants"))]
    s.debug_assert_invariants();
}

#[test]
#[should_panic]
fn orientation_mismatch_panics_in_debug() {
    let mut s = InMemoryOrientedSieve::<u32, (), BitFlip>::default();
    s.add_arrow_o(1, 2, (), BitFlip(true));
    s.adjacency_in.get_mut(&2).unwrap()[0].2 = BitFlip(false);
    #[cfg(any(debug_assertions, feature = "strict-invariants"))]
    s.debug_assert_invariants();
}

#[test]
#[should_panic]
fn stack_mirror_missing_panics_in_debug() {
    let mut st = InMemoryStack::<u32, u32, ()>::new();
    st.add_arrow(1, 2, ()).unwrap();
    st.down.get_mut(&2).unwrap().clear();
    #[cfg(any(debug_assertions, feature = "strict-invariants"))]
    st.debug_assert_invariants();
}
