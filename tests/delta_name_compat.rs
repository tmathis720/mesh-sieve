use mesh_sieve::data::refine::delta::{Delta, SliceDelta};
use mesh_sieve::topology::arrow::Orientation;

#[test]
fn impls_satisfy_both_names() {
    fn needs_slice_delta<T: SliceDelta<u8>>() {}
    fn needs_delta<T: Delta<u8>>() {}

    needs_slice_delta::<Orientation>();
    needs_delta::<Orientation>();
}

#[test]
fn call_apply_via_reexport() {
    let src = [1u8, 2, 3];
    let mut dst = [0u8; 3];
    let o = Orientation::Reverse;
    <Orientation as Delta<u8>>::apply(&o, &src, &mut dst).unwrap();
    assert_eq!(&dst, &[3, 2, 1]);
}
