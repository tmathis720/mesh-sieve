use mesh_sieve::algs::communicator::CommTag;
use mesh_sieve::algs::wire::{expect_exact_len, WirePoint};
use mesh_sieve::topology::point::PointId;

#[test]
fn commtag_offset_wrap() {
    let t = CommTag::new(u16::MAX).offset(1);
    assert_eq!(t.as_u16(), 0);
}

#[test]
fn wirepoint_roundtrip() {
    let ids = [1u64, 1u64 << 31, (1u64 << 63) - 1];
    for raw in ids {
        let p = PointId::new(raw).unwrap();
        let w = p.to_wire();
        let back = PointId::from_wire(w);
        assert_eq!(back, p);
    }
}

#[test]
fn expect_exact_len_err() {
    assert!(expect_exact_len(3,4).is_err());
    assert!(expect_exact_len(4,4).is_ok());
}
