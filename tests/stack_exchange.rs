use mesh_sieve::algs::communicator::{CommTag, NoComm, StackCommTags};
use mesh_sieve::algs::completion::{complete_stack, stack_exchange::WirePoint};
use mesh_sieve::mesh_error::MeshSieveError;
use mesh_sieve::overlap::overlap::Overlap;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::stack::InMemoryStack;

#[test]
fn wire_point_roundtrip() {
    let vals = [
        PointId::new(u64::MAX / 2).unwrap(),
        PointId::new(u64::MAX / 2 + 1).unwrap(),
    ];
    for &p in &vals {
        let w = p.to_wire();
        let p2 = <PointId as WirePoint>::from_wire(w);
        assert_eq!(p, p2);
    }
}

#[test]
fn rank_bounds_error() {
    #[derive(Copy, Clone, Default, Eq, PartialEq, Hash, Debug, Ord, PartialOrd)]
    struct U(u64);
    impl WirePoint for U {
        fn to_wire(self) -> u64 { self.0 }
        fn from_wire(w: u64) -> Self { U(w) }
    }

    let mut stack = InMemoryStack::<U, U, u8>::new();
    let mut overlap = Overlap::default();
    let p = PointId::new(1).unwrap();
    overlap.add_link(p, 1, p); // neighbor rank 1

    let comm = NoComm;
    let tags = StackCommTags::from_base(CommTag::new(0x10));
    let err = complete_stack(&mut stack, &overlap, &comm, 0, 1, tags).unwrap_err();
    match err {
        MeshSieveError::CommError { neighbor, .. } => assert_eq!(neighbor, 1),
        _ => panic!("unexpected error"),
    }
}

