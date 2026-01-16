mod util;
use util::*;

use bytemuck::{Pod, Zeroable, cast_slice};
use mesh_sieve::algs::communicator::{Communicator, NoComm, Wait};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug, PartialEq, Eq)]
struct WireU64 {
    x: u64,
}

#[test]
fn no_comm_is_nop() {
    let comm = NoComm;
    assert!(comm.is_no_comm());
    let mut buf = [0u8; 8];
    let h = comm.irecv(0, 123, &mut buf);
    assert!(h.wait().is_none());
    let s = comm.isend(0, 123, &[]);
    assert!(s.wait().is_none());
}

#[test]
fn rayon_comm_roundtrip_and_tag_isolation() {
    let (c0, c1) = rayons();

    const TAG_A: u16 = 0xA100;
    const TAG_B: u16 = 0xB200;

    let mut buf_a = [0u8; core::mem::size_of::<WireU64>()];
    let mut buf_b = [0u8; core::mem::size_of::<WireU64>()];
    let rxa = c1.irecv(0, TAG_A, &mut buf_a);
    let rxb = c1.irecv(0, TAG_B, &mut buf_b);

    let wa = [WireU64 {
        x: 0xDEAD_BEEF_F00D_F00D,
    }];
    let wb = [WireU64 {
        x: 0x0123_4567_89AB_CDEF,
    }];
    c0.isend(1, TAG_B, cast_slice(&wb));
    c0.isend(1, TAG_A, cast_slice(&wa));

    let ra = rxa.wait().expect("rxa");
    let rb = rxb.wait().expect("rxb");
    assert_eq!(&ra[..], cast_slice(&wa));
    assert_eq!(&rb[..], cast_slice(&wb));
}

#[cfg(feature = "mpi-support")]
#[test]
fn mpi_comm_smoke_if_available() {
    use mesh_sieve::algs::communicator::MpiComm;
    let world = MpiComm::new().expect("MPI initialization failed");
    let me = world.rank();
    let n = world.size();
    const TAG: u16 = 0xCAFE;
    let to = (me + 1) % n;
    let from = (me + n - 1) % n;
    let tx = [42u8, me as u8, 0, 0];
    let mut rx = [0u8; 4];
    let r = world.irecv(from, TAG, &mut rx);
    let s = world.isend(to, TAG, &tx);
    let got = r.wait().expect("mpi rx");
    assert_eq!(got, tx);
    let _ = s.wait();
}
