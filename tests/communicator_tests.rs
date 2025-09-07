use mesh_sieve::algs::communicator::{CommTag, Communicator, RayonComm, Wait};

#[test]
fn rayon_round_trip() {
    let tag = CommTag(0x1000);
    let c0 = RayonComm::new(0, 2);
    let c1 = RayonComm::new(1, 2);

    let msg = b"hello";
    let _s = c0.isend(1, tag.base(), msg);

    let mut buf = [0u8; 5];
    let h = c1.irecv(0, tag.base(), &mut buf);
    let got = h.wait().unwrap();
    assert_eq!(&got, msg);
}

#[test]
fn rayon_fifo_order() {
    let tag = CommTag(0x1001);
    let c0 = RayonComm::new(0, 2);
    let c1 = RayonComm::new(1, 2);

    for i in 0..10u8 {
        let _ = c0.isend(1, tag.base(), &[i]);
    }
    let mut out = Vec::new();
    for _ in 0..10 {
        let mut b = [0u8; 1];
        let h = c1.irecv(0, tag.base(), &mut b);
        out.push(h.wait().unwrap()[0]);
    }
    assert_eq!(out, (0u8..10u8).collect::<Vec<_>>());
}

#[test]
fn truncation_is_ok() {
    let tag = CommTag(0x1002);
    let c0 = RayonComm::new(0, 2);
    let c1 = RayonComm::new(1, 2);

    let _ = c0.isend(1, tag.base(), &[1, 2, 3, 4, 5, 6]);
    let mut b = [0u8; 4];
    let h = c1.irecv(0, tag.base(), &mut b);
    let got = h.wait().unwrap();
    assert_eq!(got, vec![1, 2, 3, 4]);
}

