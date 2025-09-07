use std::collections::{HashMap, HashSet};

use mesh_sieve::algs::communicator::{CommTag, Communicator, NoComm, RayonComm, Wait};
use mesh_sieve::algs::completion::size_exchange::{
    exchange_sizes, exchange_sizes_symmetric,
};

#[test]
fn zero_neighbors_asymmetric() {
    let links: HashMap<usize, Vec<u8>> = HashMap::new();
    let res = exchange_sizes(&links, &NoComm, CommTag::new(0x10));
    assert!(res.unwrap().is_empty());
}

#[test]
fn zero_neighbors_symmetric() {
    let links: HashMap<usize, Vec<u8>> = HashMap::new();
    let neighbors: HashSet<usize> = HashSet::new();
    let res = exchange_sizes_symmetric(&links, &NoComm, CommTag::new(0x11), &neighbors);
    assert!(res.unwrap().is_empty());
}

#[test]
fn mismatch_drain() {
    let tag = CommTag::new(0x12);
    let c0 = RayonComm::new(0, 3);
    let c1 = RayonComm::new(1, 3);
    let c2 = RayonComm::new(2, 3);

    // Neighbor 1 sends malformed count (3 bytes)
    let _ = c1.isend(0, tag.as_u16(), &[1, 2, 3]);
    let mut r1 = [0u8; 4];
    let h1 = c1.irecv(0, tag.as_u16(), &mut r1);

    // Neighbor 2 sends correct 4-byte count
    let _ = c2.isend(0, tag.as_u16(), &[0, 0, 0, 0]);
    let mut r2 = [0u8; 4];
    let h2 = c2.irecv(0, tag.as_u16(), &mut r2);

    let mut links: HashMap<usize, Vec<u8>> = HashMap::new();
    links.insert(1, vec![]);
    links.insert(2, vec![]);

    let res = exchange_sizes(&links, &c0, tag);
    assert!(res.is_err());

    // Our sends to both neighbors should complete
    assert_eq!(h1.wait().unwrap().len(), 4);
    assert_eq!(h2.wait().unwrap().len(), 4);
}

#[test]
fn commtag_round_trip() {
    let val = 0xABCD;
    let tag = CommTag::new(val);
    assert_eq!(tag.as_u16(), val);
}

