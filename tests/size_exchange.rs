use mesh_sieve::algs::communicator::{CommTag, Communicator, NoComm, Wait};
use mesh_sieve::algs::completion::size_exchange::{
    exchange_sizes, exchange_sizes_symmetric,
};
use mesh_sieve::mesh_error::MeshSieveError;
use std::collections::{HashMap, HashSet};
use std::sync::{atomic::{AtomicBool, Ordering}, Arc, Mutex};

#[derive(Clone)]
struct DummySendHandle {
    waited: Arc<AtomicBool>,
}

impl Wait for DummySendHandle {
    fn wait(self) -> Option<Vec<u8>> {
        self.waited.store(true, Ordering::SeqCst);
        None
    }
}

#[derive(Clone)]
struct DummyRecvHandle {
    waited: Arc<AtomicBool>,
    resp: Option<Vec<u8>>,
}

impl Wait for DummyRecvHandle {
    fn wait(self) -> Option<Vec<u8>> {
        self.waited.store(true, Ordering::SeqCst);
        self.resp
    }
}

struct DummyComm {
    responses: HashMap<(usize, u16), Option<Vec<u8>>>,
    send_flags: Arc<Mutex<Vec<Arc<AtomicBool>>>>,
    recv_flags: Arc<Mutex<Vec<Arc<AtomicBool>>>>,
}

impl DummyComm {
    fn new(responses: HashMap<(usize, u16), Option<Vec<u8>>>) -> Self {
        Self {
            responses,
            send_flags: Arc::new(Mutex::new(Vec::new())),
            recv_flags: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

impl Communicator for DummyComm {
    type SendHandle = DummySendHandle;
    type RecvHandle = DummyRecvHandle;

    fn isend(&self, peer: usize, tag: u16, _buf: &[u8]) -> Self::SendHandle {
        let flag = Arc::new(AtomicBool::new(false));
        self.send_flags.lock().unwrap().push(flag.clone());
        // ensure tag lookup exists to mimic using tag
        let _ = peer;
        let _ = tag;
        DummySendHandle { waited: flag }
    }

    fn irecv(&self, peer: usize, tag: u16, _buf: &mut [u8]) -> Self::RecvHandle {
        let flag = Arc::new(AtomicBool::new(false));
        self.recv_flags.lock().unwrap().push(flag.clone());
        let resp = self.responses.get(&(peer, tag)).cloned().unwrap_or(None);
        DummyRecvHandle { waited: flag, resp }
    }

    fn rank(&self) -> usize {
        0
    }

    fn size(&self) -> usize {
        2
    }
}

#[test]
fn zero_neighbors_ok() {
    let links: HashMap<usize, Vec<u8>> = HashMap::new();
    let comm = NoComm;
    let res = exchange_sizes(&links, &comm, CommTag::new(1)).unwrap();
    assert!(res.is_empty());

    let all_neighbors: HashSet<usize> = HashSet::new();
    let res2 =
        exchange_sizes_symmetric(&links, &comm, CommTag::new(1), &all_neighbors).unwrap();
    assert!(res2.is_empty());
}

#[test]
fn mismatch_drains_all() {
    let tag = CommTag::new(7);
    let mut responses = HashMap::new();
    // neighbor 1 sends correct 4-byte count
    responses.insert((1, tag.as_u16()), Some(vec![0u8; 4]));
    // neighbor 2 sends only 2 bytes -> mismatch
    responses.insert((2, tag.as_u16()), Some(vec![0u8; 2]));
    let comm = DummyComm::new(responses);

    let mut links: HashMap<usize, Vec<u8>> = HashMap::new();
    links.insert(1, vec![1, 2, 3]);
    links.insert(2, vec![4, 5]);

    let err = exchange_sizes(&links, &comm, tag).unwrap_err();
    match err {
        MeshSieveError::CommError { neighbor, .. } => assert_eq!(neighbor, 2),
        _ => panic!("unexpected error"),
    }

    // ensure all waits drained
    let sends = comm.send_flags.lock().unwrap();
    assert!(sends.iter().all(|f| f.load(Ordering::SeqCst)));
    let recvs = comm.recv_flags.lock().unwrap();
    assert!(recvs.iter().all(|f| f.load(Ordering::SeqCst)));
}

#[test]
fn commtag_roundtrip() {
    let t = CommTag::new(0xABCD);
    assert_eq!(t.as_u16(), 0xABCD);
}

