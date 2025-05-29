//! Thin façade over intra-process (Rayon) or inter-process (MPI) message passing.
//!
//! Messages are *contiguous byte slices* (no zero-copy guarantees).
//! All handles are **waitable** but non-blocking -– completion.rs calls
//! `.wait()` before it trusts that the buffer is ready.

use std::sync::mpsc::{self, Sender, Receiver};
use std::sync::{Arc, Mutex};
use dashmap::DashMap;
use bytes::Bytes;
use once_cell::sync::Lazy;
use std::thread::JoinHandle;

/// Non-blocking communication interface (minimal by design).
pub trait Communicator: Send + Sync + 'static {
    /// Handle returned by `isend`.
    type SendHandle: Wait;
    /// Handle returned by `irecv`.
    type RecvHandle: Wait;

    fn isend(&self, peer: usize, tag: u16, buf: &[u8]) -> Self::SendHandle;
    fn irecv(&self, peer: usize, tag: u16, buf: &mut [u8]) -> Self::RecvHandle;
}

/// Anything that can be waited on.
pub trait Wait {
    /// Wait for completion and return the received data (if any).
    fn wait(self) -> Option<Vec<u8>>;
}

/// Compile-time no-op comm for pure serial unit tests.
#[derive(Clone, Debug, Default)]
pub struct NoComm;

impl Wait for () {
    fn wait(self) -> Option<Vec<u8>> { None }
}

impl Communicator for NoComm {
    type SendHandle = ();
    type RecvHandle = ();

    fn isend(&self, _peer: usize, _tag: u16, _buf: &[u8]) -> () { () }
    fn irecv(&self, _peer: usize, _tag: u16, _buf: &mut [u8]) -> () { () }
}

// --- RayonComm: intra-process / multi-thread ---
type Key = (usize, usize, u16); // (src, dst, tag)

static MAILBOX: Lazy<DashMap<Key, Bytes>> = Lazy::new(DashMap::new);

pub struct LocalHandle {
    buf: Arc<Mutex<Option<Vec<u8>>>>,
    handle: Option<JoinHandle<()>>,
}

impl Wait for LocalHandle {
    fn wait(mut self) -> Option<Vec<u8>> {
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        let mut guard = self.buf.lock().unwrap();
        guard.take()
    }
}

#[derive(Clone, Debug)]
pub struct RayonComm {
    rank: usize,
}

impl RayonComm {
    pub fn new(rank: usize) -> Self { Self { rank } }
}

impl Communicator for RayonComm {
    type SendHandle = ();
    type RecvHandle = LocalHandle;

    fn isend(&self, peer: usize, tag: u16, buf: &[u8]) -> Self::SendHandle {
        let key = (self.rank, peer, tag);
        let data = buf.to_vec();
        MAILBOX.insert(key, Bytes::from(data));
    }

    fn irecv(&self, peer: usize, tag: u16, buf: &mut [u8]) -> Self::RecvHandle {
        let key = (peer, self.rank, tag);
        let buf_arc = Arc::new(Mutex::new(None));
        let buf_arc_clone = buf_arc.clone();
        let buf_len = buf.len();
        let handle = std::thread::spawn(move || {
            loop {
                if let Some(bytes) = MAILBOX.remove(&key).map(|(_, v)| v) {
                    let mut guard = buf_arc_clone.lock().unwrap();
                    *guard = Some(bytes[..buf_len].to_vec());
                    break;
                }
                std::thread::yield_now();
            }
        });
        LocalHandle { buf: buf_arc, handle: Some(handle) }
    }
}

// --- MPI backend (feature = "mpi") ---
#[cfg(feature = "mpi")]
mod mpi_backend {
    use super::*;
    use rsmpi::{traits::*, topology::SystemCommunicator};
    use std::sync::atomic::{AtomicU16, Ordering::Relaxed};

    static TAG_COUNTER: AtomicU16 = AtomicU16::new(42);

    #[derive(Clone)]
    pub struct MpiComm {
        pub world: SystemCommunicator,
        pub rank:  usize,
    }

    impl MpiComm {
        pub fn new() -> Self {
            let world = rsmpi::initialize().unwrap();
            let rank  = world.rank() as usize;
            Self { world, rank }
        }
        fn next_tag() -> i32 { TAG_COUNTER.fetch_add(1, Relaxed) as i32 }
    }

    pub struct MpiHandle(rsmpi::request::Request<'static>);

    impl Wait for MpiHandle {
        fn wait(self) { self.0.wait(); }
    }

    impl Communicator for MpiComm {
        type SendHandle = MpiHandle;
        type RecvHandle = MpiHandle;

        fn isend(&self, peer: usize, _tag: u16, buf: &[u8]) -> MpiHandle {
            let tag = Self::next_tag();
            let req = unsafe { self.world.process_at_rank(peer as i32).immediate_send(buf, tag) };
            MpiHandle(req)
        }

        fn irecv(&self, peer: usize, _tag: u16, buf: &mut [u8]) -> MpiHandle {
            let tag = Self::next_tag();
            let req = unsafe { self.world.process_at_rank(peer as i32).immediate_receive_into(buf, tag) };
            MpiHandle(req)
        }
    }
}

#[cfg(feature = "mpi")]
pub use mpi_backend::MpiComm;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rayon_roundtrip_two_ranks() {
        // Simulate rank 0 and rank 1 in the same process:
        let comm0 = RayonComm::new(0);
        let comm1 = RayonComm::new(1);

        // Prepare a 4-byte receive buffer on “rank 1”:
        let mut recv_buf = [0u8; 4];

        // On rank 1: post the receive for data from peer 0 with tag 7
        let recv_handle = comm1.irecv(0, 7, &mut recv_buf);

        // On rank 0: send the 4 bytes [1,2,3,4] to peer 1 with tag 7
        let send_handle = comm0.isend(1, 7, &[1,2,3,4]);

        // Wait for the send to “complete” (it's a no-op for RayonComm::SendHandle=()):
        send_handle.wait();

        // Wait for the receive to fire, grab the data, and copy it into recv_buf
        let data = recv_handle
            .wait()
            .expect("Expected to receive data from rank 0");
        recv_buf.copy_from_slice(&data);

        // Verify
        assert_eq!(&recv_buf, &[1,2,3,4]);
    }

    #[cfg(feature = "mpi")]
    #[test]
    fn mpi_roundtrip() {
        use rsmpi::traits::*;
        let comm = MpiComm::new();
        let size = comm.world.size() as usize;
        let nbr  = (comm.rank + 1) % size;
        let mut recv = [0u8; 1];
        let r = comm.irecv(nbr, 9, &mut recv);
        let s = comm.isend(nbr, 9, &[42]);
        s.wait(); r.wait();
        assert_eq!(recv[0], 42);
    }
}
