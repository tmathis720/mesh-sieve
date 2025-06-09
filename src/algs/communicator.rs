//! Thin façade over intra-process (Rayon) or inter-process (MPI) message passing.
//!
//! Messages are *contiguous byte slices* (no zero-copy guarantees).
//! All handles are **waitable** but non-blocking -– completion.rs calls
//! `.wait()` before it trusts that the buffer is ready.

use bytes::Bytes;
use dashmap::DashMap;
use once_cell::sync::Lazy;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

/// Non-blocking communication interface (minimal by design).
pub trait Communicator: Send + Sync + 'static {
    /// Handle returned by `isend`.
    type SendHandle: Wait;
    /// Handle returned by `irecv`.
    type RecvHandle: Wait;

    fn isend(&self, peer: usize, tag: u16, buf: &[u8]) -> Self::SendHandle;
    fn irecv(&self, peer: usize, tag: u16, buf: &mut [u8]) -> Self::RecvHandle;

    /// Returns true if this communicator is NoComm (for test logic)
    fn is_no_comm(&self) -> bool {
        false
    }
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
    fn wait(self) -> Option<Vec<u8>> {
        None
    }
}

impl Communicator for NoComm {
    type SendHandle = ();
    type RecvHandle = ();

    fn isend(&self, _peer: usize, _tag: u16, _buf: &[u8]) {
        // no-op
    }
    fn irecv(&self, _peer: usize, _tag: u16, _buf: &mut [u8]) {
        // no-op
    }
    fn is_no_comm(&self) -> bool {
        true
    }
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
    pub fn new(rank: usize) -> Self {
        Self { rank }
    }
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
        LocalHandle {
            buf: buf_arc,
            handle: Some(handle),
        }
    }
}

// --- MPI backend ---
mod mpi_backend {
    use super::*;
    use mpi::environment::Universe;
    use mpi::request::Request;
    use mpi::request::StaticScope;
    use mpi::topology::{Communicator as _, SimpleCommunicator};
    use mpi::traits::*;

    pub struct MpiComm {
        _universe: Universe, // keep alive until drop
        pub world: SimpleCommunicator,
        pub rank: usize,
    }

    impl Default for MpiComm {
        fn default() -> Self {
            Self::new()
        }
    }

    impl MpiComm {
        pub fn new() -> Self {
            let universe = mpi::initialize().unwrap();
            let world = universe.world();
            let rank = world.rank() as usize;
            MpiComm {
                _universe: universe,
                world,
                rank,
            }
        }
    }

    pub struct MpiHandle {
        req: Request<'static, [u8], StaticScope>,
        buf: *mut [u8],
    }

    impl Wait for MpiHandle {
        fn wait(self) -> Option<Vec<u8>> {
            let _ = self.req.wait();
            // SAFETY: We own the leaked buffer, so it's safe to reconstruct and take ownership
            let buf = unsafe { Box::from_raw(self.buf) };
            Some(buf.to_vec())
        }
    }

    impl crate::algs::communicator::Communicator for MpiComm {
        type SendHandle = ();
        type RecvHandle = MpiHandle;

        fn isend(&self, peer: usize, _tag: u16, buf: &[u8]) {
            self.world.process_at_rank(peer as i32).send(buf);
        }

        fn irecv(&self, peer: usize, _tag: u16, buf: &mut [u8]) -> MpiHandle {
            let len = buf.len();
            let mut v = vec![0u8; len];
            let static_buf: &'static mut [u8] = Box::leak(v.into_boxed_slice());
            let buf_ptr = static_buf as *mut [u8];
            let req = self
                .world
                .process_at_rank(peer as i32)
                .immediate_receive_into(StaticScope, unsafe { &mut *buf_ptr });
            MpiHandle { req, buf: buf_ptr }
        }
    }
}

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
        let send_handle = comm0.isend(1, 7, &[1, 2, 3, 4]);

        // Wait for the send to “complete” (it's a no-op for RayonComm::SendHandle=()):
        send_handle.wait();

        // Wait for the receive to fire, grab the data, and copy it into recv_buf
        let data = recv_handle
            .wait()
            .expect("Expected to receive data from rank 0");
        recv_buf.copy_from_slice(&data);

        // Verify
        assert_eq!(&recv_buf, &[1, 2, 3, 4]);
    }

    #[test]
    fn mpi_roundtrip() {
        use mpi::traits::*;
        let comm = MpiComm::new();
        let size = comm.world.size() as usize;
        if size < 2 {
            eprintln!("mpi_roundtrip test requires at least 2 MPI ranks; skipping.");
            return;
        }
        let nbr_send = (comm.rank + 1) % size;
        let nbr_recv = (comm.rank + size - 1) % size;
        let mut recv = [0u8; 1];
        let r = comm.irecv(nbr_recv, 9, &mut recv);
        let s = comm.isend(nbr_send, 9, &[comm.rank as u8]);
        s.wait();
        r.wait();
        assert_eq!(
            recv[0], nbr_recv as u8,
            "Rank {} expected to receive {} from {} but got {}",
            comm.rank, nbr_recv as u8, nbr_recv, recv[0]
        );
    }
}
