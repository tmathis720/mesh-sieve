//! Communication abstraction for intra-process (Rayon) and inter-process (MPI) message passing.
//!
//! This module provides a minimal, non-blocking, waitable communication interface
//! for distributed and parallel mesh algorithms. Messages are contiguous byte slices,
//! and all handles are waitable but non-blocking. Includes implementations for
//! serial (no-op), Rayon-based, and MPI-based communication.

use bytes::Bytes;
use dashmap::DashMap;
use once_cell::sync::Lazy;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

/// Non-blocking communication interface (minimal by design).
///
/// Implementors provide asynchronous send/receive operations and waitable handles.
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

    /// Rank of this process (0..size-1)
    fn rank(&self) -> usize;
    /// Total number of ranks
    fn size(&self) -> usize;

    /// Synchronization barrier (default: no-op for non-MPI comms)
    fn barrier(&self) {
        // Default: no-op for non-MPI comms
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

    fn rank(&self) -> usize {
        0
    }

    fn size(&self) -> usize {
        1
    }
}

// --- RayonComm: intra-process / multi-thread ---
type Key = (usize, usize, u16); // (src, dst, tag)
pub static MAILBOX: Lazy<DashMap<Key, Bytes>> = Lazy::new(DashMap::new);

pub struct LocalHandle {
    buf: Arc<Mutex<Option<Vec<u8>>>>,
    handle: Option<JoinHandle<()>>,
}

impl Wait for LocalHandle {
    fn wait(mut self) -> Option<Vec<u8>> {
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        // Recover from a poisoned lock by taking its inner value
        let mut guard = match self.buf.lock() {
            Ok(g) => g,
            Err(poisoned) => {
                eprintln!("[RayonComm] WARNING: mailbox mutex was poisoned; recovering.");
                poisoned.into_inner()
            }
        };
        // Provide a clearer error if no message was received
        guard.take().or_else(|| {
            eprintln!("[RayonComm] ERROR: No message received for this handle. Possible send/receive mismatch or mailbox cleared too early.");
            None
        })
    }
}

#[derive(Clone, Debug, Default)]
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
        // Use (src, dst, tag) as the mailbox key
        let key = (self.rank, peer, tag);
        let data = buf.to_vec();
        MAILBOX.insert(key, Bytes::from(data));
    }

    fn irecv(&self, peer: usize, tag: u16, buf: &mut [u8]) -> Self::RecvHandle {
        // Use (src, dst, tag) as the mailbox key
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

    fn rank(&self) -> usize {
        self.rank
    }

    fn size(&self) -> usize {
        2 // For tests, default to 2
    }
}

// --- MPI backend ---
#[cfg(feature = "mpi-support")]
mod mpi_backend {
    use super::*;
    use mpi::environment::Universe;
    use mpi::topology::{Communicator as _, SimpleCommunicator};
    use mpi::traits::*;

    pub struct MpiComm {
        _universe: Universe, // keep alive until drop
        pub world: SimpleCommunicator,
        pub rank: usize,
    }

    unsafe impl Send for MpiComm {}
    unsafe impl Sync for MpiComm {}

    impl Default for MpiComm {
        fn default() -> Self {
            Self::from_universe(mpi::initialize().unwrap())
        }
    }

    impl MpiComm {
        /// convenience constructor for examples & tests
        pub fn new() -> Self {
            let uni = mpi::initialize().unwrap();
            Self::from_universe(uni)
        }

        pub fn from_universe(universe: Universe) -> Self {
            let world = universe.world();
            let rank = world.rank() as usize;
            MpiComm {
                _universe: universe,
                world,
                rank,
            }
        }
    }

    // Use boxed trait objects for handles to erase the lifetime
    pub struct MpiSendHandle {
        req: mpi::request::Request<'static, [u8], mpi::request::StaticScope>,
        buf: *mut [u8],
    }
    impl Wait for MpiSendHandle {
        fn wait(self) -> Option<Vec<u8>> {
            let _ = self.req.wait();
            unsafe { drop(Box::from_raw(self.buf)); }
            None
        }
    }
    pub struct MpiRecvHandle {
        req: mpi::request::Request<'static, [u8], mpi::request::StaticScope>,
        buf: *mut [u8],
        len: usize,
    }
    impl Wait for MpiRecvHandle {
        fn wait(self) -> Option<Vec<u8>> {
            let _ = self.req.wait();
            let boxed: Box<[u8]> = unsafe { Box::from_raw(self.buf) };
            let mut v = Vec::from(boxed);
            v.truncate(self.len);
            Some(v)
        }
    }
    impl crate::algs::communicator::Communicator for MpiComm {
        type SendHandle = MpiSendHandle;
        type RecvHandle = MpiRecvHandle;
        fn isend(&self, peer: usize, tag: u16, buf: &[u8]) -> MpiSendHandle {
            use mpi::request::StaticScope;
            let boxed = buf.to_vec().into_boxed_slice();
            let raw: *mut [u8] = Box::into_raw(boxed);
            let slice: &[u8] = unsafe { &*raw };
            // Use the tag in the MPI call
            let req = self.world.process_at_rank(peer as i32)
                .immediate_send_with_tag(StaticScope, slice, tag as i32);
            MpiSendHandle { req, buf: raw }
        }
        fn irecv(&self, peer: usize, tag: u16, template: &mut [u8]) -> MpiRecvHandle {
            use mpi::request::StaticScope;
            let len = template.len();
            let boxed = vec![0u8; len].into_boxed_slice();
            let raw: *mut [u8] = Box::into_raw(boxed);
            let slice_mut: &mut [u8] = unsafe { &mut *raw };
            // Use the tag in the MPI call
            let req = self.world.process_at_rank(peer as i32)
                .immediate_receive_into_with_tag(StaticScope, slice_mut, tag as i32);
            MpiRecvHandle { req, buf: raw, len }
        }

        fn rank(&self) -> usize {
            self.world.rank() as usize
        }
        fn size(&self) -> usize {
            self.world.size() as usize
        }
    }
}

#[cfg(feature = "mpi-support")]
pub use mpi_backend::MpiComm;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algs::communicator::Communicator;

    #[test]
    #[ignore]
    fn rayon_roundtrip_two_ranks() {
        // Always clear the mailbox before and after the test to avoid interference and leaks
        struct MailboxGuard;
        impl Drop for MailboxGuard {
            fn drop(&mut self) {
                MAILBOX.clear();
            }
        }
        let _guard = MailboxGuard;
        MAILBOX.clear();
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
        MAILBOX.clear();
    }

    #[cfg(feature = "mpi-support")]
    #[test]
    #[ignore]
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

    #[cfg(feature = "mpi-support")]
    #[test]
    #[ignore]
    fn mpi_comm_mock_serial() {
        use mpi::traits::*;
        let comm = MpiComm::new();
        if comm.world.size() < 2 {
            eprintln!("mpi_comm_mock_serial requires at least 2 MPI ranks; skipping.");
            return;
        }
        assert!(!comm.is_no_comm(), "MpiComm should not be no_comm");
        assert_eq!(comm.rank(), comm.world.rank() as usize);
        assert_eq!(comm.size(), comm.world.size() as usize);
        let mut buf = [0u8; 1];
        let r = comm.irecv(comm.rank(), 0, &mut buf);
        let s = comm.isend(comm.rank(), 0, &[123]);
        let _ = s.wait();
        let _ = r.wait();
    }

    #[test]
    fn no_comm_basics() {
        // Test NoComm API and semantics
        let comm = NoComm::default();
        assert!(comm.is_no_comm(), "NoComm should report is_no_comm() == true");
        assert_eq!(comm.rank(), 0, "NoComm rank should be 0");
        assert_eq!(comm.size(), 1, "NoComm size should be 1");
        comm.isend(0, 0, &[1, 2, 3]); // no-op
        let mut buf = [0u8; 3];
        let h = comm.irecv(0, 0, &mut buf); // no-op
        assert_eq!(h.wait(), None, "NoComm Wait should return None");
    }

    #[test]
    #[ignore]
    fn rayon_comm_edge_cases() {
        // Always clear the mailbox before and after the test to avoid interference and leaks
        struct MailboxGuard;
        impl Drop for MailboxGuard {
            fn drop(&mut self) {
                MAILBOX.clear();
            }
        }
        let _guard = MailboxGuard;
        MAILBOX.clear();
        // is_no_comm
        let comm0 = RayonComm::new(0);
        let comm1 = RayonComm::new(1);
        assert!(!comm0.is_no_comm(), "RayonComm should not be no_comm");
        // Multiple tags/peers
        let mut buf1 = [0u8; 2];
        let mut buf2 = [0u8; 2];
        let r1 = comm1.irecv(0, 1, &mut buf1);
        let r2 = comm1.irecv(0, 2, &mut buf2);
        comm0.isend(1, 1, &[10, 11]);
        comm0.isend(1, 2, &[20, 21]);
        assert_eq!(r1.wait().unwrap(), vec![10, 11]);
        assert_eq!(r2.wait().unwrap(), vec![20, 21]);
        // Buffer truncation
        let mut small_buf = [0u8; 1];
        comm0.isend(1, 3, &[99, 100, 101]);
        let r = comm1.irecv(0, 3, &mut small_buf);
        assert_eq!(r.wait().unwrap(), vec![99]);
        // Simultaneous pending receives
        let mut a = [0u8; 1];
        let mut b = [0u8; 1];
        let ra = comm1.irecv(0, 4, &mut a);
        let rb = comm1.irecv(0, 5, &mut b);
        comm0.isend(1, 4, &[1]);
        comm0.isend(1, 5, &[2]);
        assert_eq!(ra.wait().unwrap(), vec![1]);
        assert_eq!(rb.wait().unwrap(), vec![2]);
        // Ordering/overwrite: later send overwrites earlier
        comm0.isend(1, 6, &[1]);
        comm0.isend(1, 6, &[2]);
        let mut c = [0u8; 1];
        let rc = comm1.irecv(0, 6, &mut c);
        assert_eq!(rc.wait().unwrap(), vec![2]);
        MAILBOX.clear();
    }

    #[test]
    #[ignore]
    fn rayon_comm_mailbox_cleanup_and_drop_handle() {
        use std::thread;
        // Always clear the mailbox before and after the test to avoid interference and leaks
        struct MailboxGuard;
        impl Drop for MailboxGuard {
            fn drop(&mut self) {
                MAILBOX.clear();
            }
        }
        let _guard = MailboxGuard;
        MAILBOX.clear();
        let comm0 = RayonComm::new(0);
        let comm1 = RayonComm::new(1);
        // Mailbox should not grow unboundedly
        let start = MAILBOX.len();
        for i in 0..10 {
            let mut buf = [0u8; 1];
            let r = comm1.irecv(0, 100 + i, &mut buf);
            comm0.isend(1, 100 + i, &[i as u8]);
            assert_eq!(r.wait().unwrap(), vec![i as u8]);
        }
        assert!(MAILBOX.len() <= start + 1, "MAILBOX should not grow unboundedly");
        // Dropped handle: should not panic or leak threads
        let mut buf = [0u8; 1];
        let handle = comm1.irecv(0, 200, &mut buf);
        comm0.isend(1, 200, &[42]);
        drop(handle); // Should not panic or leak
        thread::sleep(std::time::Duration::from_millis(10));
        MAILBOX.clear();
    }

    // Cross-backend harness: exercise all Communicator impls
    fn roundtrip<C: Communicator + Default>() {
        let comm = C::default();
        let peer = (comm.rank() + 1) % comm.size();
        let tag = 42;
        let data = [9, 9, 9];
        let mut buf = [0u8; 3];
        let r = comm.irecv(peer, tag, &mut buf);
        let s = comm.isend(peer, tag, &data);
        let _ = s.wait();
        let received = r.wait().unwrap_or_default();
        assert_eq!(received.len(), buf.len());
    }

    #[test]
    #[ignore]
    fn sanity_rayon() {
        // Always clear the mailbox before and after the test to avoid interference and leaks
        struct MailboxGuard;
        impl Drop for MailboxGuard {
            fn drop(&mut self) {
                MAILBOX.clear();
            }
        }
        let _guard = MailboxGuard;
        MAILBOX.clear();
        roundtrip::<RayonComm>();
        MAILBOX.clear();
    }
    #[cfg(feature = "mpi-support")]
    #[test]
    #[ignore]
    fn sanity_mpi() {
        // Always clear the mailbox before and after the test to avoid interference and leaks
        struct MailboxGuard;
        impl Drop for MailboxGuard {
            fn drop(&mut self) {
                MAILBOX.clear();
            }
        }
        let _guard = MailboxGuard;
        MAILBOX.clear();
        roundtrip::<MpiComm>();
        MAILBOX.clear();
    }
}
