//! Communication abstraction for intra-process (Rayon) and inter-process (MPI) message passing.
//!
//! Wire format conventions (for higher-level protocols):
//! - All integers are LE fixed width (u32 counts/tags/ranks, u64 IDs).
//! - Structs are #[repr(C)] and bytemuck::Pod-safe; no #[repr(packed)].
//! - Receivers may truncate to their provided buffer length; higher layers must
//!   exchange sizes first if exact lengths are required.

use once_cell::sync::Lazy;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Condvar, Mutex};

/// Anything that can be waited on.
pub trait Wait {
    /// Wait for completion and return the received data (if any).
    fn wait(self) -> Option<Vec<u8>>;
}

/// Non-blocking completion test.
pub trait PollWait {
    /// Return `Some(bytes)` if the operation has completed, otherwise `None`.
    fn try_wait(&mut self) -> Option<Vec<u8>>;
}

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
    fn barrier(&self) {}
}

/// Tag newtype for safer tag arithmetic.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct CommTag(u16);

impl CommTag {
    /// Create a new tag from a raw `u16`.
    #[inline]
    pub const fn new(tag: u16) -> Self {
        Self(tag)
    }

    /// Return the underlying `u16` value.
    #[inline]
    pub const fn as_u16(self) -> u16 {
        self.0
    }

    /// Safely offset the tag by `dx`, wrapping on overflow.
    #[inline]
    pub const fn offset(self, dx: u16) -> Self {
        Self(self.0.wrapping_add(dx))
    }
}

impl From<u16> for CommTag {
    #[inline]
    fn from(x: u16) -> Self {
        CommTag::new(x)
    }
}

/// Convenience bundle of tags for the multi-phase section completion.
#[derive(Copy, Clone, Debug)]
pub struct SectionCommTags {
    /// Tag used during the size-exchange phase.
    pub sizes: CommTag,
    /// Tag used during the data-exchange phase.
    pub data: CommTag,
}

impl SectionCommTags {
    /// Construct tags from a base, assigning deterministic offsets per phase.
    #[inline]
    pub const fn from_base(base: CommTag) -> Self {
        Self {
            sizes: base,
            data: base.offset(1),
        }
    }
}

/// Convenience bundle of tags for sieve completion.
#[derive(Copy, Clone, Debug)]
pub struct SieveCommTags {
    /// Tag used during the size exchange phase.
    pub sizes: CommTag,
    /// Tag used during the data exchange phase.
    pub data: CommTag,
}

impl SieveCommTags {
    /// Construct tags from a base, assigning deterministic offsets per phase.
    #[inline]
    pub const fn from_base(base: CommTag) -> Self {
        Self {
            sizes: base,
            data: base.offset(1),
        }
    }
}

/// Convenience bundle of tags for stack completion.
#[derive(Copy, Clone, Debug)]
pub struct StackCommTags {
    /// Tag used during the size exchange phase.
    pub sizes: CommTag,
    /// Tag used during the data exchange phase.
    pub data: CommTag,
}

impl StackCommTags {
    /// Construct tags from a base, assigning deterministic offsets per phase.
    #[inline]
    pub const fn from_base(base: CommTag) -> Self {
        Self {
            sizes: base,
            data: base.offset(1),
        }
    }
}

/// Compile-time no-op comm for pure serial unit tests.
#[derive(Clone, Debug, Default)]
pub struct NoComm;

impl Wait for () {
    fn wait(self) -> Option<Vec<u8>> {
        None
    }
}

impl PollWait for () {
    fn try_wait(&mut self) -> Option<Vec<u8>> {
        None
    }
}

impl Communicator for NoComm {
    type SendHandle = ();
    type RecvHandle = ();

    fn isend(&self, _peer: usize, _tag: u16, _buf: &[u8]) {}

    fn irecv(&self, _peer: usize, _tag: u16, _buf: &mut [u8]) {}

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

#[derive(Default)]
struct Slot {
    q: VecDeque<Vec<u8>>,
}

struct Mailbox {
    map: Mutex<HashMap<Key, Arc<(Mutex<Slot>, Condvar)>>>,
}

static MAILBOX: Lazy<Mailbox> = Lazy::new(|| Mailbox {
    map: Mutex::new(HashMap::new()),
});

fn mailbox_entry(key: Key) -> Arc<(Mutex<Slot>, Condvar)> {
    let mut g = MAILBOX.map.lock().expect("MAILBOX poisoned");
    g.entry(key)
        .or_insert_with(|| Arc::new((Mutex::new(Slot::default()), Condvar::new())))
        .clone()
}

pub struct LocalSendHandle;

impl Wait for LocalSendHandle {
    fn wait(self) -> Option<Vec<u8>> {
        None
    }
}

impl PollWait for LocalSendHandle {
    fn try_wait(&mut self) -> Option<Vec<u8>> {
        None
    }
}

pub struct LocalRecvHandle {
    cell: Arc<(Mutex<Slot>, Condvar)>,
    want_len: usize,
}

impl Wait for LocalRecvHandle {
    fn wait(self) -> Option<Vec<u8>> {
        let (lock, cv) = &*self.cell;
        let mut slot = lock.lock().expect("Slot poisoned");
        while slot.q.is_empty() {
            slot = cv.wait(slot).expect("Condvar poisoned");
        }
        let mut msg = slot.q.pop_front().expect("q non-empty");
        msg.truncate(self.want_len.min(msg.len()));
        Some(msg)
    }
}

impl PollWait for LocalRecvHandle {
    fn try_wait(&mut self) -> Option<Vec<u8>> {
        let (lock, _cv) = &*self.cell;
        let mut slot = lock.lock().expect("Slot poisoned");
        if slot.q.is_empty() {
            None
        } else {
            let mut msg = slot.q.pop_front().expect("q non-empty");
            msg.truncate(self.want_len.min(msg.len()));
            Some(msg)
        }
    }
}

#[derive(Clone, Debug)]
pub struct RayonComm {
    rank: usize,
    size: usize,
}

impl RayonComm {
    pub fn new(rank: usize, size: usize) -> Self {
        Self { rank, size }
    }
}

impl Communicator for RayonComm {
    type SendHandle = LocalSendHandle;
    type RecvHandle = LocalRecvHandle;

    fn isend(&self, peer: usize, tag: u16, buf: &[u8]) -> Self::SendHandle {
        let key = (self.rank, peer, tag);
        let entry = mailbox_entry(key);
        let (lock, cv) = &*entry;
        {
            let mut slot = lock.lock().expect("Slot poisoned");
            slot.q.push_back(buf.to_vec());
        }
        cv.notify_all();
        LocalSendHandle
    }

    fn irecv(&self, peer: usize, tag: u16, buf: &mut [u8]) -> Self::RecvHandle {
        let key = (peer, self.rank, tag);
        LocalRecvHandle {
            cell: mailbox_entry(key),
            want_len: buf.len(),
        }
    }

    fn rank(&self) -> usize {
        self.rank
    }

    fn size(&self) -> usize {
        self.size
    }

    fn barrier(&self) {
        #[cfg(test)]
        {
            test_barrier::set_size(self.size);
            test_barrier::wait();
        }
    }
}

// Optional test barrier for deterministic multi-thread tests.
#[cfg(test)]
mod test_barrier {
    use once_cell::sync::Lazy;
    use std::sync::{Condvar, Mutex};

    pub struct EpochBarrier {
        size: usize,
        arrived: usize,
        epoch: usize,
    }

    static BARRIER: Lazy<(Mutex<EpochBarrier>, Condvar)> = Lazy::new(|| {
        (
            Mutex::new(EpochBarrier {
                size: 1,
                arrived: 0,
                epoch: 0,
            }),
            Condvar::new(),
        )
    });

    pub fn set_size(size: usize) {
        let (lock, _) = &*BARRIER;
        let mut b = lock.lock().unwrap();
        b.size = size;
    }

    pub fn wait() {
        let (lock, cv) = &*BARRIER;
        let mut b = lock.lock().unwrap();
        let e = b.epoch;
        b.arrived += 1;
        if b.arrived == b.size {
            b.arrived = 0;
            b.epoch += 1;
            cv.notify_all();
        } else {
            while e == b.epoch {
                b = cv.wait(b).unwrap();
            }
        }
    }
}

// --- MPI backend ---
#[cfg(feature = "mpi-support")]
mod mpi_backend {
    use super::*;
    use core::ptr::NonNull;
    use mpi::collective::CommunicatorCollectives;
    use mpi::environment::Universe;
    use mpi::point_to_point::{Destination, Source};
    use mpi::topology::{Communicator as _, SimpleCommunicator};

    pub struct MpiComm {
        _universe: Universe,
        pub world: SimpleCommunicator,
        rank: usize,
        size: usize,
    }

    unsafe impl Send for MpiComm {}
    unsafe impl Sync for MpiComm {}

    impl Default for MpiComm {
        fn default() -> Self {
            let uni = mpi::initialize().unwrap();
            let world = uni.world();
            let rank = world.rank() as usize;
            let size = world.size() as usize;
            Self {
                _universe: uni,
                world,
                rank,
                size,
            }
        }
    }

    impl Communicator for MpiComm {
        type SendHandle = MpiSendHandle;
        type RecvHandle = MpiRecvHandle;

        fn isend(&self, peer: usize, tag: u16, buf: &[u8]) -> Self::SendHandle {
            use mpi::request::StaticScope;
            let boxed = buf.to_vec().into_boxed_slice();
            let raw: *mut [u8] = Box::into_raw(boxed);
            let slice: &[u8] = unsafe { &*raw };
            let req = self
                .world
                .process_at_rank(peer as i32)
                .immediate_send_with_tag(StaticScope, slice, tag as i32);
            MpiSendHandle {
                req: Some(req),
                buf: Some(unsafe { NonNull::new_unchecked(raw) }),
            }
        }

        fn irecv(&self, peer: usize, tag: u16, template: &mut [u8]) -> Self::RecvHandle {
            use mpi::request::StaticScope;
            let len = template.len();
            let boxed = vec![0u8; len].into_boxed_slice();
            let raw: *mut [u8] = Box::into_raw(boxed);
            let slice_mut: &mut [u8] = unsafe { &mut *raw };
            let req = self
                .world
                .process_at_rank(peer as i32)
                .immediate_receive_into_with_tag(StaticScope, slice_mut, tag as i32);
            MpiRecvHandle {
                req: Some(req),
                buf: Some(unsafe { NonNull::new_unchecked(raw) }),
                len,
            }
        }

        fn rank(&self) -> usize {
            self.rank
        }
        fn size(&self) -> usize {
            self.size
        }
        fn barrier(&self) {
            self.world.barrier();
        }
    }

    pub struct MpiSendHandle {
        req: Option<mpi::request::Request<'static, [u8], mpi::request::StaticScope>>,
        buf: Option<NonNull<[u8]>>,
    }
    impl Wait for MpiSendHandle {
        fn wait(mut self) -> Option<Vec<u8>> {
            if let Some(r) = self.req.take() {
                let _ = r.wait();
            }
            if let Some(ptr) = self.buf.take() {
                unsafe {
                    drop(Box::from_raw(ptr.as_ptr()));
                }
            }
            None
        }
    }
    impl Drop for MpiSendHandle {
        fn drop(&mut self) {
            if let Some(r) = self.req.take() {
                let _ = r.test();
                #[cfg(debug_assertions)]
                eprintln!("[MpiSendHandle::drop] send not explicitly waited");
            }
            if let Some(ptr) = self.buf.take() {
                unsafe {
                    drop(Box::from_raw(ptr.as_ptr()));
                }
            }
        }
    }

    pub struct MpiRecvHandle {
        req: Option<mpi::request::Request<'static, [u8], mpi::request::StaticScope>>,
        buf: Option<NonNull<[u8]>>,
        len: usize,
    }
    impl Wait for MpiRecvHandle {
        fn wait(mut self) -> Option<Vec<u8>> {
            if let Some(r) = self.req.take() {
                let _ = r.wait();
            }
            let ptr = self.buf.take().expect("buffer missing");
            let boxed: Box<[u8]> = unsafe { Box::from_raw(ptr.as_ptr()) };
            let mut v = Vec::from(boxed);
            v.truncate(self.len);
            Some(v)
        }
    }
    impl Drop for MpiRecvHandle {
        fn drop(&mut self) {
            if let Some(r) = self.req.take() {
                let _ = r.test();
                #[cfg(debug_assertions)]
                eprintln!("[MpiRecvHandle::drop] recv not explicitly waited");
            }
            if let Some(ptr) = self.buf.take() {
                unsafe {
                    drop(Box::from_raw(ptr.as_ptr()));
                }
            }
        }
    }
}

#[cfg(feature = "mpi-support")]
pub use mpi_backend::MpiComm;
