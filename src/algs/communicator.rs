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

    /// Broadcast a byte buffer from `root` to all ranks.
    fn broadcast(&self, root: usize, buf: &mut [u8]) {
        if self.size() <= 1 {
            return;
        }
        if self.rank() == root {
            let mut sends = Vec::with_capacity(self.size().saturating_sub(1));
            for peer in 0..self.size() {
                if peer != root {
                    sends.push(self.isend(peer, COLLECTIVE_TAG_BROADCAST, buf));
                }
            }
            for send in sends {
                let _ = send.wait();
            }
        } else {
            let mut tmp = vec![0u8; buf.len()];
            let recv = self.irecv(root, COLLECTIVE_TAG_BROADCAST, &mut tmp);
            if let Some(data) = recv.wait() {
                let copy_len = buf.len().min(data.len());
                buf[..copy_len].copy_from_slice(&data[..copy_len]);
            }
        }
    }

    /// All-reduce sum for `u64` buffers.
    fn allreduce_sum(&self, values: &mut [u64]) {
        let size = self.size();
        if size <= 1 {
            return;
        }
        let root = 0;
        let encoded = encode_u64_le(values);
        if self.rank() == root {
            let mut accum = values.to_vec();
            let mut recvs = Vec::with_capacity(size.saturating_sub(1));
            for peer in 0..size {
                if peer != root {
                    let mut tmp = vec![0u8; encoded.len()];
                    let handle = self.irecv(peer, COLLECTIVE_TAG_ALLREDUCE_GATHER, &mut tmp);
                    recvs.push(handle);
                }
            }
            for recv in recvs {
                if let Some(data) = recv.wait() {
                    add_u64_le(&data, &mut accum);
                }
            }
            values.copy_from_slice(&accum);
            let out_bytes = encode_u64_le(values);
            let mut sends = Vec::with_capacity(size.saturating_sub(1));
            for peer in 0..size {
                if peer != root {
                    sends.push(self.isend(peer, COLLECTIVE_TAG_ALLREDUCE_BROADCAST, &out_bytes));
                }
            }
            for send in sends {
                let _ = send.wait();
            }
        } else {
            let send = self.isend(root, COLLECTIVE_TAG_ALLREDUCE_GATHER, &encoded);
            let mut tmp = vec![0u8; encoded.len()];
            let recv = self.irecv(root, COLLECTIVE_TAG_ALLREDUCE_BROADCAST, &mut tmp);
            let _ = send.wait();
            if let Some(data) = recv.wait() {
                decode_u64_le(&data, values);
            }
        }
    }

    /// All-gather fixed-size byte buffers into `recvbuf` (rank-major order).
    fn allgather(&self, sendbuf: &[u8], recvbuf: &mut [u8]) {
        let size = self.size();
        let chunk = sendbuf.len();
        if size == 0 || chunk == 0 {
            return;
        }
        assert_eq!(
            recvbuf.len(),
            size * chunk,
            "recvbuf must be size * sendbuf.len()"
        );
        let rank = self.rank();
        let start = rank * chunk;
        recvbuf[start..start + chunk].copy_from_slice(sendbuf);
        if size <= 1 {
            return;
        }
        let mut sends = Vec::with_capacity(size.saturating_sub(1));
        let mut recvs = Vec::with_capacity(size.saturating_sub(1));
        for peer in 0..size {
            if peer == rank {
                continue;
            }
            sends.push(self.isend(peer, COLLECTIVE_TAG_ALLGATHER, sendbuf));
            let mut tmp = vec![0u8; chunk];
            let recv = self.irecv(peer, COLLECTIVE_TAG_ALLGATHER, &mut tmp);
            recvs.push((peer, recv));
        }
        for send in sends {
            let _ = send.wait();
        }
        for (peer, recv) in recvs {
            if let Some(data) = recv.wait() {
                let offset = peer * chunk;
                assert_eq!(
                    data.len(),
                    chunk,
                    "allgather received unexpected buffer length"
                );
                recvbuf[offset..offset + chunk].copy_from_slice(&data);
            }
        }
    }
}

const COLLECTIVE_TAG_BROADCAST: u16 = u16::MAX - 1;
const COLLECTIVE_TAG_ALLGATHER: u16 = u16::MAX - 2;
const COLLECTIVE_TAG_ALLREDUCE_GATHER: u16 = u16::MAX - 3;
const COLLECTIVE_TAG_ALLREDUCE_BROADCAST: u16 = u16::MAX - 4;

fn encode_u64_le(values: &[u64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * core::mem::size_of::<u64>());
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

fn decode_u64_le(bytes: &[u8], out: &mut [u64]) {
    debug_assert_eq!(bytes.len(), out.len() * core::mem::size_of::<u64>());
    for (chunk, slot) in bytes
        .chunks_exact(core::mem::size_of::<u64>())
        .zip(out.iter_mut())
    {
        let mut raw = [0u8; 8];
        raw.copy_from_slice(chunk);
        *slot = u64::from_le_bytes(raw);
    }
}

fn add_u64_le(bytes: &[u8], accum: &mut [u64]) {
    debug_assert_eq!(bytes.len(), accum.len() * core::mem::size_of::<u64>());
    for (chunk, slot) in bytes
        .chunks_exact(core::mem::size_of::<u64>())
        .zip(accum.iter_mut())
    {
        let mut raw = [0u8; 8];
        raw.copy_from_slice(chunk);
        *slot += u64::from_le_bytes(raw);
    }
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
    use crate::mesh_error::{CommError, MeshSieveError};
    use core::ptr::NonNull;
    use mpi::collective::{CommunicatorCollectives, Root, SystemOperation};
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

    impl MpiComm {
        pub fn new() -> Result<Self, MeshSieveError> {
            let uni = mpi::initialize().ok_or_else(|| {
                MeshSieveError::Communication(CommError("MPI initialization failed".to_string()))
            })?;
            let world = uni.world();
            let rank = world.rank() as usize;
            let size = world.size() as usize;
            Ok(Self {
                _universe: uni,
                world,
                rank,
                size,
            })
        }
    }

    impl Default for MpiComm {
        fn default() -> Self {
            Self::new().expect("MPI initialization failed")
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

        fn broadcast(&self, root: usize, buf: &mut [u8]) {
            self.world.process_at_rank(root as i32).broadcast_into(buf);
        }

        fn allreduce_sum(&self, values: &mut [u64]) {
            let mut out = vec![0u64; values.len()];
            self.world
                .all_reduce_into(values, &mut out, SystemOperation::sum());
            values.copy_from_slice(&out);
        }

        fn allgather(&self, sendbuf: &[u8], recvbuf: &mut [u8]) {
            self.world.all_gather_into(sendbuf, recvbuf);
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
                let _ = r.wait();
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
                let _ = r.wait();
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
