//! Fixed, versioned, little-endian wire types for completion paths.

use bytemuck::{Pod, Zeroable};
use std::mem::{align_of, size_of};

pub fn cast_slice<T: Pod>(v: &[T]) -> &[u8] {
    bytemuck::cast_slice(v)
}

pub fn cast_slice_mut<T: Pod>(v: &mut [T]) -> &mut [u8] {
    bytemuck::cast_slice_mut(v)
}

pub fn cast_slice_from<T: Pod>(v: &[u8]) -> &[T] {
    bytemuck::cast_slice(v)
}

pub fn cast_slice_from_mut<T: Pod>(v: &mut [u8]) -> &mut [T] {
    bytemuck::cast_slice_mut(v)
}

pub fn expect_exact_len(actual: usize, expected: usize) -> Result<(), String> {
    if actual == expected {
        Ok(())
    } else {
        Err(format!("expected {expected} bytes, got {actual}"))
    }
}

#[repr(transparent)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct WireCountU32(pub u32);

pub trait WirePoint: Copy {
    fn to_wire(self) -> u64;
    fn from_wire(w: u64) -> Self;
}

/// Bump when the layout or semantics change in incompatible ways.
pub const WIRE_VERSION: u16 = 1;

/// All multi-byte integers in these structs are **little-endian** on the wire.
/// We store them pre-LE with `.to_le()` and decode with `.from_le()`.

// ===== Common records ======================================================

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct WireHdr {
    pub version_le: u16,  // = WIRE_VERSION.to_le()
    pub kind_le: u16,     // 1 = Cone, 2 = Support, etc.
    pub reserved_le: u32, // future use; keep zero
}

impl WireHdr {
    pub fn new(kind: u16) -> Self {
        Self {
            version_le: WIRE_VERSION.to_le(),
            kind_le: kind.to_le(),
            reserved_le: 0,
        }
    }
    pub fn kind(&self) -> u16 {
        u16::from_le(self.kind_le)
    }
    pub fn version(&self) -> u16 {
        u16::from_le(self.version_le)
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct WireCount {
    pub n_le: u32, // count of following records
}
impl WireCount {
    pub fn new(n: usize) -> Self {
        Self {
            n_le: (n as u32).to_le(),
        }
    }
    pub fn get(&self) -> usize {
        u32::from_le(self.n_le) as usize
    }
}

/// A point id (u64) carried on the wire.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct WirePointRepr {
    pub id_le: u64,
}
impl WirePointRepr {
    pub fn of(id: u64) -> Self {
        Self { id_le: id.to_le() }
    }
    pub fn get(&self) -> u64 {
        u64::from_le(self.id_le)
    }
}

/// An adjacency pair (src, dst) — used in closure/support replies.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct WireAdj {
    pub src_le: u64,
    pub dst_le: u64,
}
impl WireAdj {
    pub fn new(src: u64, dst: u64) -> Self {
        Self {
            src_le: src.to_le(),
            dst_le: dst.to_le(),
        }
    }
    pub fn src(&self) -> u64 {
        u64::from_le(self.src_le)
    }
    pub fn dst(&self) -> u64 {
        u64::from_le(self.dst_le)
    }
}

// ===== Sieve completion ====================================================

/// Minimal arrow payload `(src, dst)` in receiver-local IDs.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct WireArrow {
    pub src_le: u64,
    pub dst_le: u64,
}
impl WireArrow {
    pub fn new(src: u64, dst: u64) -> Self {
        Self {
            src_le: src.to_le(),
            dst_le: dst.to_le(),
        }
    }
    pub fn src(&self) -> u64 {
        u64::from_le(self.src_le)
    }
    pub fn dst(&self) -> u64 {
        u64::from_le(self.dst_le)
    }
}

/// Legacy arrow triple used by sieve completion.
/// NOTE: `rank_le` is u32 (never usize) on the wire.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct WireArrowTriple {
    pub src_le: u64,
    pub dst_le: u64,
    pub remote_point_le: u64,
    pub rank_le: u32, // remote rank
    pub _pad: u32,    // pad to 8-byte alignment (explicit)
}
impl WireArrowTriple {
    pub const SIZE: usize = 32; // 3*8 + 4 + 4
    pub fn new(src: u64, dst: u64, remote: u64, rank: u32) -> Self {
        Self {
            src_le: src.to_le(),
            dst_le: dst.to_le(),
            remote_point_le: remote.to_le(),
            rank_le: rank.to_le(),
            _pad: 0,
        }
    }
    pub fn decode(&self) -> (u64, u64, u64, u32) {
        (
            u64::from_le(self.src_le),
            u64::from_le(self.dst_le),
            u64::from_le(self.remote_point_le),
            u32::from_le(self.rank_le),
        )
    }
}

// ===== Stack completion (base, cap, payload) ==============================

/// If your payload is "opaque bytes", model it as a fixed-size byte array.
/// If it’s numeric, define a dedicated, fixed-width structure for it.
pub const WIRE_PAYLOAD_MAX: usize = 16; // example cap; set to your actual need

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct WireStackTriple {
    pub base_le: u64,
    pub cap_le: u64,
    /// Opaque payload bytes. Producer and consumer must agree on its meaning.
    pub pay: [u8; WIRE_PAYLOAD_MAX],
}
impl WireStackTriple {
    pub fn new(base: u64, cap: u64, pay: &[u8]) -> Self {
        let mut buf = [0u8; WIRE_PAYLOAD_MAX];
        let n = pay.len().min(WIRE_PAYLOAD_MAX);
        buf[..n].copy_from_slice(&pay[..n]);
        Self {
            base_le: base.to_le(),
            cap_le: cap.to_le(),
            pay: buf,
        }
    }
    pub fn base(&self) -> u64 {
        u64::from_le(self.base_le)
    }
    pub fn cap(&self) -> u64 {
        u64::from_le(self.cap_le)
    }
}

// ===== Compile-time sanity checks =========================================

const _: () = {
    // Pod/Zeroable ensures no padding contains uninit when cast to bytes.
    assert!(size_of::<WireHdr>() == 8);
    assert!(size_of::<WireCount>() == 4);
    assert!(size_of::<WirePointRepr>() == 8);
    assert!(size_of::<WireAdj>() == 16);
    assert!(size_of::<WireArrow>() == 16);
    assert!(size_of::<WireArrowTriple>() == WireArrowTriple::SIZE);
    assert!(align_of::<WireArrowTriple>() == 8);
};

impl WirePoint for crate::topology::point::PointId {
    #[inline]
    fn to_wire(self) -> u64 {
        self.get()
    }
    #[inline]
    fn from_wire(w: u64) -> Self {
        crate::topology::point::PointId::new(w).expect("invalid PointId on wire")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::{cast_slice, cast_slice_mut};

    #[test]
    fn roundtrip_adj() {
        let v = vec![WireAdj::new(1, 2), WireAdj::new(3, 4)];
        let bytes: Vec<u8> = cast_slice(&v).to_vec();
        let mut out = vec![WireAdj::zeroed(); v.len()];
        cast_slice_mut(&mut out).copy_from_slice(&bytes);
        assert_eq!(out[0].src(), 1);
        assert_eq!(out[1].dst(), 4);
    }

    #[test]
    fn roundtrip_wire_arrow() {
        let v = vec![WireArrow::new(10, 20), WireArrow::new(30, 40)];
        let bytes: Vec<u8> = cast_slice(&v).to_vec();
        let mut out = vec![WireArrow::zeroed(); v.len()];
        cast_slice_mut(&mut out).copy_from_slice(&bytes);
        assert_eq!(out[0].src(), 10);
        assert_eq!(out[1].dst(), 40);
    }

    #[test]
    fn roundtrip_arrow() {
        let t = WireArrowTriple::new(1, 2, 3, 4);
        let bytes: Vec<u8> = cast_slice(&[t]).to_vec();
        let mut out = vec![WireArrowTriple::zeroed(); 1];
        cast_slice_mut(&mut out).copy_from_slice(&bytes);
        assert_eq!(out[0].decode(), (1, 2, 3, 4));
        assert_eq!(
            WireArrowTriple::SIZE,
            std::mem::size_of::<WireArrowTriple>()
        );
    }

    #[test]
    fn roundtrip_stack() {
        let pay = [1u8, 2, 3, 4];
        let t = WireStackTriple::new(10, 20, &pay);
        let bytes: Vec<u8> = cast_slice(&[t]).to_vec();
        let mut out = vec![WireStackTriple::zeroed(); 1];
        cast_slice_mut(&mut out).copy_from_slice(&bytes);
        assert_eq!(out[0].base(), 10);
        assert_eq!(out[0].cap(), 20);
        assert_eq!(&out[0].pay[..4], &pay);
    }

    #[test]
    fn version_guard() {
        let hdr = WireHdr::new(1);
        assert_eq!(hdr.version(), WIRE_VERSION);
    }
}
