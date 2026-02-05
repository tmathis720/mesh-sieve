//! Value delta: rules for fusing data across overlaps
//!
//! This module defines the [`ValueDelta`] trait and several implementations for
//! restricting and fusing values across overlaps in distributed mesh computations.
//! It is distinct from [`crate::data::refine::delta::SliceDelta`], which handles
//! slice-to-slice transformations during mesh refinement.

use crate::topology::cell_type::CellType;

/// *Delta* encapsulates restriction & fusion for a section value `V`.
///
/// This trait defines how to extract a part of a value for communication
/// and how to merge an incoming fragment into the local value.
pub trait ValueDelta<V>: Sized {
    /// What a *restricted* value looks like (often identical to `V`).
    type Part: Send;

    /// Extract the part of `v` that travels on one arrow.
    fn restrict(v: &V) -> Self::Part;

    /// Merge an incoming fragment into the local value.
    fn fuse(local: &mut V, incoming: Self::Part);
}

/// Identity delta for cloneable values (copy-overwrites-local).
///
/// This implementation simply clones the value for restriction and overwrites
/// the local value on fusion.
#[derive(Copy, Clone)]
pub struct CopyDelta;

impl<V: Clone + Send> ValueDelta<V> for CopyDelta {
    type Part = V;
    #[inline]
    fn restrict(v: &V) -> V {
        v.clone()
    }
    #[inline]
    fn fuse(local: &mut V, incoming: V) {
        *local = incoming;
    }
}

/// No-op delta: skips fusion, always returns default for restrict.
///
/// This implementation always returns the default value for restriction and
/// does nothing on fusion.
#[derive(Copy, Clone)]
pub struct ZeroDelta;

impl<V: Clone + Default + Send> ValueDelta<V> for ZeroDelta {
    type Part = V;
    #[inline]
    fn restrict(_v: &V) -> V {
        V::default()
    }
    #[inline]
    fn fuse(_local: &mut V, _incoming: V) {
        // do nothing
    }
}

/// Additive delta for summation/balancing fields.
///
/// This implementation restricts by copying and fuses by addition.
#[derive(Copy, Clone)]
pub struct AddDelta;

impl<V> ValueDelta<V> for AddDelta
where
    V: std::ops::AddAssign + Copy + Send,
{
    type Part = V;
    #[inline]
    fn restrict(v: &V) -> V {
        *v
    }
    #[inline]
    fn fuse(local: &mut V, incoming: V) {
        *local += incoming;
    }
}

/// Encode/decode delta for `CellType`, using a compact pod-safe integer.
#[derive(Copy, Clone)]
pub struct CellTypeDelta;

impl ValueDelta<CellType> for CellTypeDelta {
    type Part = i32;

    #[inline]
    fn restrict(v: &CellType) -> Self::Part {
        cell_type_to_code(*v)
    }

    #[inline]
    fn fuse(local: &mut CellType, incoming: Self::Part) {
        *local = code_to_cell_type(incoming);
    }
}

const CELL_TAG_VERTEX: u32 = 1;
const CELL_TAG_SEGMENT: u32 = 2;
const CELL_TAG_TRIANGLE: u32 = 3;
const CELL_TAG_QUAD: u32 = 4;
const CELL_TAG_TET: u32 = 5;
const CELL_TAG_HEX: u32 = 6;
const CELL_TAG_PRISM: u32 = 7;
const CELL_TAG_PYRAMID: u32 = 8;
const CELL_TAG_POLYGON: u32 = 9;
const CELL_TAG_SIMPLEX: u32 = 10;
const CELL_TAG_POLYHEDRON: u32 = 11;

#[inline]
fn pack_cell_code(tag: u32, payload: u8) -> i32 {
    ((tag << 8) | (payload as u32)) as i32
}

#[inline]
fn cell_type_to_code(cell_type: CellType) -> i32 {
    match cell_type {
        CellType::Vertex => pack_cell_code(CELL_TAG_VERTEX, 0),
        CellType::Segment => pack_cell_code(CELL_TAG_SEGMENT, 0),
        CellType::Triangle => pack_cell_code(CELL_TAG_TRIANGLE, 0),
        CellType::Quadrilateral => pack_cell_code(CELL_TAG_QUAD, 0),
        CellType::Tetrahedron => pack_cell_code(CELL_TAG_TET, 0),
        CellType::Hexahedron => pack_cell_code(CELL_TAG_HEX, 0),
        CellType::Prism => pack_cell_code(CELL_TAG_PRISM, 0),
        CellType::Pyramid => pack_cell_code(CELL_TAG_PYRAMID, 0),
        CellType::Polygon(sides) => pack_cell_code(CELL_TAG_POLYGON, sides),
        CellType::Simplex(dim) => pack_cell_code(CELL_TAG_SIMPLEX, dim),
        CellType::Polyhedron => pack_cell_code(CELL_TAG_POLYHEDRON, 0),
    }
}

#[inline]
fn code_to_cell_type(code: i32) -> CellType {
    let raw = code as u32;
    let tag = raw >> 8;
    let payload = (raw & 0xFF) as u8;
    match tag {
        CELL_TAG_VERTEX => CellType::Vertex,
        CELL_TAG_SEGMENT => CellType::Segment,
        CELL_TAG_TRIANGLE => CellType::Triangle,
        CELL_TAG_QUAD => CellType::Quadrilateral,
        CELL_TAG_TET => CellType::Tetrahedron,
        CELL_TAG_HEX => CellType::Hexahedron,
        CELL_TAG_PRISM => CellType::Prism,
        CELL_TAG_PYRAMID => CellType::Pyramid,
        CELL_TAG_POLYGON => CellType::Polygon(payload),
        CELL_TAG_SIMPLEX => CellType::Simplex(payload),
        CELL_TAG_POLYHEDRON => CellType::Polyhedron,
        _ => CellType::Vertex,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_copy_delta_roundtrip() {
        let mut v = 5;
        let part = CopyDelta::restrict(&v);
        CopyDelta::fuse(&mut v, part);
        assert_eq!(v, 5);
    }

    #[test]
    fn test_add_delta_forward_and_fuse() {
        let mut v = 2;
        let part = AddDelta::restrict(&3);
        AddDelta::fuse(&mut v, part);
        assert_eq!(v, 5);
    }

    #[test]
    fn test_zero_delta_noop() {
        let mut v = 7;
        let part = ZeroDelta::restrict(&v);
        assert_eq!(part, 0);
        ZeroDelta::fuse(&mut v, 42);
        assert_eq!(v, 7);
    }

    #[test]
    fn zero_delta_handles_non_numeric() {
        let mut v = String::from("hello");
        let part = ZeroDelta::restrict(&v);
        assert!(part.is_empty());
        ZeroDelta::fuse(&mut v, String::from("world"));
        assert_eq!(v, "hello");
    }
}

/// Backward-compatible alias for [`ValueDelta`].
#[deprecated(note = "Renamed to ValueDelta; Delta will be removed in a future release")]
pub use ValueDelta as Delta;
