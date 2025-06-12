//! Delta trait: rules for fusing data across overlaps
//!
//! This module defines the [`Delta`] trait and several implementations for
//! restricting and fusing values across overlaps in distributed mesh computations.

/// *Delta* encapsulates restriction & fusion for a section value `V`.
///
/// This trait defines how to extract a part of a value for communication
/// and how to merge an incoming fragment into the local value.
pub trait Delta<V>: Sized {
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

impl<V: Clone + Send> Delta<V> for CopyDelta {
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

impl<V: Clone + Default + Send> Delta<V> for ZeroDelta {
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

impl<V> Delta<V> for AddDelta
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
}
