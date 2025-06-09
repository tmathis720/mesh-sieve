//! Delta trait: rules for fusing data across overlaps

/// *Delta* encapsulates restriction & fusion for a section value `V`.
pub trait Delta<V>: Sized {
    /// What a *restricted* value looks like (often identical to `V`).
    type Part: Send;

    /// Extract the part of `v` that travels on one arrow.
    fn restrict(v: &V) -> Self::Part;

    /// Merge an incoming fragment into the local value.
    fn fuse(local: &mut V, incoming: Self::Part);
}

/// Identity delta for cloneable values (copy-overwrites-local).
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

/// Additive delta for summation/balancing fields.
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
