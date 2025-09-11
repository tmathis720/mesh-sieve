//! Pluggable storage for Section buffers.
//!
//! This trait abstracts how Section's flat buffer is stored (e.g., Vec, mmap, GPU).
//! The initial design keeps CPU-slice semantics to minimize churn. Future steps
//! can add async/mapped variants without touching Section's public API.

use core::fmt::{self, Debug};

use crate::mesh_error::MeshSieveError;

/// Contiguous, indexable storage for `V` with slice access.
///
/// Notes:
/// - Returning slices keeps this step localized. A GPU/mmap backend can be
///   implemented later by staging into host memory (or by extending the trait).
pub trait Storage<V>: Debug {
    /// Construct a buffer of `len`, filled with `fill`.
    fn with_len(len: usize, fill: V) -> Self
    where
        V: Clone;

    /// Current length in elements.
    fn len(&self) -> usize;

    /// Resize to `new_len`, filling new cells with `fill`.
    fn resize(&mut self, new_len: usize, fill: V)
    where
        V: Clone;

    /// Entire read-only buffer.
    fn as_slice(&self) -> &[V];

    /// Entire mutable buffer.
    fn as_mut_slice(&mut self) -> &mut [V];

    /// Copy `src` into the range `[offset .. offset + src.len())`.
    fn write_at(&mut self, offset: usize, src: &[V]) -> Result<(), MeshSieveError>
    where
        V: Clone,
    {
        let end =
            offset
                .checked_add(src.len())
                .ok_or_else(|| MeshSieveError::ScatterChunkMismatch {
                    offset,
                    len: src.len(),
                })?;
        let buf = self.as_mut_slice();
        let dst = buf
            .get_mut(offset..end)
            .ok_or(MeshSieveError::ScatterChunkMismatch {
                offset,
                len: src.len(),
            })?;
        dst.clone_from_slice(src);
        Ok(())
    }

    /// Read the range `[offset .. offset + len)` into `dst`.
    fn read_into(&self, offset: usize, len: usize, dst: &mut [V]) -> Result<(), MeshSieveError>
    where
        V: Clone,
    {
        if dst.len() != len {
            return Err(MeshSieveError::ScatterLengthMismatch {
                expected: len,
                found: dst.len(),
            });
        }
        let end = offset
            .checked_add(len)
            .ok_or_else(|| MeshSieveError::ScatterChunkMismatch { offset, len })?;
        let buf = self.as_slice();
        let src = buf
            .get(offset..end)
            .ok_or(MeshSieveError::ScatterChunkMismatch { offset, len })?;
        dst.clone_from_slice(src);
        Ok(())
    }
}

/// `Vec`-backed storage (default).
#[derive(Clone)]
pub struct VecStorage<V>(pub(crate) Vec<V>);

impl<V> Debug for VecStorage<V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VecStorage")
            .field("len", &self.0.len())
            .finish()
    }
}

impl<V> Storage<V> for VecStorage<V> {
    fn with_len(len: usize, fill: V) -> Self
    where
        V: Clone,
    {
        Self(vec![fill; len])
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn resize(&mut self, new_len: usize, fill: V)
    where
        V: Clone,
    {
        self.0.resize(new_len, fill);
    }

    fn as_slice(&self) -> &[V] {
        &self.0
    }

    fn as_mut_slice(&mut self) -> &mut [V] {
        &mut self.0
    }
}

impl<V> From<Vec<V>> for VecStorage<V> {
    fn from(v: Vec<V>) -> Self {
        Self(v)
    }
}

impl<V> VecStorage<V> {
    pub fn into_inner(self) -> Vec<V> {
        self.0
    }
}
