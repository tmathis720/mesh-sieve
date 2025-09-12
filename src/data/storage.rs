//! Pluggable storage for Section buffers.
//!
//! This trait abstracts how Section's flat buffer is stored (e.g., Vec, mmap, GPU).
//! The initial design keeps CPU-slice semantics to minimize churn. Future steps
//! can add async/mapped variants without touching Section's public API.

use core::fmt::{self, Debug};

use crate::data::refine::delta::SliceDelta;
use crate::data::slice_storage::SliceStorage;
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

impl<V> SliceStorage<V> for VecStorage<V>
where
    V: Clone + Default + Send + Sync,
{
    fn total_len(&self) -> usize {
        self.0.len()
    }

    fn resize(&mut self, new_len: usize) -> Result<(), MeshSieveError> {
        self.0.resize(new_len, V::default());
        Ok(())
    }

    fn read_slice(&self, offset: usize, len: usize) -> Result<Vec<V>, MeshSieveError> {
        let end = offset
            .checked_add(len)
            .ok_or(MeshSieveError::ScatterChunkMismatch { offset, len })?;
        let src = self
            .0
            .get(offset..end)
            .ok_or(MeshSieveError::ScatterChunkMismatch { offset, len })?;
        Ok(src.to_vec())
    }

    fn write_slice(&mut self, offset: usize, src: &[V]) -> Result<(), MeshSieveError> {
        let end = offset
            .checked_add(src.len())
            .ok_or(MeshSieveError::ScatterChunkMismatch {
                offset,
                len: src.len(),
            })?;
        let dst = self
            .0
            .get_mut(offset..end)
            .ok_or(MeshSieveError::ScatterChunkMismatch {
                offset,
                len: src.len(),
            })?;
        dst.clone_from_slice(src);
        Ok(())
    }

    fn apply_delta<D: SliceDelta<V> + 'static>(
        &mut self,
        src_off: usize,
        dst_off: usize,
        len: usize,
        delta: &D,
    ) -> Result<(), MeshSieveError> {
        if len == 0 {
            return Ok(());
        }
        let src_end = src_off
            .checked_add(len)
            .ok_or(MeshSieveError::ScatterChunkMismatch {
                offset: src_off,
                len,
            })?;
        let dst_end = dst_off
            .checked_add(len)
            .ok_or(MeshSieveError::ScatterChunkMismatch {
                offset: dst_off,
                len,
            })?;
        if src_end > self.0.len() {
            return Err(MeshSieveError::ScatterChunkMismatch {
                offset: src_off,
                len,
            });
        }
        if dst_end > self.0.len() {
            return Err(MeshSieveError::ScatterChunkMismatch {
                offset: dst_off,
                len,
            });
        }
        let disjoint = src_end <= dst_off || dst_end <= src_off;
        if disjoint {
            if src_off < dst_off {
                let (a, b) = self.0.split_at_mut(dst_off);
                let src = &a[src_off..src_end];
                let dst = &mut b[..len];
                delta.apply(src, dst)?;
            } else {
                let (a, b) = self.0.split_at_mut(src_off);
                let dst = &mut a[dst_off..dst_end];
                let src = &b[..len];
                delta.apply(src, dst)?;
            }
        } else {
            let src_copy: Vec<V> = self.0[src_off..src_end].to_vec();
            let dst = &mut self.0[dst_off..dst_end];
            delta.apply(&src_copy, dst)?;
        }
        Ok(())
    }
}
