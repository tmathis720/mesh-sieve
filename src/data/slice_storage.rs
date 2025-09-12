use crate::mesh_error::MeshSieveError;
use crate::data::refine::delta::SliceDelta;

/// Storage of a flat array of `V` with slice-oriented operations.
pub trait SliceStorage<V: Clone>: Send + Sync {
    /// Total number of elements stored.
    fn total_len(&self) -> usize;

    /// Resize total elements; preserve prefix `[0..min(old,new)]`.
    fn resize(&mut self, new_len: usize) -> Result<(), MeshSieveError>;

    /// Read `[offset .. offset+len]` into a host `Vec`.
    fn read_slice(&self, offset: usize, len: usize) -> Result<Vec<V>, MeshSieveError>;

    /// Write `src` into `[offset .. offset+src.len()]`.
    fn write_slice(&mut self, offset: usize, src: &[V]) -> Result<(), MeshSieveError>;

    /// Apply a delta from a source slice to a destination slice (may overlap).
    fn apply_delta<D: SliceDelta<V> + 'static>(
        &mut self,
        src_off: usize,
        dst_off: usize,
        len: usize,
        delta: &D,
    ) -> Result<(), MeshSieveError>;
}
