//! Data module: section and atlas
#![warn(missing_docs)]

pub mod atlas;
pub mod bundle;
pub mod section;
pub mod storage;
pub mod slice_storage;
#[cfg(feature = "wgpu")]
pub mod wgpu;

mod _debug_invariants;
pub mod refine;

pub(crate) use _debug_invariants::DebugInvariants;

pub use storage::{Storage, VecStorage};
pub use slice_storage::SliceStorage;
#[cfg(feature = "wgpu")]
pub use wgpu::WgpuStorage;
pub use section::Section;

/// Alias for the common Vec-backed section.
pub type CpuSection<V> = section::Section<V, VecStorage<V>>;
