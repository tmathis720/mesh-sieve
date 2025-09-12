//! Data module: section and atlas
#![warn(missing_docs)]

pub mod atlas;
pub mod bundle;
pub mod section;
pub mod slice_storage;
pub mod storage;
#[cfg(feature = "wgpu")]
pub mod wgpu;

mod _debug_invariants;
pub mod refine;

#[allow(deprecated)]
pub use crate::debug_invariants::DebugInvariants;

pub use section::Section;
pub use slice_storage::SliceStorage;
pub use storage::{Storage, VecStorage};
#[cfg(feature = "wgpu")]
pub use wgpu::WgpuStorage;

/// Alias for the common Vec-backed section.
pub type CpuSection<V> = section::Section<V, VecStorage<V>>;
