//! Data module: section and atlas
#![warn(missing_docs)]

pub mod atlas;
pub mod bundle;
pub mod constrained_section;
pub mod coordinates;
pub mod discretization;
pub mod global_map;
pub mod mixed_section;
pub mod section;
pub mod slice_storage;
pub mod storage;
#[cfg(feature = "wgpu")]
pub mod wgpu;

mod _debug_invariants;
pub mod refine;

#[allow(deprecated)]
pub use crate::debug_invariants::DebugInvariants;

pub use constrained_section::{
    ConstrainedSection, ConstraintSet, DofConstraint, apply_constraints_to_section,
};
pub use discretization::{Discretization, DofLayout, FieldDiscretization, RegionKey};
pub use global_map::LocalToGlobalMap;
pub use mixed_section::{MixedScalar, MixedSectionStore, ScalarType, TaggedSection};
pub use section::Section;
pub use slice_storage::SliceStorage;
pub use storage::{Storage, VecStorage};
#[cfg(feature = "wgpu")]
pub use wgpu::WgpuStorage;

/// Alias for the common Vec-backed section.
pub type CpuSection<V> = section::Section<V, VecStorage<V>>;

/// Coordinate storage wrapper with an attached dimension.
pub use coordinates::Coordinates;
/// Higher-order coordinate storage wrapper.
pub use coordinates::HighOrderCoordinates;

/// Alias for the common Vec-backed coordinates bundle.
pub type CpuCoordinates<V> = coordinates::Coordinates<V, VecStorage<V>>;

/// Section alias for storing `CellType` data over points.
pub type CellTypeSection<S> = section::Section<crate::topology::cell_type::CellType, S>;
