#![cfg_attr(docsrs, feature(doc_cfg))]
//! # mesh-sieve
//!
//! mesh-sieve is a modular, high-performance Rust library for mesh and data management, designed for scientific computing and PDE codes. It provides abstractions for mesh topology, field data, parallel partitioning, and communication, supporting both serial and MPI-based distributed workflows.
//!
//! ## Features
//! - Mesh topology and Sieve data structures for flexible mesh connectivity
//! - Atlas and Section types for mapping mesh points to data arrays
//! - Pluggable communication backends (serial, Rayon, MPI) for ghost exchange and mesh distribution
//! - Built-in support for graph partitioning (Metis, custom algorithms)
//! - MPI integration for distributed mesh and data exchange
//! - Extensive serial, parallel, and property-based testing
//!
//! ## Determinism
//!
//! All randomized decisions use `SmallRng` seeds drawn from configuration so runs are
//! reproducible. Unit tests fix seeds explicitly to ensure deterministic behavior.
//!
//! ## Usage
//! Add `mesh-sieve` as a dependency in your `Cargo.toml` and enable features as needed:
//!
//! ```toml
//! [dependencies]
//! mesh-sieve = "1.2.1"
//! # Optional features:
//! # features = ["mpi-support","rayon","metis-support"]
//! ```
//!
//! For a complete API reference and usage guide, see [API_Guide.md](API_Guide.md).
//!
//! ## Migration highlights
//! See [docs/MIGRATING.md](docs/MIGRATING.md) for more details on recent API
//! changes:
//! - Fallible map access via `try_restrict_*` helpers and the [`FallibleMap`]
//!   trait.
//! - [`Bundle::assemble`] now averages element-wise; use
//!   [`Bundle::assemble_with`] with a [`SliceReducer`](crate::data::bundle::SliceReducer)
//!   for custom behavior.
//! - `data::refine::delta::SliceDelta` replaces the deprecated `Delta` alias.
//!
//! ## Shared payloads
//! When payloads are large or shared across many arrows, instantiate a sieve with `Payload = Arc<T>`.
//! Traversal and algorithms will clone the `Arc` handle (cheap) without copying `T`.
//! Use type aliases [`topology::sieve::InMemorySieveArc`],
//! [`topology::sieve::InMemoryOrientedSieveArc`], and [`topology::sieve::InMemoryStackArc`] for convenience.
//! Avoid wrappers that convert between `T` and `Arc<T>` on the fly; they add allocations and defeat sharing.

//! Public prelude for mesh-sieve: mesh/data-management library for PDE codes
//!
//! All Sieve implementations provide `points()`, `base_points()`, and `cap_points()` iterators for global point set access.

// Re-export our major subsystems:
pub mod algs;
pub mod data;
pub mod debug_invariants;
pub mod mesh_error;
pub mod overlap;
#[cfg(feature = "mpi-support")]
pub mod partitioning;
pub mod section;
pub mod topology;

pub use debug_invariants::DebugInvariants;

/// A convenient prelude to import the most-used traits & types:
pub mod prelude {
    pub use crate::algs::communicator::Communicator;
    #[cfg(feature = "mpi-support")]
    pub use crate::algs::communicator::MpiComm;
    #[cfg(feature = "rayon")]
    pub use crate::algs::communicator::RayonComm;
    pub use crate::algs::completion::{
        complete_section, complete_sieve, complete_stack, complete_stack_with_tags,
    };
    pub use crate::algs::rcm::distributed_rcm;
    pub use crate::data::atlas::Atlas;
    pub use crate::data::coordinates::Coordinates;
    #[cfg(feature = "map-adapter")]
    pub use crate::data::section::Map;
    pub use crate::data::section::Section;
    pub use crate::debug_invariants::DebugInvariants;
    pub use crate::overlap::delta::{AddDelta, CopyDelta, ValueDelta};
    pub use crate::overlap::overlap::Overlap;
    pub use crate::topology::bounds::{PayloadLike, PointLike};
    pub use crate::topology::cell_type::CellType;
    pub use crate::topology::labels::LabelSet;
    pub use crate::topology::point::PointId;
    pub use crate::topology::sieve::{
        InMemoryOrientedSieve, InMemoryOrientedSieveArc, InMemorySieve, InMemorySieveArc,
        InMemorySieveDeterministic, InMemoryStackArc, MutableSieve, Orientation, OrientedSieve,
        Sieve, SieveBuildExt, SieveQueryExt,
    };
    pub use crate::topology::stack::{InMemoryStack, Stack};
}
