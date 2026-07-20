# mesh-sieve

`mesh-sieve` is a Rust library for mesh topology, field data, and mesh-aware
data movement in scientific computing codes. It provides DMPlex-inspired sieve
topologies, atlas/section storage for degrees of freedom, mesh labels, geometry
metadata, refinement and assembly utilities, overlap-driven exchange, and
feature-gated MPI, METIS, and GPU support.

The crate is currently version `3.8.1`, uses Rust edition `2024`, and is licensed
under MIT.

## Status

This is an active research and infrastructure library. The core topology and
data APIs are broad and heavily tested, but some integration surfaces are still
evolving:

- Public APIs are `Result`-oriented for data access and layout mutation. Legacy
  panicking map helpers are hidden behind the `map-adapter` feature.
- MPI, METIS, Rayon, CGNS, WGPU, and native CUDA support are opt-in feature flags.
- CGNS/HDF5 reading is experimental and gated by `cgns`; writing CGNS is not
  implemented.
- Triangle and TetGen feature flags are present, but external generator
  integration is not part of the stable default workflow.
- The repository includes many examples and tests; prefer those over older
  snippets or migration notes when checking behavior.

## What the Library Provides

### Topology

Mesh entities are represented as nonzero `PointId` values. Directed incidence is
stored in `Sieve` implementations:

- `InMemorySieve` for generic directed topology.
- `InMemoryOrientedSieve` and `MeshSieve` for orientation-preserving mesh
  topology.
- `InMemorySieveDeterministic` for deterministic map ordering.
- `FrozenSieveCsr` for cache-friendly frozen traversal.
- `InMemoryStack` for vertical base-to-cap relations used by refinement and
  assembly workflows.

The `Sieve` trait supports cone/support queries, point-only traversal helpers,
closure/star traversal, meet/join-style lattice operations, strata, and
arrow-level mutation. A sieve stores at most one arrow for a `(src, dst)` pair;
inserting the same edge replaces its payload.

### Field Data

Field data is stored as an `Atlas` plus a `Section`:

- `Atlas` maps each `PointId` to a contiguous slice `(offset, len)`.
- `Section<V, S>` stores values over that atlas using a storage backend.
- `VecStorage` is the normal CPU backend.
- `WgpuStorage` is available with the `wgpu` feature for suitable POD values.
- `DeviceSection` provides explicit accelerator upload/refresh/download without
  pretending device memory is a host slice.
- `MultiSection`, `FieldSection`, and `ConstrainedSection` support multi-field
  layouts and constrained degrees of freedom.
- `MixedSectionStore` stores named sections with heterogeneous scalar types.

Atlas and section mutations preserve contiguous layouts and expose explicit
errors for duplicate points, missing points, zero-length slices, stale scatter
plans, and slice length mismatches.

### Mesh Metadata and Geometry

The crate includes structures for mesh labels, ownership, cell types,
coordinates, discretization metadata, and DMPlex-style mesh management:

- `LabelSet` for integer labels on mesh points.
- `CellType` for typed element metadata.
- `Coordinates` and `HighOrderCoordinates` for geometry sections.
- `Discretization` and runtime element metadata.
- `CoordinateDM` and `MeshDM` for higher-level mesh, coordinate, label,
  section, distribution, and solver-numbering workflows.
- FVM and FE helpers in `geometry`, `physics`, and `discretization`.

### Algorithms

The `algs` module contains topology and data algorithms used by mesh workflows:

- mesh generation helpers
- interpolation and extrusion
- boundary classification and submesh extraction
- adjacency and dual graph construction
- partitioning adapters and distribution
- PointSF-style migration and field distribution
- overlap completion for topology, stacks, and sections
- reverse Cuthill-McKee renumbering
- assembly and preallocation utilities
- field transfer and reduction

### Overlap and Parallel Workflows

`Overlap` records local point sharing with neighbor ranks. It supports
structural links, later remote-point resolution, closure expansion, pruning of
detached ranks, and invariant validation.

Communication is abstracted through `Communicator` implementations:

- serial `NoComm`
- Rayon-backed in-process communication with `rayon`
- MPI communication with `mpi-support`

Partitioning support includes in-tree algorithms and optional METIS helpers.
The `partitioning` module itself is compiled with `mpi-support`.

### I/O

Readers and writers are trait based through `SieveSectionReader` and
`SieveSectionWriter`. `MeshData` is the common container for topology,
coordinates, labels, cell types, named sections, mixed sections, and
discretization metadata.

Implemented or partially implemented formats include:

- Gmsh
- VTK
- PLY
- Fluent
- Exodus II
- XDMF
- HDF5 and PETSc HDF5
- partitioned mesh metadata and bundles
- experimental CGNS/HDF5 reading with `--features cgns`

See [docs/io-format-capabilities.md](docs/io-format-capabilities.md) and
[docs/hdf5-layout.md](docs/hdf5-layout.md) for current format coverage.

## Installation

```toml
[dependencies]
mesh-sieve = "3.8.0"
```

When working from this repository:

```sh
cargo check
cargo test
```

The crate currently depends on HDF5 support through `hdf5` and `hdf5-sys`.
The `hdf5-sys` dependency is configured with its `static` feature.

## Feature Flags

The default feature set is empty.

| Feature | Purpose |
| --- | --- |
| `strict-invariants` | Enable deeper invariant checks outside debug-only paths. |
| `check-invariants` | Alias that enables `strict-invariants`. |
| `map-adapter` | Expose legacy infallible map helpers. Prefer `try_*` APIs. |
| `rayon` | Enable Rayon-backed parallel utilities and communicator support. |
| `mpi-support` | Enable MPI support; also enables `rayon`, `rand`, and `ahash`. |
| `mpi-derive` | Enable MPI derive support from the `mpi` crate. |
| `metis-support` | Enable the vendored METIS partitioning backend through `metis-sys`. |
| `fast-hash` | Use `ahash` in selected hot paths. |
| `deterministic-order` | Prefer deterministic map/set ordering where supported. |
| `deterministic-owners` | Deterministic ownership-related behavior in gated paths. |
| `wgpu` | Enable GPU storage support for sections. |
| `cuda` | Enable native CUDA 12 execution through dynamically loaded `cudarc` driver/NVRTC libraries. |
| `cuda-cublas` | Add cuBLAS bindings to the CUDA backend. |
| `cuda-cusparse` | Add cuSPARSE bindings to the CUDA backend. |
| `cuda-cusolver` | Add cuSOLVER bindings to the CUDA backend. |
| `cuda-nccl` | Add NCCL bindings for downstream multi-GPU collectives. |
| `cgns` | Enable the experimental CGNS/HDF5 reader. |
| `gmsh-support` | Reserved feature flag for Gmsh-related integration paths. |
| `triangle-support` | Reserved feature flag for Triangle integration paths. |
| `tetgen-support` | Reserved feature flag for TetGen integration paths. |
| `check-empty-part` | Extra overlap invariant checking for empty rank nodes. |
| `check-graph-edges` | Extra graph edge checking in gated code paths. |
| `expensive-checks` | Enable additional expensive validation paths. |
| `binpack-retry` | Enable retry behavior in bin-packing partition paths. |
| `exact-metrics` | Enable exact partition metric helpers where available. |
| `mem-snapshot` | Enable memory snapshot instrumentation paths. |
| `sparse-bitset` | Reserved sparse bitset feature flag. |
| `log` | Enable log-gated paths in the crate. |

Common development lanes:

```sh
cargo test
cargo test --features strict-invariants
cargo test --features rayon
cargo test --features mpi-support
cargo check --features cuda
cargo test --features cuda --test cuda_fvm
```

## CUDA execution

CUDA is a plan-based execution layer, not a `Section` storage adapter. Mutable
topology, labels, coordinates, and layout remain authoritative on the host:

```text
mutable mesh -> FrozenSieveCsr/FvmInputs -> persistent device plan -> batched kernels
```

`DeviceMeshPlan` uploads dense cone/support CSR arrays. `DeviceFvmPlan` compiles
cell/face geometry and cell-to-face CSR, while `DeviceFvmState` keeps scalar
state, face workspaces, gradients, and residuals resident across iterations.
CUDA currently provides `f32` and `f64` scalar upwind/central convection,
orthogonal diffusion, Green--Gauss gradients, wet/dry masking, and deterministic
cell gathering without floating-point atomics. The CPU backend executes the
same packed plans for parity testing.

The `cuda` feature builds without link-time CUDA dependencies, but execution
requires the NVIDIA driver and NVRTC shared libraries at runtime. It targets
the CUDA 12 ABI by default. `ComputeBackend::Auto` probes both the device and
NVRTC during initialization and reports why it selected CUDA or CPU; it never
falls back after a kernel error. WGPU remains available as an explicit backend
for operations that support it.

Plans capture topology, atlas, and geometry epochs. Callers must increment the
geometry epoch after coordinate changes and the topology version after mesh
adaptation; stale plans return `AcceleratorError` before launch.

MPI examples normally need an MPI launcher:

```sh
cargo mpirun -n 2 --features mpi-support --example mpi_complete
```

## Quick Start

### Topology

```rust
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::sieve::{InMemorySieve, Sieve};

fn main() -> Result<(), mesh_sieve::mesh_error::MeshSieveError> {
    let mut sieve = InMemorySieve::<PointId, ()>::default();

    let cell = PointId::new(1)?;
    let face = PointId::new(2)?;

    sieve.add_arrow(cell, face, ());

    let cone: Vec<_> = sieve.cone_points(cell).collect();
    let closure: Vec<_> = sieve.closure_iter([cell]).collect();

    assert_eq!(cone, vec![face]);
    assert!(closure.contains(&face));

    Ok(())
}
```

### Atlas and Section

```rust
use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::section::Section;
use mesh_sieve::data::storage::VecStorage;
use mesh_sieve::topology::point::PointId;

fn main() -> Result<(), mesh_sieve::mesh_error::MeshSieveError> {
    let p = PointId::new(7)?;

    let mut atlas = Atlas::default();
    atlas.try_insert(p, 3)?;

    let mut section = Section::<f64, VecStorage<f64>>::new(atlas);
    section.try_set(p, &[1.0, 2.0, 3.0])?;

    assert_eq!(section.try_restrict(p)?, &[1.0, 2.0, 3.0]);

    Ok(())
}
```

### Labels

```rust
use mesh_sieve::topology::labels::LabelSet;
use mesh_sieve::topology::point::PointId;

fn main() -> Result<(), mesh_sieve::mesh_error::MeshSieveError> {
    let wall = PointId::new(11)?;

    let mut labels = LabelSet::new();
    labels.set_label(wall, "boundary", 1);

    assert_eq!(labels.get_label(wall, "boundary"), Some(1));

    Ok(())
}
```

### Overlap

```rust
use mesh_sieve::overlap::overlap::Overlap;
use mesh_sieve::topology::point::PointId;

fn main() -> Result<(), mesh_sieve::mesh_error::MeshSieveError> {
    let local = PointId::new(42)?;
    let remote = PointId::new(42042)?;

    let mut overlap = Overlap::default();
    overlap.add_link_structural_one(local, 1);
    overlap.resolve_remote_point(local, 1, remote)?;

    let links: Vec<_> = overlap.links_to_resolved(1).collect();
    assert_eq!(links, vec![(local, remote)]);

    Ok(())
}
```

## Examples

The repository contains focused examples under [examples](examples):

- `meshgen_basic.rs`
- `gmsh_io.rs`
- `gmsh_interpolate_quality.rs`
- `poisson_segment.rs`
- `poisson_quad.rs`
- `refine_triangle.rs`
- `adapt_with_metric.rs`
- `dm_metric_label_adapt.rs`
- `distribute_with_overlap.rs`
- `periodic_2d_wrap.rs`
- `e2e_showcase.rs`
- `external_mesh_workflows.rs`
- `comprehensive_mesh_workflow.rs`
- `extrude_boundary_constraints.rs`

MPI-oriented examples include:

- `mpi_complete.rs`
- `mpi_complete_stack.rs`
- `mpi_complete_no_overlap.rs`
- `mpi_complete_multiple_neighbors.rs`
- `distribute_mpi.rs`
- `mesh_distribute_two_ranks.rs`
- `mpi_partition_exchange.rs`
- `mpi_partitioned_io.rs`
- `distributed_rcm.rs`
- `petsc_sf_composed_pipeline.rs`
- `refine_distribute_complete_section.rs`
- `fvm_distributed_qa.rs`

Example commands:

```sh
cargo run --example meshgen_basic
cargo run --example gmsh_io
cargo run --features rayon --example e2e_showcase
cargo mpirun -n 2 --features mpi-support --example mesh_distribute_two_ranks
```

## Project Layout

```text
src/
  topology/        core points, arrows, sieves, labels, ownership, orientation
  data/            atlas, section, constraints, coordinates, mixed sections
  algs/            mesh algorithms, distribution, completion, assembly
  overlap/         rank-sharing structures and overlap deltas
  io/              mesh readers, writers, bundles, partitioned I/O
  dm.rs            DMPlex-like facade
  geometry/        metrics, quality checks, point location, FVM geometry
  physics/         FE and FVM helper routines
  discretization/  runtime discretization metadata and element helpers
  mesh_generation/ mesh construction helpers
  partitioning/    MPI-gated partitioning implementation
```

## Documentation

Additional repository documentation:

- [API_Guide.md](API_Guide.md)
- [Physics_Guide.md](Physics_Guide.md)
- [docs/topology-api.md](docs/topology-api.md)
- [docs/fe-setup.md](docs/fe-setup.md)
- [docs/geometry-quality.md](docs/geometry-quality.md)
- [docs/io-format-capabilities.md](docs/io-format-capabilities.md)
- [docs/exodus.md](docs/exodus.md)
- [docs/hdf5-layout.md](docs/hdf5-layout.md)
- [docs/coastal-mesh-labels.md](docs/coastal-mesh-labels.md)

API documentation is configured for <https://docs.rs/mesh-sieve>.

## Testing

The test suite covers topology traversal, deterministic ordering, atlas/section
invariants, scatter and assembly behavior, I/O round trips, mesh distribution,
overlap determinism, refinement/coarsening, FVM/FE helpers, diagnostics, and
feature-gated MPI/GPU paths.

Useful commands:

```sh
cargo test
cargo test --features strict-invariants
cargo test --features rayon
cargo test --features cgns
cargo test --features wgpu
cargo test --features mpi-support
```

Some feature combinations require system tools or libraries, such as MPI,
GPU drivers, or HDF5-compatible runtime support.

## License

MIT. See [LICENSE](LICENSE).
