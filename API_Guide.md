# mesh-sieve API Guide

This guide describes the current public API shape of `mesh-sieve` as it exists
in this repository. The crate is version `3.8.0`, uses Rust edition `2024`, and
has no default feature flags.

`mesh-sieve` is organized around a small set of mesh abstractions:

- `PointId`: nonzero mesh entity identifiers.
- `Sieve`: directed incidence graphs for topology.
- `Atlas`: point-to-slice layout metadata.
- `Section<V, S>`: values stored over an atlas with a storage backend.
- `LabelSet`: integer labels on mesh points.
- `Coordinates`: coordinate sections with topological and embedding dimensions.
- `Overlap`: local-to-rank sharing information.
- `Communicator`: serial, Rayon, and MPI communication abstraction.
- `MeshData` and `MeshBundle`: I/O containers for topology and data.
- `MeshDM`: a DMPlex-like facade over topology, coordinates, labels, sections,
  distribution, and solver numbering.

The preferred style is fallible and explicit: use `try_*` methods, propagate
`MeshSieveError`, and enable `strict-invariants` in at least one test lane.

## 1. Crate Setup

```toml
[dependencies]
mesh-sieve = "3.8.0"
```

Common local commands:

```sh
cargo check
cargo test
cargo test --features strict-invariants
```

The default feature set is empty. Important opt-in features:

| Feature | Use |
| --- | --- |
| `strict-invariants` | Run deeper invariant checks outside debug-only paths. |
| `check-invariants` | Alias for `strict-invariants`. |
| `map-adapter` | Expose legacy infallible map helpers. Prefer `try_*`. |
| `rayon` | Enable Rayon-backed parallel utilities and `RayonComm`. |
| `mpi-support` | Enable MPI communication and MPI distribution paths. |
| `mpi-derive` | Enable derive support from the `mpi` crate. |
| `metis-support` | Enable the vendored METIS backend through `metis-sys`. |
| `fast-hash` | Enable `ahash` in selected hot paths. |
| `deterministic-order` | Prefer deterministic map/set ordering where supported. |
| `wgpu` | Enable GPU-backed section storage. |
| `cgns` | Enable the experimental CGNS/HDF5 reader. |

Feature-gated paths may require system support: MPI, HDF5, GPU drivers,
or a compatible WGPU adapter.

## 2. Points and Arrows

Every mesh entity is a `PointId`. `PointId::new(0)` is invalid because zero is
reserved as a sentinel.

```rust
use mesh_sieve::topology::point::PointId;

fn main() -> Result<(), mesh_sieve::mesh_error::MeshSieveError> {
    let cell = PointId::new(1)?;
    let face = PointId::new(2)?;
    assert_eq!(cell.get(), 1);
    assert_ne!(cell, face);
    Ok(())
}
```

`Arrow<P>` represents a directed incidence `src -> dst` carrying an arbitrary
payload. For vertical stack arrows, `Polarity::{Forward, Reverse}` represents
slice orientation during refinement and assembly. For group-valued mesh
orientation, use types in `topology::orientation` and `OrientedSieve`.

## 3. Sieve Topology

The core trait is `topology::sieve::Sieve`. A sieve is a directed graph with
bidirectional incidence queries:

- `cone(p)`: outgoing arrows from `p`.
- `support(p)`: incoming arrows to `p`.
- `cone_points(p)` and `support_points(p)`: point-only adapters.
- `add_arrow(src, dst, payload)`: insert or replace one edge.
- `remove_arrow(src, dst)`: remove one edge.
- `closure_iter(seeds)`, `star_iter(seeds)`, `closure_both_iter(seeds)`.
- `height`, `depth`, `diameter`, `height_stratum`, `depth_stratum`.
- `base_points`, `cap_points`, `points`, `points_sorted`, `points_chart_order`.

A sieve stores at most one arrow for a `(src, dst)` pair. Adding the same edge
again replaces its payload rather than creating a parallel edge.

Common implementations and aliases:

- `InMemorySieve<P, T>`: generic in-memory topology.
- `InMemoryOrientedSieve<P, T, O>`: per-arrow orientation.
- `MeshSieve`: default orientation-preserving mesh topology alias.
- `InMemorySieveDeterministic`: deterministic in-memory variant.
- `FrozenSieveCsr`: frozen CSR representation for cache-friendly traversal.
- `InMemoryStack<B, C, T>`: vertical base-to-cap relation.

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

Use `SieveBuildExt` and `SieveReserveExt` for bulk construction and
preallocation. Use `SieveRef` traversal variants when you need borrowed
payloads rather than cloned payloads.

## 4. Labels

`LabelSet` stores integer labels by point and name. It is used for boundaries,
regions, material IDs, imported mesh tags, and selecting submeshes.

Key methods:

- `set_label(point, name, value)`
- `get_label(point, name)`
- `stratum_points(name, value)`
- `stratum_values(name)`
- `stratum_points_in_range(name, range)`
- `stratum_union`, `stratum_intersection`, `stratum_difference`
- `propagate_label_set_closure`, `propagate_label_set_star`

```rust
use mesh_sieve::topology::labels::LabelSet;
use mesh_sieve::topology::point::PointId;

fn main() -> Result<(), mesh_sieve::mesh_error::MeshSieveError> {
    let wall = PointId::new(11)?;

    let mut labels = LabelSet::new();
    labels.set_label(wall, "boundary", 1);

    assert_eq!(labels.get_label(wall, "boundary"), Some(1));
    assert_eq!(labels.stratum_points("boundary", 1), vec![wall]);
    Ok(())
}
```

## 5. Atlas and Section

`Atlas` is layout only. It maps each point to a contiguous slice in a flat data
buffer.

Key `Atlas` methods:

- `try_insert(point, len) -> Result<offset, MeshSieveError>`
- `get(point) -> Option<(offset, len)>`
- `contains(point)`
- `remove_point(point)`
- `points()`, `iter_entries()`, `iter_spans()`
- `total_len()`, `len()`, `is_empty()`
- `version()`, `build_scatter_plan()`

`Section<V, S>` couples an atlas with storage. The storage bound is
`data::storage::Storage<V>`. `VecStorage<V>` is the normal CPU backend and
`CpuSection<V>` is the crate alias for `Section<V, VecStorage<V>>`.

Key `Section` methods:

- `new(atlas)`
- `atlas()`
- `try_restrict(point) -> Result<&[V], MeshSieveError>`
- `try_restrict_mut(point) -> Result<&mut [V], MeshSieveError>`
- `try_set(point, values)`
- `try_add_point(point, len)`
- `try_remove_point(point)`
- `with_atlas_mut(|atlas| ...)`
- `with_atlas_resize(policy, |atlas| ...)`
- `try_apply_delta_between_points(src, dst, delta)`
- `gather_in_order()`
- `try_scatter_in_order(buf)`
- `try_scatter_from(buf, spans)`
- `try_scatter_with_plan(buf, plan)`

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

### Layout Mutation

Use `with_atlas_mut` when existing points keep their slice lengths. If an
existing point changes length, the mutation is rolled back and returns
`MeshSieveError::AtlasPointLengthChanged`.

Use `with_atlas_resize` when length changes are intentional:

```rust
use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::section::{ResizePolicy, Section};
use mesh_sieve::data::storage::VecStorage;
use mesh_sieve::topology::point::PointId;

fn main() -> Result<(), mesh_sieve::mesh_error::MeshSieveError> {
    let p = PointId::new(7)?;
    let q = PointId::new(8)?;

    let mut atlas = Atlas::default();
    atlas.try_insert(p, 2)?;

    let mut section = Section::<f64, VecStorage<f64>>::new(atlas);
    section.try_set(p, &[1.0, 2.0])?;

    section.with_atlas_resize(ResizePolicy::PreservePrefix, |atlas| {
        let _ = atlas.try_insert(q, 3);
    })?;

    Ok(())
}
```

Available resize policies:

- `ZeroInit`
- `PreservePrefix`
- `PreserveSuffix`
- `PadWith(value)`

### Storage Traits

`Storage<V>` provides the storage interface used by `Section`: construction by
length, resize, immutable/mutable flat slices, and delta/scatter helpers.

`SliceStorage<V>` is a lower-level trait implemented by `VecStorage` and, when
enabled, `WgpuStorage`. Most users should program against `Section<V, S>` and
`Storage<V>` rather than using `SliceStorage` directly.

## 6. Deltas

There are two delta concepts:

- `data::refine::delta::SliceDelta<V>` transforms one slice into another. It is
  used by section-internal copy/refine paths and by `Polarity`.
- `overlap::delta::ValueDelta<V>` restricts and fuses individual values for
  exchange across overlaps. Built-ins include `CopyDelta`, `AddDelta`,
  `ZeroDelta`, and `CellTypeDelta`.

`Delta` aliases may still exist for compatibility, but new code should use
`SliceDelta` and `ValueDelta` by name.

## 7. Coordinates and Geometry Data

`Coordinates<V, S>` wraps a section and validates that every point has exactly
`embedding_dimension` values. It also records the mesh topological dimension.

Related types:

- `Coordinates<V, S>`
- `CpuCoordinates<V>`
- `HighOrderCoordinates<V, S>`
- `MeshVelocity<V, S>`
- `CoordinateDM<V, S>`

```rust
use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::coordinates::{Coordinates, HighOrderCoordinates};
use mesh_sieve::data::storage::VecStorage;
use mesh_sieve::topology::point::PointId;

fn main() -> Result<(), mesh_sieve::mesh_error::MeshSieveError> {
    let p = PointId::new(1)?;

    let mut atlas = Atlas::default();
    atlas.try_insert(p, 3)?;

    let mut coords = Coordinates::<f64, VecStorage<f64>>::try_new(2, 3, atlas)?;
    coords.try_restrict_mut(p)?.copy_from_slice(&[0.0, 1.0, 0.0]);

    let mut ho_atlas = Atlas::default();
    ho_atlas.try_insert(p, 9)?;
    let high_order = HighOrderCoordinates::<f64, VecStorage<f64>>::try_new(3, ho_atlas)?;
    coords.set_high_order(high_order)?;

    Ok(())
}
```

Geometry and physics helpers live in:

- `geometry::metrics`
- `geometry::quality`
- `geometry::locate`
- `geometry::fvm`
- `physics::fe`
- `physics::fvm`
- `discretization::runtime`

See [docs/geometry-quality.md](docs/geometry-quality.md) and
[Physics_Guide.md](Physics_Guide.md).

## 8. Discretization and Section Layouts

There are two related but separate API areas:

- `data::discretization`: basis and quadrature metadata keyed by regions.
- `data::section_layout`: DOF layout construction helpers.

`Discretization` stores named `FieldDiscretization` records. Each field maps a
`RegionKey` to `DiscretizationMetadata`.

```rust
use mesh_sieve::data::discretization::{
    Discretization, DiscretizationMetadata, FieldDiscretization,
};
use mesh_sieve::topology::cell_type::CellType;

let mut fields = Discretization::new();

let tri_p1 = DiscretizationMetadata::new("lagrange", "gauss")
    .with_basis_metadata(1, ["phi0", "phi1", "phi2"])
    .with_quadrature_metadata(
        2,
        vec![vec![1.0 / 3.0, 1.0 / 3.0]],
        vec![0.5],
    );

let mut velocity = FieldDiscretization::new();
velocity.set_cell_type_metadata(CellType::Triangle, tri_p1);
velocity.set_label_metadata(
    "fluid",
    1,
    DiscretizationMetadata::new("lagrange", "gauss"),
);

fields.insert_field("velocity", velocity);
```

For actual DOF vector layouts, use `data::section_layout` helpers such as
`DofLayout`, `build_layout_with`, `layout_for_section_with_constraints_and_periodic`,
`layout_for_multi_section_with_periodic`, `local_vector_for_section`, and
`local_vector_for_layout`.

## 9. Constraints and Multi-Field Data

`ConstrainedSection<V, S>` wraps a section and stores fixed DOF values by point
and component. It is useful for Dirichlet-style constraints.

```rust
use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::constrained_section::ConstrainedSection;
use mesh_sieve::data::section::Section;
use mesh_sieve::data::storage::VecStorage;
use mesh_sieve::topology::point::PointId;

fn main() -> Result<(), mesh_sieve::mesh_error::MeshSieveError> {
    let p = PointId::new(7)?;

    let mut atlas = Atlas::default();
    atlas.try_insert(p, 3)?;

    let section = Section::<f64, VecStorage<f64>>::new(atlas);
    let mut constrained = ConstrainedSection::new(section);

    constrained.insert_constraint(p, 2, 0.0)?;
    constrained.apply_constraints()?;

    Ok(())
}
```

`MultiSection<V, S>` stacks multiple named `FieldSection<V, S>` values and
computes field offsets/spans for a point:

- `FieldSection::new(name, section)`
- `FieldSection::insert_constraint(point, index, value)`
- `MultiSection::new(fields)`
- `field_offset(point, field_index)`
- `field_span(point, field_index)`
- `field_offset_by_name(point, name)`
- `apply_constraints()`

```rust
use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::multi_section::{FieldSection, MultiSection};
use mesh_sieve::data::section::Section;
use mesh_sieve::data::storage::VecStorage;
use mesh_sieve::topology::point::PointId;

fn main() -> Result<(), mesh_sieve::mesh_error::MeshSieveError> {
    let p = PointId::new(7)?;

    let mut atlas = Atlas::default();
    atlas.try_insert(p, 3)?;

    let velocity = Section::<f64, VecStorage<f64>>::new(atlas.clone());
    let pressure = Section::<f64, VecStorage<f64>>::new(atlas);

    let fields = vec![
        FieldSection::new("velocity", velocity),
        FieldSection::new("pressure", pressure),
    ];

    let multi = MultiSection::new(fields)?;
    let (_offset, dof_count) = multi.field_span(p, 0)?;
    assert_eq!(dof_count, 3);

    Ok(())
}
```

Hanging-node constraints are available through `HangingNodeConstraints`,
`HangingDofConstraint`, `LinearConstraintTerm`,
`constraints_from_topological_anchors`, and `apply_hanging_constraints_to_section`.

## 10. Mixed Scalar Sections

`MixedSectionStore` keeps named sections with different scalar types. Supported
scalar tags are `f64`, `f32`, `i32`, `i64`, `u32`, and `u64`.

```rust
use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::mixed_section::MixedSectionStore;
use mesh_sieve::data::section::Section;
use mesh_sieve::data::storage::VecStorage;

let atlas = Atlas::default();

let mut store = MixedSectionStore::new();
store.insert("temperature", Section::<f64, VecStorage<f64>>::new(atlas.clone()));
store.insert("material_id", Section::<i32, VecStorage<i32>>::new(atlas));

let temp = store.get::<f64>("temperature");
assert!(temp.is_some());
```

Use `get_tagged`, `iter`, `gather_in_order`, and `try_scatter_in_order` when
the scalar type is not known statically.

## 11. Bundles, Refinement, and Assembly

`Bundle<V, S, D>` combines:

- `stack: InMemoryStack<PointId, PointId, Polarity>`
- `section: Section<V, S>`
- `delta: D`

`Bundle::refine(bases)` pushes data from base points to cap points using stack
orientation. `Bundle::assemble_with(bases, reducer)` pulls cap slices back to
base slices using a reducer such as `AverageReducer`.

Common APIs:

- `refine(bases)`
- `refine_with_constraints(bases, constraints)`
- `assemble(bases)`
- `assemble_with(bases, reducer)`
- `assemble_with_constraints(bases, reducer, constraints)`
- `apply_constraints(constraints)`

Reducers implement `SliceReducer<V>`. `AverageReducer` is the default reducer
used by `assemble`.

## 12. Overlap

`Overlap` stores local point sharing with remote ranks. Internally it is a
bipartite graph between local points and `Part(rank)` nodes, with a
`Remote { rank, remote_point }` payload.

Key APIs:

- `add_link_structural_one(local, rank)`
- `try_add_link_structural_one(local, rank)`
- `add_links_structural_bulk(iter)` and `try_add_links_structural_bulk(iter)`
- `resolve_remote_point(local, rank, remote)`
- `resolve_remote_points(iter)`
- `neighbor_ranks()`
- `links_to(rank)`
- `links_to_resolved(rank)`
- `links_to_sorted(rank)`
- `links_to_resolved_sorted(rank)`
- `remove_neighbor_rank(rank)`
- `retain_neighbor_ranks(keep)`
- `prune_empty_parts()`
- `validate_invariants()`

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

Use completion helpers such as `ensure_closure_of_support`,
`ensure_closure_of_support_from_seeds`, and `expand_one_layer_mesh` to grow
overlap along topology. After deleting links or recomputing neighbor sets, call
`prune_empty_parts()` if you use the `check-empty-part` invariant feature.

## 13. Communication and Distribution

`algs::communicator::Communicator` is the byte-oriented communication trait.
Implementations include:

- `NoComm`: serial/no-op communicator.
- `RayonComm`: available with `rayon`.
- `MpiComm`: available with `mpi-support`.

Important communicator operations:

- `isend(peer, tag, bytes)`
- `irecv(peer, tag, buffer)`
- `rank()`, `size()`
- `barrier()`
- `broadcast(root, buffer)`
- `allreduce_sum(values)`

Higher-level distribution and PETSc-SF-style APIs live in:

- `algs::point_sf`
- `algs::distribute`
- `algs::completion`
- `algs::assembly`

Selected exports:

- `PointSF`, `PointSfLeaf`, `RemotePoint`, `SfDistribution`
- `create_point_sf`, `create_process_sf`, `create_migration_sf`
- `distribute_topology`, `distribute_section`, `distribute_labels`,
  `distribute_field`
- `DistributionConfig`, `ProvidedPartition`, `CustomPartitioner`
- `distribute_mesh`, `distribute_with_overlap`,
  `distribute_with_overlap_periodic`
- `complete_sieve`, `complete_sieve_with_tags`,
  `complete_sieve_until_converged`
- `complete_section`, `complete_section_with_ownership`,
  `complete_section_with_tags`, `complete_section_with_tags_and_ownership`
- `complete_stack`, `complete_stack_with_tags`

MPI examples usually need an MPI launcher:

```sh
mpirun -n 2 cargo run --features mpi-support --example mpi_complete
```

## 14. Partitioning

The crate has two partitioning surfaces:

- `algs::partition` and `algs::metis_partition` for algorithm-level helpers.
- `partitioning` for MPI-gated in-tree partitioning algorithms and metrics.

`metis-support` enables the vendored METIS bindings. The `partitioning` module is compiled
with `mpi-support`.

Related APIs and examples:

- `algs::dual_graph`
- `algs::adjacency_graph`
- `algs::rcm::distributed_rcm`
- `partitioning::{partition, PartitionerConfig}` when `mpi-support` is enabled
- examples `partition.rs`, `distributed_rcm.rs`, and `mpi_partition_exchange.rs`

## 15. Mesh Containers and I/O

`io::MeshData<S, V, St, CtSt>` is the common container returned by readers:

- `sieve`
- `coordinates`
- `sections`
- `mixed_sections`
- `labels`
- `cell_types`
- `discretization`

Readers and writers use:

- `SieveSectionReader`
- `SieveSectionWriter`

Available format modules:

- `io::gmsh`
- `io::vtk`
- `io::ply`
- `io::fluent`
- `io::exodus`
- `io::xdmf`
- `io::hdf5`
- `io::petsc_hdf5`
- `io::partitioned`
- `io::bundle`
- `io::cgns` with `cgns`

```rust
use mesh_sieve::data::section::Section;
use mesh_sieve::data::storage::VecStorage;
use mesh_sieve::io::MeshData;
use mesh_sieve::topology::cell_type::CellType;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::sieve::InMemorySieve;

let sieve = InMemorySieve::<PointId, ()>::default();
let mut mesh: MeshData<_, f64, VecStorage<f64>, VecStorage<CellType>> = MeshData::new(sieve);
mesh.sections
    .insert("temperature".to_string(), Section::new(Default::default()));
```

`MeshBundle` stores multiple `MeshData` values and can synchronize labels or
coordinates across them with `sync_labels()` and `sync_coordinates()`.

See:

- [docs/io-format-capabilities.md](docs/io-format-capabilities.md)
- [docs/hdf5-layout.md](docs/hdf5-layout.md)
- [docs/exodus.md](docs/exodus.md)

## 16. MeshDM

`MeshDM` is a higher-level DMPlex-like facade. It coordinates topology,
coordinates, labels, named sections, distribution metadata, and solver
preallocation workflows.

Commonly used exported types:

- `MeshDM`
- `MeshDMBuilder`
- `MeshDMOptions`
- `MeshDMDistribution`
- `MeshDMLabelSelection`
- `MeshDMSubmesh`
- `MeshPointChart`
- `MeshVector`
- `MeshVectorInsertMode`
- `PreallocationGraph`

Use `MeshDM` when you want one object to manage a mesh workflow end to end. Use
the lower-level `Sieve`, `Atlas`, `Section`, `Coordinates`, and `LabelSet` APIs
when you need direct control over topology and data layout.

The facade includes DMPlex-style point-chart, cone/support, height/depth, and
transitive-closure queries. `create_section_from_depth()` builds a named local
section from per-depth DOF counts, with a label-restricted variant for field
support. Local vectors created by `create_local_vector()` support oriented
closure gather and insert/add operations through `get_local_vector_closure()`
and `set_local_vector_closure()`.

## 17. Algorithms Overview

The `algs` module re-exports the main mesh algorithms:

- `meshgen`: mesh generation helpers.
- `interpolate`: topology interpolation.
- `extrude`: boundary and mesh extrusion workflows.
- `boundary`: boundary classification and label creation.
- `submesh`: submesh extraction.
- `transform`: coordinate transforms.
- `field_transfer`: nearest-point, nearest-cell, refinement-map, and
  label-based transfers.
- `renumber`: point renumbering and stratified permutations.
- `rcm`: reverse Cuthill-McKee ordering, including distributed paths.
- `reduction`: reductions over mesh data.
- `wire`: wire encoding utilities for communication.

Use the examples directory for complete workflows:

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

## 18. WGPU Storage

With `--features wgpu`, `data::WgpuStorage<V>` is available for `V:
bytemuck::Pod + bytemuck::Zeroable + Send + Sync + 'static`.

The WGPU backend implements the lower-level slice storage operations used by
sections: GPU buffer allocation, host staging reads/writes, forward copy,
temporary-buffer copy for overlapping ranges, and a reverse-copy compute path
for `Polarity::Reverse`.

This backend is optional and lower-level than normal `VecStorage` use. Most
code should remain generic over `Section<V, S>` and `Storage<V>`, using
`VecStorage` unless a workflow explicitly needs GPU-backed storage.

## 18.1 CUDA Execution Plans

With `--features cuda`, the `accelerator` module exposes explicit device
buffers and immutable execution plans rather than a CUDA-backed `Section`.
`DeviceFvmOperator` combines a `DeviceFvmPlan`, `FiniteVolumeMetadata`, scheme
settings, standard boundary coefficients, and least-squares weights.
`DeviceFvmState::upload_components` uses component-major storage and retains
all flux, gradient, source, and residual workspaces across evaluations.

Use `DeviceReduction` for resident sum, dot, L2, and maximum-absolute-value
reductions. `DeviceCsrMatrix` validates symbolic/global CSR patterns and offers
CPU SpMV; enable `cuda-cusparse` for CUDA SpMV. CUDA stream events are exposed
through `CudaBackend::{record_event,wait_event,synchronize_event,elapsed_ms}`.

## 19. Error Model

Most public operations return `Result<_, MeshSieveError>`.

Common error families:

- Point and topology: invalid point IDs, missing points, duplicate points,
  cyclic topology, invalid strata.
- Atlas and section: zero-length slices, missing atlas points, slice length
  mismatches, stale scatter plans, changed atlas lengths.
- Constraints: constraint index out of bounds.
- Tags and mixed sections: missing section names or scalar type mismatch.
- Geometry: invalid dimensions and geometry validation failures.
- Overlap: non-bipartite links, rank mismatches, duplicate overlap edges, empty
  parts when that check is enabled, strict mirror mismatches.
- Communication and I/O: backend-specific communication, parsing, and format
  errors.

Use the exact error variants in `mesh_error.rs` when matching errors in tests.
Deprecated compatibility variants may still exist; new code should match the
current non-deprecated variants.

## 20. Testing Recommendations

At minimum:

```sh
cargo test
cargo test --features strict-invariants
```

Useful feature lanes:

```sh
cargo test --features rayon
cargo test --features cgns
cargo test --features wgpu
cargo test --features mpi-support
```

For API-level tests, cover:

- atlas and section invariants after insertion, removal, scatter, and resize
- missing-point and length-mismatch errors
- deterministic traversal or sorted traversal where output order matters
- label propagation through closure/star
- overlap mirror validation under `strict-invariants`
- serial versus Rayon or MPI parity where feature-gated paths are used

## 21. Quick Reference

| Area | Main APIs |
| --- | --- |
| Points | `PointId::new`, `PointId::get` |
| Topology | `Sieve`, `InMemorySieve`, `InMemoryOrientedSieve`, `MeshSieve`, `FrozenSieveCsr` |
| Traversal | `cone`, `support`, `closure_iter`, `star_iter`, `points_chart_order` |
| Labels | `LabelSet`, `stratum_points`, `stratum_values`, label propagation helpers |
| Data layout | `Atlas`, `Section`, `CpuSection`, `VecStorage`, `Storage` |
| Layout mutation | `with_atlas_mut`, `with_atlas_resize`, `ResizePolicy` |
| Coordinates | `Coordinates`, `HighOrderCoordinates`, `MeshVelocity`, `CoordinateDM` |
| Discretization | `Discretization`, `FieldDiscretization`, `DiscretizationMetadata`, `RegionKey` |
| DOF layouts | `DofLayout`, `build_layout_with`, section layout helpers |
| Constraints | `ConstrainedSection`, `ConstraintSet`, hanging-node constraints |
| Multi-field data | `FieldSection`, `MultiSection` |
| Mixed scalar data | `MixedSectionStore`, `TaggedSection`, `TaggedSectionBuffer`, `ScalarType` |
| Refinement workflow | `Bundle`, `SliceReducer`, `AverageReducer`, `Polarity` |
| Overlap | `Overlap`, `Remote`, `CopyDelta`, `AddDelta`, `ZeroDelta` |
| Communication | `Communicator`, `NoComm`, `RayonComm`, `MpiComm` |
| Distribution | `PointSF`, `distribute_topology`, `distribute_section`, `distribute_with_overlap` |
| Completion | `complete_sieve`, `complete_section`, `complete_stack` |
| I/O | `MeshData`, `MeshBundle`, `SieveSectionReader`, `SieveSectionWriter` |
| Facade | `MeshDM`, `MeshDMBuilder`, `MeshDMOptions` |

## 22. Additional Documentation

- [README.md](README.md)
- [Physics_Guide.md](Physics_Guide.md)
- [docs/topology-api.md](docs/topology-api.md)
- [docs/fe-setup.md](docs/fe-setup.md)
- [docs/geometry-quality.md](docs/geometry-quality.md)
- [docs/io-format-capabilities.md](docs/io-format-capabilities.md)
- [docs/hdf5-layout.md](docs/hdf5-layout.md)
- [docs/exodus.md](docs/exodus.md)
- [docs/coastal-mesh-labels.md](docs/coastal-mesh-labels.md)
