# Physics Guide for mesh-sieve

This guide explains how to use `mesh-sieve` as a foundation for scientific and
engineering codes. It is intentionally written at the modeling and workflow
level. For exact type signatures, see [API_Guide.md](API_Guide.md) and the
Rust API docs.

`mesh-sieve` is not a complete solver framework. It is a mesh, field, geometry,
and data-movement library that helps you build solver infrastructure for
finite-element, finite-volume, discontinuous Galerkin, domain-decomposed, and
moving-mesh workflows.

## 1. The Central Idea

Most physics solvers repeatedly answer the same questions:

- What topological entities exist: cells, faces, edges, vertices?
- Which entities touch each other?
- Where are the unknowns stored?
- How do local contributions move into a global vector or residual?
- Which points are boundary, material, wet/dry, owned, or ghosted?
- How do fields move across refinement levels or process boundaries?

`mesh-sieve` separates those concerns:

| Scientific concept | Library concept | What it gives you |
| --- | --- | --- |
| Mesh entities | `PointId` | A uniform handle for cells, faces, edges, vertices, and DOFs. |
| Incidence and adjacency | `Sieve` | Cone/support/closure/star traversal independent of element shape. |
| Field layout | `Atlas` | Per-point DOF counts and contiguous offsets. |
| Field values | `Section<V, S>` | Values stored over an atlas, usually with `VecStorage`. |
| Regions and boundaries | `LabelSet` | Named integer tags for physics decisions. |
| Coordinates | `Coordinates` | Geometry attached to mesh points, including high-order coordinates. |
| Element metadata | `DiscretizationMetadata` | Basis and quadrature choices by field and region. |
| Local-to-global structure | closure and assembly helpers | DMPlex-style closure ordering, DOF maps, and preallocation. |
| Refinement transfer | `Stack` and `Bundle` | Push values down or assemble them back up. |
| Ghost exchange | `Overlap`, `PointSF`, completion | Distributed topology and data movement. |
| Solver facade | `MeshDM` | A higher-level object for mesh, labels, sections, distribution, and numbering. |

The practical design rule is simple: keep physics decisions in labels,
metadata, and field sections; keep mesh connectivity in sieves; let closure,
assembly, overlap, and distribution helpers move data through those structures.

## 2. Choosing the Right Workflow

Use this as a first decision map.

| Problem type | Recommended path |
| --- | --- |
| Linear FEM or scalar elliptic PDEs | `Coordinates`, `DiscretizationMetadata`, FE helpers, closure-aware assembly. |
| Cell-centered finite volume | `FvmInputs`, `FluxStencil`, FV geometry, convective/diffusive flux assembly. |
| DG or trace-based methods | Sieve closure/star traversal, oriented closure extraction, custom trace transforms. |
| Multiphysics fields | `MultiSection`, named sections, per-field discretization metadata, labels by region. |
| Boundary-condition-heavy models | `LabelSet`, coastal labels, constrained sections, FV boundary policies. |
| Adaptive or multilevel methods | `Stack`, `Bundle`, refinement/coarsening helpers, field transfer helpers. |
| Moving mesh or ALE-style workflows | `Coordinates`, `MeshVelocity`, coordinate transforms, geometry checks. |
| Parallel domain decomposition | `Overlap`, `PointSF`, completion, MPI/Rayon communicators. |
| Mesh import/export workflows | `MeshData`, `MeshBundle`, Gmsh/VTK/Exodus/XDMF/HDF5/PETSc HDF5 readers and writers. |

## 3. Geometry Comes First

Before assembling equations, make the geometry explicit and check it.

Use:

- `Coordinates<f64, S>` for point coordinates.
- `HighOrderCoordinates` when element geometry is curved or higher-order.
- `geometry::metrics` for volumes, normals, maps, Jacobians, and reference-to-physical transforms.
- `geometry::quality` for cell quality and validation.
- `geometry::locate` for point location, interpolation at points, and periodic localization.
- `geometry::fvm` for FV face metrics.

For engineering work, geometry checks should be part of ingestion and regression
tests. A bad Jacobian, inverted cell, or inconsistent face normal will usually
corrupt every later physical quantity. The library exposes geometry failures as
`MeshSieveError` rather than silently accepting invalid data.

Typical geometry pipeline:

1. Read or generate a mesh into `MeshData`.
2. Confirm `coordinates` and `cell_types` are present.
3. Validate supported `CellType` values for your discretization.
4. Run cell quality or metric checks on representative meshes.
5. Build FE element geometry or FV face/cell metrics once, then reuse them in
   solver loops.

See [docs/geometry-quality.md](docs/geometry-quality.md) for geometry checks.

## 4. Finite Element Workflows

Finite-element codes usually need four layers:

1. Topology: cells and their closure.
2. Coordinates: physical geometry for vertices or high-order geometry nodes.
3. Discretization metadata: basis and quadrature.
4. Assembly maps: local closure DOFs to global rows/columns.

Current high-level FE entry points include:

- `physics::fe::evaluate_reference_element`
- `physics::fe::integrate_reference_scalar`
- `physics::fe::assemble_element_matrices`
- `physics::fe::assemble_element_matrices_from_closure`
- `physics::fe::extract_element_closure`
- `physics::fe::extract_oriented_element_closure`
- `discretization::runtime::runtime_from_metadata`
- `discretization::runtime::tabulate_element`
- `discretization::runtime::local_stiffness_matrix`
- `discretization::runtime::local_load_vector`
- `discretization::runtime::assemble_local_matrix`
- `discretization::runtime::assemble_local_vector`
- `discretization::runtime::closure_dof_map`
- `algs::assembly::preallocation_csr_from_closure`

### Basis and Quadrature

`DiscretizationMetadata` describes basis and quadrature. The runtime supports a
PetscFE-like matrix of common cell families:

- segments
- triangles
- quadrilaterals
- tetrahedra
- hexahedra
- prisms
- pyramids

Use `supported_discretizations()` and `discretization_capability(cell_type)` to
check what a workflow can support before you assemble a problem.

Example metadata:

```rust
use mesh_sieve::data::discretization::DiscretizationMetadata;

let metadata = DiscretizationMetadata::new("lagrange", "gauss2")
    .with_basis_metadata(2, [] as [&str; 0]);
```

### Element Assembly

For simple element kernels, assemble from cell nodes:

```rust
use mesh_sieve::physics::fe::assemble_element_matrices;
use mesh_sieve::topology::cell_type::CellType;

let element = assemble_element_matrices(
    &coordinates,
    CellType::Triangle,
    &cell_nodes,
    &metadata,
    |_| 1.0,
)?;
```

For DMPlex-style workflows, prefer closure-aware assembly:

```rust
use mesh_sieve::data::closure::ClosureOrder;
use mesh_sieve::physics::fe::assemble_element_matrices_from_closure;
use mesh_sieve::topology::cell_type::CellType;

let element = assemble_element_matrices_from_closure(
    &topology,
    &coordinates,
    CellType::Triangle,
    cell,
    topology_version,
    &ClosureOrder::BreadthFirstDmpLex,
    &metadata,
    |_| 1.0,
)?;
```

Closure-aware assembly is the better mental model for production solvers
because it keeps element shape, closure ordering, orientation, and global
numbering explicit.

### Practical FE Pattern

For a scalar Poisson-like problem:

1. Put unknowns on vertices or other chosen closure points with an `Atlas`.
2. Store the field in a `Section<f64, VecStorage<f64>>`.
3. Describe basis and quadrature with `DiscretizationMetadata`.
4. For each cell, gather closure coordinates and closure field values.
5. Compute local stiffness and load.
6. Scatter into a matrix/vector section or external sparse matrix.
7. Apply constraints or boundary conditions using labels and constrained sections.

The example [examples/poisson_segment.rs](examples/poisson_segment.rs) shows
this pattern on a small interval mesh.

## 5. Finite Volume Workflows

Finite-volume codes are organized around conservative face fluxes. The
`physics::fvm` module is the intended high-level entry point for FV users.

Current FV concepts:

- `FluxStencil`: face, owner cell, optional neighbor cell.
- `FvmFaceLoops`: internal and boundary face loops.
- `FvmInputs`: stencils plus cell and face geometry.
- `PackedFvmInputs` and `FvmPackedCache`: reusable packed hot-loop data.
- `FvmFieldSections`: cell scalar, cell gradient, and face mass flux maps.
- `FvmSchemeSettings`: convective, reconstruction, and diffusion settings.
- `FvBoundaryPolicy`: maps boundary branches to boundary conditions.
- `FvmAssemblyOutput`: convective, diffusive, and total residual/source data.

### Building FV Inputs

The usual FV setup is:

1. Identify faces.
2. Classify each face as internal or boundary with `classify_face_loops`.
3. Build cell geometry and face geometry.
4. Construct `FvmInputs`.
5. Optionally build or reuse `FvmPackedCache` for hot loops.

The library uses support relationships to classify faces. A face with one
supporting cell is a boundary face; a face with two supporting cells is an
internal face; more than two is a non-manifold error.

### Flux Assembly

High-level assembly functions:

- `assemble_fv_system`
- `assemble_convective_fluxes`
- `assemble_convective_fluxes_with_reconstruction`
- `assemble_convective_fluxes_masked`
- `assemble_diffusive_fluxes`
- `assemble_diffusive_fluxes_with_hooks`

Supported modeling controls include:

- `ConvectiveScheme::Upwind`
- `ConvectiveScheme::Central`
- `ConvectiveScheme::BoundedLinear`
- `ConvectiveScheme::BlendUpwindCentral`
- `ConvectiveScheme::HighResolution`
- `ReconstructionSettings`
- `SlopeLimiterFamily`
- `DiffusionSettings`
- `NonOrthogonalCorrectionMode`
- `BoundaryCondition::{Dirichlet, Neumann, Robin}`

For a conservative update, interpret the output residual map as per-cell net
flux. Internal face contributions are applied with opposite signs to owner and
neighbor cells.

### Boundary Branches

Boundary labels let science code choose physical closure behavior without
hard-coding mesh IDs. The FV module understands coastal-oriented branches:

- `Open`
- `Inflow`
- `Outflow`
- `Tidal`
- `Bed`
- `FreeSurface`

Use:

- `boundary_branch_for_face`
- `boundary_branch_for_face_checked`
- `flux_activity_mask_from_wet_dry`

This is especially useful for ocean, coastal, shallow-water, atmospheric, or
hydraulic models where wet/dry state and boundary role affect flux behavior.

### Extending FV Physics

Use hooks when the base convective/diffusive forms are not enough:

- `FvmPhysicsHooks`
- `FvmSourceFluxHook`

Hooks are a good fit for turbulence closures, free-surface terms, source terms,
or application-specific face flux corrections. Keep these hooks physically
meaningful and deterministic; use labels and metadata to select where they
apply.

## 6. DG and Trace-Based Methods

DG methods often need both cell-local values and oriented traces on faces.
`mesh-sieve` supports this style through topology and closure tools rather than
a single DG-specific facade.

Useful building blocks:

- `Sieve` closure/star traversal for cell and face neighborhoods.
- `OrientedSieve` for orientation-aware topology.
- `extract_oriented_element_closure` for closure values with orientation
  symmetries.
- `SectionSym` implementations for orientation-specific value transforms.
- `Polarity` and `SliceDelta` for simple forward/reverse slice transforms.
- `LabelSet` to select interfaces, walls, materials, and boundary types.

Recommended DG pattern:

1. Store per-cell and trace fields in separate sections or fields.
2. Use face support to identify left/right cells.
3. Extract oriented cell closure data for each side.
4. Compute numerical fluxes using labels and boundary policies.
5. Accumulate residuals back into cell-local sections or a global vector.

This keeps numerical flux code independent from the concrete mesh storage.

## 7. Boundary Conditions and Constraints

Boundary conditions should be represented as data, not as scattered conditionals
over point IDs.

Use:

- `LabelSet` for boundary, material, region, and role tags.
- `ConstrainedSection` for fixed DOF values.
- `ConstraintSet` and `apply_constraints_to_section`.
- `data::bc` helpers for Dirichlet and coastal boundary assembly.
- FV boundary policies for face-flux closure.

Practical approach:

1. Import or assign labels once.
2. Convert labels into constraints or boundary policies.
3. Apply constraints after refinement, assembly, or field transfer.
4. Test label propagation over closure/star when derived regions are needed.

This makes boundary behavior reproducible and easier to audit.

## 8. Multiphysics and Multiple Fields

Many scientific codes solve several fields on different supports: velocity on
faces, pressure in cells, temperature on vertices, tracers in cells, and so on.

Use:

- `MultiSection` for multiple named fields with field offsets.
- `FieldSection` for each field.
- `MixedSectionStore` when fields have different scalar types.
- `Discretization` and `FieldDiscretization` for per-field metadata.
- `RegionKey::label` and `RegionKey::cell_type` to attach metadata to physical
  regions.

High-level model:

1. Define each field by where it lives and how many DOFs it has.
2. Attach discretization metadata by field and region.
3. Keep physical regions in labels.
4. Build local kernels around field names and region metadata rather than raw
   offsets.

This is the path to extensible multiphysics without losing performance in the
core sections.

## 9. Moving Meshes and ALE-Style Problems

Moving-mesh problems need coordinate data to be as structured as solution data.

Use:

- `Coordinates` for current mesh coordinates.
- `MeshVelocity` for velocity fields aligned with coordinate dimension.
- `Coordinates::advance_with_velocity` for simple coordinate updates.
- `algs::transform` for coordinate transforms.
- `geometry::quality` to validate the mesh after motion.
- `geometry::locate` for remapping or point tracking.

Recommended loop:

1. Compute or update mesh velocity.
2. Advance coordinates.
3. Validate geometry and quality.
4. Rebuild geometry metrics.
5. Transfer or update fields if topology or coordinates changed.

Do not treat mesh motion as an unrelated side channel. Keeping coordinates in
sections allows the same validation, transfer, and I/O machinery to apply.

## 10. Adaptivity, Refinement, and Field Transfer

Adaptive workflows are about preserving meaning while topology changes.

Relevant APIs:

- `adapt`
- `topology::adapt`
- `topology::refine`
- `data::refine`
- `Stack`
- `Bundle`
- `field_transfer`
- `hanging_node_constraints`

Use `Bundle` when data moves through a base-to-cap relation:

- `refine` pushes values from coarse/base points to refined/cap points.
- `assemble` or `assemble_with` pulls values back with a reducer.
- `AverageReducer` is available; custom reducers can encode sum, min/max, or
  conservative accumulation semantics.

Use `field_transfer` helpers when data moves between related but not strictly
hierarchical meshes:

- nearest point
- nearest cell centroid
- refinement map
- shared labels

For constrained adaptivity, apply constraints after transfer or assembly. For
hanging-node methods, use the hanging-node constraint helpers to express linear
relations instead of directly overwriting field values.

## 11. Distributed Physics

Distributed physics is mostly a question of ownership, ghost state, and
consistent exchange.

High-level tools:

- `Overlap`: records which local points are shared with neighbor ranks.
- `PointSF`: PETSc-SF-style root/leaf distribution maps.
- `complete_sieve`: exchange missing topology across overlap.
- `complete_section`: exchange section data across overlap.
- `complete_stack`: exchange vertical stack data.
- `distribute_topology`, `distribute_section`, `distribute_labels`,
  `distribute_field`: distribution helpers.
- `NoComm`, `RayonComm`, `MpiComm`: communication backends.

Workflow:

1. Partition or provide a partition.
2. Create ownership and point-SF data.
3. Distribute topology, labels, coordinates, and sections.
4. Build overlap for ghosted points.
5. Complete topology and field data before kernels that need ghost closure.
6. Assemble local residuals.
7. Exchange or reduce shared contributions with the appropriate delta.

Use `RayonComm` for local multi-rank style testing and `MpiComm` for real MPI
runs. MPI support is behind the `mpi-support` feature.

## 12. I/O and Reproducible Science

`MeshData` is the common in-memory container for imported or generated meshes.
It carries:

- topology
- coordinates
- named sections
- mixed sections
- labels
- cell types
- discretization metadata

Use `MeshBundle` for multi-mesh workflows such as partitions, time slices, or
related meshes that need synchronized labels or coordinates.

Relevant format support includes Gmsh, VTK, PLY, Fluent, Exodus II, XDMF, HDF5,
PETSc HDF5, partitioned metadata, and experimental CGNS/HDF5 reading with the
`cgns` feature.

For reproducibility:

- Prefer labels and metadata over implicit conventions.
- Use deterministic traversal or sorted outputs when comparing files.
- Keep one CI lane with `strict-invariants`.
- Store discretization metadata alongside mesh data when exporting workflows.

See [docs/io-format-capabilities.md](docs/io-format-capabilities.md).

## 13. Solver Integration Strategy

`mesh-sieve` deliberately stops short of owning your linear solver or nonlinear
time integrator. The recommended boundary is:

- Use `mesh-sieve` to manage topology, geometry, labels, field layout, local
  closure extraction, preallocation, I/O, and distributed exchange.
- Use your solver stack to own sparse matrix storage, linear solves,
  nonlinear iterations, time integration, and physical model coupling.

Good integration points:

- Build CSR preallocation with closure-based assembly helpers.
- Export local element maps through `DofMap` and global maps.
- Use `MeshDM` when a DMPlex-like facade improves solver code readability.
- Keep solver vectors mirrored as sections during setup/debug, then bridge to
  external dense or sparse storage for production solves.

## 14. Modeling Checklist

Before writing kernels, answer these questions:

1. What are my topological entities and how are they connected?
2. Which `CellType` values do I support?
3. Where does each unknown live: vertex, edge, face, cell, closure, or trace?
4. Which labels select boundaries, materials, wet/dry state, or equations?
5. Which basis, quadrature, reconstruction, limiter, or flux scheme applies by region?
6. What geometry metrics must be valid before assembly?
7. What needs to be ghosted in parallel: topology, coordinates, labels, fields, or all of them?
8. How will values transfer after refinement, coarsening, remeshing, or motion?
9. Which invariants should be checked in tests?

If these answers are represented as mesh data, labels, sections, and metadata,
your physics code will be easier to test and easier to move between serial,
threaded, and MPI workflows.

## 15. Performance Guidance

Start with clear data representation, then optimize the loops that profiling
identifies.

Useful practices:

- Build geometry and face/cell metrics once per mesh state.
- Use closure indices or DOF maps rather than recomputing local ordering inside
  every kernel.
- Use `PackedFvmInputs` or `FvmPackedCache` for repeated FV face loops.
- Use point-only traversal such as `cone_points` and `support_points` when
  payloads are not needed.
- Use sorted or deterministic paths only where reproducibility matters.
- Use `fast-hash` where iteration order is not part of the result.
- Avoid mutating atlas layout inside hot loops.
- Keep GPU/WGPU use for workflows that can amortize transfer and staging costs.

The library provides low-level controls, but most scientific users should first
optimize at the level of geometry reuse, closure reuse, packed FV loops, and
communication volume.

## 16. Where to Look Next

- [API_Guide.md](API_Guide.md): current API map.
- [docs/fe-setup.md](docs/fe-setup.md): focused FE setup and assembly note.
- [docs/geometry-quality.md](docs/geometry-quality.md): geometry checks.
- [docs/coastal-mesh-labels.md](docs/coastal-mesh-labels.md): coastal labels and boundary semantics.
- [examples/poisson_segment.rs](examples/poisson_segment.rs): small FE assembly example.
- [examples/poisson_quad.rs](examples/poisson_quad.rs): quadrilateral Poisson example.
- [examples/fvm_distributed_qa.rs](examples/fvm_distributed_qa.rs): distributed FV QA example.
- [examples/distribute_with_overlap.rs](examples/distribute_with_overlap.rs): overlap distribution example.

The key mental model is: use topology for relationships, labels for physical
meaning, sections for fields, metadata for numerical choices, and overlap for
distributed state. That separation is what lets the same mesh representation
support FE, FV, DG-style, adaptive, and parallel scientific workflows.
