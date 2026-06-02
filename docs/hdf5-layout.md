# HDF5 mesh layouts

This repository supports two HDF5 layouts:

1. `src/io/hdf5.rs` keeps the original compact Mesh Sieve layout used by older
   XDMF/HDF5 round-trips.
2. `src/io/petsc_hdf5.rs` writes and reads a PETSc DMPlex HDF5 v3-style layout
   intended for interoperability with PETSc/Firedrake/FEniCSx-like workflows.

## PETSc DMPlex HDF5 v2/v3-compatible layout

The PETSc-compatible reader accepts DMPlex storage versions `2` and `3` in
strict mode.  mesh-sieve writes version `3` (`DMPLEX_STORAGE_VERSION`) because
that is the layout used by PETSc's current DMPlex HDF5 workflow for named
topologies, DMs, sections, and vectors.  Version `2` files are treated as legacy
DMPlex-compatible input when they preserve the same topology/DM hierarchy and
permutation semantics.

The PETSc-compatible writer stores a version marker at the file root and then
nests mesh topology, labels, sections, and vectors beneath a named topology:

- `/dmplex_storage_version`
  - `i32[1]`, currently written as `3`; strict reads accept `2` and `3`.
- `/topologies/<mesh>/topology/permutation`
  - `i64[num_points]` containing the preserved global `PointId` order.
  - Cone arrays refer to points by zero-based index into this permutation, which
    allows topology, section, and vector data to be loaded separately and
    re-associated.
- `/topologies/<mesh>/topology/strata/<depth>/cone_sizes`
  - `i32[num_points_at_depth]`; one cone size per point in permutation order for
    that depth.
- `/topologies/<mesh>/topology/strata/<depth>/cones`
  - `i64[sum(cone_sizes)]`; flattened cone point indices into `permutation`.
- `/topologies/<mesh>/topology/strata/<depth>/orientations`
  - `i32[sum(cone_sizes)]`; one orientation per cone entry.
- `/topologies/<mesh>/topology/cell_types`
  - `i32[num_points]`; DMPlex cell type code for points that carry a cell type,
    or `-1` for points without one.
- `/topologies/<mesh>/labels/<label>/<value>/points`
  - `i64[num_label_points]`; sorted global `PointId`s for each integer label
    stratum.
- `/topologies/<mesh>/dms/<dm>/section/<name>/points`
  - `i64[num_section_points]`; global point IDs in the preserved mesh order.
- `/topologies/<mesh>/dms/<dm>/section/<name>/dofs`
  - `i32[num_section_points]`; number of scalar degrees of freedom for each
    listed point.
- `/topologies/<mesh>/dms/<dm>/section/<name>/offsets`
  - `i64[num_section_points]`; offsets into the matching vector dataset.
- `/topologies/<mesh>/dms/<dm>/vecs/<name>`
  - `f64[sum(dofs)]`; vector values for the named section.

The public PETSc-compatible API is exposed as `mesh_sieve::io::petsc_hdf5` and
provides `PetscHdf5Reader`, `PetscHdf5Writer`, `write_mesh_to_petsc_hdf5`, and
`read_mesh_from_petsc_hdf5`.


## PETSc parallel provenance and cross-rank validation

PETSc DMPlex can save a mesh, sections, and vectors collectively and load them
with a different communicator size.  mesh-sieve records the same workflow with
root-level rank counts and serialized point-SF migration maps:

- `/saved_rank_count`
  - `i32[1]`; number of ranks that wrote the file.
- `/loaded_rank_count`
  - `i32[1]`; number of ranks represented by the loaded/redistributed view.
- `/migration/load_sf`
  - Flattened `i64` rows `[local_point, remote_rank, remote_point, owner_rank, is_ghost]`
    describing the PETSc load SF.
- `/migration/redistribute_sf`
  - Optional flattened rows for the redistribution step when save/load rank
    counts differ.
- `/migration/section_sf`
  - Flattened rows for section/vector migration after topology movement.

`read_mesh_and_migration_provenance` validates that rank counts are non-zero,
all referenced ranks are less than `loaded_rank_count`, duplicate SF leaves are
rejected, a rank-count change has an explicit redistribution SF, and the section
SF covers every point carrying section, coordinate, or cell-type data.  The same
coherence rules are exposed on `MeshDMProvenance::validate_consistency()` for DM
setup pipelines that combine load, redistribution, and section migration maps.

The test suite keeps deterministic HDF5 fixtures for serial, partitioned, and
cross-rank DMPlex-like files.  When `petsc4py` is installed, additional tests
exercise the PETSc HDF5 viewer path so mesh-sieve-written files are opened by
PETSc and PETSc-materialized HDF5 hierarchies are read back by mesh-sieve.

## Known incompatibilities and limits

- Only DMPlex storage versions `2` and `3` are accepted in strict mode.  Use
  permissive mode only for diagnostics of newer or partially compatible files.
- The topology reader expects a `permutation` dataset and stratum cone arrays;
  PETSc files that omit the preserved point order cannot be re-associated with
  sections/vectors safely.
- Section/vector support currently covers scalar `f64` sections with explicit
  `points`, `dofs`, `offsets`, and matching `/vecs/<name>` data.
- Parallel HDF5 files are validated through serialized SF provenance metadata;
  mesh-sieve does not replay PETSc's collective MPI-IO calls internally.
- Unknown DMPlex cell-type integer codes are rejected unless mapped to a
  supported `CellType`.

## Legacy compact Mesh Sieve layout

The legacy `Hdf5Writer`/`Hdf5Reader` in `src/io/hdf5.rs` store:

- `/topology`
  - `cell_ids` (`i64`, shape = `num_cells`)
  - `point_ids` (`i64`, shape = `num_points`)
  - `cell_types` (`i32`, shape = `num_cells`)
  - `cell_offsets` (`i64`, shape = `num_cells + 1`)
  - `cells` (`i64`, shape = `num_cell_connectivity`)
  - `mixed` (`i64`, shape = `num_mixed_entries`)
- `/geometry`
  - `coordinates` (`f64`, shape = `num_points x 3`)
  - `topological_dimension` (`i32`, shape = `1`)
  - `embedding_dimension` (`i32`, shape = `1`)
  - `point_ids` (`i64`, shape = `num_points`)
- `/sections/<name>`
  - `ids` (`i64`, shape = `num_entries`)
  - `values` (`f64`, shape = `num_entries x num_components`)
  - `num_components` (`i32`, shape = `1`)
- `/labels/<name>`
  - `ids` (`i64`, shape = `num_entries`)
  - `values` (`i32`, shape = `num_entries`)

## Cell type codes (DMPlex)

| DMPlex code | CellType |
| --- | --- |
| 0 | Vertex |
| 1 | Segment |
| 2 | Triangle |
| 3 | Quadrilateral |
| 4 | Tetrahedron |
| 5 | Hexahedron |
| 6 | Prism |
| 7 | Pyramid |

## XDMF/HDF5 linkage

When the XDMF writer is configured to emit HDF5-backed `DataItem`s, the XDMF
file references datasets using `Format="HDF"` and a dataset URI such as
`mesh.h5:/geometry/coordinates` for the legacy layout. PETSc-compatible files
should reference the DMPlex hierarchy under `/topologies/<mesh>`.
