# HDF5 mesh layouts

This repository supports two HDF5 layouts:

1. `src/io/hdf5.rs` keeps the original compact Mesh Sieve layout used by older
   XDMF/HDF5 round-trips.
2. `src/io/petsc_hdf5.rs` writes and reads a PETSc DMPlex HDF5 v3-style layout
   intended for interoperability with PETSc/Firedrake/FEniCSx-like workflows.

## PETSc DMPlex HDF5 v3-compatible layout

The PETSc-compatible writer stores a version marker at the file root and then
nests mesh topology, labels, sections, and vectors beneath a named topology:

- `/dmplex_storage_version`
  - `i32[1]`, currently `3`.
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
