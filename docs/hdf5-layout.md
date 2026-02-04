# HDF5 mesh layout (PETSc/DMPlex conventions)

This project stores mesh topology and metadata in an HDF5 file using a
DMPlex-style layout so it can be consumed by PETSc-compatible tooling and
XDMF front-ends.

## Groups

- `/topology`
  - `cell_ids` (`i64`, shape = `num_cells`)
    - Original `PointId` for each cell.
  - `point_ids` (`i64`, shape = `num_points`)
    - Original `PointId` for each vertex/point.
  - `cell_types` (`i32`, shape = `num_cells`)
    - DMPlex cell type codes (see mapping below).
  - `cell_offsets` (`i64`, shape = `num_cells + 1`)
    - Offsets into the flattened `cells` connectivity array.
  - `cells` (`i64`, shape = `num_cell_connectivity`)
    - Flattened connectivity, storing `PointId` values.
  - `mixed` (`i64`, shape = `num_mixed_entries`)
    - XDMF Mixed topology encoding (`cell_code, v0, v1, ...`).
- `/geometry`
  - `coordinates` (`f64`, shape = `num_points x 3`)
    - XYZ coordinates padded to 3 entries per point.
  - `topological_dimension` (`i32`, shape = `1`)
  - `embedding_dimension` (`i32`, shape = `1`)
  - `point_ids` (`i64`, shape = `num_points`)
- `/sections/<name>`
  - `ids` (`i64`, shape = `num_entries`)
    - `PointId` values that own section data.
  - `values` (`f64`, shape = `num_entries x num_components`)
  - `num_components` (`i32`, shape = `1`)
- `/labels/<name>`
  - `ids` (`i64`, shape = `num_entries`)
  - `values` (`i32`, shape = `num_entries`)

## Cell type codes (DMPlex)

The `cell_types` array uses PETSc/DMPlex numbering:

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
file references the datasets above using `Format="HDF"` and a dataset URI such
as `mesh.h5:/geometry/coordinates`.
