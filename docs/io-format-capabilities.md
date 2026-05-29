# Mesh I/O format capabilities

This page summarizes mesh-sieve's DMPlex-class import/export coverage.  The
rows describe the topology, coordinate, label, section, higher-order, and
mixed-dimensional features that are represented by the public readers/writers in
`src/io/`.

| Format | Reader | Writer | Topology | Coordinates | Labels / sets | Sections / fields | Higher-order nodes | Mixed-dimensional meshes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gmsh `.msh` | `GmshReader` | `GmshWriter` | ASCII v2.2 plus ASCII/binary v4.x points, lines, triangles, quads, tets, hexes, prisms, pyramids, and mixed element blocks. | 1D/2D/3D coordinates with inferred mesh dimension. | Physical names, physical/entity tags, element type tags, and extra element tags are preserved as labels. | Element data is imported into mixed sections where practical. | Common curved Gmsh element variants are accepted; base topology is stored in `cell_types` and extra geometry is stored as high-order coordinates. | Supported, with an option to reject mixed topological dimensions during validation. |
| Exodus II / NetCDF-HDF5 | `ExodusReader` / `ExodusIiReader` | `ExodusWriter` | Common linear blocks: point, bar, tri, quad, tet, hex, wedge/prism, and pyramid. | Coordinate arrays are read and written with their stored coordinate dimension. | Element blocks, node sets, side sets, and related Exodus IDs are mapped to labels. | Nodal and elemental variables are represented as sections. | Not currently expanded beyond supported linear block types. | Multiple element blocks can carry different cell types; mixed-dimensional validation is caller-controlled. |
| PETSc DMPlex HDF5 | `PetscHdf5Reader` | `PetscHdf5Writer` | DMPlex cone-size/cone/permutation topology with cell-type sections. | Coordinate DMs/sections are represented through the DMPlex HDF5 section/vector layout. | DMPlex labels and strata are preserved. | DMPlex sections/vectors are imported and exported by name. | Stored if represented by user sections; no special curved-cell interpretation is applied. | Supported when encoded by the DMPlex topology and labels. |
| CGNS / HDF5 | `CgnsReader` behind `--features cgns` | Not yet | Unstructured `Elements_t` sections for common fixed-size CGNS element codes and `MIXED` connectivity. | `GridCoordinates_t` / `GridCoordinates` arrays `CoordinateX`, `CoordinateY`, and `CoordinateZ`. | `ZoneBC_t` / `BC_t` `PointList` and `PointRange` entries become `cgns:bc` and `cgns:bc:<name>` labels. | `FlowSolution_t` scalar/vector data arrays are imported when their cardinality matches vertices or imported cells. | Curved CGNS element families are not yet expanded into high-order coordinate storage. | Mixed element sections are supported; mixed-dimensional meshes are represented through per-cell `cell_types`. |
| Fluent `.msh` | `FluentReader` | Not yet | Practical ASCII subset: vertex coordinate sections and explicit cell connectivity sections; compact fixtures are also accepted. | 2D/3D coordinates are imported into coordinate sections. | Boundary face zones are not yet mapped to labels in the current subset. | Field/solution import is not yet implemented. | Not currently supported. | Mixed polygonal cell arity is represented through `cell_types`. |
| PLY | `PlyReader` | Not yet | ASCII polygon surface meshes with `vertex` and `face` elements. | `x y z` vertex properties are imported as 3-component coordinates. | PLY comments/properties are not converted to labels. | Additional PLY properties are not converted to sections. | Not currently supported. | Surface faces can mix triangles, quads, and polygons. |
| VTK / XDMF / internal HDF5 | `VtkReader`, `XdmfReader`, HDF5 helpers | Writers available for supported subsets | Project-specific structured/unstructured subsets used by existing tests. | Supported according to each backend's layout. | Supported where the backend has a label representation. | Supported where the backend has a section/vector representation. | Backend-specific. | Backend-specific. |

## Notes for DMPlex parity

- The DMPlex ecosystem exposes a broad I/O surface. mesh-sieve now has reader
  entry points for the common DMPlex import formats: Gmsh, Exodus II, CGNS,
  Fluent, PLY, and PETSc HDF5.
- CGNS and Fluent remain intentionally conservative subsets because both formats
  have many dialects. Unsupported encodings fail with parse errors rather than
  silently dropping topology.
- Round-trip coverage exists for writer-backed formats. Reader-only formats are
  covered by fixture import tests until exporters are added.
