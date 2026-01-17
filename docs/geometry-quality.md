# Geometry Quality Utilities

This document describes the geometry quality utilities in `mesh-sieve`, the
expected coordinate layouts, and examples of optional validation in I/O and
refinement flows.

## Coordinate layout

Coordinate data is stored in a `data::coordinates::Coordinates` section, which
wraps a `Section` with a fixed slice length per point. The quality routines
accept **dimension 2 or 3** and interpret the slices as:

- `dim = 2`: `(x, y)`
- `dim = 3`: `(x, y, z)`

For surface meshes embedded in 3D, project or pre-process your coordinates so
that the quality checks can evaluate a consistent plane (the default quality
checks operate in the XY plane).

## Supported cell layouts

The quality helpers support the following cell types and vertex ordering:

| Cell type | Vertex ordering |
| --- | --- |
| Triangle | `[v0, v1, v2]` counter-clockwise in XY |
| Quadrilateral | `[v0, v1, v2, v3]` counter-clockwise in XY |
| Tetrahedron | `[v0, v1, v2, v3]` |
| Hexahedron | `[v0, v1, v2, v3, v4, v5, v6, v7]` with bottom face `[0,1,2,3]` and top `[4,5,6,7]` |
| Prism | `[v0, v1, v2, v3, v4, v5]` with bottom triangle `[0,1,2]` and top `[3,4,5]` |
| Pyramid | `[v0, v1, v2, v3, v4]` with base quad `[0,1,2,3]` and apex `v4` |

These layouts match the expected connectivity ordering used by the Gmsh reader
for linear elements and the refinement templates in `topology::refine`.

## Examples

### Compute quality metrics for a triangle

```rust
use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::coordinates::Coordinates;
use mesh_sieve::data::storage::VecStorage;
use mesh_sieve::geometry::quality::cell_quality_from_section;
use mesh_sieve::topology::cell_type::CellType;
use mesh_sieve::topology::point::PointId;

let mut atlas = Atlas::default();
let v0 = PointId::new(1)?;
let v1 = PointId::new(2)?;
let v2 = PointId::new(3)?;

atlas.try_insert(v0, 2)?;
atlas.try_insert(v1, 2)?;
atlas.try_insert(v2, 2)?;

let mut coords = Coordinates::<f64, VecStorage<f64>>::try_new(2, atlas)?;
coords.try_restrict_mut(v0)?.copy_from_slice(&[0.0, 0.0]);
coords.try_restrict_mut(v1)?.copy_from_slice(&[1.0, 0.0]);
coords.try_restrict_mut(v2)?.copy_from_slice(&[0.0, 1.0]);

let q = cell_quality_from_section(CellType::Triangle, &[v0, v1, v2], &coords)?;
println!("aspect ratio = {}", q.aspect_ratio);
# Ok::<(), mesh_sieve::mesh_error::MeshSieveError>(())
```

### Validate geometry during Gmsh import

```rust
use mesh_sieve::io::gmsh::{GmshReader, GmshReadOptions};

let reader = GmshReader::default();
let opts = GmshReadOptions { check_geometry: true };
let mesh = reader.read_with_options(std::fs::File::open("mesh.msh")?, opts)?;
```

### Validate geometry before refinement

```rust
use mesh_sieve::topology::refine::{refine_mesh_with_options, RefineOptions};

let opts = RefineOptions { check_geometry: true };
let refined = refine_mesh_with_options(
    &mut coarse_sieve,
    &cell_types,
    Some(&coordinates),
    opts,
)?;
```
