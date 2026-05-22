# Coastal Mesh Label Schema

This document recommends a consistent metadata schema for coastal/ocean meshes when using `mesh-sieve` labels.

## Canonical label names

- `boundary_class`
  - `1` = free surface
  - `2` = bed
  - `3` = open boundary
- `boundary_role` (used for points/entities that are also `boundary_class=3`)
  - `11` = inflow
  - `12` = outflow
  - `13` = tidal
- `vertical_layer`
  - integer index for sigma/z layer (for example: `0..nz-1`)
- `wet_dry_mask` (optional)
  - `1` = wet
  - `0` = dry

These constants are available in `src/topology/coastal.rs`.

## Recommended usage

1. Assign exactly one `boundary_class` value to each boundary entity where classification is required.
2. For `boundary_class=3` entities, add one `boundary_role` value.
3. Assign `vertical_layer` to points/entities where vertical indexing is required by your discretization.
4. Optionally add `wet_dry_mask` for inundation/exposure workflows.

## Query helpers

`CoastalLabelQueries` provides convenience methods on `LabelSet`:

- `free_surface_points()`
- `bed_points()`
- `open_boundary_points()`
- `inflow_points()`, `outflow_points()`, `tidal_points()`
- `vertical_layer_points(layer_index)`
- `wet_points()`, `dry_points()`

## Consistency validation

Use `validate_coastal_metadata(...)` with `CoastalValidationOptions`:

- open-boundary role checks (`boundary_role` should only appear on open boundaries)
- optional complete boundary partition checks against an expected boundary set
- optional complete vertical coverage checks against an expected point set

## Example

```rust
use mesh_sieve::topology::{
    LabelSet, BOUNDARY_CLASS_LABEL, BOUNDARY_ROLE_LABEL, VERTICAL_LAYER_LABEL,
    BoundaryClass, OpenBoundaryRole, CoastalValidationOptions, validate_coastal_metadata,
};

let mut labels = LabelSet::new();
let p = mesh_sieve::topology::point::PointId::new(101)?;
labels.set_label(p, BOUNDARY_CLASS_LABEL, BoundaryClass::Open.code());
labels.set_label(p, BOUNDARY_ROLE_LABEL, OpenBoundaryRole::Tidal.code());
labels.set_label(p, VERTICAL_LAYER_LABEL, 0);

let opts = CoastalValidationOptions::default();
validate_coastal_metadata(&labels, None, None, opts)?;
# Ok::<(), mesh_sieve::topology::CoastalMetadataError>(())
```
