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

## Finite-volume boundary mapping

For FV boundary assembly (`src/physics/fvm.rs`), coastal labels map to boundary branches as:

- `boundary_class=1` (`FreeSurface`) → `FvBoundaryBranch::FreeSurface`
- `boundary_class=2` (`Bed`) → `FvBoundaryBranch::Bed`
- `boundary_class=3` (`Open`) + no role → `FvBoundaryBranch::Open`
- `boundary_class=3` + `boundary_role=11` (`Inflow`) → `FvBoundaryBranch::Inflow`
- `boundary_class=3` + `boundary_role=12` (`Outflow`) → `FvBoundaryBranch::Outflow`
- `boundary_class=3` + `boundary_role=13` (`Tidal`) → `FvBoundaryBranch::Tidal`

Use `data::bc::coastal_boundary_face_sets(...)` to fetch per-class/role boundary face IDs from labels and a boundary-face iterator.

Use `data::bc::map_coastal_boundary_conditions(...)` to produce per-face `BoundaryCondition` values from a closure over `FvBoundaryBranch`.

### Example: coastal labels + FV assembly

```rust
use mesh_sieve::data::bc::{coastal_boundary_face_sets, map_coastal_boundary_conditions};
use mesh_sieve::physics::fvm::{
    ConvectiveScheme, assemble_convective_fluxes_masked, flux_activity_mask_from_wet_dry,
};

let face_sets = coastal_boundary_face_sets(&labels, inputs.boundary_faces().map(|s| s.face));
assert!(!face_sets.open.is_empty() || !face_sets.bed.is_empty() || !face_sets.free_surface.is_empty());

let bcs = map_coastal_boundary_conditions(&labels, inputs.boundary_faces().map(|s| s.face), |branch, _face| {
    match branch {
        mesh_sieve::physics::fvm::FvBoundaryBranch::Inflow => mesh_sieve::physics::fvm::BoundaryCondition::Dirichlet { value: 1.0 },
        mesh_sieve::physics::fvm::FvBoundaryBranch::Outflow => mesh_sieve::physics::fvm::BoundaryCondition::Neumann { gradient: 0.0 },
        mesh_sieve::physics::fvm::FvBoundaryBranch::Tidal => mesh_sieve::physics::fvm::BoundaryCondition::Dirichlet { value: 0.2 },
        mesh_sieve::physics::fvm::FvBoundaryBranch::Bed => mesh_sieve::physics::fvm::BoundaryCondition::Neumann { gradient: 0.0 },
        mesh_sieve::physics::fvm::FvBoundaryBranch::FreeSurface => mesh_sieve::physics::fvm::BoundaryCondition::Neumann { gradient: 0.0 },
        mesh_sieve::physics::fvm::FvBoundaryBranch::Open => mesh_sieve::physics::fvm::BoundaryCondition::Neumann { gradient: 0.0 },
    }
});

let wet_dry_mask = flux_activity_mask_from_wet_dry(&inputs, &labels);
let flux = assemble_convective_fluxes_masked(
    &inputs,
    &cell_scalar,
    &face_mass_flux,
    &bcs,
    ConvectiveScheme::Upwind,
    Some(&wet_dry_mask),
)?;
```

`FluxActivityMask` gates both boundary faces and near-boundary internal faces, so dry cells can suppress boundary and adjacent flux contributions.
