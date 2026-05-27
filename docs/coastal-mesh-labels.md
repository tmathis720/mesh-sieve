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

### Example: coastal labels → boundary policy → FV assembly

```rust
use mesh_sieve::data::bc::{coastal_boundary_face_sets, map_coastal_boundary_conditions};
use mesh_sieve::physics::fvm::{
    assemble_fv_system, BoundaryCondition, ConvectiveScheme, DiffusionSettings, FvBoundaryPolicy,
    FvBoundaryBranch, FvmFieldSections, FvmSchemeSettings, ReconstructionSettings,
    UnsupportedBoundaryBehavior,
};
use std::collections::{HashMap, HashSet};

let face_sets = coastal_boundary_face_sets(&labels, inputs.boundary_faces().map(|s| s.face));
assert!(!face_sets.open.is_empty() || !face_sets.bed.is_empty() || !face_sets.free_surface.is_empty());

let boundary_face_branches = map_coastal_boundary_conditions(
    &labels,
    inputs.boundary_faces().map(|s| s.face),
    |branch, _face| branch,
);

let boundary_policy = FvBoundaryPolicy {
    boundary_face_branches,
    allowed_branches: HashSet::from([
        FvBoundaryBranch::Inflow,
        FvBoundaryBranch::Outflow,
        FvBoundaryBranch::Tidal,
        FvBoundaryBranch::Bed,
        FvBoundaryBranch::FreeSurface,
        FvBoundaryBranch::Open,
    ]),
    convective_branch_hooks: HashMap::from([
        (FvBoundaryBranch::Inflow, BoundaryCondition::Dirichlet { value: 1.0 }),
        (FvBoundaryBranch::Outflow, BoundaryCondition::Neumann { gradient: 0.0 }),
        (FvBoundaryBranch::Tidal, BoundaryCondition::Dirichlet { value: 0.2 }),
        (FvBoundaryBranch::Bed, BoundaryCondition::Neumann { gradient: 0.0 }),
        (FvBoundaryBranch::FreeSurface, BoundaryCondition::Neumann { gradient: 0.0 }),
        (FvBoundaryBranch::Open, BoundaryCondition::Neumann { gradient: 0.0 }),
    ]),
    diffusive_branch_hooks: HashMap::from([
        (FvBoundaryBranch::Inflow, BoundaryCondition::Neumann { gradient: 0.0 }),
        (FvBoundaryBranch::Outflow, BoundaryCondition::Neumann { gradient: 0.0 }),
        (FvBoundaryBranch::Tidal, BoundaryCondition::Neumann { gradient: 0.0 }),
        (FvBoundaryBranch::Bed, BoundaryCondition::Neumann { gradient: 0.0 }),
        (FvBoundaryBranch::FreeSurface, BoundaryCondition::Neumann { gradient: 0.0 }),
        (FvBoundaryBranch::Open, BoundaryCondition::Neumann { gradient: 0.0 }),
    ]),
    unsupported_behavior: UnsupportedBoundaryBehavior::Error,
};

let out = assemble_fv_system(
    &inputs,
    &FvmFieldSections {
        cell_scalar,
        cell_gradient,
        face_mass_flux,
    },
    &boundary_policy,
    FvmSchemeSettings {
        convective: ConvectiveScheme::Upwind,
        reconstruction: ReconstructionSettings::default(),
        diffusion: DiffusionSettings::default(),
    },
)?;
```
