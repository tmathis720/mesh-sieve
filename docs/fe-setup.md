# Finite Element Setup (Discretization + Assembly)

This note summarizes a typical FE workflow with mesh-sieve. It complements the
high-level material in [`Physics_Guide.md`](../Physics_Guide.md).

## 1) Describe basis and quadrature metadata

Define the basis and quadrature identifiers for each region, optionally adding
orders, shape-function labels, and explicit quadrature points/weights:

```rust
use mesh_sieve::data::discretization::{DiscretizationMetadata, FieldDiscretization, RegionKey};
use mesh_sieve::topology::cell_type::CellType;

let metadata = DiscretizationMetadata::new("lagrange_p1", "gauss2")
    .with_basis_metadata(1, ["N1", "N2"])
    .with_quadrature_metadata(
        2,
        vec![vec![-0.5773502692], vec![0.5773502692]],
        vec![1.0, 1.0],
    );

let mut field = FieldDiscretization::new();
field.set_metadata(RegionKey::cell_type(CellType::Segment), metadata);
```

The explicit quadrature points/weights override the label-based lookup used by
runtime assembly.

## 2) Evaluate basis + quadrature on a reference element

```rust
use mesh_sieve::physics::fe::{evaluate_reference_element, integrate_reference_scalar};
use mesh_sieve::topology::cell_type::CellType;

let eval = evaluate_reference_element(CellType::Segment, &metadata)?;
let one = integrate_reference_scalar(&eval, |_| 1.0)?;
```

## 3) Assemble element matrices from coordinates

```rust
use mesh_sieve::physics::fe::assemble_element_matrices;
use mesh_sieve::topology::point::PointId;

let cell_nodes = vec![PointId::new(1)?, PointId::new(2)?];
let element = assemble_element_matrices(&coordinates, CellType::Segment, &cell_nodes, &metadata, |_| 1.0)?;

// element.stiffness and element.load are ready to be scattered into a global system.
```

The assembly helper uses the discretization metadata to select basis and
quadrature data, then evaluates element geometry using the supplied
`Coordinates`.
