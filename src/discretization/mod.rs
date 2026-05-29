//! Runtime discretization helpers for basis/quadrature evaluation and assembly.

pub mod runtime;

pub use runtime::{
    Basis, BasisFamily, BasisTabulation, ClosureDof, DiscretizationCapability, DofMap,
    ElementRuntime, ElementTabulation, QuadratureFamily, QuadratureRule, assemble_local_matrix,
    assemble_local_vector, cell_vertices, discretization_capability,
    ensure_geometry_order_supported, local_load_vector, local_stiffness_matrix,
    runtime_from_metadata, supported_discretizations, tabulate_element,
};
