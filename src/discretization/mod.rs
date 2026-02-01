//! Runtime discretization helpers for basis/quadrature evaluation and assembly.

pub mod runtime;

pub use runtime::{
    Basis, BasisTabulation, DofMap, ElementRuntime, ElementTabulation, QuadratureRule,
    assemble_local_matrix, assemble_local_vector, cell_vertices, local_load_vector,
    local_stiffness_matrix, runtime_from_metadata, tabulate_element,
};
