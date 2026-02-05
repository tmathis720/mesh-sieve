//! Finite-element utilities for evaluation and assembly.

use crate::data::coordinates::Coordinates;
use crate::data::discretization::DiscretizationMetadata;
use crate::data::storage::Storage;
use crate::discretization::runtime::{
    Basis, BasisTabulation, QuadratureRule, local_load_vector, local_stiffness_matrix,
    runtime_from_metadata, tabulate_element,
};
use crate::mesh_error::MeshSieveError;
use crate::topology::cell_type::CellType;
use crate::topology::point::PointId;

/// Basis and quadrature evaluation on the reference element.
#[derive(Clone, Debug)]
pub struct ReferenceElementEvaluation {
    /// Basis implementation used for evaluation.
    pub basis: Basis,
    /// Quadrature rule on the reference element.
    pub quadrature: QuadratureRule,
    /// Basis tabulation (values and gradients) on the reference element.
    pub tabulation: BasisTabulation,
}

/// Evaluate basis functions at quadrature points on the reference element.
pub fn evaluate_reference_element(
    cell_type: CellType,
    metadata: &DiscretizationMetadata,
) -> Result<ReferenceElementEvaluation, MeshSieveError> {
    let runtime = runtime_from_metadata(metadata, cell_type)?;
    let tabulation = runtime.basis.tabulate(&runtime.quadrature.points)?;
    Ok(ReferenceElementEvaluation {
        basis: runtime.basis,
        quadrature: runtime.quadrature,
        tabulation,
    })
}

/// Integrate a scalar function on the reference element using quadrature.
pub fn integrate_reference_scalar<F>(
    evaluation: &ReferenceElementEvaluation,
    f: F,
) -> Result<f64, MeshSieveError>
where
    F: Fn(&[f64]) -> f64,
{
    if evaluation.quadrature.points.len() != evaluation.quadrature.weights.len() {
        return Err(MeshSieveError::InvalidGeometry(
            "quadrature points/weights length mismatch".to_string(),
        ));
    }
    let mut total = 0.0;
    for (point, weight) in evaluation
        .quadrature
        .points
        .iter()
        .zip(evaluation.quadrature.weights.iter())
    {
        total += weight * f(point);
    }
    Ok(total)
}

/// Local element matrices and vectors for a scalar Poisson-like operator.
#[derive(Clone, Debug)]
pub struct ElementMatrices {
    /// Element stiffness matrix in row-major order.
    pub stiffness: Vec<f64>,
    /// Element load vector.
    pub load: Vec<f64>,
}

/// Assemble element matrices from coordinates, metadata, and a source term.
pub fn assemble_element_matrices<S, F>(
    coordinates: &Coordinates<f64, S>,
    cell_type: CellType,
    cell_nodes: &[PointId],
    metadata: &DiscretizationMetadata,
    rhs: F,
) -> Result<ElementMatrices, MeshSieveError>
where
    S: Storage<f64>,
    F: Fn(&[f64]) -> f64,
{
    let node_coords = gather_node_coordinates(coordinates, cell_nodes)?;
    let runtime = runtime_from_metadata(metadata, cell_type)?;
    let tabulation = tabulate_element(&runtime, &node_coords)?;
    Ok(ElementMatrices {
        stiffness: local_stiffness_matrix(&tabulation),
        load: local_load_vector(&tabulation, rhs),
    })
}

fn gather_node_coordinates<S: Storage<f64>>(
    coordinates: &Coordinates<f64, S>,
    cell_nodes: &[PointId],
) -> Result<Vec<Vec<f64>>, MeshSieveError> {
    let mut node_coords = Vec::with_capacity(cell_nodes.len());
    for node in cell_nodes {
        let slice = coordinates.section().try_restrict(*node)?;
        node_coords.push(slice.to_vec());
    }
    Ok(node_coords)
}
