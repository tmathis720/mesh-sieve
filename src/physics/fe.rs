//! Finite-element utilities for evaluation and assembly.

use crate::data::closure::{
    ClosureOrder, IdentitySectionSym, SectionSym, build_closure_index,
    build_closure_index_unoriented, get_closure,
};
use crate::data::coordinates::Coordinates;
use crate::data::discretization::DiscretizationMetadata;
use crate::data::global_map::LocalToGlobalMap;
use crate::data::section::Section;
use crate::data::storage::Storage;
use crate::discretization::runtime::{
    Basis, BasisTabulation, QuadratureRule, local_load_vector, local_stiffness_matrix,
    runtime_from_metadata, tabulate_element,
};
use crate::mesh_error::MeshSieveError;
use crate::topology::cell_type::CellType;
use crate::topology::point::PointId;
use crate::topology::sieve::{Orientation, OrientedSieve, Sieve};

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

/// Assemble element matrices using vertices selected from a DMPlex-style closure.
///
/// This is the closure-aware counterpart to [`assemble_element_matrices`].  The
/// closure order is explicit, so high-order and tensor-product callers can share
/// the same predictable ordering used for solution-vector closure access.
pub fn assemble_element_matrices_from_closure<T, S, F>(
    topology: &T,
    coordinates: &Coordinates<f64, S>,
    cell_type: CellType,
    cell: PointId,
    topology_version: u64,
    order: &ClosureOrder,
    metadata: &DiscretizationMetadata,
    rhs: F,
) -> Result<ElementMatrices, MeshSieveError>
where
    T: Sieve<Point = PointId>,
    S: Storage<f64>,
    F: Fn(&[f64]) -> f64,
{
    let index = build_closure_index_unoriented(
        topology,
        coordinates.section(),
        cell,
        topology_version,
        order,
        &IdentitySectionSym,
    )?;
    let mut cell_nodes = Vec::new();
    for point in index.point_order() {
        if topology.cone_points(point).next().is_none() {
            cell_nodes.push(point);
        }
    }
    assemble_element_matrices(coordinates, cell_type, &cell_nodes, metadata, rhs)
}

/// Orientation-aware FE closure data for element kernels.
#[derive(Clone, Debug)]
pub struct ElementClosureData<V> {
    /// Cell whose closure was extracted.
    pub cell: PointId,
    /// Flattened closure values after applying point orientation symmetries.
    pub values: Vec<V>,
    /// Local point/slot to global-index map for each closure scalar DOF, when provided.
    pub global_indices: Option<Vec<u64>>,
}

/// Extract closure values for a FE cell using DMPLEX-like ordering.
pub fn extract_element_closure<T, V, Sct>(
    topology: &T,
    section: &Section<V, Sct>,
    cell: PointId,
    topology_version: u64,
    order: &ClosureOrder,
) -> Result<ElementClosureData<V>, MeshSieveError>
where
    T: Sieve<Point = PointId>,
    V: Clone,
    Sct: Storage<V>,
{
    let index = build_closure_index_unoriented(
        topology,
        section,
        cell,
        topology_version,
        order,
        &IdentitySectionSym,
    )?;
    let values = get_closure(section, &index)?;
    Ok(ElementClosureData {
        cell,
        values,
        global_indices: None,
    })
}

/// Extract oriented closure values and optional global indices for a FE cell.
pub fn extract_oriented_element_closure<T, V, Sct, O, Sym>(
    topology: &T,
    section: &Section<V, Sct>,
    global_map: Option<&LocalToGlobalMap>,
    cell: PointId,
    topology_version: u64,
    order: &ClosureOrder,
    sym: &Sym,
) -> Result<ElementClosureData<V>, MeshSieveError>
where
    T: OrientedSieve<Point = PointId, Orient = O>,
    V: Clone,
    Sct: Storage<V>,
    O: Orientation + Eq + std::hash::Hash,
    Sym: SectionSym<O>,
{
    let index = build_closure_index(topology, section, cell, topology_version, order, sym)?;
    let values = get_closure(section, &index)?;
    let global_indices = if let Some(map) = global_map {
        let mut indices = Vec::with_capacity(index.len);
        for entry in &index.points {
            for &local_dof in &entry.permutation {
                indices.push(map.global_index(entry.point, local_dof)?);
            }
        }
        Some(indices)
    } else {
        None
    };
    Ok(ElementClosureData {
        cell,
        values,
        global_indices,
    })
}
