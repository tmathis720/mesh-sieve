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
    Basis, BasisTabulation, QuadratureRule, ensure_geometry_order_supported, local_load_vector,
    local_stiffness_matrix, runtime_from_metadata, tabulate_element,
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

fn closure_vertex_coordinates<T, S>(
    topology: &T,
    coordinates: &Coordinates<f64, S>,
    cell: PointId,
    topology_version: u64,
    order: &ClosureOrder,
) -> Result<Vec<Vec<f64>>, MeshSieveError>
where
    T: Sieve<Point = PointId>,
    S: Storage<f64>,
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
    gather_node_coordinates(coordinates, &cell_nodes)
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
    let runtime = runtime_from_metadata(metadata, cell_type)?;
    let node_coords = if let Some(high_order) = coordinates.high_order() {
        if high_order.section().atlas().contains(cell) {
            let values = high_order.section().try_restrict(cell)?;
            let dim = high_order.dimension();
            let geometry_nodes = values.len() / dim;
            let geometry_order = metadata
                .basis_order
                .unwrap_or(runtime.basis.degree())
                .max(1);
            ensure_geometry_order_supported(cell_type, geometry_order)?;
            if geometry_nodes == runtime.basis.num_nodes() {
                values.chunks(dim).map(|tuple| tuple.to_vec()).collect()
            } else {
                return Err(MeshSieveError::InvalidGeometry(format!(
                    "high-order coordinates for {cell:?} provide {geometry_nodes} nodes, but {:?} P{} requires {}",
                    cell_type,
                    runtime.basis.degree(),
                    runtime.basis.num_nodes()
                )));
            }
        } else {
            closure_vertex_coordinates(topology, coordinates, cell, topology_version, order)?
        }
    } else {
        closure_vertex_coordinates(topology, coordinates, cell, topology_version, order)?
    };
    let tabulation = tabulate_element(&runtime, &node_coords)?;
    Ok(ElementMatrices {
        stiffness: local_stiffness_matrix(&tabulation),
        load: local_load_vector(&tabulation, rhs),
    })
}

/// Orientation-aware FE closure data for element kernels.
#[derive(Clone, Debug)]
pub struct ElementClosureData<V> {
    /// Cell whose closure was extracted.
    pub cell: PointId,
    /// Flattened closure values after applying point orientation symmetries.
    pub values: Vec<V>,
    /// Local point/slot to global-index map for each closure scalar DOF, when provided.
    ///
    /// Constrained/eliminated DOFs that have no owned global slot are recorded
    /// as `u64::MAX`; constraint-aware insertion helpers distribute those
    /// entries through their parent equations instead of using the sentinel.
    pub global_indices: Option<Vec<u64>>,
    /// Closure point and local-DOF order corresponding to `values`.
    pub points: Vec<ElementClosurePoint>,
}

/// One point contribution in an element closure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ElementClosurePoint {
    /// Point in the closure.
    pub point: PointId,
    /// Local DOF indices in oriented closure order.
    pub local_dofs: Vec<usize>,
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
    let points = index
        .points
        .iter()
        .map(|entry| ElementClosurePoint {
            point: entry.point,
            local_dofs: entry.permutation.clone(),
        })
        .collect();
    Ok(ElementClosureData {
        cell,
        values,
        global_indices: None,
        points,
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
                match map.global_index(entry.point, local_dof) {
                    Ok(global) => indices.push(global),
                    Err(MeshSieveError::ConstraintIndexOutOfBounds { .. }) => {
                        indices.push(u64::MAX)
                    }
                    Err(err) => return Err(err),
                }
            }
        }
        Some(indices)
    } else {
        None
    };
    let points = index
        .points
        .iter()
        .map(|entry| ElementClosurePoint {
            point: entry.point,
            local_dofs: entry.permutation.clone(),
        })
        .collect();
    Ok(ElementClosureData {
        cell,
        values,
        global_indices,
        points,
    })
}

/// Scatter an element residual into a global vector while honoring hanging-node constraints.
///
/// Free closure DOFs are inserted at their own global index. Contributions for a
/// hanging DOF are distributed to its parent DOFs using the constraint weights,
/// matching the transpose action of linear constraint elimination.
pub fn insert_element_residual_with_hanging_constraints<V>(
    closure: &ElementClosureData<V>,
    element_residual: &[V],
    global_map: &LocalToGlobalMap,
    constraints: &crate::data::hanging_node_constraints::HangingNodeConstraints<V>,
    global_residual: &mut [V],
) -> Result<(), MeshSieveError>
where
    V: Clone + Default + core::ops::AddAssign + core::ops::Mul<Output = V>,
{
    if closure.values.len() != element_residual.len() {
        return Err(MeshSieveError::InvalidGeometry(format!(
            "element residual length {} does not match closure length {} for {:?}",
            element_residual.len(),
            closure.values.len(),
            closure.cell
        )));
    }

    let Some(global_indices) = &closure.global_indices else {
        return Err(MeshSieveError::InvalidGeometry(
            "global indices are required for constrained residual insertion".to_string(),
        ));
    };
    if global_indices.len() != element_residual.len() {
        return Err(MeshSieveError::InvalidGeometry(format!(
            "global-index length {} does not match residual length {} for {:?}",
            global_indices.len(),
            element_residual.len(),
            closure.cell
        )));
    }

    let mut cursor = 0usize;
    for entry in &closure.points {
        for &local_dof in &entry.local_dofs {
            let value = element_residual[cursor].clone();
            if let Some(point_constraints) = constraints.constraints_for(entry.point)
                && let Some(constraint) = point_constraints.iter().find(|c| c.index == local_dof)
            {
                for term in &constraint.terms {
                    let global = global_map.global_index(term.point, term.index)? as usize;
                    let len = global_residual.len();
                    let slot = global_residual.get_mut(global).ok_or_else(|| {
                        MeshSieveError::InvalidGeometry(format!(
                            "global residual index {global} out of bounds (len={len})"
                        ))
                    })?;
                    *slot += value.clone() * term.weight.clone();
                }
            } else {
                let global = global_indices[cursor] as usize;
                let len = global_residual.len();
                let slot = global_residual.get_mut(global).ok_or_else(|| {
                    MeshSieveError::InvalidGeometry(format!(
                        "global residual index {global} out of bounds (len={len})"
                    ))
                })?;
                *slot += value;
            }
            cursor += 1;
        }
    }
    Ok(())
}
