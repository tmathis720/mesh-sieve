//! Runtime basis/quadrature lookup and element assembly utilities.

use crate::data::discretization::DiscretizationMetadata;
use crate::data::section::Section;
use crate::data::storage::Storage;
use crate::mesh_error::MeshSieveError;
use crate::topology::cell_type::CellType;
use crate::topology::point::PointId;
use crate::topology::sieve::Sieve;
use std::collections::HashMap;

/// Supported basis implementations.
#[derive(Clone, Debug)]
pub enum Basis {
    /// Linear Lagrange basis on a segment.
    LagrangeP1Segment,
    /// Bilinear Lagrange basis on a quadrilateral.
    LagrangeQ1Quadrilateral,
}

impl Basis {
    /// Resolve a basis implementation from a metadata label and cell type.
    pub fn from_metadata(name: &str, cell_type: CellType) -> Result<Self, MeshSieveError> {
        let normalized = name.to_lowercase();
        match cell_type {
            CellType::Segment => match normalized.as_str() {
                "lagrange_p1" | "p1" | "lagrange1" | "linear" => Ok(Self::LagrangeP1Segment),
                _ => Err(MeshSieveError::InvalidGeometry(format!(
                    "unsupported basis '{name}' for segment"
                ))),
            },
            CellType::Quadrilateral => match normalized.as_str() {
                "lagrange_q1" | "q1" | "bilinear" => Ok(Self::LagrangeQ1Quadrilateral),
                _ => Err(MeshSieveError::InvalidGeometry(format!(
                    "unsupported basis '{name}' for quadrilateral"
                ))),
            },
            _ => Err(MeshSieveError::InvalidGeometry(format!(
                "unsupported cell type {cell_type:?} for basis '{name}'"
            ))),
        }
    }

    /// Reference dimension of the basis.
    pub fn dimension(&self) -> usize {
        match self {
            Basis::LagrangeP1Segment => 1,
            Basis::LagrangeQ1Quadrilateral => 2,
        }
    }

    /// Number of basis functions per element.
    pub fn num_nodes(&self) -> usize {
        match self {
            Basis::LagrangeP1Segment => 2,
            Basis::LagrangeQ1Quadrilateral => 4,
        }
    }

    /// The cell type supported by this basis.
    pub fn cell_type(&self) -> CellType {
        match self {
            Basis::LagrangeP1Segment => CellType::Segment,
            Basis::LagrangeQ1Quadrilateral => CellType::Quadrilateral,
        }
    }

    /// Evaluate basis values and gradients at reference points.
    pub fn tabulate(&self, points: &[Vec<f64>]) -> Result<BasisTabulation, MeshSieveError> {
        match self {
            Basis::LagrangeP1Segment => tabulate_p1_segment(points),
            Basis::LagrangeQ1Quadrilateral => tabulate_q1_quad(points),
        }
    }
}

/// Basis function tabulation on the reference element.
#[derive(Clone, Debug)]
pub struct BasisTabulation {
    /// Basis values per quadrature point: `[qp][basis]`.
    pub values: Vec<Vec<f64>>,
    /// Reference gradients per quadrature point: `[qp][basis][dim]`.
    pub gradients: Vec<Vec<Vec<f64>>>,
}

/// Quadrature rule on the reference element.
#[derive(Clone, Debug)]
pub struct QuadratureRule {
    /// Name for diagnostics.
    pub name: String,
    /// Quadrature points in reference coordinates.
    pub points: Vec<Vec<f64>>,
    /// Quadrature weights.
    pub weights: Vec<f64>,
}

impl QuadratureRule {
    /// Construct a quadrature rule from a metadata label and cell type.
    pub fn from_metadata(name: &str, cell_type: CellType) -> Result<Self, MeshSieveError> {
        let normalized = name.to_lowercase();
        match cell_type {
            CellType::Segment => match normalized.as_str() {
                "gauss1" | "midpoint" => Ok(gauss_legendre_1d(1, "gauss1")),
                "gauss2" => Ok(gauss_legendre_1d(2, "gauss2")),
                _ => Err(MeshSieveError::InvalidGeometry(format!(
                    "unsupported quadrature '{name}' for segment"
                ))),
            },
            CellType::Quadrilateral => match normalized.as_str() {
                "gauss1" | "midpoint" | "gauss1x1" => Ok(tensor_product_quadrature(
                    &gauss_legendre_1d(1, "gauss1"),
                    &gauss_legendre_1d(1, "gauss1"),
                    "gauss1x1",
                )),
                "gauss2" | "gauss2x2" => Ok(tensor_product_quadrature(
                    &gauss_legendre_1d(2, "gauss2"),
                    &gauss_legendre_1d(2, "gauss2"),
                    "gauss2x2",
                )),
                _ => Err(MeshSieveError::InvalidGeometry(format!(
                    "unsupported quadrature '{name}' for quadrilateral"
                ))),
            },
            _ => Err(MeshSieveError::InvalidGeometry(format!(
                "unsupported cell type {cell_type:?} for quadrature '{name}'"
            ))),
        }
    }

    /// Dimension of the quadrature points.
    pub fn dimension(&self) -> usize {
        self.points.first().map(|p| p.len()).unwrap_or(0)
    }
}

/// Combined runtime basis/quadrature for an element.
#[derive(Clone, Debug)]
pub struct ElementRuntime {
    pub basis: Basis,
    pub quadrature: QuadratureRule,
}

/// Resolve runtime basis/quadrature from metadata.
pub fn runtime_from_metadata(
    metadata: &DiscretizationMetadata,
    cell_type: CellType,
) -> Result<ElementRuntime, MeshSieveError> {
    let basis = Basis::from_metadata(&metadata.basis, cell_type)?;
    let quadrature = QuadratureRule::from_metadata(&metadata.quadrature, cell_type)?;
    Ok(ElementRuntime { basis, quadrature })
}

/// Tabulation data on a physical element.
#[derive(Clone, Debug)]
pub struct ElementTabulation {
    /// Quadrature points in reference space.
    pub reference_points: Vec<Vec<f64>>,
    /// Quadrature weights (reference).
    pub weights: Vec<f64>,
    /// Quadrature points in physical space.
    pub physical_points: Vec<Vec<f64>>,
    /// Basis values per quadrature point.
    pub basis_values: Vec<Vec<f64>>,
    /// Basis gradients per quadrature point in physical coordinates.
    pub basis_gradients: Vec<Vec<Vec<f64>>>,
    /// Jacobian matrix per quadrature point, row-major.
    pub jacobians: Vec<Vec<f64>>,
    /// Absolute value of the Jacobian determinant per quadrature point.
    pub jacobian_dets: Vec<f64>,
}

/// Build element tabulation including geometric terms.
pub fn tabulate_element(
    runtime: &ElementRuntime,
    node_coords: &[Vec<f64>],
) -> Result<ElementTabulation, MeshSieveError> {
    let basis = &runtime.basis;
    let num_nodes = basis.num_nodes();
    if node_coords.len() != num_nodes {
        return Err(MeshSieveError::InvalidGeometry(format!(
            "expected {num_nodes} node coordinates, found {}",
            node_coords.len()
        )));
    }
    let dim = basis.dimension();
    let embed_dim = node_coords
        .first()
        .map(|coords| coords.len())
        .ok_or_else(|| MeshSieveError::InvalidGeometry("missing node coordinates".to_string()))?;
    if dim != embed_dim {
        return Err(MeshSieveError::InvalidGeometry(format!(
            "basis dimension {dim} does not match coordinate dimension {embed_dim}"
        )));
    }
    for coords in node_coords {
        if coords.len() != embed_dim {
            return Err(MeshSieveError::InvalidGeometry(
                "inconsistent coordinate dimension".to_string(),
            ));
        }
    }

    let quad = &runtime.quadrature;
    if quad.dimension() != dim {
        return Err(MeshSieveError::InvalidGeometry(format!(
            "quadrature dimension {} does not match basis dimension {dim}",
            quad.dimension()
        )));
    }
    let basis_tab = basis.tabulate(&quad.points)?;
    let mut jacobians = Vec::with_capacity(quad.points.len());
    let mut jacobian_dets = Vec::with_capacity(quad.points.len());
    let mut basis_gradients = Vec::with_capacity(quad.points.len());
    let mut physical_points = Vec::with_capacity(quad.points.len());

    for (qp, ref_grads) in basis_tab.gradients.iter().enumerate() {
        let jac = build_jacobian(dim, node_coords, ref_grads)?;
        let (det, inv) = invert_jacobian(dim, &jac)?;
        jacobian_dets.push(det.abs());
        jacobians.push(jac.clone());

        let mut grad_qp = Vec::with_capacity(num_nodes);
        for ref_grad in ref_grads {
            let mut phys_grad = vec![0.0; dim];
            for phys_dim in 0..dim {
                let mut value = 0.0;
                for ref_dim in 0..dim {
                    value += inv[ref_dim * dim + phys_dim] * ref_grad[ref_dim];
                }
                phys_grad[phys_dim] = value;
            }
            grad_qp.push(phys_grad);
        }
        basis_gradients.push(grad_qp);

        let mut x = vec![0.0; dim];
        for (node, value) in node_coords.iter().zip(basis_tab.values[qp].iter()) {
            for d in 0..dim {
                x[d] += value * node[d];
            }
        }
        physical_points.push(x);
    }

    Ok(ElementTabulation {
        reference_points: quad.points.clone(),
        weights: quad.weights.clone(),
        physical_points,
        basis_values: basis_tab.values,
        basis_gradients,
        jacobians,
        jacobian_dets,
    })
}

/// Compute a local Poisson stiffness matrix from tabulation data.
pub fn local_stiffness_matrix(tabulation: &ElementTabulation) -> Vec<f64> {
    let num_nodes = tabulation
        .basis_values
        .first()
        .map(|v| v.len())
        .unwrap_or(0);
    let mut matrix = vec![0.0; num_nodes * num_nodes];
    for (qp, grads) in tabulation.basis_gradients.iter().enumerate() {
        let weight = tabulation.weights[qp] * tabulation.jacobian_dets[qp];
        for i in 0..num_nodes {
            for j in 0..num_nodes {
                let dot = grads[i]
                    .iter()
                    .zip(grads[j].iter())
                    .map(|(a, b)| a * b)
                    .sum::<f64>();
                matrix[i * num_nodes + j] += weight * dot;
            }
        }
    }
    matrix
}

/// Compute a local load vector for a source term `rhs`.
pub fn local_load_vector<F>(tabulation: &ElementTabulation, rhs: F) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64,
{
    let num_nodes = tabulation
        .basis_values
        .first()
        .map(|v| v.len())
        .unwrap_or(0);
    let mut vector = vec![0.0; num_nodes];
    for (qp, values) in tabulation.basis_values.iter().enumerate() {
        let weight = tabulation.weights[qp] * tabulation.jacobian_dets[qp];
        let f_val = rhs(&tabulation.physical_points[qp]);
        for i in 0..num_nodes {
            vector[i] += weight * f_val * values[i];
        }
    }
    vector
}

/// Mapping from mesh points to contiguous DOF indices.
#[derive(Clone, Debug)]
pub struct DofMap {
    dofs: Vec<PointId>,
    indices: HashMap<PointId, usize>,
}

impl DofMap {
    /// Build a DOF map from an ordered list of points.
    pub fn new(dofs: Vec<PointId>) -> Self {
        let indices = dofs.iter().enumerate().map(|(idx, &p)| (p, idx)).collect();
        Self { dofs, indices }
    }

    /// Return the ordered DOF points.
    pub fn dofs(&self) -> &[PointId] {
        &self.dofs
    }

    /// Total number of DOFs.
    pub fn len(&self) -> usize {
        self.dofs.len()
    }

    /// Lookup the index for a point.
    pub fn index(&self, point: PointId) -> Option<usize> {
        self.indices.get(&point).copied()
    }
}

/// Add a local vector contribution into a global section.
pub fn assemble_local_vector<S: Storage<f64>>(
    section: &mut Section<f64, S>,
    dofs: &[PointId],
    local: &[f64],
) -> Result<(), MeshSieveError> {
    if dofs.len() != local.len() {
        return Err(MeshSieveError::InvalidGeometry(
            "local vector length mismatch".to_string(),
        ));
    }
    for (point, value) in dofs.iter().zip(local.iter()) {
        let slice = section.try_restrict_mut(*point)?;
        if slice.len() != 1 {
            return Err(MeshSieveError::InvalidGeometry(format!(
                "expected scalar slice for {point:?}, found {}",
                slice.len()
            )));
        }
        slice[0] += value;
    }
    Ok(())
}

/// Add a local matrix contribution into a global dense-matrix section.
pub fn assemble_local_matrix<S: Storage<f64>>(
    section: &mut Section<f64, S>,
    dofs: &[PointId],
    dof_map: &DofMap,
    local: &[f64],
) -> Result<(), MeshSieveError> {
    let num_nodes = dofs.len();
    if local.len() != num_nodes * num_nodes {
        return Err(MeshSieveError::InvalidGeometry(
            "local matrix length mismatch".to_string(),
        ));
    }
    for (row, point) in dofs.iter().enumerate() {
        let row_slice = section.try_restrict_mut(*point)?;
        if row_slice.len() != dof_map.len() {
            return Err(MeshSieveError::InvalidGeometry(format!(
                "expected row length {}, found {} for {point:?}",
                dof_map.len(),
                row_slice.len()
            )));
        }
        for (col, col_point) in dofs.iter().enumerate() {
            let col_idx = dof_map.index(*col_point).ok_or_else(|| {
                MeshSieveError::InvalidGeometry(format!("missing DOF index for {col_point:?}"))
            })?;
            row_slice[col_idx] += local[row * num_nodes + col];
        }
    }
    Ok(())
}

/// Retrieve the vertices that define a cell, in cone order when possible.
pub fn cell_vertices<S: Sieve<Point = PointId>>(
    sieve: &S,
    cell: PointId,
    expected: usize,
) -> Result<Vec<PointId>, MeshSieveError> {
    let cone: Vec<PointId> = sieve.cone_points(cell).collect();
    if cone.len() == expected && cone.iter().all(|p| sieve.cone_points(*p).next().is_none()) {
        return Ok(cone);
    }

    let mut vertices = Vec::new();
    for point in sieve.closure(std::iter::once(cell)) {
        if sieve.cone_points(point).next().is_none() {
            vertices.push(point);
        }
    }
    vertices.sort();
    vertices.dedup();
    if vertices.len() != expected {
        return Err(MeshSieveError::InvalidGeometry(format!(
            "expected {expected} vertices for cell {cell:?}, found {}",
            vertices.len()
        )));
    }
    Ok(vertices)
}

fn tabulate_p1_segment(points: &[Vec<f64>]) -> Result<BasisTabulation, MeshSieveError> {
    let mut values = Vec::with_capacity(points.len());
    let mut gradients = Vec::with_capacity(points.len());
    for point in points {
        if point.len() != 1 {
            return Err(MeshSieveError::InvalidGeometry(
                "segment quadrature must be 1D".to_string(),
            ));
        }
        let xi = point[0];
        values.push(vec![0.5 * (1.0 - xi), 0.5 * (1.0 + xi)]);
        gradients.push(vec![vec![-0.5], vec![0.5]]);
    }
    Ok(BasisTabulation { values, gradients })
}

fn tabulate_q1_quad(points: &[Vec<f64>]) -> Result<BasisTabulation, MeshSieveError> {
    let mut values = Vec::with_capacity(points.len());
    let mut gradients = Vec::with_capacity(points.len());
    for point in points {
        if point.len() != 2 {
            return Err(MeshSieveError::InvalidGeometry(
                "quadrilateral quadrature must be 2D".to_string(),
            ));
        }
        let xi = point[0];
        let eta = point[1];
        let n1 = 0.25 * (1.0 - xi) * (1.0 - eta);
        let n2 = 0.25 * (1.0 + xi) * (1.0 - eta);
        let n3 = 0.25 * (1.0 + xi) * (1.0 + eta);
        let n4 = 0.25 * (1.0 - xi) * (1.0 + eta);
        values.push(vec![n1, n2, n3, n4]);

        let dndxi = vec![
            vec![-0.25 * (1.0 - eta), -0.25 * (1.0 - xi)],
            vec![0.25 * (1.0 - eta), -0.25 * (1.0 + xi)],
            vec![0.25 * (1.0 + eta), 0.25 * (1.0 + xi)],
            vec![-0.25 * (1.0 + eta), 0.25 * (1.0 - xi)],
        ];
        gradients.push(dndxi);
    }
    Ok(BasisTabulation { values, gradients })
}

fn gauss_legendre_1d(order: usize, name: &str) -> QuadratureRule {
    match order {
        1 => QuadratureRule {
            name: name.to_string(),
            points: vec![vec![0.0]],
            weights: vec![2.0],
        },
        2 => {
            let pt = 1.0_f64 / 3.0_f64.sqrt();
            QuadratureRule {
                name: name.to_string(),
                points: vec![vec![-pt], vec![pt]],
                weights: vec![1.0, 1.0],
            }
        }
        _ => QuadratureRule {
            name: name.to_string(),
            points: vec![],
            weights: vec![],
        },
    }
}

fn tensor_product_quadrature(a: &QuadratureRule, b: &QuadratureRule, name: &str) -> QuadratureRule {
    let mut points = Vec::with_capacity(a.points.len() * b.points.len());
    let mut weights = Vec::with_capacity(a.points.len() * b.points.len());
    for (pa, wa) in a.points.iter().zip(a.weights.iter()) {
        for (pb, wb) in b.points.iter().zip(b.weights.iter()) {
            let mut pt = Vec::with_capacity(pa.len() + pb.len());
            pt.extend_from_slice(pa);
            pt.extend_from_slice(pb);
            points.push(pt);
            weights.push(wa * wb);
        }
    }
    QuadratureRule {
        name: name.to_string(),
        points,
        weights,
    }
}

fn build_jacobian(
    dim: usize,
    node_coords: &[Vec<f64>],
    ref_grads: &[Vec<f64>],
) -> Result<Vec<f64>, MeshSieveError> {
    let mut jac = vec![0.0; dim * dim];
    for (node, grad) in node_coords.iter().zip(ref_grads.iter()) {
        for phys_dim in 0..dim {
            for ref_dim in 0..dim {
                jac[phys_dim * dim + ref_dim] += node[phys_dim] * grad[ref_dim];
            }
        }
    }
    Ok(jac)
}

fn invert_jacobian(dim: usize, jac: &[f64]) -> Result<(f64, Vec<f64>), MeshSieveError> {
    match dim {
        1 => {
            let det = jac[0];
            if det.abs() < f64::EPSILON {
                return Err(MeshSieveError::InvalidGeometry(
                    "zero Jacobian determinant".to_string(),
                ));
            }
            Ok((det, vec![1.0 / det]))
        }
        2 => {
            let a = jac[0];
            let b = jac[1];
            let c = jac[2];
            let d = jac[3];
            let det = a * d - b * c;
            if det.abs() < f64::EPSILON {
                return Err(MeshSieveError::InvalidGeometry(
                    "zero Jacobian determinant".to_string(),
                ));
            }
            Ok((det, vec![d / det, -b / det, -c / det, a / det]))
        }
        3 => {
            let a = jac[0];
            let b = jac[1];
            let c = jac[2];
            let d = jac[3];
            let e = jac[4];
            let f = jac[5];
            let g = jac[6];
            let h = jac[7];
            let i = jac[8];
            let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
            if det.abs() < f64::EPSILON {
                return Err(MeshSieveError::InvalidGeometry(
                    "zero Jacobian determinant".to_string(),
                ));
            }
            let inv = vec![
                (e * i - f * h) / det,
                (c * h - b * i) / det,
                (b * f - c * e) / det,
                (f * g - d * i) / det,
                (a * i - c * g) / det,
                (c * d - a * f) / det,
                (d * h - e * g) / det,
                (b * g - a * h) / det,
                (a * e - b * d) / det,
            ];
            Ok((det, inv))
        }
        _ => Err(MeshSieveError::InvalidGeometry(format!(
            "unsupported Jacobian dimension {dim}"
        ))),
    }
}
