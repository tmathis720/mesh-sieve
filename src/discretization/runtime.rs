//! Runtime basis/quadrature lookup and element assembly utilities.

use crate::data::closure::{
    build_closure_index_unoriented, ClosureIndex, ClosureOrder, IdentitySectionSym,
};
use crate::data::coordinates::Coordinates;
use crate::data::discretization::DiscretizationMetadata;
use crate::data::section::Section;
use crate::data::storage::Storage;
use crate::mesh_error::MeshSieveError;
use crate::topology::cell_type::CellType;
use crate::topology::point::PointId;
use crate::topology::sieve::Sieve;
use std::collections::{BTreeSet, HashMap};

/// Supported finite-element basis implementations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Basis {
    /// Linear Lagrange basis on a segment (legacy spelling).
    LagrangeP1Segment,
    /// Bilinear Lagrange basis on a quadrilateral (legacy spelling).
    LagrangeQ1Quadrilateral,
    /// Runtime Lagrange basis on a supported reference cell.
    Lagrange {
        /// Reference cell shape.
        cell_type: CellType,
        /// Polynomial/interpolation degree.
        degree: usize,
        /// Shape family used to place nodes and choose interpolation monomials.
        family: BasisFamily,
    },
}

/// Lagrange basis family, mirroring PetscFE simplex/tensor-product setup choices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BasisFamily {
    /// Simplex-complete polynomial basis on segment/triangle/tetrahedron.
    Simplex,
    /// Tensor-product basis on segment/quadrilateral/hexahedron.
    TensorProduct,
    /// Tensor product of triangle and segment bases for prisms.
    Prism,
    /// Layered tensor-product basis for pyramids.
    Pyramid,
}

impl Basis {
    /// Resolve a basis implementation from a metadata label and cell type.
    pub fn from_metadata(name: &str, cell_type: CellType) -> Result<Self, MeshSieveError> {
        let normalized = normalize_name(name);
        let explicit_degree = parse_trailing_degree(&normalized);
        let degree = explicit_degree.unwrap_or(1);
        let is_lagrange = normalized.contains("lagrange")
            || normalized == "p"
            || normalized == "q"
            || normalized.starts_with('p')
            || normalized.starts_with('q')
            || normalized == "linear"
            || normalized == "bilinear"
            || normalized == "trilinear";
        if !is_lagrange {
            return Err(MeshSieveError::InvalidGeometry(format!(
                "unsupported basis '{name}' for {cell_type:?}"
            )));
        }
        if matches!(cell_type, CellType::Segment) && degree == 1 {
            return Ok(Self::LagrangeP1Segment);
        }
        if matches!(cell_type, CellType::Quadrilateral) && degree == 1 {
            return Ok(Self::LagrangeQ1Quadrilateral);
        }
        Self::lagrange(cell_type, degree)
    }

    /// Build a Lagrange basis with a degree inferred from metadata.
    pub fn from_metadata_with_order(
        metadata: &DiscretizationMetadata,
        cell_type: CellType,
    ) -> Result<Self, MeshSieveError> {
        let mut basis = Self::from_metadata(&metadata.basis, cell_type)?;
        if let Some(order) = metadata.basis_order {
            basis = Self::lagrange(cell_type, order)?;
        }
        Ok(basis)
    }

    /// Construct a configurable Lagrange basis for PetscFE-like setup paths.
    pub fn lagrange(cell_type: CellType, degree: usize) -> Result<Self, MeshSieveError> {
        let family = default_lagrange_family(cell_type)?;
        Self::lagrange_with_family(cell_type, degree, family)
    }

    /// Construct a Lagrange basis with an explicit node-placement family.
    pub fn lagrange_with_family(
        cell_type: CellType,
        degree: usize,
        family: BasisFamily,
    ) -> Result<Self, MeshSieveError> {
        ensure_lagrange_supported(cell_type, degree, family)?;
        Ok(Self::Lagrange {
            cell_type,
            degree,
            family,
        })
    }

    /// Reference dimension of the basis.
    pub fn dimension(&self) -> usize {
        match self {
            Basis::LagrangeP1Segment => 1,
            Basis::LagrangeQ1Quadrilateral => 2,
            Basis::Lagrange { cell_type, .. } => cell_type.dimension() as usize,
        }
    }

    /// Polynomial/interpolation degree.
    pub fn degree(&self) -> usize {
        match self {
            Basis::LagrangeP1Segment | Basis::LagrangeQ1Quadrilateral => 1,
            Basis::Lagrange { degree, .. } => *degree,
        }
    }

    /// Number of basis functions per element.
    pub fn num_nodes(&self) -> usize {
        match self {
            Basis::LagrangeP1Segment => 2,
            Basis::LagrangeQ1Quadrilateral => 4,
            Basis::Lagrange {
                cell_type,
                degree,
                family,
            } => reference_nodes(*cell_type, *degree, *family)
                .map(|n| n.len())
                .unwrap_or(0),
        }
    }

    /// The cell type supported by this basis.
    pub fn cell_type(&self) -> CellType {
        match self {
            Basis::LagrangeP1Segment => CellType::Segment,
            Basis::LagrangeQ1Quadrilateral => CellType::Quadrilateral,
            Basis::Lagrange { cell_type, .. } => *cell_type,
        }
    }

    /// Evaluate basis values and gradients at reference points.
    pub fn tabulate(&self, points: &[Vec<f64>]) -> Result<BasisTabulation, MeshSieveError> {
        match self {
            Basis::LagrangeP1Segment => tabulate_p1_segment(points),
            Basis::LagrangeQ1Quadrilateral => tabulate_q1_quad(points),
            Basis::Lagrange {
                cell_type,
                degree,
                family,
            } => tabulate_lagrange(*cell_type, *degree, *family, points),
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
        let normalized = normalize_name(name);
        let order =
            parse_trailing_degree(&normalized).unwrap_or_else(|| match normalized.as_str() {
                "midpoint" | "centroid" => 1,
                _ => 2,
            });
        match canonical_cell_type(cell_type) {
            CellType::Vertex => Ok(point_quadrature(name)),
            CellType::Segment => Ok(gauss_legendre_1d(order.min(3), name)),
            CellType::Quadrilateral => Ok(tensor_power_quadrature(2, order.min(3), name)),
            CellType::Hexahedron => Ok(tensor_power_quadrature(3, order.min(3), name)),
            CellType::Triangle => simplex_quadrature(2, order, name),
            CellType::Tetrahedron => simplex_quadrature(3, order, name),
            CellType::Prism => prism_quadrature(order, name),
            CellType::Pyramid => pyramid_quadrature(order, name),
            canonical => Err(MeshSieveError::InvalidGeometry(format!(
                "unsupported cell topology {canonical:?} for quadrature '{name}'"
            ))),
        }
    }

    /// Construct a quadrature rule from explicit points and weights.
    pub fn from_explicit(
        name: impl Into<String>,
        points: Vec<Vec<f64>>,
        weights: Vec<f64>,
    ) -> Result<Self, MeshSieveError> {
        if points.len() != weights.len() {
            return Err(MeshSieveError::InvalidGeometry(
                "quadrature points/weights length mismatch".to_string(),
            ));
        }
        Ok(Self {
            name: name.into(),
            points,
            weights,
        })
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
    let basis = Basis::from_metadata_with_order(metadata, cell_type)?;
    let quadrature = if metadata.has_quadrature_data() {
        QuadratureRule::from_explicit(
            metadata.quadrature.clone(),
            metadata.quadrature_points.clone(),
            metadata.quadrature_weights.clone(),
        )?
    } else {
        QuadratureRule::from_metadata(&metadata.quadrature, cell_type)?
    };
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

/// Finite-volume discretization metadata resembling PetscFV setup state.
#[derive(Clone, Debug, PartialEq)]
pub struct FiniteVolumeMetadata {
    /// Number of conserved/scalar components per cell.
    pub components: usize,
    /// Reconstruction order for face states.
    pub reconstruction_order: usize,
    /// Optional limiter name used by callers.
    pub limiter: Option<String>,
}

impl FiniteVolumeMetadata {
    /// Construct cell-centered FV metadata.
    pub fn new(components: usize) -> Self {
        Self {
            components,
            reconstruction_order: 1,
            limiter: None,
        }
    }

    /// Set reconstruction order.
    pub fn with_reconstruction_order(mut self, order: usize) -> Self {
        self.reconstruction_order = order;
        self
    }

    /// Set limiter label.
    pub fn with_limiter(mut self, limiter: impl Into<String>) -> Self {
        self.limiter = Some(limiter.into());
        self
    }
}

/// Cell geometry used by finite-volume residuals.
#[derive(Clone, Debug, PartialEq)]
pub struct CellGeometry {
    /// Cell centroid in physical coordinates.
    pub centroid: Vec<f64>,
    /// Cell measure (length/area/volume).
    pub volume: f64,
}

/// Face geometry used by finite-volume flux residuals.
#[derive(Clone, Debug, PartialEq)]
pub struct FaceGeometry {
    /// Face identifier.
    pub face: PointId,
    /// Face centroid.
    pub centroid: Vec<f64>,
    /// Outward normal scaled by face measure for the owner/left cell.
    pub normal: Vec<f64>,
    /// Face measure.
    pub area: f64,
    /// Adjacent cells in deterministic support order.
    pub neighbors: Vec<PointId>,
}

/// Cell-centered finite-volume stencil for one face.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FluxStencil {
    /// Face connecting the stencil cells.
    pub face: PointId,
    /// Left/owner cell.
    pub left: PointId,
    /// Optional right neighbor; `None` means boundary face.
    pub right: Option<PointId>,
}

/// Compute an affine cell geometry from vertices.
pub fn cell_geometry_from_vertices(vertices: &[Vec<f64>]) -> Result<CellGeometry, MeshSieveError> {
    if vertices.is_empty() {
        return Err(MeshSieveError::InvalidGeometry(
            "cell geometry requires vertices".to_string(),
        ));
    }
    let dim = vertices[0].len();
    let mut centroid = vec![0.0; dim];
    for vertex in vertices {
        if vertex.len() != dim {
            return Err(MeshSieveError::InvalidGeometry(
                "inconsistent vertex dimension".to_string(),
            ));
        }
        for d in 0..dim {
            centroid[d] += vertex[d];
        }
    }
    for value in &mut centroid {
        *value /= vertices.len() as f64;
    }
    let volume = match dim {
        1 => (vertices
            .iter()
            .map(|v| v[0])
            .fold(f64::NEG_INFINITY, f64::max)
            - vertices.iter().map(|v| v[0]).fold(f64::INFINITY, f64::min))
        .abs(),
        2 => polygon_area(vertices),
        3 => bounding_box_volume(vertices),
        _ => 0.0,
    };
    Ok(CellGeometry { centroid, volume })
}

/// Compute face geometry from face vertices and support cells.
pub fn face_geometry_from_vertices(
    face: PointId,
    vertices: &[Vec<f64>],
    neighbors: Vec<PointId>,
) -> Result<FaceGeometry, MeshSieveError> {
    let cell = cell_geometry_from_vertices(vertices)?;
    let dim = cell.centroid.len();
    let (normal, area) = match dim {
        1 => (vec![1.0], 1.0),
        2 => {
            if vertices.len() < 2 {
                return Err(MeshSieveError::InvalidGeometry(
                    "2D face requires two vertices".to_string(),
                ));
            }
            let dx = vertices[1][0] - vertices[0][0];
            let dy = vertices[1][1] - vertices[0][1];
            let length = (dx * dx + dy * dy).sqrt();
            (vec![dy, -dx], length)
        }
        3 => {
            if vertices.len() < 3 {
                return Err(MeshSieveError::InvalidGeometry(
                    "3D face requires at least three vertices".to_string(),
                ));
            }
            let a = sub(&vertices[1], &vertices[0]);
            let b = sub(&vertices[2], &vertices[0]);
            let cross = vec![
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ];
            let area = norm(&cross) * 0.5;
            (cross, area)
        }
        _ => (vec![], 0.0),
    };
    Ok(FaceGeometry {
        face,
        centroid: cell.centroid,
        normal,
        area,
        neighbors,
    })
}

/// Build neighbor flux stencils from face supports.
pub fn flux_stencils<T>(topology: &T, faces: impl IntoIterator<Item = PointId>) -> Vec<FluxStencil>
where
    T: Sieve<Point = PointId>,
{
    let mut stencils = Vec::new();
    for face in faces {
        let mut cells: Vec<_> = topology.support_points(face).collect();
        cells.sort_unstable();
        if let Some(&left) = cells.first() {
            stencils.push(FluxStencil {
                face,
                left,
                right: cells.get(1).copied(),
            });
        }
    }
    stencils
}

/// One scalar degree of freedom in an element closure.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ClosureDof {
    /// Mesh point that owns this scalar DOF.
    pub point: PointId,
    /// Section-local DOF slot on `point`.
    pub local_dof: usize,
}

/// Mapping from closure-local scalar DOFs to contiguous assembly indices.
#[derive(Clone, Debug)]
pub struct DofMap {
    dofs: Vec<PointId>,
    closure_dofs: Vec<ClosureDof>,
    indices: HashMap<ClosureDof, usize>,
    first_point_indices: HashMap<PointId, usize>,
}

impl DofMap {
    /// Build a scalar DOF map from an ordered list of points.
    pub fn new(dofs: Vec<PointId>) -> Self {
        let closure_dofs: Vec<_> = dofs
            .iter()
            .map(|&point| ClosureDof {
                point,
                local_dof: 0,
            })
            .collect();
        Self::from_closure_dofs(closure_dofs)
    }

    /// Build a DOF map from orientation-correct closure DOF slots.
    pub fn from_closure_dofs(closure_dofs: Vec<ClosureDof>) -> Self {
        let dofs = closure_dofs.iter().map(|dof| dof.point).collect();
        let mut indices = HashMap::with_capacity(closure_dofs.len());
        let mut first_point_indices = HashMap::new();
        for (idx, &dof) in closure_dofs.iter().enumerate() {
            indices.insert(dof, idx);
            first_point_indices.entry(dof.point).or_insert(idx);
        }
        Self {
            dofs,
            closure_dofs,
            indices,
            first_point_indices,
        }
    }

    /// Return the ordered DOF owner points.
    pub fn dofs(&self) -> &[PointId] {
        &self.dofs
    }

    /// Return the ordered point/local-slot DOFs consumed by closure kernels.
    pub fn closure_dofs(&self) -> &[ClosureDof] {
        &self.closure_dofs
    }

    /// Total number of scalar DOFs.
    pub fn len(&self) -> usize {
        self.closure_dofs.len()
    }

    /// Returns true when this map has no DOFs.
    pub fn is_empty(&self) -> bool {
        self.closure_dofs.is_empty()
    }

    /// Lookup the first index for a point.
    pub fn index(&self, point: PointId) -> Option<usize> {
        self.first_point_indices.get(&point).copied()
    }

    /// Lookup the assembly index for a point-local DOF slot.
    pub fn slot_index(&self, point: PointId, local_dof: usize) -> Option<usize> {
        self.indices.get(&ClosureDof { point, local_dof }).copied()
    }
}

/// Build an assembly DOF map from a cached/un-cached DMPlex-style closure order.
pub fn closure_dof_map<T, V, Sct>(
    topology: &T,
    section: &Section<V, Sct>,
    cell: PointId,
    topology_version: u64,
    order: &ClosureOrder,
) -> Result<DofMap, MeshSieveError>
where
    T: Sieve<Point = PointId>,
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
    Ok(dof_map_from_closure_index(&index))
}

/// Convert a closure index to the point-level DOF map used by scalar assembly.
pub fn dof_map_from_closure_index<O>(index: &ClosureIndex<O>) -> DofMap {
    let mut dofs = Vec::with_capacity(index.len);
    for entry in &index.points {
        for &local_dof in &entry.permutation {
            dofs.push(ClosureDof {
                point: entry.point,
                local_dof,
            });
        }
    }
    DofMap::from_closure_dofs(dofs)
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

/// Add a local vector contribution into a section using point/local-slot closure DOFs.
pub fn assemble_local_vector_closure<S: Storage<f64>>(
    section: &mut Section<f64, S>,
    dofs: &[ClosureDof],
    local: &[f64],
) -> Result<(), MeshSieveError> {
    if dofs.len() != local.len() {
        return Err(MeshSieveError::InvalidGeometry(
            "local vector length mismatch".to_string(),
        ));
    }
    for (dof, value) in dofs.iter().zip(local.iter()) {
        let slice = section.try_restrict_mut(dof.point)?;
        if dof.local_dof >= slice.len() {
            return Err(MeshSieveError::InvalidGeometry(format!(
                "missing local DOF {} for {:?}",
                dof.local_dof, dof.point
            )));
        }
        slice[dof.local_dof] += value;
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

/// Gather coordinate vectors for points.
pub fn gather_point_coordinates<S: Storage<f64>>(
    coordinates: &Coordinates<f64, S>,
    points: &[PointId],
) -> Result<Vec<Vec<f64>>, MeshSieveError> {
    let mut coords = Vec::with_capacity(points.len());
    for point in points {
        coords.push(coordinates.section().try_restrict(*point)?.to_vec());
    }
    Ok(coords)
}

fn normalize_name(name: &str) -> String {
    name.to_ascii_lowercase().replace(['-', ' '], "_")
}

fn parse_trailing_degree(name: &str) -> Option<usize> {
    let digits: String = name
        .chars()
        .rev()
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>()
        .chars()
        .rev()
        .collect();
    (!digits.is_empty()).then(|| digits.parse().ok()).flatten()
}

/// Runtime finite-element support policy.
///
/// The table below is the single source of truth for combinations implemented by
/// this module. Unsupported entries are intentionally rejected before node,
/// quadrature, or Jacobian code is reached.
///
/// | Cell topology | Lagrange families | Quadrature | Jacobian |
/// | --- | --- | --- | --- |
/// | `Vertex`, `Simplex(0)` | `Simplex`, `TensorProduct` | point rule | 0D |
/// | `Segment`, `Simplex(1)` | `Simplex`, `TensorProduct` | Gauss-Legendre | 1D |
/// | `Triangle`, `Simplex(2)` | `Simplex` | simplex triangle | 2D |
/// | `Quadrilateral` | `TensorProduct` | tensor-product Gauss | 2D |
/// | `Tetrahedron`, `Simplex(3)` | `Simplex` | simplex tetrahedron | 3D |
/// | `Hexahedron` | `TensorProduct` | tensor-product Gauss | 3D |
/// | `Prism` | `Prism` | triangle × segment product | 3D |
/// | `Pyramid` | `Pyramid` | one-point pyramid rule | 3D |
const RUNTIME_CAPABILITIES: &[RuntimeCapability] = &[
    RuntimeCapability::new(
        CellType::Vertex,
        &[BasisFamily::Simplex, BasisFamily::TensorProduct],
    ),
    RuntimeCapability::new(
        CellType::Segment,
        &[BasisFamily::Simplex, BasisFamily::TensorProduct],
    ),
    RuntimeCapability::new(CellType::Triangle, &[BasisFamily::Simplex]),
    RuntimeCapability::new(CellType::Quadrilateral, &[BasisFamily::TensorProduct]),
    RuntimeCapability::new(CellType::Tetrahedron, &[BasisFamily::Simplex]),
    RuntimeCapability::new(CellType::Hexahedron, &[BasisFamily::TensorProduct]),
    RuntimeCapability::new(CellType::Prism, &[BasisFamily::Prism]),
    RuntimeCapability::new(CellType::Pyramid, &[BasisFamily::Pyramid]),
];

#[derive(Clone, Copy, Debug)]
struct RuntimeCapability {
    cell_type: CellType,
    families: &'static [BasisFamily],
}

impl RuntimeCapability {
    const fn new(cell_type: CellType, families: &'static [BasisFamily]) -> Self {
        Self {
            cell_type,
            families,
        }
    }
}

fn canonical_cell_type(cell_type: CellType) -> CellType {
    match cell_type {
        CellType::Simplex(0) => CellType::Vertex,
        CellType::Simplex(1) => CellType::Segment,
        CellType::Simplex(2) => CellType::Triangle,
        CellType::Simplex(3) => CellType::Tetrahedron,
        other => other,
    }
}

fn capability_for(cell_type: CellType) -> Option<&'static RuntimeCapability> {
    let canonical = canonical_cell_type(cell_type);
    RUNTIME_CAPABILITIES
        .iter()
        .find(|capability| capability.cell_type == canonical)
}

fn default_lagrange_family(cell_type: CellType) -> Result<BasisFamily, MeshSieveError> {
    capability_for(cell_type)
        .and_then(|capability| capability.families.first().copied())
        .ok_or_else(|| {
            MeshSieveError::InvalidGeometry(format!(
                "unsupported cell type {cell_type:?} for Lagrange basis"
            ))
        })
}

fn ensure_lagrange_supported(
    cell_type: CellType,
    degree: usize,
    family: BasisFamily,
) -> Result<(), MeshSieveError> {
    if degree == 0 && canonical_cell_type(cell_type) != CellType::Vertex {
        return Err(MeshSieveError::InvalidGeometry(
            "Lagrange degree must be at least one for non-vertex cells".to_string(),
        ));
    }
    let capability = capability_for(cell_type).ok_or_else(|| {
        MeshSieveError::InvalidGeometry(format!(
            "unsupported cell type {cell_type:?} for Lagrange basis"
        ))
    })?;
    if !capability.families.contains(&family) {
        return Err(MeshSieveError::InvalidGeometry(format!(
            "unsupported Lagrange family {family:?} for {cell_type:?}; supported families: {:?}",
            capability.families
        )));
    }
    Ok(())
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
        gradients.push(vec![
            vec![-0.25 * (1.0 - eta), -0.25 * (1.0 - xi)],
            vec![0.25 * (1.0 - eta), -0.25 * (1.0 + xi)],
            vec![0.25 * (1.0 + eta), 0.25 * (1.0 + xi)],
            vec![-0.25 * (1.0 + eta), 0.25 * (1.0 - xi)],
        ]);
    }
    Ok(BasisTabulation { values, gradients })
}

fn tabulate_lagrange(
    cell_type: CellType,
    degree: usize,
    family: BasisFamily,
    points: &[Vec<f64>],
) -> Result<BasisTabulation, MeshSieveError> {
    let nodes = reference_nodes(cell_type, degree, family)?;
    let dim = cell_type.dimension() as usize;
    for point in points {
        if point.len() != dim {
            return Err(MeshSieveError::InvalidGeometry(format!(
                "quadrature point dimension {} does not match {dim}",
                point.len()
            )));
        }
    }
    let exponents =
        interpolation_exponents(canonical_cell_type(cell_type), degree, family, nodes.len());
    let vandermonde: Vec<Vec<f64>> = nodes
        .iter()
        .map(|node| monomials(node, &exponents))
        .collect();
    let inv = invert_square(vandermonde)?;
    let mut values = Vec::with_capacity(points.len());
    let mut gradients = Vec::with_capacity(points.len());
    for point in points {
        let mons = monomials(point, &exponents);
        let dmons = monomial_gradients(point, &exponents);
        let mut vals = vec![0.0; nodes.len()];
        let mut grads = vec![vec![0.0; dim]; nodes.len()];
        for basis in 0..nodes.len() {
            for m in 0..exponents.len() {
                let coeff = inv[m][basis];
                vals[basis] += coeff * mons[m];
                for d in 0..dim {
                    grads[basis][d] += coeff * dmons[m][d];
                }
            }
        }
        values.push(vals);
        gradients.push(grads);
    }
    Ok(BasisTabulation { values, gradients })
}

fn reference_nodes(
    cell_type: CellType,
    degree: usize,
    family: BasisFamily,
) -> Result<Vec<Vec<f64>>, MeshSieveError> {
    ensure_lagrange_supported(cell_type, degree, family)?;
    let p = degree;
    let canonical = canonical_cell_type(cell_type);
    let denom = p.max(1) as f64;
    let nodes = match (canonical, family) {
        (CellType::Vertex, BasisFamily::Simplex | BasisFamily::TensorProduct) => vec![Vec::new()],
        (CellType::Segment, BasisFamily::Simplex | BasisFamily::TensorProduct) => (0..=p)
            .map(|i| vec![-1.0 + 2.0 * i as f64 / denom])
            .collect(),
        (CellType::Triangle, BasisFamily::Simplex) => {
            let mut out = Vec::new();
            for i in 0..=p {
                for j in 0..=p - i {
                    out.push(vec![i as f64 / denom, j as f64 / denom]);
                }
            }
            out
        }
        (CellType::Tetrahedron, BasisFamily::Simplex) => {
            let mut out = Vec::new();
            for i in 0..=p {
                for j in 0..=p - i {
                    for k in 0..=p - i - j {
                        out.push(vec![i as f64 / denom, j as f64 / denom, k as f64 / denom]);
                    }
                }
            }
            out
        }
        (CellType::Quadrilateral, BasisFamily::TensorProduct) => tensor_nodes(2, p),
        (CellType::Hexahedron, BasisFamily::TensorProduct) => tensor_nodes(3, p),
        (CellType::Prism, BasisFamily::Prism) => {
            let tri = reference_nodes(CellType::Triangle, p, BasisFamily::Simplex)?;
            let seg = reference_nodes(CellType::Segment, p, BasisFamily::Simplex)?;
            let mut out = Vec::new();
            for t in &tri {
                for z in &seg {
                    out.push(vec![t[0], t[1], z[0]]);
                }
            }
            out
        }
        (CellType::Pyramid, BasisFamily::Pyramid) => {
            let mut out = Vec::new();
            for k in 0..=p {
                let layer = p - k;
                let z = -1.0 + 2.0 * k as f64 / denom;
                if layer == 0 {
                    out.push(vec![0.0, 0.0, z]);
                } else {
                    for i in 0..=layer {
                        for j in 0..=layer {
                            let scale = layer as f64 / denom;
                            out.push(vec![
                                scale * (-1.0 + 2.0 * i as f64 / layer as f64),
                                scale * (-1.0 + 2.0 * j as f64 / layer as f64),
                                z,
                            ]);
                        }
                    }
                }
            }
            out
        }
        _ => {
            return Err(MeshSieveError::InvalidGeometry(format!(
                "unsupported node layout for {cell_type:?} / {family:?}"
            )));
        }
    };
    Ok(nodes)
}

fn tensor_nodes(dim: usize, degree: usize) -> Vec<Vec<f64>> {
    let mut out = Vec::new();
    let mut cur = vec![0.0; dim];
    fn rec(out: &mut Vec<Vec<f64>>, cur: &mut [f64], axis: usize, degree: usize) {
        if axis == cur.len() {
            out.push(cur.to_vec());
            return;
        }
        for i in 0..=degree {
            cur[axis] = -1.0 + 2.0 * i as f64 / degree as f64;
            rec(out, cur, axis + 1, degree);
        }
    }
    rec(&mut out, &mut cur, 0, degree);
    out
}

fn interpolation_exponents(
    cell_type: CellType,
    degree: usize,
    family: BasisFamily,
    count: usize,
) -> Vec<Vec<usize>> {
    let exponents = match (cell_type, family) {
        (CellType::Vertex, BasisFamily::Simplex | BasisFamily::TensorProduct) => vec![Vec::new()],
        (CellType::Segment, BasisFamily::Simplex | BasisFamily::TensorProduct) => {
            (0..=degree).map(|i| vec![i]).collect()
        }
        (CellType::Triangle, BasisFamily::Simplex) => simplex_exponents(2, degree),
        (CellType::Tetrahedron, BasisFamily::Simplex) => simplex_exponents(3, degree),
        (CellType::Quadrilateral, BasisFamily::TensorProduct) => tensor_exponents(2, degree),
        (CellType::Hexahedron, BasisFamily::TensorProduct) => tensor_exponents(3, degree),
        (CellType::Prism, BasisFamily::Prism) => {
            let mut out = Vec::new();
            for tri in simplex_exponents(2, degree) {
                for z in 0..=degree {
                    out.push(vec![tri[0], tri[1], z]);
                }
            }
            out
        }
        (CellType::Pyramid, BasisFamily::Pyramid) => {
            let mut out = Vec::new();
            for z in 0..=degree {
                let layer_degree = degree - z;
                for x in 0..=layer_degree {
                    for y in 0..=layer_degree {
                        out.push(vec![x, y, z]);
                    }
                }
            }
            out
        }
        _ => monomial_exponents(cell_type.dimension() as usize, count),
    };
    debug_assert_eq!(exponents.len(), count);
    exponents
}

fn simplex_exponents(dim: usize, degree: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    for total in 0..=degree {
        let mut cur = vec![0; dim];
        gen_exponents_of_total(dim, total, 0, &mut cur, &mut out, usize::MAX);
    }
    out
}

fn tensor_exponents(dim: usize, degree: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    let mut cur = vec![0; dim];
    fn rec(out: &mut Vec<Vec<usize>>, cur: &mut [usize], axis: usize, degree: usize) {
        if axis == cur.len() {
            out.push(cur.to_vec());
            return;
        }
        for i in 0..=degree {
            cur[axis] = i;
            rec(out, cur, axis + 1, degree);
        }
    }
    rec(&mut out, &mut cur, 0, degree);
    out
}

fn monomial_exponents(dim: usize, count: usize) -> Vec<Vec<usize>> {
    if dim == 0 {
        return vec![Vec::new(); count];
    }
    let mut exps = Vec::new();
    let mut total = 0;
    while exps.len() < count {
        let mut cur = vec![0; dim];
        gen_exponents_of_total(dim, total, 0, &mut cur, &mut exps, count);
        total += 1;
    }
    exps
}

fn gen_exponents_of_total(
    dim: usize,
    total: usize,
    axis: usize,
    cur: &mut [usize],
    out: &mut Vec<Vec<usize>>,
    limit: usize,
) {
    if out.len() >= limit {
        return;
    }
    if axis + 1 == dim {
        cur[axis] = total;
        out.push(cur.to_vec());
        return;
    }
    for v in 0..=total {
        cur[axis] = v;
        gen_exponents_of_total(dim, total - v, axis + 1, cur, out, limit);
    }
}

fn monomials(point: &[f64], exponents: &[Vec<usize>]) -> Vec<f64> {
    exponents
        .iter()
        .map(|exp| {
            point
                .iter()
                .zip(exp.iter())
                .map(|(x, e)| x.powi(*e as i32))
                .product()
        })
        .collect()
}

fn monomial_gradients(point: &[f64], exponents: &[Vec<usize>]) -> Vec<Vec<f64>> {
    exponents
        .iter()
        .map(|exp| {
            (0..point.len())
                .map(|d| {
                    if exp[d] == 0 {
                        0.0
                    } else {
                        let mut value = exp[d] as f64;
                        for (axis, (x, e)) in point.iter().zip(exp.iter()).enumerate() {
                            let pow = if axis == d { e - 1 } else { *e };
                            value *= x.powi(pow as i32);
                        }
                        value
                    }
                })
                .collect()
        })
        .collect()
}

fn invert_square(mut a: Vec<Vec<f64>>) -> Result<Vec<Vec<f64>>, MeshSieveError> {
    let n = a.len();
    if a.iter().any(|row| row.len() != n) {
        return Err(MeshSieveError::InvalidGeometry(
            "non-square interpolation matrix".to_string(),
        ));
    }
    let mut inv = vec![vec![0.0; n]; n];
    for (i, row) in inv.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&a_row, &b_row| a[a_row][col].abs().total_cmp(&a[b_row][col].abs()))
            .unwrap();
        if a[pivot][col].abs() < 1e-12 {
            return Err(MeshSieveError::InvalidGeometry(
                "singular interpolation matrix for basis nodes".to_string(),
            ));
        }
        a.swap(col, pivot);
        inv.swap(col, pivot);
        let scale = a[col][col];
        for j in 0..n {
            a[col][j] /= scale;
            inv[col][j] /= scale;
        }
        for r in 0..n {
            if r == col {
                continue;
            }
            let factor = a[r][col];
            if factor == 0.0 {
                continue;
            }
            for j in 0..n {
                a[r][j] -= factor * a[col][j];
                inv[r][j] -= factor * inv[col][j];
            }
        }
    }
    Ok(inv)
}

fn point_quadrature(name: &str) -> QuadratureRule {
    QuadratureRule {
        name: name.to_string(),
        points: vec![Vec::new()],
        weights: vec![1.0],
    }
}

fn gauss_legendre_1d(order: usize, name: &str) -> QuadratureRule {
    match order {
        1 => QuadratureRule {
            name: name.to_string(),
            points: vec![vec![0.0]],
            weights: vec![2.0],
        },
        3 => {
            let pt = (3.0_f64 / 5.0).sqrt();
            QuadratureRule {
                name: name.to_string(),
                points: vec![vec![-pt], vec![0.0], vec![pt]],
                weights: vec![5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0],
            }
        }
        _ => {
            let pt = 1.0_f64 / 3.0_f64.sqrt();
            QuadratureRule {
                name: name.to_string(),
                points: vec![vec![-pt], vec![pt]],
                weights: vec![1.0, 1.0],
            }
        }
    }
}

fn tensor_power_quadrature(dim: usize, order: usize, name: &str) -> QuadratureRule {
    let one = gauss_legendre_1d(order, name);
    let mut points = vec![Vec::new()];
    let mut weights = vec![1.0];
    for _ in 0..dim {
        let mut next_points = Vec::new();
        let mut next_weights = Vec::new();
        for (prefix, pw) in points.iter().zip(weights.iter()) {
            for (p, w) in one.points.iter().zip(one.weights.iter()) {
                let mut q = prefix.clone();
                q.push(p[0]);
                next_points.push(q);
                next_weights.push(pw * w);
            }
        }
        points = next_points;
        weights = next_weights;
    }
    QuadratureRule {
        name: name.to_string(),
        points,
        weights,
    }
}

fn simplex_quadrature(
    dim: usize,
    order: usize,
    name: &str,
) -> Result<QuadratureRule, MeshSieveError> {
    match dim {
        0 => Ok(point_quadrature(name)),
        1 => Ok(gauss_legendre_1d(order.min(3), name)),
        2 if order > 1 => Ok(QuadratureRule {
            name: name.to_string(),
            points: vec![
                vec![1.0 / 6.0, 1.0 / 6.0],
                vec![2.0 / 3.0, 1.0 / 6.0],
                vec![1.0 / 6.0, 2.0 / 3.0],
            ],
            weights: vec![1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0],
        }),
        2 => Ok(QuadratureRule {
            name: name.to_string(),
            points: vec![vec![1.0 / 3.0, 1.0 / 3.0]],
            weights: vec![0.5],
        }),
        3 => Ok(QuadratureRule {
            name: name.to_string(),
            points: vec![vec![0.25, 0.25, 0.25]],
            weights: vec![1.0 / 6.0],
        }),
        _ => Err(MeshSieveError::InvalidGeometry(format!(
            "unsupported simplex quadrature dimension {dim}"
        ))),
    }
}

fn prism_quadrature(order: usize, name: &str) -> Result<QuadratureRule, MeshSieveError> {
    let tri = simplex_quadrature(2, order, name)?;
    let seg = gauss_legendre_1d(order.min(3), name);
    let mut points = Vec::new();
    let mut weights = Vec::new();
    for (tp, tw) in tri.points.iter().zip(tri.weights.iter()) {
        for (sp, sw) in seg.points.iter().zip(seg.weights.iter()) {
            points.push(vec![tp[0], tp[1], sp[0]]);
            weights.push(tw * sw);
        }
    }
    Ok(QuadratureRule {
        name: name.to_string(),
        points,
        weights,
    })
}

fn pyramid_quadrature(_order: usize, name: &str) -> Result<QuadratureRule, MeshSieveError> {
    Ok(QuadratureRule {
        name: name.to_string(),
        points: vec![vec![0.0, 0.0, -0.5]],
        weights: vec![8.0 / 3.0],
    })
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
        0 => Ok((1.0, Vec::new())),
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

fn polygon_area(vertices: &[Vec<f64>]) -> f64 {
    if vertices.len() < 3 {
        return 0.0;
    }
    let mut area = 0.0;
    for i in 0..vertices.len() {
        let j = (i + 1) % vertices.len();
        area += vertices[i][0] * vertices[j][1] - vertices[j][0] * vertices[i][1];
    }
    0.5 * area.abs()
}

fn bounding_box_volume(vertices: &[Vec<f64>]) -> f64 {
    let mut mins = vec![f64::INFINITY; 3];
    let mut maxs = vec![f64::NEG_INFINITY; 3];
    for v in vertices {
        for d in 0..3 {
            mins[d] = mins[d].min(v[d]);
            maxs[d] = maxs[d].max(v[d]);
        }
    }
    (0..3).map(|d| maxs[d] - mins[d]).product::<f64>().abs()
}

fn sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Build a CSR row/column structure from element closure DOFs.
pub fn csr_from_element_dof_maps(maps: impl IntoIterator<Item = DofMap>) -> CsrPattern {
    let mut rows: HashMap<ClosureDof, BTreeSet<ClosureDof>> = HashMap::new();
    for map in maps {
        let dofs = map.closure_dofs().to_vec();
        for row in &dofs {
            let cols = rows.entry(*row).or_default();
            for col in &dofs {
                cols.insert(*col);
            }
        }
    }
    let mut row_dofs: Vec<_> = rows.keys().copied().collect();
    row_dofs.sort_unstable();
    let mut xadj = Vec::with_capacity(row_dofs.len() + 1);
    let mut adjncy = Vec::new();
    xadj.push(0);
    for row in &row_dofs {
        if let Some(cols) = rows.get(row) {
            adjncy.extend(cols.iter().copied());
        }
        xadj.push(adjncy.len());
    }
    CsrPattern {
        xadj,
        adjncy,
        rows: row_dofs,
    }
}

/// Symbolic CSR pattern over closure DOF rows and columns.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CsrPattern {
    /// CSR offsets by row.
    pub xadj: Vec<usize>,
    /// CSR columns as closure DOFs.
    pub adjncy: Vec<ClosureDof>,
    /// Row closure DOFs.
    pub rows: Vec<ClosureDof>,
}

#[cfg(test)]
mod tests {
    use super::*;

    const CELL_CASES: &[CellType] = &[
        CellType::Vertex,
        CellType::Segment,
        CellType::Triangle,
        CellType::Quadrilateral,
        CellType::Tetrahedron,
        CellType::Hexahedron,
        CellType::Prism,
        CellType::Pyramid,
        CellType::Simplex(0),
        CellType::Simplex(1),
        CellType::Simplex(2),
        CellType::Simplex(3),
        CellType::Simplex(4),
        CellType::Polygon(5),
        CellType::Polyhedron,
    ];

    const FAMILY_CASES: &[BasisFamily] = &[
        BasisFamily::Simplex,
        BasisFamily::TensorProduct,
        BasisFamily::Prism,
        BasisFamily::Pyramid,
    ];

    fn capability_supports(cell_type: CellType, family: BasisFamily) -> bool {
        capability_for(cell_type)
            .map(|capability| capability.families.contains(&family))
            .unwrap_or(false)
    }

    #[test]
    fn capability_table_is_the_lagrange_family_policy() {
        for &cell_type in CELL_CASES {
            for &family in FAMILY_CASES {
                let result = Basis::lagrange_with_family(cell_type, 1, family);
                assert_eq!(
                    result.is_ok(),
                    capability_supports(cell_type, family),
                    "{cell_type:?} / {family:?} should match capability table"
                );
            }
        }
    }

    #[test]
    fn supported_basis_quadrature_and_node_layouts_tabulate() {
        for capability in RUNTIME_CAPABILITIES {
            for &family in capability.families {
                for degree in 1..=2 {
                    let basis =
                        Basis::lagrange_with_family(capability.cell_type, degree, family).unwrap();
                    let quad =
                        QuadratureRule::from_metadata("gauss2", capability.cell_type).unwrap();
                    assert_eq!(quad.dimension(), basis.dimension());

                    let tabulation = basis.tabulate(&quad.points).unwrap();
                    assert_eq!(tabulation.values.len(), quad.points.len());
                    assert_eq!(tabulation.gradients.len(), quad.points.len());
                    for values in &tabulation.values {
                        assert_eq!(values.len(), basis.num_nodes());
                        assert!((values.iter().sum::<f64>() - 1.0).abs() < 1e-10);
                    }
                }
            }
        }
    }

    #[test]
    fn simplex_aliases_use_canonical_capabilities() {
        let aliases = [
            (CellType::Simplex(0), CellType::Vertex, 1),
            (CellType::Simplex(1), CellType::Segment, 2),
            (CellType::Simplex(2), CellType::Triangle, 3),
            (CellType::Simplex(3), CellType::Tetrahedron, 4),
        ];
        for (alias, canonical, expected_nodes) in aliases {
            let basis = Basis::lagrange(alias, 1).unwrap();
            assert_eq!(basis.num_nodes(), expected_nodes);
            let quad = QuadratureRule::from_metadata("gauss2", alias).unwrap();
            assert_eq!(quad.dimension(), canonical.dimension() as usize);
        }
    }

    #[test]
    fn unsupported_combinations_are_explicit_policy_errors() {
        let bad_family =
            Basis::lagrange_with_family(CellType::Triangle, 1, BasisFamily::TensorProduct)
                .unwrap_err();
        assert!(format!("{bad_family}").contains("unsupported Lagrange family"));

        let bad_cell =
            Basis::lagrange_with_family(CellType::Polygon(5), 1, BasisFamily::Simplex).unwrap_err();
        assert!(format!("{bad_cell}").contains("unsupported cell type"));

        let bad_quadrature =
            QuadratureRule::from_metadata("gauss2", CellType::Simplex(4)).unwrap_err();
        assert!(format!("{bad_quadrature}").contains("unsupported cell topology"));
    }

    #[test]
    fn jacobian_inversion_policy_covers_zero_through_three_dimensions() {
        let cases = [
            (0, Vec::new()),
            (1, vec![2.0]),
            (2, vec![2.0, 0.0, 0.0, 3.0]),
            (3, vec![2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0]),
        ];
        for (dim, jacobian) in cases {
            let (det, inverse) = invert_jacobian(dim, &jacobian).unwrap();
            assert!(det > 0.0);
            assert_eq!(inverse.len(), dim * dim);
        }
        assert!(invert_jacobian(4, &[0.0; 16]).is_err());
    }

    #[test]
    fn vertex_element_tabulates_with_zero_dimensional_jacobian() {
        let runtime = ElementRuntime {
            basis: Basis::lagrange(CellType::Vertex, 1).unwrap(),
            quadrature: QuadratureRule::from_metadata("point", CellType::Vertex).unwrap(),
        };
        let tabulation = tabulate_element(&runtime, &[Vec::new()]).unwrap();
        assert_eq!(tabulation.reference_points, vec![Vec::<f64>::new()]);
        assert_eq!(tabulation.physical_points, vec![Vec::<f64>::new()]);
        assert_eq!(tabulation.jacobians, vec![Vec::<f64>::new()]);
        assert_eq!(tabulation.jacobian_dets, vec![1.0]);
    }
}
