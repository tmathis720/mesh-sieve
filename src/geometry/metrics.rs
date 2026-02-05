//! Geometry metrics and mappings for mesh cells.
//!
//! The reference elements use the following vertex ordering:
//! - Segment: `[v0, v1]` with `\xi \in [0, 1]`.
//! - Triangle: `[v0, v1, v2]` with `(r, s)` in the unit right triangle.
//! - Quadrilateral: `[v0, v1, v2, v3]` with `(r, s)` in `[0, 1]^2`.
//! - Tetrahedron: `[v0, v1, v2, v3]` with `(r, s, t)` in the unit tetrahedron.
//! - Hexahedron: `[v0, v1, v2, v3, v4, v5, v6, v7]` with `(r, s, t)` in `[0, 1]^3`.
//! - Prism: `[v0, v1, v2, v3, v4, v5]` with `(r, s)` in the unit triangle and `t` in `[0, 1]`.
//! - Pyramid: `[v0, v1, v2, v3, v4]` with `(r, s)` in `[0, 1]^2` and apex at `t = 1`.
//!
//! Polygonal and polyhedral cells are not supported by the mapping and Jacobian
//! utilities; use explicit triangulations for those cases.

use crate::data::coordinates::Coordinates;
use crate::data::storage::Storage;
use crate::mesh_error::MeshSieveError;
use crate::topology::cell_type::CellType;
use crate::topology::point::PointId;

const EPS: f64 = 1e-12;

/// Compute the signed cell volume/area/length for the given vertices.
///
/// For 2D cells embedded in 3D, the returned area is the magnitude of the
/// area vector (unsigned).
pub fn cell_volume(cell_type: CellType, vertices: &[[f64; 3]]) -> Result<f64, MeshSieveError> {
    let expected = expected_vertex_count(cell_type).ok_or_else(|| {
        MeshSieveError::InvalidGeometry(format!("unsupported cell type: {cell_type:?}"))
    })?;
    if vertices.len() != expected {
        return Err(MeshSieveError::InvalidGeometry(format!(
            "vertex count mismatch: expected {expected}, got {}",
            vertices.len()
        )));
    }
    match cell_type {
        CellType::Vertex => Ok(0.0),
        CellType::Segment => Ok(norm(sub(vertices[1], vertices[0]))),
        CellType::Triangle => Ok(0.5
            * norm(cross(
                sub(vertices[1], vertices[0]),
                sub(vertices[2], vertices[0]),
            ))),
        CellType::Quadrilateral => Ok(0.5
            * norm(cross(
                sub(vertices[1], vertices[0]),
                sub(vertices[2], vertices[0]),
            ))
            + 0.5
                * norm(cross(
                    sub(vertices[2], vertices[0]),
                    sub(vertices[3], vertices[0]),
                ))),
        CellType::Tetrahedron => {
            Ok(signed_volume(vertices[0], vertices[1], vertices[2], vertices[3]).abs())
        }
        CellType::Hexahedron => Ok(hex_volume(vertices).abs()),
        CellType::Prism => Ok(prism_volume(vertices).abs()),
        CellType::Pyramid => Ok(pyramid_volume(vertices).abs()),
        CellType::Simplex(1) => Ok(norm(sub(vertices[1], vertices[0]))),
        CellType::Simplex(2) => Ok(0.5
            * norm(cross(
                sub(vertices[1], vertices[0]),
                sub(vertices[2], vertices[0]),
            ))),
        CellType::Simplex(3) => {
            Ok(signed_volume(vertices[0], vertices[1], vertices[2], vertices[3]).abs())
        }
        _ => Err(MeshSieveError::InvalidGeometry(format!(
            "unsupported cell type: {cell_type:?}"
        ))),
    }
}

/// Compute the cell volume/area/length using point IDs and a coordinate section.
pub fn cell_volume_from_section<S>(
    cell_type: CellType,
    vertex_ids: &[PointId],
    coordinates: &Coordinates<f64, S>,
) -> Result<f64, MeshSieveError>
where
    S: Storage<f64>,
{
    let vertices = gather_vertices(vertex_ids, coordinates)?;
    cell_volume(cell_type, &vertices)
}

/// Compute outward-facing unit normals for a cell.
///
/// For surface cells (triangles/quads), a single normal is returned. For volume
/// cells, one normal per face is returned, following the standard face ordering.
pub fn cell_normals(
    cell_type: CellType,
    vertices: &[[f64; 3]],
) -> Result<Vec<[f64; 3]>, MeshSieveError> {
    let expected = expected_vertex_count(cell_type).ok_or_else(|| {
        MeshSieveError::InvalidGeometry(format!("unsupported cell type: {cell_type:?}"))
    })?;
    if vertices.len() != expected {
        return Err(MeshSieveError::InvalidGeometry(format!(
            "vertex count mismatch: expected {expected}, got {}",
            vertices.len()
        )));
    }
    match cell_type {
        CellType::Triangle => Ok(vec![unit_normal(vertices[0], vertices[1], vertices[2])?]),
        CellType::Quadrilateral => Ok(vec![unit_normal(vertices[0], vertices[1], vertices[2])?]),
        CellType::Tetrahedron | CellType::Hexahedron | CellType::Prism | CellType::Pyramid => {
            let faces = faces_for_cell(cell_type).ok_or_else(|| {
                MeshSieveError::InvalidGeometry(format!("unsupported cell type: {cell_type:?}"))
            })?;
            let mut normals = Vec::with_capacity(faces.len());
            for face in faces {
                let n = face.len();
                if n < 3 {
                    return Err(MeshSieveError::InvalidGeometry(
                        "face has fewer than 3 vertices".into(),
                    ));
                }
                let a = vertices[face[0]];
                let b = vertices[face[1]];
                let c = vertices[face[2]];
                normals.push(unit_normal(a, b, c)?);
            }
            Ok(normals)
        }
        _ => Ok(Vec::new()),
    }
}

/// Compute unit normals for a cell using point IDs and a coordinate section.
pub fn cell_normals_from_section<S>(
    cell_type: CellType,
    vertex_ids: &[PointId],
    coordinates: &Coordinates<f64, S>,
) -> Result<Vec<[f64; 3]>, MeshSieveError>
where
    S: Storage<f64>,
{
    let vertices = gather_vertices(vertex_ids, coordinates)?;
    cell_normals(cell_type, &vertices)
}

/// Map a point in reference coordinates to physical coordinates.
pub fn reference_to_physical(
    cell_type: CellType,
    vertices: &[[f64; 3]],
    reference_point: &[f64],
) -> Result<[f64; 3], MeshSieveError> {
    let (weights, _) = shape_functions(cell_type, reference_point)?;
    if vertices.len() != weights.len() {
        return Err(MeshSieveError::InvalidGeometry(format!(
            "vertex count mismatch: expected {}, got {}",
            weights.len(),
            vertices.len()
        )));
    }
    let mut out = [0.0; 3];
    for (weight, vertex) in weights.iter().zip(vertices.iter()) {
        out[0] += weight * vertex[0];
        out[1] += weight * vertex[1];
        out[2] += weight * vertex[2];
    }
    Ok(out)
}

/// Map a point in reference coordinates to physical coordinates using point IDs.
pub fn reference_to_physical_from_section<S>(
    cell_type: CellType,
    vertex_ids: &[PointId],
    coordinates: &Coordinates<f64, S>,
    reference_point: &[f64],
) -> Result<[f64; 3], MeshSieveError>
where
    S: Storage<f64>,
{
    let vertices = gather_vertices(vertex_ids, coordinates)?;
    reference_to_physical(cell_type, &vertices, reference_point)
}

/// Compute the Jacobian matrix at a reference point.
///
/// The returned matrix is stored row-major with shape `(3, cell_dim)`.
pub fn jacobian(
    cell_type: CellType,
    vertices: &[[f64; 3]],
    reference_point: &[f64],
) -> Result<Vec<f64>, MeshSieveError> {
    let (_, grads) = shape_functions(cell_type, reference_point)?;
    if vertices.len() != grads.len() {
        return Err(MeshSieveError::InvalidGeometry(format!(
            "vertex count mismatch: expected {}, got {}",
            grads.len(),
            vertices.len()
        )));
    }
    let dim = grads.get(0).map(|g| g.len()).unwrap_or(0);
    let mut out = vec![0.0; 3 * dim];
    for (vertex, grad) in vertices.iter().zip(grads.iter()) {
        for ref_dim in 0..dim {
            out[0 * dim + ref_dim] += vertex[0] * grad[ref_dim];
            out[1 * dim + ref_dim] += vertex[1] * grad[ref_dim];
            out[2 * dim + ref_dim] += vertex[2] * grad[ref_dim];
        }
    }
    Ok(out)
}

/// Compute the Jacobian matrix using point IDs and a coordinate section.
pub fn jacobian_from_section<S>(
    cell_type: CellType,
    vertex_ids: &[PointId],
    coordinates: &Coordinates<f64, S>,
    reference_point: &[f64],
) -> Result<Vec<f64>, MeshSieveError>
where
    S: Storage<f64>,
{
    let vertices = gather_vertices(vertex_ids, coordinates)?;
    jacobian(cell_type, &vertices, reference_point)
}

/// Push a vector from reference space into physical space using the Jacobian.
pub fn push_forward_vector(
    cell_type: CellType,
    vertices: &[[f64; 3]],
    reference_point: &[f64],
    reference_vector: &[f64],
) -> Result<[f64; 3], MeshSieveError> {
    let jac = jacobian(cell_type, vertices, reference_point)?;
    let dim = reference_vector.len();
    if jac.len() != 3 * dim {
        return Err(MeshSieveError::InvalidGeometry(
            "reference vector dimension mismatch".into(),
        ));
    }
    let mut out = [0.0; 3];
    for ref_dim in 0..dim {
        out[0] += jac[0 * dim + ref_dim] * reference_vector[ref_dim];
        out[1] += jac[1 * dim + ref_dim] * reference_vector[ref_dim];
        out[2] += jac[2 * dim + ref_dim] * reference_vector[ref_dim];
    }
    Ok(out)
}

/// Pull a physical vector back into reference space using a least-squares solve.
pub fn pull_back_vector(
    cell_type: CellType,
    vertices: &[[f64; 3]],
    reference_point: &[f64],
    physical_vector: &[f64; 3],
) -> Result<Vec<f64>, MeshSieveError> {
    let jac = jacobian(cell_type, vertices, reference_point)?;
    let dim = jac.len() / 3;
    if dim == 0 {
        return Ok(Vec::new());
    }
    let cols = jacobian_columns(&jac, dim);
    match dim {
        1 => {
            let col = cols[0];
            let denom = dot(col, col);
            if denom.abs() <= EPS {
                return Err(MeshSieveError::InvalidGeometry(
                    "degenerate jacobian".into(),
                ));
            }
            Ok(vec![dot(col, *physical_vector) / denom])
        }
        2 => {
            let a = dot(cols[0], cols[0]);
            let b = dot(cols[0], cols[1]);
            let c = dot(cols[1], cols[1]);
            let det = a * c - b * b;
            if det.abs() <= EPS {
                return Err(MeshSieveError::InvalidGeometry(
                    "degenerate jacobian".into(),
                ));
            }
            let rhs0 = dot(cols[0], *physical_vector);
            let rhs1 = dot(cols[1], *physical_vector);
            let inv_det = 1.0 / det;
            let x0 = (c * rhs0 - b * rhs1) * inv_det;
            let x1 = (-b * rhs0 + a * rhs1) * inv_det;
            Ok(vec![x0, x1])
        }
        3 => {
            let mut mat = [0.0; 9];
            for i in 0..3 {
                for j in 0..3 {
                    mat[i * 3 + j] = dot(cols[i], cols[j]);
                }
            }
            let rhs = [
                dot(cols[0], *physical_vector),
                dot(cols[1], *physical_vector),
                dot(cols[2], *physical_vector),
            ];
            let inv = invert_3x3(mat)?;
            Ok(vec![
                inv[0] * rhs[0] + inv[1] * rhs[1] + inv[2] * rhs[2],
                inv[3] * rhs[0] + inv[4] * rhs[1] + inv[5] * rhs[2],
                inv[6] * rhs[0] + inv[7] * rhs[1] + inv[8] * rhs[2],
            ])
        }
        _ => Err(MeshSieveError::InvalidGeometry(format!(
            "unsupported reference dimension: {dim}"
        ))),
    }
}

/// Map a physical point back to reference coordinates using Newton iteration.
///
/// This is intended for linear elements; nonlinear mappings may require
/// additional iterations or a better initial guess.
pub fn physical_to_reference(
    cell_type: CellType,
    vertices: &[[f64; 3]],
    physical_point: &[f64; 3],
) -> Result<Vec<f64>, MeshSieveError> {
    let dim = cell_type.dimension() as usize;
    if dim == 0 {
        return Ok(Vec::new());
    }
    let mut ref_point = vec![0.5; dim];
    for _ in 0..20 {
        let mapped = reference_to_physical(cell_type, vertices, &ref_point)?;
        let residual = sub(mapped, *physical_point);
        if norm(residual) <= 1e-10 {
            return Ok(ref_point);
        }
        let correction = pull_back_vector(cell_type, vertices, &ref_point, &residual)?;
        for (r, c) in ref_point.iter_mut().zip(correction.iter()) {
            *r -= c;
        }
    }
    Ok(ref_point)
}

fn shape_functions(
    cell_type: CellType,
    reference_point: &[f64],
) -> Result<(Vec<f64>, Vec<Vec<f64>>), MeshSieveError> {
    match cell_type {
        CellType::Vertex => Ok((vec![1.0], vec![Vec::new()])),
        CellType::Segment | CellType::Simplex(1) => {
            if reference_point.len() != 1 {
                return Err(MeshSieveError::InvalidGeometry(
                    "segment reference point must have 1 component".into(),
                ));
            }
            let r = reference_point[0];
            let weights = vec![1.0 - r, r];
            let grads = vec![vec![-1.0], vec![1.0]];
            Ok((weights, grads))
        }
        CellType::Triangle | CellType::Simplex(2) => {
            if reference_point.len() != 2 {
                return Err(MeshSieveError::InvalidGeometry(
                    "triangle reference point must have 2 components".into(),
                ));
            }
            let r = reference_point[0];
            let s = reference_point[1];
            let weights = vec![1.0 - r - s, r, s];
            let grads = vec![vec![-1.0, -1.0], vec![1.0, 0.0], vec![0.0, 1.0]];
            Ok((weights, grads))
        }
        CellType::Quadrilateral => {
            if reference_point.len() != 2 {
                return Err(MeshSieveError::InvalidGeometry(
                    "quad reference point must have 2 components".into(),
                ));
            }
            let r = reference_point[0];
            let s = reference_point[1];
            let weights = vec![(1.0 - r) * (1.0 - s), r * (1.0 - s), r * s, (1.0 - r) * s];
            let grads = vec![
                vec![-(1.0 - s), -(1.0 - r)],
                vec![1.0 - s, -r],
                vec![s, r],
                vec![-s, 1.0 - r],
            ];
            Ok((weights, grads))
        }
        CellType::Tetrahedron | CellType::Simplex(3) => {
            if reference_point.len() != 3 {
                return Err(MeshSieveError::InvalidGeometry(
                    "tet reference point must have 3 components".into(),
                ));
            }
            let r = reference_point[0];
            let s = reference_point[1];
            let t = reference_point[2];
            let weights = vec![1.0 - r - s - t, r, s, t];
            let grads = vec![
                vec![-1.0, -1.0, -1.0],
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ];
            Ok((weights, grads))
        }
        CellType::Hexahedron => {
            if reference_point.len() != 3 {
                return Err(MeshSieveError::InvalidGeometry(
                    "hex reference point must have 3 components".into(),
                ));
            }
            let r = reference_point[0];
            let s = reference_point[1];
            let t = reference_point[2];
            let rm = 1.0 - r;
            let sm = 1.0 - s;
            let tm = 1.0 - t;
            let weights = vec![
                rm * sm * tm,
                r * sm * tm,
                r * s * tm,
                rm * s * tm,
                rm * sm * t,
                r * sm * t,
                r * s * t,
                rm * s * t,
            ];
            let grads = vec![
                vec![-sm * tm, -rm * tm, -rm * sm],
                vec![sm * tm, -r * tm, -r * sm],
                vec![s * tm, r * tm, -r * s],
                vec![-s * tm, rm * tm, -rm * s],
                vec![-sm * t, -rm * t, rm * sm],
                vec![sm * t, -r * t, r * sm],
                vec![s * t, r * t, r * s],
                vec![-s * t, rm * t, rm * s],
            ];
            Ok((weights, grads))
        }
        CellType::Prism => {
            if reference_point.len() != 3 {
                return Err(MeshSieveError::InvalidGeometry(
                    "prism reference point must have 3 components".into(),
                ));
            }
            let r = reference_point[0];
            let s = reference_point[1];
            let t = reference_point[2];
            let rm = 1.0 - r - s;
            let tm = 1.0 - t;
            let weights = vec![rm * tm, r * tm, s * tm, rm * t, r * t, s * t];
            let grads = vec![
                vec![-tm, -tm, -rm],
                vec![tm, 0.0, -r],
                vec![0.0, tm, -s],
                vec![-t, -t, rm],
                vec![t, 0.0, r],
                vec![0.0, t, s],
            ];
            Ok((weights, grads))
        }
        CellType::Pyramid => {
            if reference_point.len() != 3 {
                return Err(MeshSieveError::InvalidGeometry(
                    "pyramid reference point must have 3 components".into(),
                ));
            }
            let r = reference_point[0];
            let s = reference_point[1];
            let t = reference_point[2];
            let tm = 1.0 - t;
            let rm = 1.0 - r;
            let sm = 1.0 - s;
            let weights = vec![tm * rm * sm, tm * r * sm, tm * r * s, tm * rm * s, t];
            let grads = vec![
                vec![-tm * sm, -tm * rm, -rm * sm],
                vec![tm * sm, -tm * r, -r * sm],
                vec![tm * s, tm * r, -r * s],
                vec![-tm * s, tm * rm, -rm * s],
                vec![0.0, 0.0, 1.0],
            ];
            Ok((weights, grads))
        }
        _ => Err(MeshSieveError::InvalidGeometry(format!(
            "unsupported cell type: {cell_type:?}"
        ))),
    }
}

fn expected_vertex_count(cell_type: CellType) -> Option<usize> {
    match cell_type {
        CellType::Vertex => Some(1),
        CellType::Segment => Some(2),
        CellType::Triangle => Some(3),
        CellType::Quadrilateral => Some(4),
        CellType::Tetrahedron => Some(4),
        CellType::Hexahedron => Some(8),
        CellType::Prism => Some(6),
        CellType::Pyramid => Some(5),
        CellType::Simplex(d) => match d {
            1 => Some(2),
            2 => Some(3),
            3 => Some(4),
            _ => None,
        },
        _ => None,
    }
}

fn gather_vertices<S>(
    vertex_ids: &[PointId],
    coordinates: &Coordinates<f64, S>,
) -> Result<Vec<[f64; 3]>, MeshSieveError>
where
    S: Storage<f64>,
{
    let dim = coordinates.dimension();
    if dim != 2 && dim != 3 {
        return Err(MeshSieveError::InvalidGeometry(format!(
            "unsupported coordinate dimension: {dim}"
        )));
    }
    let mut out = Vec::with_capacity(vertex_ids.len());
    for point in vertex_ids {
        let slice = coordinates.try_restrict(*point)?;
        if slice.len() != dim {
            return Err(MeshSieveError::InvalidGeometry(format!(
                "coordinate slice length mismatch for {point:?}: expected {dim}, got {}",
                slice.len()
            )));
        }
        let xyz = if dim == 2 {
            [slice[0], slice[1], 0.0]
        } else {
            [slice[0], slice[1], slice[2]]
        };
        out.push(xyz);
    }
    Ok(out)
}

fn signed_volume(a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]) -> f64 {
    let ab = sub(b, a);
    let ac = sub(c, a);
    let ad = sub(d, a);
    dot(ab, cross(ac, ad)) / 6.0
}

fn prism_volume(vertices: &[[f64; 3]]) -> f64 {
    signed_volume(vertices[0], vertices[1], vertices[2], vertices[3])
        + signed_volume(vertices[1], vertices[4], vertices[2], vertices[3])
        + signed_volume(vertices[2], vertices[4], vertices[5], vertices[3])
}

fn pyramid_volume(vertices: &[[f64; 3]]) -> f64 {
    signed_volume(vertices[0], vertices[1], vertices[2], vertices[4])
        + signed_volume(vertices[0], vertices[2], vertices[3], vertices[4])
}

fn hex_volume(vertices: &[[f64; 3]]) -> f64 {
    signed_volume(vertices[0], vertices[1], vertices[3], vertices[4])
        + signed_volume(vertices[1], vertices[2], vertices[3], vertices[6])
        + signed_volume(vertices[1], vertices[3], vertices[4], vertices[6])
        + signed_volume(vertices[1], vertices[4], vertices[5], vertices[6])
        + signed_volume(vertices[3], vertices[4], vertices[6], vertices[7])
}

fn unit_normal(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> Result<[f64; 3], MeshSieveError> {
    let n = cross(sub(b, a), sub(c, a));
    let len = norm(n);
    if len <= EPS {
        return Err(MeshSieveError::InvalidGeometry("degenerate normal".into()));
    }
    Ok([n[0] / len, n[1] / len, n[2] / len])
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

fn jacobian_columns(jac: &[f64], dim: usize) -> Vec<[f64; 3]> {
    let mut cols = Vec::with_capacity(dim);
    for ref_dim in 0..dim {
        cols.push([
            jac[0 * dim + ref_dim],
            jac[1 * dim + ref_dim],
            jac[2 * dim + ref_dim],
        ]);
    }
    cols
}

fn invert_3x3(mat: [f64; 9]) -> Result<[f64; 9], MeshSieveError> {
    let det = mat[0] * (mat[4] * mat[8] - mat[5] * mat[7])
        - mat[1] * (mat[3] * mat[8] - mat[5] * mat[6])
        + mat[2] * (mat[3] * mat[7] - mat[4] * mat[6]);
    if det.abs() <= EPS {
        return Err(MeshSieveError::InvalidGeometry(
            "degenerate jacobian".into(),
        ));
    }
    let inv_det = 1.0 / det;
    Ok([
        (mat[4] * mat[8] - mat[5] * mat[7]) * inv_det,
        (mat[2] * mat[7] - mat[1] * mat[8]) * inv_det,
        (mat[1] * mat[5] - mat[2] * mat[4]) * inv_det,
        (mat[5] * mat[6] - mat[3] * mat[8]) * inv_det,
        (mat[0] * mat[8] - mat[2] * mat[6]) * inv_det,
        (mat[2] * mat[3] - mat[0] * mat[5]) * inv_det,
        (mat[3] * mat[7] - mat[4] * mat[6]) * inv_det,
        (mat[1] * mat[6] - mat[0] * mat[7]) * inv_det,
        (mat[0] * mat[4] - mat[1] * mat[3]) * inv_det,
    ])
}

fn faces_for_cell(cell_type: CellType) -> Option<&'static [&'static [usize]]> {
    match cell_type {
        CellType::Tetrahedron => Some(&TET_FACES),
        CellType::Hexahedron => Some(&HEX_FACES),
        CellType::Prism => Some(&PRISM_FACES),
        CellType::Pyramid => Some(&PYRAMID_FACES),
        _ => None,
    }
}

const TET_FACE_0: [usize; 3] = [0, 1, 2];
const TET_FACE_1: [usize; 3] = [0, 1, 3];
const TET_FACE_2: [usize; 3] = [1, 2, 3];
const TET_FACE_3: [usize; 3] = [0, 2, 3];
const HEX_FACE_0: [usize; 4] = [0, 1, 2, 3];
const HEX_FACE_1: [usize; 4] = [4, 5, 6, 7];
const HEX_FACE_2: [usize; 4] = [0, 1, 5, 4];
const HEX_FACE_3: [usize; 4] = [1, 2, 6, 5];
const HEX_FACE_4: [usize; 4] = [2, 3, 7, 6];
const HEX_FACE_5: [usize; 4] = [3, 0, 4, 7];
const PRISM_FACE_0: [usize; 3] = [0, 1, 2];
const PRISM_FACE_1: [usize; 3] = [3, 4, 5];
const PRISM_FACE_2: [usize; 4] = [0, 1, 4, 3];
const PRISM_FACE_3: [usize; 4] = [1, 2, 5, 4];
const PRISM_FACE_4: [usize; 4] = [2, 0, 3, 5];
const PYRAMID_FACE_0: [usize; 4] = [0, 1, 2, 3];
const PYRAMID_FACE_1: [usize; 3] = [0, 1, 4];
const PYRAMID_FACE_2: [usize; 3] = [1, 2, 4];
const PYRAMID_FACE_3: [usize; 3] = [2, 3, 4];
const PYRAMID_FACE_4: [usize; 3] = [3, 0, 4];

const TET_FACES: [&[usize]; 4] = [&TET_FACE_0, &TET_FACE_1, &TET_FACE_2, &TET_FACE_3];
const HEX_FACES: [&[usize]; 6] = [
    &HEX_FACE_0,
    &HEX_FACE_1,
    &HEX_FACE_2,
    &HEX_FACE_3,
    &HEX_FACE_4,
    &HEX_FACE_5,
];
const PRISM_FACES: [&[usize]; 5] = [
    &PRISM_FACE_0,
    &PRISM_FACE_1,
    &PRISM_FACE_2,
    &PRISM_FACE_3,
    &PRISM_FACE_4,
];
const PYRAMID_FACES: [&[usize]; 5] = [
    &PYRAMID_FACE_0,
    &PYRAMID_FACE_1,
    &PYRAMID_FACE_2,
    &PYRAMID_FACE_3,
    &PYRAMID_FACE_4,
];

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-10
    }

    #[test]
    fn segment_metrics() {
        let vertices = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let volume = cell_volume(CellType::Segment, &vertices).unwrap();
        assert!(approx(volume, 2.0));
        let mapped = reference_to_physical(CellType::Segment, &vertices, &[0.5]).unwrap();
        assert!(approx(mapped[0], 1.0));
    }

    #[test]
    fn triangle_metrics() {
        let vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let area = cell_volume(CellType::Triangle, &vertices).unwrap();
        assert!(approx(area, 0.5));
        let normals = cell_normals(CellType::Triangle, &vertices).unwrap();
        assert!(approx(normals[0][2], 1.0));
        let jac = jacobian(CellType::Triangle, &vertices, &[0.0, 0.0]).unwrap();
        assert!(approx(jac[0], 1.0));
        assert!(approx(jac[3], 1.0));
    }

    #[test]
    fn quad_metrics() {
        let vertices = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let area = cell_volume(CellType::Quadrilateral, &vertices).unwrap();
        assert!(approx(area, 1.0));
        let mapped =
            reference_to_physical(CellType::Quadrilateral, &vertices, &[0.5, 0.5]).unwrap();
        assert!(approx(mapped[0], 0.5));
        assert!(approx(mapped[1], 0.5));
    }

    #[test]
    fn tetra_metrics() {
        let vertices = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let vol = cell_volume(CellType::Tetrahedron, &vertices).unwrap();
        assert!(approx(vol, 1.0 / 6.0));
        let ref_point =
            physical_to_reference(CellType::Tetrahedron, &vertices, &[0.25, 0.25, 0.25]).unwrap();
        assert!(approx(ref_point[0], 0.25));
        assert!(approx(ref_point[1], 0.25));
        assert!(approx(ref_point[2], 0.25));
    }

    #[test]
    fn hex_metrics() {
        let vertices = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ];
        let vol = cell_volume(CellType::Hexahedron, &vertices).unwrap();
        assert!(approx(vol, 1.0));
        let jac = jacobian(CellType::Hexahedron, &vertices, &[0.0, 0.0, 0.0]).unwrap();
        assert!(approx(jac[0], 1.0));
        assert!(approx(jac[4], 1.0));
        assert!(approx(jac[8], 1.0));
    }

    #[test]
    fn prism_metrics() {
        let vertices = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 2.0],
        ];
        let vol = cell_volume(CellType::Prism, &vertices).unwrap();
        assert!(approx(vol, 1.0));
    }

    #[test]
    fn pyramid_metrics() {
        let vertices = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 1.0],
        ];
        let vol = cell_volume(CellType::Pyramid, &vertices).unwrap();
        assert!(approx(vol, 1.0 / 3.0));
        let mapped = reference_to_physical(CellType::Pyramid, &vertices, &[0.5, 0.5, 1.0]).unwrap();
        assert!(approx(mapped[2], 1.0));
    }
}
