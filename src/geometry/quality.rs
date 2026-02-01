//! Cell quality utilities based on coordinate sections.
//!
//! # Coordinate layout
//! Coordinate sections store a slice per point with a fixed embedding dimension.
//! Quality checks accept embedding dimensions **2** or **3** and interpret the
//! slice as `(x, y)` or `(x, y, z)` respectively. For 2D meshes embedded in 3D,
//! project to the XY plane before evaluation.
//!
//! # Supported cell types
//! The quality routines support the following cell types with the vertex
//! ordering shown below:
//!
//! - **Triangle**: `[v0, v1, v2]` (counter-clockwise in XY).
//! - **Quadrilateral**: `[v0, v1, v2, v3]` (counter-clockwise in XY).
//! - **Tetrahedron**: `[v0, v1, v2, v3]`.
//! - **Hexahedron**: `[v0, v1, v2, v3, v4, v5, v6, v7]` with
//!   bottom face `[0, 1, 2, 3]` and top face `[4, 5, 6, 7]`.
//! - **Prism**: `[v0, v1, v2, v3, v4, v5]` with bottom triangle `[0, 1, 2]`
//!   and top triangle `[3, 4, 5]`.
//! - **Pyramid**: `[v0, v1, v2, v3, v4]` with base quad `[0, 1, 2, 3]`
//!   and apex `v4`.
//!
//! # Examples
//! ```rust
//! use mesh_sieve::data::atlas::Atlas;
//! use mesh_sieve::data::coordinates::Coordinates;
//! use mesh_sieve::data::storage::VecStorage;
//! use mesh_sieve::geometry::quality::{cell_quality_from_section, validate_cell_geometry};
//! use mesh_sieve::topology::cell_type::CellType;
//! use mesh_sieve::topology::point::PointId;
//!
//! let mut atlas = Atlas::default();
//! let v0 = PointId::new(1)?;
//! let v1 = PointId::new(2)?;
//! let v2 = PointId::new(3)?;
//! atlas.try_insert(v0, 2)?;
//! atlas.try_insert(v1, 2)?;
//! atlas.try_insert(v2, 2)?;
//!
//! let mut coords = Coordinates::<f64, VecStorage<f64>>::try_new(2, 2, atlas)?;
//! coords.try_restrict_mut(v0)?.copy_from_slice(&[0.0, 0.0]);
//! coords.try_restrict_mut(v1)?.copy_from_slice(&[1.0, 0.0]);
//! coords.try_restrict_mut(v2)?.copy_from_slice(&[0.0, 1.0]);
//!
//! let quality = cell_quality_from_section(
//!     CellType::Triangle,
//!     &[v0, v1, v2],
//!     &coords,
//! )?;
//! assert!(quality.jacobian_sign > 0.0);
//!
//! // Use the validator to error on inverted or degenerate elements.
//! validate_cell_geometry(CellType::Triangle, &[v0, v1, v2], &coords)?;
//! # Ok::<(), mesh_sieve::mesh_error::MeshSieveError>(())
//! ```

use crate::data::coordinates::Coordinates;
use crate::data::storage::Storage;
use crate::mesh_error::MeshSieveError;
use crate::topology::cell_type::CellType;
use crate::topology::point::PointId;
use std::f64::consts::PI;

const EPS: f64 = 1e-12;

/// Basic quality metrics for a single cell.
#[derive(Clone, Copy, Debug)]
pub struct CellQuality {
    /// Ratio of the longest edge length to the shortest edge length.
    pub aspect_ratio: f64,
    /// Minimum corner angle (degrees) across all faces.
    pub min_angle_deg: f64,
    /// Signed Jacobian (area for 2D, volume for 3D). Negative values indicate
    /// inverted orientation; zero indicates degenerate geometry.
    pub jacobian_sign: f64,
}

/// Compute quality metrics for a cell, using point IDs and a coordinate section.
///
/// Returns an error for unsupported cell types or invalid geometry inputs.
pub fn cell_quality_from_section<S>(
    cell_type: CellType,
    vertex_ids: &[PointId],
    coordinates: &Coordinates<f64, S>,
) -> Result<CellQuality, MeshSieveError>
where
    S: Storage<f64>,
{
    let vertices = gather_vertices(vertex_ids, coordinates)?;
    cell_quality(cell_type, &vertices)
}

/// Validate that a cell is not inverted or degenerate.
///
/// Returns the computed quality metrics on success.
pub fn validate_cell_geometry<S>(
    cell_type: CellType,
    vertex_ids: &[PointId],
    coordinates: &Coordinates<f64, S>,
) -> Result<CellQuality, MeshSieveError>
where
    S: Storage<f64>,
{
    let quality = cell_quality_from_section(cell_type, vertex_ids, coordinates)?;
    if !quality.jacobian_sign.is_finite() || quality.jacobian_sign.abs() <= EPS {
        return Err(MeshSieveError::InvalidGeometry(format!(
            "degenerate geometry: jacobian sign = {}",
            quality.jacobian_sign
        )));
    }
    if quality.jacobian_sign < 0.0 {
        return Err(MeshSieveError::InvalidGeometry(format!(
            "inverted geometry: jacobian sign = {}",
            quality.jacobian_sign
        )));
    }
    if !quality.min_angle_deg.is_finite() || quality.min_angle_deg <= 0.0 {
        return Err(MeshSieveError::InvalidGeometry(format!(
            "invalid geometry: min angle = {}",
            quality.min_angle_deg
        )));
    }
    if !quality.aspect_ratio.is_finite() || quality.aspect_ratio <= 0.0 {
        return Err(MeshSieveError::InvalidGeometry(format!(
            "invalid geometry: aspect ratio = {}",
            quality.aspect_ratio
        )));
    }
    Ok(quality)
}

/// Compute quality metrics from explicit vertex coordinates.
pub fn cell_quality(
    cell_type: CellType,
    vertices: &[[f64; 3]],
) -> Result<CellQuality, MeshSieveError> {
    let expected = expected_vertex_count(cell_type).ok_or_else(|| {
        MeshSieveError::InvalidGeometry(format!("unsupported cell type: {cell_type:?}"))
    })?;
    if vertices.len() != expected {
        return Err(MeshSieveError::InvalidGeometry(format!(
            "vertex count mismatch: expected {expected}, got {}",
            vertices.len()
        )));
    }
    let aspect_ratio = aspect_ratio(cell_type, vertices)?;
    let min_angle_deg = min_angle(cell_type, vertices)?;
    let jacobian_sign = jacobian_sign(cell_type, vertices)?;
    Ok(CellQuality {
        aspect_ratio,
        min_angle_deg,
        jacobian_sign,
    })
}

fn expected_vertex_count(cell_type: CellType) -> Option<usize> {
    match cell_type {
        CellType::Triangle => Some(3),
        CellType::Quadrilateral => Some(4),
        CellType::Tetrahedron => Some(4),
        CellType::Hexahedron => Some(8),
        CellType::Prism => Some(6),
        CellType::Pyramid => Some(5),
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

fn aspect_ratio(cell_type: CellType, vertices: &[[f64; 3]]) -> Result<f64, MeshSieveError> {
    let edges = edges_for_cell(cell_type).ok_or_else(|| {
        MeshSieveError::InvalidGeometry(format!("unsupported cell type: {cell_type:?}"))
    })?;
    let mut min_len = f64::INFINITY;
    let mut max_len = 0.0f64;
    for (a, b) in edges {
        let len = norm(sub(vertices[*a], vertices[*b]));
        if len <= EPS {
            return Err(MeshSieveError::InvalidGeometry(
                "zero-length edge detected".into(),
            ));
        }
        min_len = min_len.min(len);
        max_len = max_len.max(len);
    }
    Ok(max_len / min_len)
}

fn min_angle(cell_type: CellType, vertices: &[[f64; 3]]) -> Result<f64, MeshSieveError> {
    let faces = faces_for_cell(cell_type).ok_or_else(|| {
        MeshSieveError::InvalidGeometry(format!("unsupported cell type: {cell_type:?}"))
    })?;
    let mut min_angle = f64::INFINITY;
    for face in faces {
        let n = face.len();
        for i in 0..n {
            let prev = face[(i + n - 1) % n];
            let curr = face[i];
            let next = face[(i + 1) % n];
            let v1 = sub(vertices[prev], vertices[curr]);
            let v2 = sub(vertices[next], vertices[curr]);
            let angle = angle_deg(v1, v2)?;
            min_angle = min_angle.min(angle);
        }
    }
    Ok(min_angle)
}

fn jacobian_sign(cell_type: CellType, vertices: &[[f64; 3]]) -> Result<f64, MeshSieveError> {
    match cell_type {
        CellType::Triangle => Ok(signed_area_xy(vertices[0], vertices[1], vertices[2])),
        CellType::Quadrilateral => Ok(signed_area_xy(vertices[0], vertices[1], vertices[2])
            + signed_area_xy(vertices[0], vertices[2], vertices[3])),
        CellType::Tetrahedron => Ok(signed_volume(
            vertices[0],
            vertices[1],
            vertices[2],
            vertices[3],
        )),
        CellType::Hexahedron => Ok(hex_signed_volume(vertices)),
        CellType::Prism => Ok(prism_signed_volume(vertices)),
        CellType::Pyramid => Ok(pyramid_signed_volume(vertices)),
        _ => Err(MeshSieveError::InvalidGeometry(format!(
            "unsupported cell type: {cell_type:?}"
        ))),
    }
}

fn signed_area_xy(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    let abx = b[0] - a[0];
    let aby = b[1] - a[1];
    let acx = c[0] - a[0];
    let acy = c[1] - a[1];
    0.5 * (abx * acy - aby * acx)
}

fn signed_volume(a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]) -> f64 {
    let ab = sub(b, a);
    let ac = sub(c, a);
    let ad = sub(d, a);
    dot(ab, cross(ac, ad)) / 6.0
}

fn prism_signed_volume(vertices: &[[f64; 3]]) -> f64 {
    signed_volume(vertices[0], vertices[1], vertices[2], vertices[3])
        + signed_volume(vertices[1], vertices[4], vertices[2], vertices[3])
        + signed_volume(vertices[2], vertices[4], vertices[5], vertices[3])
}

fn pyramid_signed_volume(vertices: &[[f64; 3]]) -> f64 {
    signed_volume(vertices[0], vertices[1], vertices[2], vertices[4])
        + signed_volume(vertices[0], vertices[2], vertices[3], vertices[4])
}

fn hex_signed_volume(vertices: &[[f64; 3]]) -> f64 {
    signed_volume(vertices[0], vertices[1], vertices[3], vertices[4])
        + signed_volume(vertices[1], vertices[2], vertices[3], vertices[6])
        + signed_volume(vertices[1], vertices[3], vertices[4], vertices[6])
        + signed_volume(vertices[1], vertices[4], vertices[5], vertices[6])
        + signed_volume(vertices[3], vertices[4], vertices[6], vertices[7])
}

fn edges_for_cell(cell_type: CellType) -> Option<&'static [(usize, usize)]> {
    match cell_type {
        CellType::Triangle => Some(&TRI_EDGES),
        CellType::Quadrilateral => Some(&QUAD_EDGES),
        CellType::Tetrahedron => Some(&TET_EDGES),
        CellType::Hexahedron => Some(&HEX_EDGES),
        CellType::Prism => Some(&PRISM_EDGES),
        CellType::Pyramid => Some(&PYRAMID_EDGES),
        _ => None,
    }
}

fn faces_for_cell(cell_type: CellType) -> Option<&'static [&'static [usize]]> {
    match cell_type {
        CellType::Triangle => Some(&TRI_FACES),
        CellType::Quadrilateral => Some(&QUAD_FACES),
        CellType::Tetrahedron => Some(&TET_FACES),
        CellType::Hexahedron => Some(&HEX_FACES),
        CellType::Prism => Some(&PRISM_FACES),
        CellType::Pyramid => Some(&PYRAMID_FACES),
        _ => None,
    }
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

fn angle_deg(a: [f64; 3], b: [f64; 3]) -> Result<f64, MeshSieveError> {
    let na = norm(a);
    let nb = norm(b);
    if na <= EPS || nb <= EPS {
        return Err(MeshSieveError::InvalidGeometry(
            "zero-length edge detected".into(),
        ));
    }
    let mut cos = dot(a, b) / (na * nb);
    cos = cos.clamp(-1.0, 1.0);
    Ok(cos.acos() * 180.0 / PI)
}

const TRI_EDGES: [(usize, usize); 3] = [(0, 1), (1, 2), (2, 0)];
const QUAD_EDGES: [(usize, usize); 4] = [(0, 1), (1, 2), (2, 3), (3, 0)];
const TET_EDGES: [(usize, usize); 6] = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)];
const HEX_EDGES: [(usize, usize); 12] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
];
const PRISM_EDGES: [(usize, usize); 9] = [
    (0, 1),
    (1, 2),
    (2, 0),
    (3, 4),
    (4, 5),
    (5, 3),
    (0, 3),
    (1, 4),
    (2, 5),
];
const PYRAMID_EDGES: [(usize, usize); 8] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (0, 4),
    (1, 4),
    (2, 4),
    (3, 4),
];

const TRI_FACE: [usize; 3] = [0, 1, 2];
const QUAD_FACE: [usize; 4] = [0, 1, 2, 3];
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

const TRI_FACES: [&[usize]; 1] = [&TRI_FACE];
const QUAD_FACES: [&[usize]; 1] = [&QUAD_FACE];
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
