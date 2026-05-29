use crate::data::coordinates::Coordinates;
use crate::data::section::Section;
use crate::data::storage::Storage;
use crate::mesh_error::MeshSieveError;
use crate::topology::cell_type::CellType;
use crate::topology::point::PointId;
use crate::topology::sieve::Sieve;
use std::cmp::Ordering;

const EPS: f64 = 1e-12;
/// Assumption for polygonal faces:
/// vertices are provided in a boundary-walk order (clockwise or counterclockwise).
/// Faces may be mildly non-planar; centroid and area are computed by fan triangulation
/// around the arithmetic mean projected along a Newell normal surrogate.

#[derive(Clone, Debug)]
pub struct FvmFaceMetrics {
    pub face: PointId,
    pub owner: PointId,
    pub neighbor: Option<PointId>,
    pub centroid: [f64; 3],
    pub area_vector: [f64; 3],
    pub area_magnitude: f64,
    pub owner_to_neighbor: Option<[f64; 3]>,
    pub orthogonal_distance: f64,
    pub non_orthogonality_deg: Option<f64>,
    pub skewness_vector: [f64; 3],
    pub owner_to_face: [f64; 3],
    pub neighbor_to_face: Option<[f64; 3]>,
}

pub fn build_fvm_face_metrics<S, CT, CS>(
    sieve: &S,
    cell_types: &Section<CellType, CT>,
    coordinates: &Coordinates<f64, CS>,
) -> Result<Vec<FvmFaceMetrics>, MeshSieveError>
where
    S: Sieve<Point = PointId>,
    CT: Storage<CellType>,
    CS: Storage<f64>,
{
    let mut faces: Vec<PointId> = sieve
        .points()
        .filter(|p| is_face(*p, sieve, cell_types).unwrap_or(false))
        .collect();
    faces.sort_unstable();
    faces
        .into_iter()
        .map(|f| build_face_metric(sieve, cell_types, coordinates, f))
        .collect()
}

pub fn build_face_metric<S, CT, CS>(
    sieve: &S,
    cell_types: &Section<CellType, CT>,
    coordinates: &Coordinates<f64, CS>,
    face: PointId,
) -> Result<FvmFaceMetrics, MeshSieveError>
where
    S: Sieve<Point = PointId>,
    CT: Storage<CellType>,
    CS: Storage<f64>,
{
    let mut supports: Vec<PointId> = sieve
        .support_points(face)
        .filter(|p| is_cell(*p, cell_types).unwrap_or(false))
        .collect();
    supports.sort_unstable();
    if supports.is_empty() {
        return Err(MeshSieveError::InvalidGeometry(format!(
            "face {face:?} has no cell support"
        )));
    }
    let owner = supports[0];
    let neighbor = supports.get(1).copied();

    let vertices = face_vertices(sieve, face, coordinates)?;
    if vertices.len() < 2 {
        return Err(MeshSieveError::InvalidGeometry(
            "face with <2 vertices".into(),
        ));
    }
    let centroid = polygon_centroid(&vertices);
    let owner_centroid = cell_centroid(sieve, owner, coordinates)?;
    let mut area_vec = polygon_area_vector(&vertices);
    if let Some(n) = neighbor {
        let nc = cell_centroid(sieve, n, coordinates)?;
        if dot(area_vec, sub(nc, owner_centroid)) < 0.0 {
            area_vec = scale(area_vec, -1.0);
        }
    } else if dot(area_vec, sub(centroid, owner_centroid)) < 0.0 {
        area_vec = scale(area_vec, -1.0);
    }
    let area_mag = norm(area_vec);
    let owner_to_face = sub(centroid, owner_centroid);

    let (owner_to_neighbor, orth_dist, non_ortho, skew, neigh_to_face) = if let Some(n) = neighbor {
        let nc = cell_centroid(sieve, n, coordinates)?;
        let d = sub(nc, owner_centroid);
        let dmag = norm(d);
        let ahat = if area_mag > EPS {
            scale(area_vec, 1.0 / area_mag)
        } else {
            [0.0; 3]
        };
        let orth = dot(d, ahat).abs();
        let angle = if dmag > EPS && area_mag > EPS {
            let c = (dot(area_vec, d).abs() / (area_mag * dmag)).clamp(-1.0, 1.0);
            Some(c.acos().to_degrees())
        } else {
            None
        };
        let lambda = if dmag > EPS {
            dot(sub(centroid, owner_centroid), d) / (dmag * dmag)
        } else {
            0.5
        };
        let closest_on_d = add(owner_centroid, scale(d, lambda));
        let skew = sub(centroid, closest_on_d);
        (Some(d), orth, angle, skew, Some(sub(centroid, nc)))
    } else {
        let ahat = if area_mag > EPS {
            scale(area_vec, 1.0 / area_mag)
        } else {
            [0.0; 3]
        };
        let orth = dot(owner_to_face, ahat).abs();
        let skew = sub(
            centroid,
            add(owner_centroid, scale(ahat, dot(owner_to_face, ahat))),
        );
        (None, orth, None, skew, None)
    };

    Ok(FvmFaceMetrics {
        face,
        owner,
        neighbor,
        centroid,
        area_vector: area_vec,
        area_magnitude: area_mag,
        owner_to_neighbor,
        orthogonal_distance: orth_dist,
        non_orthogonality_deg: non_ortho,
        skewness_vector: skew,
        owner_to_face,
        neighbor_to_face: neigh_to_face,
    })
}

fn is_cell<C: Storage<CellType>>(
    p: PointId,
    cell_types: &Section<CellType, C>,
) -> Result<bool, MeshSieveError> {
    Ok(cell_types
        .try_restrict(p)
        .ok()
        .and_then(|v| v.first().copied())
        .map(|ct| ct.dimension() >= 2)
        .unwrap_or(false))
}

fn is_face<S, CT>(
    p: PointId,
    sieve: &S,
    cell_types: &Section<CellType, CT>,
) -> Result<bool, MeshSieveError>
where
    S: Sieve<Point = PointId>,
    CT: Storage<CellType>,
{
    let support_cells = sieve
        .support_points(p)
        .filter(|q| is_cell(*q, cell_types).unwrap_or(false))
        .count();
    if support_cells == 0 || support_cells > 2 {
        return Ok(false);
    }
    let cone: Vec<_> = sieve.cone_points(p).collect();
    if cone.len() < 2 {
        return Ok(false);
    }
    Ok(cone
        .into_iter()
        .all(|q| sieve.cone_points(q).next().is_none()))
}

fn face_vertices<S, C: Storage<f64>>(
    sieve: &S,
    face: PointId,
    coordinates: &Coordinates<f64, C>,
) -> Result<Vec<[f64; 3]>, MeshSieveError>
where
    S: Sieve<Point = PointId>,
{
    let mut points: Vec<_> = sieve.cone_points(face).collect();
    points.sort_unstable();
    points
        .into_iter()
        .map(|p| coord3(coordinates.try_restrict(p)?))
        .collect()
}

fn cell_centroid<S, C: Storage<f64>>(
    sieve: &S,
    cell: PointId,
    coordinates: &Coordinates<f64, C>,
) -> Result<[f64; 3], MeshSieveError>
where
    S: Sieve<Point = PointId>,
{
    let mut verts: Vec<PointId> = sieve
        .closure_iter([cell])
        .filter(|p| sieve.cone_points(*p).next().is_none())
        .collect();
    verts.sort_unstable();
    verts.dedup();
    let coords: Result<Vec<_>, _> = verts
        .into_iter()
        .map(|v| coord3(coordinates.try_restrict(v)?))
        .collect();
    Ok(mean(&coords?))
}

fn coord3(values: &[f64]) -> Result<[f64; 3], MeshSieveError> {
    match values.len().cmp(&3) {
        Ordering::Equal => Ok([values[0], values[1], values[2]]),
        Ordering::Less if values.len() == 2 => Ok([values[0], values[1], 0.0]),
        _ => Err(MeshSieveError::InvalidGeometry(
            "expected 2D/3D coordinates".into(),
        )),
    }
}

fn polygon_area_vector(vertices: &[[f64; 3]]) -> [f64; 3] {
    if vertices.len() == 2 {
        return sub(vertices[1], vertices[0]);
    }
    let c = mean(vertices);
    let mut sum = [0.0; 3];
    for i in 0..vertices.len() {
        let a = sub(vertices[i], c);
        let b = sub(vertices[(i + 1) % vertices.len()], c);
        sum = add(sum, scale(cross(a, b), 0.5));
    }
    sum
}

fn polygon_centroid(vertices: &[[f64; 3]]) -> [f64; 3] {
    if vertices.len() <= 2 {
        return mean(vertices);
    }
    let c0 = mean(vertices);
    let mut weighted = [0.0; 3];
    let mut wsum = 0.0;
    for i in 0..vertices.len() {
        let a = vertices[i];
        let b = vertices[(i + 1) % vertices.len()];
        let tri_n = cross(sub(a, c0), sub(b, c0));
        let w = 0.5 * norm(tri_n);
        if w > EPS {
            let tri_c = scale(add(add(c0, a), b), 1.0 / 3.0);
            weighted = add(weighted, scale(tri_c, w));
            wsum += w;
        }
    }
    if wsum > EPS {
        scale(weighted, 1.0 / wsum)
    } else {
        c0
    }
}

fn mean(v: &[[f64; 3]]) -> [f64; 3] {
    let mut c = [0.0; 3];
    for p in v {
        c = add(c, *p);
    }
    scale(c, 1.0 / (v.len() as f64))
}
fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
