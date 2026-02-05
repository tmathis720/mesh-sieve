//! Construct intermediate entities (edges/faces) from cell→vertex meshes.
//!
//! # Expected cell types
//! This module assumes cells reference their vertices in a standard ordering:
//!
//! - [`CellType::Triangle`]: `(0,1,2)`
//! - [`CellType::Quadrilateral`]: `(0,1,2,3)`
//! - [`CellType::Tetrahedron`]: `(0,1,2,3)`
//! - [`CellType::Hexahedron`]: `(0,1,2,3,4,5,6,7)` with
//!   `0..=3` the bottom face and `4..=7` the top face.
//! - [`CellType::Polygon(n)`]: `(0..n)` in cyclic order.
//! - [`CellType::Polyhedron`]: requires explicit face connectivity metadata.
//!
//! The interpolation routine uses these orderings to derive edges and faces,
//! then inserts the appropriate arrows so the sieve has
//! `cell → face → edge → vertex` (for 3D cells) or `cell → edge → vertex`
//! (for 2D cells) connectivity.
//!
//! # Example
//! ```rust
//! # fn try_main() -> Result<(), mesh_sieve::mesh_error::MeshSieveError> {
//! use mesh_sieve::algs::interpolate::interpolate_edges_faces;
//! use mesh_sieve::data::atlas::Atlas;
//! use mesh_sieve::data::section::Section;
//! use mesh_sieve::data::storage::VecStorage;
//! use mesh_sieve::topology::cell_type::CellType;
//! use mesh_sieve::topology::point::PointId;
//! use mesh_sieve::topology::sieve::{InMemorySieve, MutableSieve, Sieve};
//!
//! let mut sieve = InMemorySieve::<PointId, ()>::default();
//! let cell = PointId::new(1)?;
//! let v = |i| PointId::new(i).unwrap();
//! for p in [cell, v(2), v(3), v(4), v(5)] {
//!     MutableSieve::add_point(&mut sieve, p);
//! }
//! for vert in [v(2), v(3), v(4), v(5)] {
//!     sieve.add_arrow(cell, vert, ());
//! }
//!
//! let mut atlas = Atlas::default();
//! for p in [cell, v(2), v(3), v(4), v(5)] {
//!     atlas.try_insert(p, 1)?;
//! }
//! let mut cell_types = Section::<CellType, VecStorage<CellType>>::new(atlas);
//! cell_types.try_set(cell, &[CellType::Quadrilateral])?;
//! for p in [v(2), v(3), v(4), v(5)] {
//!     cell_types.try_set(p, &[CellType::Vertex])?;
//! }
//!
//! let result = interpolate_edges_faces(&mut sieve, &mut cell_types)?;
//! assert_eq!(result.edge_points.len(), 4);
//! # Ok(())
//! # }
//! ```

use crate::data::section::Section;
use crate::data::storage::Storage;
use crate::mesh_error::MeshSieveError;
use crate::topology::cell_type::CellType;
use crate::topology::point::PointId;
use crate::topology::sieve::{MutableSieve, Sieve};
use std::collections::BTreeMap;

/// Edge/face bookkeeping produced during interpolation.
#[derive(Debug, Default)]
pub struct InterpolationResult {
    /// Canonical edge key `(min,max)` → edge point.
    pub edge_points: BTreeMap<(PointId, PointId), PointId>,
    /// Canonical face key (sorted vertices) → face point.
    pub face_points: BTreeMap<Vec<PointId>, PointId>,
}

/// Optional per-cell ordering/connectivity metadata for interpolation.
#[derive(Debug, Default)]
pub struct InterpolationOrdering {
    /// Per-cell permutation of the cell's cone vertices.
    pub vertex_permutations: BTreeMap<PointId, Vec<usize>>,
    /// Per-cell face connectivity for polyhedra, expressed as vertex indices.
    pub polyhedron_faces: BTreeMap<PointId, Vec<Vec<usize>>>,
}

/// Insert intermediate edges/faces into a cell→vertex mesh.
pub fn interpolate_edges_faces<S, CtSt>(
    sieve: &mut S,
    cell_types: &mut Section<CellType, CtSt>,
) -> Result<InterpolationResult, MeshSieveError>
where
    S: MutableSieve<Point = PointId>,
    S::Payload: Default,
    CtSt: Storage<CellType> + Clone,
{
    interpolate_edges_faces_with_ordering(sieve, cell_types, None)
}

/// Insert intermediate edges/faces into a cell→vertex mesh with explicit ordering metadata.
pub fn interpolate_edges_faces_with_ordering<S, CtSt>(
    sieve: &mut S,
    cell_types: &mut Section<CellType, CtSt>,
    ordering: Option<&InterpolationOrdering>,
) -> Result<InterpolationResult, MeshSieveError>
where
    S: MutableSieve<Point = PointId>,
    S::Payload: Default,
    CtSt: Storage<CellType> + Clone,
{
    let mut max_id = 0u64;
    for p in sieve.points() {
        max_id = max_id.max(p.get());
    }
    let mut next_id = max_id
        .checked_add(1)
        .ok_or(MeshSieveError::InvalidPointId)?;

    let cells = collect_cells(cell_types)?;
    let mut result = InterpolationResult::default();

    for (cell, cell_type) in cells {
        let expected = expected_vertex_count(cell_type);
        let permutation = ordering.and_then(|meta| {
            meta.vertex_permutations
                .get(&cell)
                .map(|perm| perm.as_slice())
        });
        let vertices = cell_vertices(sieve, cell, expected, permutation)?;

        let poly_faces = ordering.and_then(|meta| {
            meta.polyhedron_faces
                .get(&cell)
                .map(|faces| faces.as_slice())
        });
        let edges = cell_edges(cell_type, &vertices, poly_faces)?;
        for [a, b] in edges {
            let edge = edge_point(
                &mut result.edge_points,
                &mut next_id,
                sieve,
                cell_types,
                a,
                b,
            )?;
            sieve.add_arrow(cell, edge, S::Payload::default());
        }

        for (face_vertices, face_type) in cell_faces(cell_type, &vertices, poly_faces)? {
            let face = face_point(
                &mut result.face_points,
                &mut next_id,
                sieve,
                cell_types,
                &face_vertices,
                face_type,
            )?;
            for [a, b] in face_edges(&face_vertices)? {
                let edge = edge_point(
                    &mut result.edge_points,
                    &mut next_id,
                    sieve,
                    cell_types,
                    a,
                    b,
                )?;
                sieve.add_arrow(face, edge, S::Payload::default());
            }
            sieve.add_arrow(cell, face, S::Payload::default());
        }
    }

    Ok(result)
}

fn collect_cells<CtSt>(
    cell_types: &Section<CellType, CtSt>,
) -> Result<Vec<(PointId, CellType)>, MeshSieveError>
where
    CtSt: Storage<CellType>,
{
    let mut cells = Vec::new();
    for (point, cell_slice) in cell_types.iter() {
        if cell_slice.len() != 1 {
            return Err(MeshSieveError::SliceLengthMismatch {
                point,
                expected: 1,
                found: cell_slice.len(),
            });
        }
        let cell_type = cell_slice[0];
        if matches!(
            cell_type,
            CellType::Triangle
                | CellType::Quadrilateral
                | CellType::Tetrahedron
                | CellType::Hexahedron
                | CellType::Prism
                | CellType::Polygon(_)
                | CellType::Polyhedron
        ) {
            cells.push((point, cell_type));
        } else if cell_type.dimension() >= 2 {
            return Err(MeshSieveError::InvalidGeometry(format!(
                "unsupported cell type: {cell_type:?}"
            )));
        }
    }
    Ok(cells)
}

fn expected_vertex_count(cell_type: CellType) -> Option<usize> {
    match cell_type {
        CellType::Triangle => Some(3),
        CellType::Quadrilateral => Some(4),
        CellType::Tetrahedron => Some(4),
        CellType::Hexahedron => Some(8),
        CellType::Polygon(n) => Some(n as usize),
        CellType::Prism => Some(6),
        CellType::Polyhedron => None,
        _ => None,
    }
}

fn cell_vertices(
    sieve: &mut impl Sieve<Point = PointId>,
    cell: PointId,
    expected: Option<usize>,
    permutation: Option<&[usize]>,
) -> Result<Vec<PointId>, MeshSieveError> {
    let cone: Vec<PointId> = sieve.cone_points(cell).collect();
    if let Some(expected) = expected {
        if cone.len() != expected {
            return Err(MeshSieveError::InvalidGeometry(format!(
                "cell {cell:?} expected {expected} vertices, got {}",
                cone.len()
            )));
        }
    }
    for v in &cone {
        if sieve.cone_points(*v).next().is_some() {
            return Err(MeshSieveError::InvalidGeometry(format!(
                "cell {cell:?} references non-vertex point {v:?}"
            )));
        }
    }
    if let Some(permutation) = permutation {
        return apply_permutation(cell, &cone, permutation);
    }
    Ok(cone)
}

fn cell_edges(
    cell_type: CellType,
    vertices: &[PointId],
    polyhedron_faces: Option<&[Vec<usize>]>,
) -> Result<Vec<[PointId; 2]>, MeshSieveError> {
    let edges = match cell_type {
        CellType::Triangle => vec![
            [vertices[0], vertices[1]],
            [vertices[1], vertices[2]],
            [vertices[2], vertices[0]],
        ],
        CellType::Quadrilateral => vec![
            [vertices[0], vertices[1]],
            [vertices[1], vertices[2]],
            [vertices[2], vertices[3]],
            [vertices[3], vertices[0]],
        ],
        CellType::Polygon(_) => polygon_edges(vertices),
        CellType::Tetrahedron => vec![
            [vertices[0], vertices[1]],
            [vertices[1], vertices[2]],
            [vertices[2], vertices[0]],
            [vertices[0], vertices[3]],
            [vertices[1], vertices[3]],
            [vertices[2], vertices[3]],
        ],
        CellType::Hexahedron => vec![
            [vertices[0], vertices[1]],
            [vertices[1], vertices[2]],
            [vertices[2], vertices[3]],
            [vertices[3], vertices[0]],
            [vertices[4], vertices[5]],
            [vertices[5], vertices[6]],
            [vertices[6], vertices[7]],
            [vertices[7], vertices[4]],
            [vertices[0], vertices[4]],
            [vertices[1], vertices[5]],
            [vertices[2], vertices[6]],
            [vertices[3], vertices[7]],
        ],
        CellType::Prism => vec![
            [vertices[0], vertices[1]],
            [vertices[1], vertices[2]],
            [vertices[2], vertices[0]],
            [vertices[3], vertices[4]],
            [vertices[4], vertices[5]],
            [vertices[5], vertices[3]],
            [vertices[0], vertices[3]],
            [vertices[1], vertices[4]],
            [vertices[2], vertices[5]],
        ],
        CellType::Polyhedron => polyhedron_edges(vertices, polyhedron_faces)?,
        _ => {
            return Err(MeshSieveError::InvalidGeometry(format!(
                "unsupported cell type: {cell_type:?}"
            )));
        }
    };
    Ok(edges)
}

fn cell_faces(
    cell_type: CellType,
    vertices: &[PointId],
    polyhedron_faces: Option<&[Vec<usize>]>,
) -> Result<Vec<(Vec<PointId>, CellType)>, MeshSieveError> {
    let faces = match cell_type {
        CellType::Triangle | CellType::Quadrilateral | CellType::Polygon(_) => Vec::new(),
        CellType::Tetrahedron => vec![
            (
                vec![vertices[0], vertices[1], vertices[2]],
                CellType::Triangle,
            ),
            (
                vec![vertices[0], vertices[1], vertices[3]],
                CellType::Triangle,
            ),
            (
                vec![vertices[1], vertices[2], vertices[3]],
                CellType::Triangle,
            ),
            (
                vec![vertices[0], vertices[2], vertices[3]],
                CellType::Triangle,
            ),
        ],
        CellType::Hexahedron => vec![
            (
                vec![vertices[0], vertices[1], vertices[2], vertices[3]],
                CellType::Quadrilateral,
            ),
            (
                vec![vertices[4], vertices[5], vertices[6], vertices[7]],
                CellType::Quadrilateral,
            ),
            (
                vec![vertices[0], vertices[1], vertices[5], vertices[4]],
                CellType::Quadrilateral,
            ),
            (
                vec![vertices[1], vertices[2], vertices[6], vertices[5]],
                CellType::Quadrilateral,
            ),
            (
                vec![vertices[2], vertices[3], vertices[7], vertices[6]],
                CellType::Quadrilateral,
            ),
            (
                vec![vertices[3], vertices[0], vertices[4], vertices[7]],
                CellType::Quadrilateral,
            ),
        ],
        CellType::Prism => vec![
            (
                vec![vertices[0], vertices[1], vertices[2]],
                CellType::Triangle,
            ),
            (
                vec![vertices[3], vertices[4], vertices[5]],
                CellType::Triangle,
            ),
            (
                vec![vertices[0], vertices[1], vertices[4], vertices[3]],
                CellType::Quadrilateral,
            ),
            (
                vec![vertices[1], vertices[2], vertices[5], vertices[4]],
                CellType::Quadrilateral,
            ),
            (
                vec![vertices[2], vertices[0], vertices[3], vertices[5]],
                CellType::Quadrilateral,
            ),
        ],
        CellType::Polyhedron => polyhedron_faces_to_vertices(vertices, polyhedron_faces)?,
        _ => {
            return Err(MeshSieveError::InvalidGeometry(format!(
                "unsupported cell type: {cell_type:?}"
            )));
        }
    };
    Ok(faces)
}

fn face_edges(vertices: &[PointId]) -> Result<Vec<[PointId; 2]>, MeshSieveError> {
    let n = vertices.len();
    if n < 3 {
        return Err(MeshSieveError::InvalidGeometry(format!(
            "unsupported face vertex count: {n}"
        )));
    }
    let mut edges = Vec::with_capacity(n);
    for i in 0..n {
        edges.push([vertices[i], vertices[(i + 1) % n]]);
    }
    Ok(edges)
}

fn edge_point<S, CtSt>(
    edges: &mut BTreeMap<(PointId, PointId), PointId>,
    next_id: &mut u64,
    sieve: &mut S,
    cell_types: &mut Section<CellType, CtSt>,
    a: PointId,
    b: PointId,
) -> Result<PointId, MeshSieveError>
where
    S: MutableSieve<Point = PointId>,
    S::Payload: Default,
    CtSt: Storage<CellType> + Clone,
{
    let key = if a < b { (a, b) } else { (b, a) };
    if let Some(p) = edges.get(&key) {
        return Ok(*p);
    }
    let edge = alloc_point(next_id)?;
    edges.insert(key, edge);
    MutableSieve::add_point(sieve, edge);
    sieve.add_arrow(edge, a, S::Payload::default());
    sieve.add_arrow(edge, b, S::Payload::default());
    cell_types.try_add_point(edge, 1)?;
    cell_types.try_set(edge, &[CellType::Segment])?;
    Ok(edge)
}

fn face_point<S, CtSt>(
    faces: &mut BTreeMap<Vec<PointId>, PointId>,
    next_id: &mut u64,
    sieve: &mut S,
    cell_types: &mut Section<CellType, CtSt>,
    vertices: &[PointId],
    face_type: CellType,
) -> Result<PointId, MeshSieveError>
where
    S: MutableSieve<Point = PointId>,
    S::Payload: Default,
    CtSt: Storage<CellType> + Clone,
{
    let mut key = vertices.to_vec();
    key.sort();
    if let Some(p) = faces.get(&key) {
        return Ok(*p);
    }
    let face = alloc_point(next_id)?;
    faces.insert(key, face);
    MutableSieve::add_point(sieve, face);
    cell_types.try_add_point(face, 1)?;
    cell_types.try_set(face, &[face_type])?;
    Ok(face)
}

fn alloc_point(next_id: &mut u64) -> Result<PointId, MeshSieveError> {
    let id = PointId::new(*next_id)?;
    *next_id = next_id
        .checked_add(1)
        .ok_or(MeshSieveError::InvalidPointId)?;
    Ok(id)
}

fn apply_permutation(
    cell: PointId,
    vertices: &[PointId],
    permutation: &[usize],
) -> Result<Vec<PointId>, MeshSieveError> {
    if permutation.len() != vertices.len() {
        return Err(MeshSieveError::InvalidGeometry(format!(
            "cell {cell:?} permutation length {} does not match vertex count {}",
            permutation.len(),
            vertices.len()
        )));
    }
    let mut seen = vec![false; vertices.len()];
    let mut ordered = Vec::with_capacity(vertices.len());
    for &idx in permutation {
        if idx >= vertices.len() {
            return Err(MeshSieveError::InvalidGeometry(format!(
                "cell {cell:?} permutation index {idx} out of bounds"
            )));
        }
        if seen[idx] {
            return Err(MeshSieveError::InvalidGeometry(format!(
                "cell {cell:?} permutation index {idx} appears multiple times"
            )));
        }
        seen[idx] = true;
        ordered.push(vertices[idx]);
    }
    Ok(ordered)
}

fn polygon_edges(vertices: &[PointId]) -> Vec<[PointId; 2]> {
    let n = vertices.len();
    let mut edges = Vec::with_capacity(n);
    for i in 0..n {
        edges.push([vertices[i], vertices[(i + 1) % n]]);
    }
    edges
}

fn polyhedron_edges(
    vertices: &[PointId],
    polyhedron_faces: Option<&[Vec<usize>]>,
) -> Result<Vec<[PointId; 2]>, MeshSieveError> {
    let faces = polyhedron_faces.ok_or_else(|| {
        MeshSieveError::InvalidGeometry("polyhedron faces metadata required".to_string())
    })?;
    let mut edges = BTreeMap::new();
    for face in faces {
        let face_vertices = face_indices_to_vertices(vertices, face)?;
        for [a, b] in face_edges(&face_vertices)? {
            let key = if a < b { (a, b) } else { (b, a) };
            edges.entry(key).or_insert([a, b]);
        }
    }
    Ok(edges.values().copied().collect())
}

fn polyhedron_faces_to_vertices(
    vertices: &[PointId],
    polyhedron_faces: Option<&[Vec<usize>]>,
) -> Result<Vec<(Vec<PointId>, CellType)>, MeshSieveError> {
    let faces = polyhedron_faces.ok_or_else(|| {
        MeshSieveError::InvalidGeometry("polyhedron faces metadata required".to_string())
    })?;
    let mut result = Vec::with_capacity(faces.len());
    for face in faces {
        let face_vertices = face_indices_to_vertices(vertices, face)?;
        let face_type = face_type_for_vertices(face_vertices.len())?;
        result.push((face_vertices, face_type));
    }
    Ok(result)
}

fn face_indices_to_vertices(
    vertices: &[PointId],
    face: &[usize],
) -> Result<Vec<PointId>, MeshSieveError> {
    if face.len() < 3 {
        return Err(MeshSieveError::InvalidGeometry(format!(
            "unsupported face vertex count: {}",
            face.len()
        )));
    }
    let mut face_vertices = Vec::with_capacity(face.len());
    for &idx in face {
        let v = vertices.get(idx).ok_or_else(|| {
            MeshSieveError::InvalidGeometry(format!(
                "face index {idx} out of bounds for {} vertices",
                vertices.len()
            ))
        })?;
        face_vertices.push(*v);
    }
    Ok(face_vertices)
}

fn face_type_for_vertices(count: usize) -> Result<CellType, MeshSieveError> {
    match count {
        3 => Ok(CellType::Triangle),
        4 => Ok(CellType::Quadrilateral),
        n => {
            let n_u8 = u8::try_from(n).map_err(|_| {
                MeshSieveError::InvalidGeometry(format!("polygon face vertex count {n} exceeds u8"))
            })?;
            Ok(CellType::Polygon(n_u8))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::atlas::Atlas;
    use crate::data::storage::VecStorage;
    use crate::topology::sieve::{InMemorySieve, Sieve};

    fn v(raw: u64) -> PointId {
        PointId::new(raw).unwrap()
    }

    #[test]
    fn interpolates_quad_edges() {
        let mut sieve = InMemorySieve::<PointId, ()>::default();
        let cell = v(1);
        let vertices = [v(2), v(3), v(4), v(5)];
        MutableSieve::add_point(&mut sieve, cell);
        for p in vertices {
            MutableSieve::add_point(&mut sieve, p);
            sieve.add_arrow(cell, p, ());
        }

        let mut atlas = Atlas::default();
        atlas.try_insert(cell, 1).unwrap();
        for p in vertices {
            atlas.try_insert(p, 1).unwrap();
        }
        let mut cell_types = Section::<CellType, VecStorage<CellType>>::new(atlas);
        cell_types
            .try_set(cell, &[CellType::Quadrilateral])
            .unwrap();
        for p in vertices {
            cell_types.try_set(p, &[CellType::Vertex]).unwrap();
        }

        let result = interpolate_edges_faces(&mut sieve, &mut cell_types).unwrap();
        assert_eq!(result.edge_points.len(), 4);
        assert!(result.face_points.is_empty());
        for ((a, b), edge) in &result.edge_points {
            assert!(sieve.has_arrow(*edge, *a));
            assert!(sieve.has_arrow(*edge, *b));
            assert!(sieve.has_arrow(cell, *edge));
        }
    }

    #[test]
    fn interpolates_mixed_elements_with_ordering() {
        let mut sieve = InMemorySieve::<PointId, ()>::default();
        let cell_quad = v(1);
        let quad_vertices = [v(2), v(3), v(4), v(5)];
        let cell_polygon = v(10);
        let polygon_vertices = [v(11), v(12), v(13), v(14), v(15)];
        let cell_polyhedron = v(20);
        let poly_vertices = [v(21), v(22), v(23), v(24)];

        let mut all_points = vec![cell_quad, cell_polygon, cell_polyhedron];
        all_points.extend(quad_vertices);
        all_points.extend(polygon_vertices);
        all_points.extend(poly_vertices);
        for p in &all_points {
            MutableSieve::add_point(&mut sieve, *p);
        }
        for p in quad_vertices {
            sieve.add_arrow(cell_quad, p, ());
        }
        for p in polygon_vertices {
            sieve.add_arrow(cell_polygon, p, ());
        }
        for p in poly_vertices {
            sieve.add_arrow(cell_polyhedron, p, ());
        }

        let mut atlas = Atlas::default();
        for p in &all_points {
            atlas.try_insert(*p, 1).unwrap();
        }
        let mut cell_types = Section::<CellType, VecStorage<CellType>>::new(atlas);
        cell_types
            .try_set(cell_quad, &[CellType::Quadrilateral])
            .unwrap();
        cell_types
            .try_set(cell_polygon, &[CellType::Polygon(5)])
            .unwrap();
        cell_types
            .try_set(cell_polyhedron, &[CellType::Polyhedron])
            .unwrap();
        for p in quad_vertices
            .into_iter()
            .chain(polygon_vertices)
            .chain(poly_vertices)
        {
            cell_types.try_set(p, &[CellType::Vertex]).unwrap();
        }

        let mut ordering = InterpolationOrdering::default();
        ordering
            .vertex_permutations
            .insert(cell_polygon, vec![0, 2, 3, 4, 1]);
        ordering.polyhedron_faces.insert(
            cell_polyhedron,
            vec![vec![0, 1, 2], vec![0, 1, 3], vec![1, 2, 3], vec![0, 2, 3]],
        );

        let result =
            interpolate_edges_faces_with_ordering(&mut sieve, &mut cell_types, Some(&ordering))
                .unwrap();

        assert_eq!(result.edge_points.len(), 15);
        assert_eq!(result.face_points.len(), 4);

        let key = if polygon_vertices[0] < polygon_vertices[2] {
            (polygon_vertices[0], polygon_vertices[2])
        } else {
            (polygon_vertices[2], polygon_vertices[0])
        };
        assert!(result.edge_points.contains_key(&key));

        for face in result.face_points.values() {
            assert!(sieve.has_arrow(cell_polyhedron, *face));
            let face_type = cell_types.try_restrict(*face).unwrap()[0];
            assert!(matches!(face_type, CellType::Triangle));
            assert!(sieve.cone_points(*face).next().is_some());
        }
    }
}
