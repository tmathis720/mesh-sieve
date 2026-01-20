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
        let expected = expected_vertex_count(cell_type).ok_or_else(|| {
            MeshSieveError::InvalidGeometry(format!("unsupported cell type: {cell_type:?}"))
        })?;
        let vertices = cell_vertices(sieve, cell, expected)?;

        let edges = cell_edges(cell_type, &vertices)?;
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

        for (face_vertices, face_type) in cell_faces(cell_type, &vertices)? {
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
        _ => None,
    }
}

fn cell_vertices(
    sieve: &mut impl Sieve<Point = PointId>,
    cell: PointId,
    expected: usize,
) -> Result<Vec<PointId>, MeshSieveError> {
    let cone: Vec<PointId> = sieve.cone_points(cell).collect();
    if cone.len() != expected {
        return Err(MeshSieveError::InvalidGeometry(format!(
            "cell {cell:?} expected {expected} vertices, got {}",
            cone.len()
        )));
    }
    for v in &cone {
        if sieve.cone_points(*v).next().is_some() {
            return Err(MeshSieveError::InvalidGeometry(format!(
                "cell {cell:?} references non-vertex point {v:?}"
            )));
        }
    }
    Ok(cone)
}

fn cell_edges(
    cell_type: CellType,
    vertices: &[PointId],
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
) -> Result<Vec<(Vec<PointId>, CellType)>, MeshSieveError> {
    let faces = match cell_type {
        CellType::Triangle | CellType::Quadrilateral => Vec::new(),
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
        _ => {
            return Err(MeshSieveError::InvalidGeometry(format!(
                "unsupported cell type: {cell_type:?}"
            )));
        }
    };
    Ok(faces)
}

fn face_edges(vertices: &[PointId]) -> Result<Vec<[PointId; 2]>, MeshSieveError> {
    match vertices.len() {
        3 => Ok(vec![
            [vertices[0], vertices[1]],
            [vertices[1], vertices[2]],
            [vertices[2], vertices[0]],
        ]),
        4 => Ok(vec![
            [vertices[0], vertices[1]],
            [vertices[1], vertices[2]],
            [vertices[2], vertices[3]],
            [vertices[3], vertices[0]],
        ]),
        n => Err(MeshSieveError::InvalidGeometry(format!(
            "unsupported face vertex count: {n}"
        ))),
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::atlas::Atlas;
    use crate::data::storage::VecStorage;
    use crate::topology::sieve::InMemorySieve;

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
}
