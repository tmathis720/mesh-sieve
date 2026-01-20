//! Extrude a 2D surface mesh into a layered 3D mesh.
//!
//! # Expected cell types
//! - [`CellType::Triangle`] surfaces extrude to [`CellType::Prism`] volumes.
//! - [`CellType::Quadrilateral`] surfaces extrude to [`CellType::Hexahedron`] volumes.
//!
//! The input mesh must be a cell→vertex topology; vertices are points with no
//! outgoing arrows. The output mesh only contains volume cells and vertices
//! (no intermediate edges/faces). Use [`crate::algs::interpolate`] if you need
//! edge/face entities after extrusion.
//!
//! # Example
//! ```rust
//! # fn try_main() -> Result<(), mesh_sieve::mesh_error::MeshSieveError> {
//! use mesh_sieve::algs::extrude::extrude_surface_layers;
//! use mesh_sieve::data::atlas::Atlas;
//! use mesh_sieve::data::coordinates::Coordinates;
//! use mesh_sieve::data::section::Section;
//! use mesh_sieve::data::storage::VecStorage;
//! use mesh_sieve::topology::cell_type::CellType;
//! use mesh_sieve::topology::point::PointId;
//! use mesh_sieve::topology::sieve::{InMemorySieve, MutableSieve};
//!
//! let mut surface = InMemorySieve::<PointId, ()>::default();
//! let cell = PointId::new(1)?;
//! let vertices = [PointId::new(2)?, PointId::new(3)?, PointId::new(4)?];
//! MutableSieve::add_point(&mut surface, cell);
//! for v in vertices {
//!     MutableSieve::add_point(&mut surface, v);
//!     surface.add_arrow(cell, v, ());
//! }
//!
//! let mut atlas = Atlas::default();
//! for p in [cell, vertices[0], vertices[1], vertices[2]] {
//!     atlas.try_insert(p, 1)?;
//! }
//! let mut cell_types = Section::<CellType, VecStorage<CellType>>::new(atlas);
//! cell_types.try_set(cell, &[CellType::Triangle])?;
//! for v in vertices {
//!     cell_types.try_set(v, &[CellType::Vertex])?;
//! }
//!
//! let mut coord_atlas = Atlas::default();
//! for v in vertices {
//!     coord_atlas.try_insert(v, 2)?;
//! }
//! let mut coords = Coordinates::<f64, VecStorage<f64>>::try_new(2, coord_atlas)?;
//! coords.section_mut().try_set(vertices[0], &[0.0, 0.0])?;
//! coords.section_mut().try_set(vertices[1], &[1.0, 0.0])?;
//! coords.section_mut().try_set(vertices[2], &[0.0, 1.0])?;
//!
//! let mesh = extrude_surface_layers(&surface, &cell_types, &coords, &[0.0, 1.0])?;
//! assert_eq!(mesh.cell_types.as_ref().unwrap().iter().count(), 7);
//! # Ok(())
//! # }
//! ```

use crate::data::atlas::Atlas;
use crate::data::coordinates::Coordinates;
use crate::data::section::Section;
use crate::data::storage::{Storage, VecStorage};
use crate::io::MeshData;
use crate::mesh_error::MeshSieveError;
use crate::topology::cell_type::CellType;
use crate::topology::point::PointId;
use crate::topology::sieve::{InMemorySieve, MutableSieve, Sieve};
use std::collections::BTreeMap;

/// Extrude a surface mesh into layered volume cells.
pub fn extrude_surface_layers<S, CtSt, Cs>(
    surface: &S,
    cell_types: &Section<CellType, CtSt>,
    coordinates: &Coordinates<f64, Cs>,
    layer_offsets: &[f64],
) -> Result<
    MeshData<InMemorySieve<PointId, ()>, f64, VecStorage<f64>, VecStorage<CellType>>,
    MeshSieveError,
>
where
    S: Sieve<Point = PointId>,
    CtSt: Storage<CellType>,
    Cs: Storage<f64>,
{
    if layer_offsets.len() < 2 {
        return Err(MeshSieveError::InvalidGeometry(
            "layer_offsets must contain at least two entries".into(),
        ));
    }
    let dimension = coordinates.dimension();
    if dimension < 2 {
        return Err(MeshSieveError::InvalidGeometry(
            "surface coordinates must be at least 2D".into(),
        ));
    }

    let (vertices, cells) = collect_surface_entities(surface, cell_types)?;
    let max_id = surface.points().map(|p| p.get()).max().unwrap_or(0);
    let mut next_id = max_id
        .checked_add(1)
        .ok_or(MeshSieveError::InvalidPointId)?;

    let mut sieve = InMemorySieve::<PointId, ()>::default();
    let mut coord_atlas = Atlas::default();
    let mut cell_atlas = Atlas::default();

    let mut layer_points: BTreeMap<(PointId, usize), PointId> = BTreeMap::new();
    for &vertex in &vertices {
        for (layer, _) in layer_offsets.iter().enumerate() {
            let new_point = alloc_point(&mut next_id)?;
            MutableSieve::add_point(&mut sieve, new_point);
            coord_atlas.try_insert(new_point, 3)?;
            cell_atlas.try_insert(new_point, 1)?;
            layer_points.insert((vertex, layer), new_point);
        }
    }

    let mut coords = Coordinates::<f64, VecStorage<f64>>::try_new(3, coord_atlas)?;
    for (&(vertex, layer), &new_point) in &layer_points {
        let base = coordinates.try_restrict(vertex)?;
        let mut xyz = [0.0; 3];
        xyz[0] = base[0];
        xyz[1] = base[1];
        if dimension > 2 {
            xyz[2] = base[2];
        }
        xyz[2] += layer_offsets[layer];
        coords.section_mut().try_set(new_point, &xyz)?;
    }

    let mut cell_types_out = Section::<CellType, VecStorage<CellType>>::new(cell_atlas);
    for &point in layer_points.values() {
        cell_types_out.try_set(point, &[CellType::Vertex])?;
    }

    for (_cell, cell_type, vertices) in cells {
        let cell_kind = match cell_type {
            CellType::Triangle => CellType::Prism,
            CellType::Quadrilateral => CellType::Hexahedron,
            _ => {
                return Err(MeshSieveError::InvalidGeometry(format!(
                    "unsupported surface cell type: {cell_type:?}"
                )));
            }
        };
        for layer in 0..(layer_offsets.len() - 1) {
            let volume_cell = alloc_point(&mut next_id)?;
            MutableSieve::add_point(&mut sieve, volume_cell);
            cell_types_out.try_add_point(volume_cell, 1)?;
            cell_types_out.try_set(volume_cell, &[cell_kind])?;

            let bottom: Vec<PointId> = vertices
                .iter()
                .map(|v| layer_points.get(&(*v, layer)).copied().unwrap())
                .collect();
            let top: Vec<PointId> = vertices
                .iter()
                .map(|v| layer_points.get(&(*v, layer + 1)).copied().unwrap())
                .collect();

            match cell_type {
                CellType::Triangle => {
                    for p in bottom.iter().chain(top.iter()) {
                        sieve.add_arrow(volume_cell, *p, ());
                    }
                }
                CellType::Quadrilateral => {
                    for p in bottom.iter().chain(top.iter()) {
                        sieve.add_arrow(volume_cell, *p, ());
                    }
                }
                _ => {}
            }
        }
    }

    Ok(MeshData {
        sieve,
        coordinates: Some(coords),
        sections: BTreeMap::new(),
        labels: None,
        cell_types: Some(cell_types_out),
    })
}

fn collect_surface_entities<CtSt>(
    surface: &impl Sieve<Point = PointId>,
    cell_types: &Section<CellType, CtSt>,
) -> Result<(Vec<PointId>, Vec<(PointId, CellType, Vec<PointId>)>), MeshSieveError>
where
    CtSt: Storage<CellType>,
{
    let mut vertices = Vec::new();
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
        if cell_type == CellType::Vertex {
            vertices.push(point);
            continue;
        }
        if matches!(cell_type, CellType::Triangle | CellType::Quadrilateral) {
            let verts: Vec<PointId> = surface.cone_points(point).collect();
            let expected = match cell_type {
                CellType::Triangle => 3,
                CellType::Quadrilateral => 4,
                _ => 0,
            };
            if verts.len() != expected {
                return Err(MeshSieveError::InvalidGeometry(format!(
                    "cell {point:?} expected {expected} vertices, got {}",
                    verts.len()
                )));
            }
            cells.push((point, cell_type, verts));
        } else if cell_type.dimension() >= 2 {
            return Err(MeshSieveError::InvalidGeometry(format!(
                "unsupported surface cell type: {cell_type:?}"
            )));
        }
    }
    Ok((vertices, cells))
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
    use crate::data::storage::VecStorage;

    fn v(raw: u64) -> PointId {
        PointId::new(raw).unwrap()
    }

    #[test]
    fn extrudes_triangle_prism() {
        let mut surface = InMemorySieve::<PointId, ()>::default();
        let cell = v(1);
        let vertices = [v(2), v(3), v(4)];
        MutableSieve::add_point(&mut surface, cell);
        for vert in vertices {
            MutableSieve::add_point(&mut surface, vert);
            surface.add_arrow(cell, vert, ());
        }

        let mut atlas = Atlas::default();
        atlas.try_insert(cell, 1).unwrap();
        for vert in vertices {
            atlas.try_insert(vert, 1).unwrap();
        }
        let mut cell_types = Section::<CellType, VecStorage<CellType>>::new(atlas);
        cell_types.try_set(cell, &[CellType::Triangle]).unwrap();
        for vert in vertices {
            cell_types.try_set(vert, &[CellType::Vertex]).unwrap();
        }

        let mut coord_atlas = Atlas::default();
        for vert in vertices {
            coord_atlas.try_insert(vert, 2).unwrap();
        }
        let mut coords = Coordinates::<f64, VecStorage<f64>>::try_new(2, coord_atlas).unwrap();
        coords
            .section_mut()
            .try_set(vertices[0], &[0.0, 0.0])
            .unwrap();
        coords
            .section_mut()
            .try_set(vertices[1], &[1.0, 0.0])
            .unwrap();
        coords
            .section_mut()
            .try_set(vertices[2], &[0.0, 1.0])
            .unwrap();

        let mesh = extrude_surface_layers(&surface, &cell_types, &coords, &[0.0, 1.0]).unwrap();
        let cell_types_out = mesh.cell_types.as_ref().unwrap();
        let prism_cells: Vec<_> = cell_types_out
            .iter()
            .filter(|(_, ty)| ty[0] == CellType::Prism)
            .collect();
        assert_eq!(prism_cells.len(), 1);
    }
}
