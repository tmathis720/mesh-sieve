//! Extrude a 2D surface mesh into a layered 3D mesh.
//!
//! # Expected cell types
//! - [`CellType::Triangle`] surfaces extrude to [`CellType::Prism`] volumes.
//! - [`CellType::Quadrilateral`] surfaces extrude to [`CellType::Hexahedron`] volumes.
//! - [`CellType::Polygon`] surfaces extrude to [`CellType::Polyhedron`] volumes.
//!
//! The input mesh must be a cell→vertex topology; vertices are points with no
//! outgoing arrows. The output mesh can optionally include intermediate
//! edges/faces; otherwise it only contains volume cells and vertices.
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
//! use mesh_sieve::topology::Sieve;
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

use crate::algs::interpolate::{InterpolationOrdering, interpolate_edges_faces_with_ordering};
use crate::data::atlas::Atlas;
use crate::data::coordinates::Coordinates;
use crate::data::discretization::Discretization;
use crate::data::section::Section;
use crate::data::storage::{Storage, VecStorage};
use crate::io::MeshData;
use crate::mesh_error::MeshSieveError;
use crate::topology::cell_type::CellType;
use crate::topology::labels::LabelSet;
use crate::topology::point::PointId;
use crate::topology::sieve::{InMemorySieve, MutableSieve, Sieve};
use std::collections::BTreeMap;

/// Options controlling extrusion output.
#[derive(Clone, Copy, Debug, Default)]
pub struct ExtrudeOptions {
    /// When true, generate intermediate edges/faces as part of the extrusion.
    pub include_intermediate_entities: bool,
}

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
    extrude_surface_layers_with_options(
        surface,
        cell_types,
        coordinates,
        layer_offsets,
        ExtrudeOptions::default(),
    )
}

/// Extrude a surface mesh with configurable output options.
pub fn extrude_surface_layers_with_options<S, CtSt, Cs>(
    surface: &S,
    cell_types: &Section<CellType, CtSt>,
    coordinates: &Coordinates<f64, Cs>,
    layer_offsets: &[f64],
    options: ExtrudeOptions,
) -> Result<
    MeshData<InMemorySieve<PointId, ()>, f64, VecStorage<f64>, VecStorage<CellType>>,
    MeshSieveError,
>
where
    S: Sieve<Point = PointId>,
    CtSt: Storage<CellType>,
    Cs: Storage<f64>,
{
    extrude_surface_layers_inner::<S, CtSt, Cs, VecStorage<f64>>(
        surface,
        cell_types,
        coordinates,
        layer_offsets,
        options,
        None,
        None,
        None,
    )
}

/// Extrude a surface mesh from a `MeshData` container and preserve metadata.
pub fn extrude_surface_mesh_layers<S, St, CtSt>(
    mesh: &MeshData<S, f64, St, CtSt>,
    layer_offsets: &[f64],
    options: ExtrudeOptions,
) -> Result<
    MeshData<InMemorySieve<PointId, ()>, f64, VecStorage<f64>, VecStorage<CellType>>,
    MeshSieveError,
>
where
    S: Sieve<Point = PointId>,
    St: Storage<f64>,
    CtSt: Storage<CellType>,
{
    let coordinates = mesh.coordinates.as_ref().ok_or_else(|| {
        MeshSieveError::InvalidGeometry("surface mesh is missing coordinates".into())
    })?;
    let cell_types = mesh.cell_types.as_ref().ok_or_else(|| {
        MeshSieveError::InvalidGeometry("surface mesh is missing cell types".into())
    })?;
    extrude_surface_layers_inner(
        &mesh.sieve,
        cell_types,
        coordinates,
        layer_offsets,
        options,
        mesh.labels.as_ref(),
        Some(&mesh.sections),
        mesh.discretization.as_ref(),
    )
}

#[allow(clippy::too_many_arguments)]
fn extrude_surface_layers_inner<S, CtSt, Cs, St>(
    surface: &S,
    cell_types: &Section<CellType, CtSt>,
    coordinates: &Coordinates<f64, Cs>,
    layer_offsets: &[f64],
    options: ExtrudeOptions,
    labels: Option<&LabelSet>,
    sections: Option<&BTreeMap<String, Section<f64, St>>>,
    discretization: Option<&Discretization>,
) -> Result<
    MeshData<InMemorySieve<PointId, ()>, f64, VecStorage<f64>, VecStorage<CellType>>,
    MeshSieveError,
>
where
    S: Sieve<Point = PointId>,
    CtSt: Storage<CellType>,
    Cs: Storage<f64>,
    St: Storage<f64>,
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
    let mut point_map: BTreeMap<PointId, Vec<PointId>> = BTreeMap::new();

    let mut layer_points: BTreeMap<(PointId, usize), PointId> = BTreeMap::new();
    for &vertex in &vertices {
        for (layer, _) in layer_offsets.iter().enumerate() {
            let new_point = alloc_point(&mut next_id)?;
            MutableSieve::add_point(&mut sieve, new_point);
            coord_atlas.try_insert(new_point, 3)?;
            cell_atlas.try_insert(new_point, 1)?;
            layer_points.insert((vertex, layer), new_point);
            point_map.entry(vertex).or_default().push(new_point);
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

    let mut ordering = InterpolationOrdering::default();
    for (cell, cell_type, vertices) in cells {
        let cell_kind = match cell_type {
            CellType::Triangle => CellType::Prism,
            CellType::Quadrilateral => CellType::Hexahedron,
            CellType::Polygon(_) => CellType::Polyhedron,
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
            point_map.entry(cell).or_default().push(volume_cell);

            let bottom: Vec<PointId> = vertices
                .iter()
                .map(|v| layer_points.get(&(*v, layer)).copied().unwrap())
                .collect();
            let top: Vec<PointId> = vertices
                .iter()
                .map(|v| layer_points.get(&(*v, layer + 1)).copied().unwrap())
                .collect();

            let mut combined = Vec::with_capacity(bottom.len() + top.len());
            combined.extend_from_slice(&bottom);
            combined.extend_from_slice(&top);
            for p in &combined {
                sieve.add_arrow(volume_cell, *p, ());
            }

            if cell_kind == CellType::Polyhedron {
                let n = bottom.len();
                let mut faces = Vec::with_capacity(n + 2);
                faces.push((0..n).collect());
                faces.push((n..(2 * n)).collect());
                for i in 0..n {
                    let next = (i + 1) % n;
                    faces.push(vec![i, next, n + next, n + i]);
                }
                ordering.polyhedron_faces.insert(volume_cell, faces);
            }
        }
    }

    if options.include_intermediate_entities {
        interpolate_edges_faces_with_ordering(&mut sieve, &mut cell_types_out, Some(&ordering))?;
    }

    let labels_out = labels
        .map(|label_set| map_labels(label_set, &point_map))
        .filter(|mapped| !mapped.is_empty());

    let sections_out = if let Some(sections) = sections {
        map_sections(sections, &point_map)?
    } else {
        BTreeMap::new()
    };

    Ok(MeshData {
        sieve,
        coordinates: Some(coords),
        sections: sections_out,
        mixed_sections: crate::data::mixed_section::MixedSectionStore::default(),
        labels: labels_out,
        cell_types: Some(cell_types_out),
        discretization: discretization.cloned(),
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
    let mut vertex_set = std::collections::HashSet::new();
    for point in surface.points() {
        if surface.cone_points(point).next().is_none() {
            vertices.push(point);
            vertex_set.insert(point);
        }
    }
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
            continue;
        }
        if matches!(
            cell_type,
            CellType::Triangle | CellType::Quadrilateral | CellType::Polygon(_)
        ) {
            let verts: Vec<PointId> = surface.cone_points(point).collect();
            let expected = match cell_type {
                CellType::Triangle => 3,
                CellType::Quadrilateral => 4,
                CellType::Polygon(n) => n as usize,
                _ => 0,
            };
            if verts.len() != expected {
                return Err(MeshSieveError::InvalidGeometry(format!(
                    "cell {point:?} expected {expected} vertices, got {}",
                    verts.len()
                )));
            }
            if let Some(non_vertex) = verts.iter().find(|v| !vertex_set.contains(v)) {
                return Err(MeshSieveError::InvalidGeometry(format!(
                    "cell {point:?} references non-vertex point {non_vertex:?}"
                )));
            }
            cells.push((point, cell_type, verts));
        } else if cell_type.dimension() >= 1 {
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

fn map_labels(labels: &LabelSet, point_map: &BTreeMap<PointId, Vec<PointId>>) -> LabelSet {
    let mut out = LabelSet::new();
    for (name, point, value) in labels.iter() {
        if let Some(targets) = point_map.get(&point) {
            for &target in targets {
                out.set_label(target, name, value);
            }
        }
    }
    out
}

fn map_sections<St>(
    sections: &BTreeMap<String, Section<f64, St>>,
    point_map: &BTreeMap<PointId, Vec<PointId>>,
) -> Result<BTreeMap<String, Section<f64, VecStorage<f64>>>, MeshSieveError>
where
    St: Storage<f64>,
{
    let mut out = BTreeMap::new();
    for (name, section) in sections {
        let mut atlas = Atlas::default();
        for (point, values) in section.iter() {
            if let Some(targets) = point_map.get(&point) {
                for &target in targets {
                    atlas.try_insert(target, values.len())?;
                }
            }
        }
        let mut mapped = Section::<f64, VecStorage<f64>>::new(atlas);
        for (point, values) in section.iter() {
            if let Some(targets) = point_map.get(&point) {
                for &target in targets {
                    mapped.try_set(target, values)?;
                }
            }
        }
        out.insert(name.clone(), mapped);
    }
    Ok(out)
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

    #[test]
    fn extrudes_polygon_polyhedron() {
        let mut surface = InMemorySieve::<PointId, ()>::default();
        let cell = v(1);
        let vertices = [v(2), v(3), v(4), v(5), v(6)];
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
        cell_types
            .try_set(cell, &[CellType::Polygon(vertices.len() as u8)])
            .unwrap();
        for vert in vertices {
            cell_types.try_set(vert, &[CellType::Vertex]).unwrap();
        }

        let mut coord_atlas = Atlas::default();
        for vert in vertices {
            coord_atlas.try_insert(vert, 2).unwrap();
        }
        let mut coords = Coordinates::<f64, VecStorage<f64>>::try_new(2, coord_atlas).unwrap();
        for (idx, vert) in vertices.iter().enumerate() {
            coords
                .section_mut()
                .try_set(*vert, &[idx as f64, 0.0])
                .unwrap();
        }

        let mesh = extrude_surface_layers(&surface, &cell_types, &coords, &[0.0, 1.0]).unwrap();
        let cell_types_out = mesh.cell_types.as_ref().unwrap();
        let poly_cells: Vec<_> = cell_types_out
            .iter()
            .filter(|(_, ty)| ty[0] == CellType::Polyhedron)
            .collect();
        assert_eq!(poly_cells.len(), 1);
    }

    #[test]
    fn extrudes_with_intermediate_entities() {
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

        let mesh = extrude_surface_layers_with_options(
            &surface,
            &cell_types,
            &coords,
            &[0.0, 1.0],
            ExtrudeOptions {
                include_intermediate_entities: true,
            },
        )
        .unwrap();
        let cell_types_out = mesh.cell_types.as_ref().unwrap();
        let has_edges = cell_types_out
            .iter()
            .any(|(_, ty)| ty[0] == CellType::Segment);
        assert!(has_edges);
    }

    #[test]
    fn extrude_preserves_labels_and_sections() {
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

        let mut labels = LabelSet::new();
        labels.set_label(vertices[0], "boundary", 7);
        labels.set_label(cell, "surface", 3);

        let mut scalar_atlas = Atlas::default();
        scalar_atlas.try_insert(cell, 1).unwrap();
        for vert in vertices {
            scalar_atlas.try_insert(vert, 1).unwrap();
        }
        let mut section = Section::<f64, VecStorage<f64>>::new(scalar_atlas);
        section.try_set(cell, &[9.0]).unwrap();
        for vert in vertices {
            section.try_set(vert, &[1.0]).unwrap();
        }
        let mut sections = BTreeMap::new();
        sections.insert("tag".to_string(), section);

        let input = MeshData {
            sieve: surface,
            coordinates: Some(coords),
            sections,
            mixed_sections: crate::data::mixed_section::MixedSectionStore::default(),
            labels: Some(labels),
            cell_types: Some(cell_types),
            discretization: None,
        };

        let mesh =
            extrude_surface_mesh_layers(&input, &[0.0, 1.0], ExtrudeOptions::default()).unwrap();

        let out_labels = mesh.labels.as_ref().unwrap();
        let boundary_points: Vec<_> = out_labels.points_with_label("boundary", 7).collect();
        assert_eq!(boundary_points.len(), 2);

        let tag_section = mesh.sections.get("tag").unwrap();
        assert_eq!(tag_section.iter().count(), 7);
    }
}
