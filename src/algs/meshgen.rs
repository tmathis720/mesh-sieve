//! Basic mesh generators for structured boxes and simple shells.

use crate::data::atlas::Atlas;
use crate::data::coordinates::Coordinates;
use crate::data::mixed_section::MixedSectionStore;
use crate::data::section::Section;
use crate::data::storage::VecStorage;
use crate::io::MeshData;
use crate::mesh_error::MeshSieveError;
use crate::topology::cell_type::CellType;
use crate::topology::labels::LabelSet;
use crate::topology::point::PointId;
use crate::topology::sieve::{InMemorySieve, MutableSieve};
use std::collections::BTreeMap;

/// Cell-type choices for structured meshes.
#[derive(Clone, Copy, Debug)]
pub enum StructuredCellType {
    Triangle,
    Quadrilateral,
    Hexahedron,
}

/// Optional metadata for mesh generators.
#[derive(Clone, Debug, Default)]
pub struct MeshGenOptions {
    pub labels: Option<LabelSet>,
}

type MeshGenResult = Result<
    MeshData<InMemorySieve<PointId, ()>, f64, VecStorage<f64>, VecStorage<CellType>>,
    MeshSieveError,
>;

fn invalid_geometry(message: impl Into<String>) -> MeshSieveError {
    MeshSieveError::InvalidGeometry(message.into())
}

fn build_mesh(
    dimension: usize,
    vertex_coords: &[Vec<f64>],
    cells: &[Vec<usize>],
    cell_type: CellType,
    labels: Option<LabelSet>,
) -> MeshGenResult {
    if dimension == 0 {
        return Err(invalid_geometry("dimension must be non-zero"));
    }
    for (idx, coord) in vertex_coords.iter().enumerate() {
        if coord.len() != dimension {
            return Err(invalid_geometry(format!(
                "vertex {idx} has dimension {}, expected {dimension}",
                coord.len()
            )));
        }
    }

    let mut sieve = InMemorySieve::<PointId, ()>::default();
    let mut next_id = 1u64;

    let mut vertex_points = Vec::with_capacity(vertex_coords.len());
    for _ in 0..vertex_coords.len() {
        let pid = PointId::new(next_id)?;
        next_id += 1;
        MutableSieve::add_point(&mut sieve, pid);
        vertex_points.push(pid);
    }

    let mut cell_points = Vec::with_capacity(cells.len());
    for _ in 0..cells.len() {
        let pid = PointId::new(next_id)?;
        next_id += 1;
        MutableSieve::add_point(&mut sieve, pid);
        cell_points.push(pid);
    }

    for (cell_idx, vertices) in cells.iter().enumerate() {
        let cell_point = cell_points[cell_idx];
        for &vidx in vertices {
            let vpoint = *vertex_points.get(vidx).ok_or_else(|| {
                invalid_geometry(format!("cell {cell_idx} references missing vertex {vidx}"))
            })?;
            sieve.add_arrow(cell_point, vpoint, ());
        }
    }
    sieve.sort_adjacency();

    let mut coord_atlas = Atlas::default();
    for &p in &vertex_points {
        coord_atlas.try_insert(p, dimension)?;
    }
    let mut coords = Coordinates::<f64, VecStorage<f64>>::try_new(dimension, coord_atlas)?;
    for (p, coord) in vertex_points.iter().zip(vertex_coords.iter()) {
        coords.section_mut().try_set(*p, coord)?;
    }

    let mut cell_atlas = Atlas::default();
    for &p in vertex_points.iter().chain(cell_points.iter()) {
        cell_atlas.try_insert(p, 1)?;
    }
    let mut cell_types = Section::<CellType, VecStorage<CellType>>::new(cell_atlas);
    for &p in &vertex_points {
        cell_types.try_set(p, &[CellType::Vertex])?;
    }
    for &p in &cell_points {
        cell_types.try_set(p, &[cell_type])?;
    }

    Ok(MeshData {
        sieve,
        coordinates: Some(coords),
        sections: BTreeMap::new(),
        mixed_sections: MixedSectionStore::default(),
        labels,
        cell_types: Some(cell_types),
        discretization: None,
    })
}

/// Generate a structured 2D box mesh over `[min, max]` with `nx`×`ny` cells.
pub fn structured_box_2d(
    nx: usize,
    ny: usize,
    min: [f64; 2],
    max: [f64; 2],
    cell_type: StructuredCellType,
    options: MeshGenOptions,
) -> MeshGenResult {
    if nx == 0 || ny == 0 {
        return Err(invalid_geometry("nx and ny must be positive"));
    }

    let cell_type = match cell_type {
        StructuredCellType::Triangle => CellType::Triangle,
        StructuredCellType::Quadrilateral => CellType::Quadrilateral,
        StructuredCellType::Hexahedron => {
            return Err(invalid_geometry("hex elements are not valid for 2D meshes"));
        }
    };

    let dx = (max[0] - min[0]) / nx as f64;
    let dy = (max[1] - min[1]) / ny as f64;
    let mut vertices = Vec::with_capacity((nx + 1) * (ny + 1));
    for j in 0..=ny {
        let y = min[1] + dy * j as f64;
        for i in 0..=nx {
            let x = min[0] + dx * i as f64;
            vertices.push(vec![x, y]);
        }
    }

    let mut cells = Vec::new();
    let row_stride = nx + 1;
    for j in 0..ny {
        for i in 0..nx {
            let v0 = j * row_stride + i;
            let v1 = v0 + 1;
            let v3 = v0 + row_stride;
            let v2 = v3 + 1;
            match cell_type {
                CellType::Triangle => {
                    cells.push(vec![v0, v1, v2]);
                    cells.push(vec![v0, v2, v3]);
                }
                CellType::Quadrilateral => {
                    cells.push(vec![v0, v1, v2, v3]);
                }
                _ => unreachable!("filtered above"),
            }
        }
    }

    build_mesh(2, &vertices, &cells, cell_type, options.labels)
}

/// Generate a structured 3D box mesh over `[min, max]` with `nx`×`ny`×`nz` cells.
pub fn structured_box_3d(
    nx: usize,
    ny: usize,
    nz: usize,
    min: [f64; 3],
    max: [f64; 3],
    cell_type: StructuredCellType,
    options: MeshGenOptions,
) -> MeshGenResult {
    if nx == 0 || ny == 0 || nz == 0 {
        return Err(invalid_geometry("nx, ny, and nz must be positive"));
    }
    let cell_type = match cell_type {
        StructuredCellType::Hexahedron => CellType::Hexahedron,
        StructuredCellType::Triangle | StructuredCellType::Quadrilateral => {
            return Err(invalid_geometry(
                "triangle/quadrilateral elements are not valid for 3D box meshes",
            ));
        }
    };

    let dx = (max[0] - min[0]) / nx as f64;
    let dy = (max[1] - min[1]) / ny as f64;
    let dz = (max[2] - min[2]) / nz as f64;
    let mut vertices = Vec::with_capacity((nx + 1) * (ny + 1) * (nz + 1));
    for k in 0..=nz {
        let z = min[2] + dz * k as f64;
        for j in 0..=ny {
            let y = min[1] + dy * j as f64;
            for i in 0..=nx {
                let x = min[0] + dx * i as f64;
                vertices.push(vec![x, y, z]);
            }
        }
    }

    let mut cells = Vec::with_capacity(nx * ny * nz);
    let row_stride = nx + 1;
    let slab_stride = row_stride * (ny + 1);
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let base = k * slab_stride + j * row_stride + i;
                let v0 = base;
                let v1 = base + 1;
                let v3 = base + row_stride;
                let v2 = v3 + 1;
                let v4 = base + slab_stride;
                let v5 = v4 + 1;
                let v7 = v4 + row_stride;
                let v6 = v7 + 1;
                cells.push(vec![v0, v1, v2, v3, v4, v5, v6, v7]);
            }
        }
    }

    build_mesh(3, &vertices, &cells, cell_type, options.labels)
}

/// Generate a spherical shell (triangulated) with `n_lat` by `n_lon` divisions.
pub fn sphere_shell(
    radius: f64,
    n_lat: usize,
    n_lon: usize,
    options: MeshGenOptions,
) -> MeshGenResult {
    if radius <= 0.0 {
        return Err(invalid_geometry("radius must be positive"));
    }
    if n_lat < 2 || n_lon < 3 {
        return Err(invalid_geometry(
            "sphere shell requires n_lat >= 2 and n_lon >= 3",
        ));
    }

    let mut vertices = Vec::new();
    vertices.push(vec![0.0, 0.0, radius]);
    let mut rings: Vec<Vec<usize>> = Vec::new();

    let two_pi = std::f64::consts::TAU;
    for lat in 1..n_lat {
        let theta = std::f64::consts::PI * (lat as f64) / (n_lat as f64);
        let sin_t = theta.sin();
        let cos_t = theta.cos();
        let mut ring = Vec::with_capacity(n_lon);
        for lon in 0..n_lon {
            let phi = two_pi * (lon as f64) / (n_lon as f64);
            let x = radius * sin_t * phi.cos();
            let y = radius * sin_t * phi.sin();
            let z = radius * cos_t;
            ring.push(vertices.len());
            vertices.push(vec![x, y, z]);
        }
        rings.push(ring);
    }
    let bottom_idx = vertices.len();
    vertices.push(vec![0.0, 0.0, -radius]);

    let mut cells = Vec::new();
    let top_idx = 0usize;
    if let Some(first_ring) = rings.first() {
        for lon in 0..n_lon {
            let next = (lon + 1) % n_lon;
            cells.push(vec![top_idx, first_ring[lon], first_ring[next]]);
        }
    }

    for band in 0..rings.len().saturating_sub(1) {
        let ring_a = &rings[band];
        let ring_b = &rings[band + 1];
        for lon in 0..n_lon {
            let next = (lon + 1) % n_lon;
            let a0 = ring_a[lon];
            let a1 = ring_a[next];
            let b0 = ring_b[lon];
            let b1 = ring_b[next];
            cells.push(vec![a0, b0, b1]);
            cells.push(vec![a0, b1, a1]);
        }
    }

    if let Some(last_ring) = rings.last() {
        for lon in 0..n_lon {
            let next = (lon + 1) % n_lon;
            cells.push(vec![last_ring[lon], bottom_idx, last_ring[next]]);
        }
    }

    build_mesh(3, &vertices, &cells, CellType::Triangle, options.labels)
}

/// Generate a cylindrical shell (quad mesh) with `n_theta` around and `n_z` along height.
pub fn cylinder_shell(
    radius: f64,
    height: f64,
    n_theta: usize,
    n_z: usize,
    options: MeshGenOptions,
) -> MeshGenResult {
    if radius <= 0.0 || height <= 0.0 {
        return Err(invalid_geometry("radius and height must be positive"));
    }
    if n_theta < 3 || n_z < 1 {
        return Err(invalid_geometry(
            "cylinder shell requires n_theta >= 3 and n_z >= 1",
        ));
    }

    let mut vertices = Vec::new();
    let mut rings: Vec<Vec<usize>> = Vec::with_capacity(n_z + 1);
    let two_pi = std::f64::consts::TAU;
    for k in 0..=n_z {
        let z = height * (k as f64) / (n_z as f64);
        let mut ring = Vec::with_capacity(n_theta);
        for t in 0..n_theta {
            let phi = two_pi * (t as f64) / (n_theta as f64);
            let x = radius * phi.cos();
            let y = radius * phi.sin();
            ring.push(vertices.len());
            vertices.push(vec![x, y, z]);
        }
        rings.push(ring);
    }

    let mut cells = Vec::new();
    for band in 0..n_z {
        let ring_a = &rings[band];
        let ring_b = &rings[band + 1];
        for t in 0..n_theta {
            let next = (t + 1) % n_theta;
            let a0 = ring_a[t];
            let a1 = ring_a[next];
            let b0 = ring_b[t];
            let b1 = ring_b[next];
            cells.push(vec![a0, a1, b1, b0]);
        }
    }

    build_mesh(
        3,
        &vertices,
        &cells,
        CellType::Quadrilateral,
        options.labels,
    )
}
