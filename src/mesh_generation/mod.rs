//! Structured mesh generators with boundary labeling and periodic equivalence.

use crate::data::atlas::Atlas;
use crate::data::coordinates::Coordinates;
use crate::data::mixed_section::MixedSectionStore;
use crate::data::section::Section;
use crate::data::storage::VecStorage;
use crate::io::MeshData;
use crate::mesh_error::MeshSieveError;
use crate::topology::cell_type::CellType;
use crate::topology::labels::LabelSet;
use crate::topology::periodic::PointEquivalence;
use crate::topology::point::PointId;
use crate::topology::sieve::{InMemorySieve, MutableSieve, Sieve};
use std::collections::BTreeMap;

/// Boundary label name for the minimum-x side.
pub const BOUNDARY_X_MIN: &str = "boundary_x_min";
/// Boundary label name for the maximum-x side.
pub const BOUNDARY_X_MAX: &str = "boundary_x_max";
/// Boundary label name for the minimum-y side.
pub const BOUNDARY_Y_MIN: &str = "boundary_y_min";
/// Boundary label name for the maximum-y side.
pub const BOUNDARY_Y_MAX: &str = "boundary_y_max";
/// Boundary label name for the minimum-z side.
pub const BOUNDARY_Z_MIN: &str = "boundary_z_min";
/// Boundary label name for the maximum-z side.
pub const BOUNDARY_Z_MAX: &str = "boundary_z_max";

type MeshDataType =
    MeshData<InMemorySieve<PointId, ()>, f64, VecStorage<f64>, VecStorage<CellType>>;

/// Optional periodic identification for structured meshes.
#[derive(Clone, Copy, Debug, Default)]
pub struct Periodicity {
    pub x: bool,
    pub y: bool,
    pub z: bool,
}

impl Periodicity {
    /// No periodic directions.
    pub fn none() -> Self {
        Self::default()
    }
}

/// Optional configuration for mesh generation.
#[derive(Clone, Copy, Debug, Default)]
pub struct MeshGenerationOptions {
    pub periodic: Periodicity,
}

/// Output from a mesh generator with optional periodic equivalence.
#[derive(Debug)]
pub struct GeneratedMesh {
    pub mesh: MeshDataType,
    pub periodic: Option<PointEquivalence>,
}

fn invalid_geometry(message: impl Into<String>) -> MeshSieveError {
    MeshSieveError::InvalidGeometry(message.into())
}

fn build_mesh(
    dimension: usize,
    vertex_coords: &[Vec<f64>],
    cells: &[Vec<usize>],
    cell_type: CellType,
) -> Result<(MeshDataType, Vec<PointId>), MeshSieveError> {
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
    let mut coords =
        Coordinates::<f64, VecStorage<f64>>::try_new(dimension, dimension, coord_atlas)?;
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

    Ok((
        MeshDataType {
            sieve,
            coordinates: Some(coords),
            sections: BTreeMap::new(),
            mixed_sections: MixedSectionStore::default(),
            labels: None,
            cell_types: Some(cell_types),
            discretization: None,
        },
        vertex_points,
    ))
}

/// Generate a 1D interval mesh with `n` segments over `[min, max]`.
pub fn interval_mesh(
    n: usize,
    min: f64,
    max: f64,
    options: MeshGenerationOptions,
) -> Result<GeneratedMesh, MeshSieveError> {
    if n == 0 {
        return Err(invalid_geometry("n must be positive"));
    }

    let dx = (max - min) / n as f64;
    let mut vertices = Vec::with_capacity(n + 1);
    for i in 0..=n {
        vertices.push(vec![min + dx * i as f64]);
    }

    let mut cells = Vec::with_capacity(n);
    for i in 0..n {
        cells.push(vec![i, i + 1]);
    }

    let (mut mesh, vertex_points) = build_mesh(1, &vertices, &cells, CellType::Segment)?;

    let mut labels = LabelSet::new();
    if let Some(&first) = vertex_points.first() {
        labels.set_label(first, BOUNDARY_X_MIN, 1);
    }
    if let Some(&last) = vertex_points.last() {
        labels.set_label(last, BOUNDARY_X_MAX, 1);
    }
    mesh.labels = Some(labels);

    let periodic = if options.periodic.x {
        let mut eq = PointEquivalence::new();
        if let (Some(&first), Some(&last)) = (vertex_points.first(), vertex_points.last()) {
            eq.add_equivalence(first, last);
        }
        Some(eq)
    } else {
        None
    };

    Ok(GeneratedMesh { mesh, periodic })
}

/// Generate a structured quadrilateral mesh over `[min, max]` with `nx`×`ny` cells.
pub fn quad_mesh(
    nx: usize,
    ny: usize,
    min: [f64; 2],
    max: [f64; 2],
    options: MeshGenerationOptions,
) -> Result<GeneratedMesh, MeshSieveError> {
    if nx == 0 || ny == 0 {
        return Err(invalid_geometry("nx and ny must be positive"));
    }

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

    let mut cells = Vec::with_capacity(nx * ny);
    let row_stride = nx + 1;
    for j in 0..ny {
        for i in 0..nx {
            let v0 = j * row_stride + i;
            let v1 = v0 + 1;
            let v3 = v0 + row_stride;
            let v2 = v3 + 1;
            cells.push(vec![v0, v1, v2, v3]);
        }
    }

    let (mut mesh, vertex_points) = build_mesh(2, &vertices, &cells, CellType::Quadrilateral)?;

    let mut labels = LabelSet::new();
    for j in 0..=ny {
        for i in 0..=nx {
            let idx = j * (nx + 1) + i;
            let point = vertex_points[idx];
            if i == 0 {
                labels.set_label(point, BOUNDARY_X_MIN, 1);
            }
            if i == nx {
                labels.set_label(point, BOUNDARY_X_MAX, 1);
            }
            if j == 0 {
                labels.set_label(point, BOUNDARY_Y_MIN, 1);
            }
            if j == ny {
                labels.set_label(point, BOUNDARY_Y_MAX, 1);
            }
        }
    }
    mesh.labels = Some(labels);

    let periodic = if options.periodic.x || options.periodic.y {
        let mut eq = PointEquivalence::new();
        if options.periodic.x {
            for j in 0..=ny {
                let left_idx = j * (nx + 1);
                let right_idx = left_idx + nx;
                eq.add_equivalence(vertex_points[left_idx], vertex_points[right_idx]);
            }
        }
        if options.periodic.y {
            for i in 0..=nx {
                let bottom_idx = i;
                let top_idx = ny * (nx + 1) + i;
                eq.add_equivalence(vertex_points[bottom_idx], vertex_points[top_idx]);
            }
        }
        Some(eq)
    } else {
        None
    };

    Ok(GeneratedMesh { mesh, periodic })
}

/// Generate a structured hexahedral mesh over `[min, max]` with `nx`×`ny`×`nz` cells.
pub fn hex_mesh(
    nx: usize,
    ny: usize,
    nz: usize,
    min: [f64; 3],
    max: [f64; 3],
    options: MeshGenerationOptions,
) -> Result<GeneratedMesh, MeshSieveError> {
    if nx == 0 || ny == 0 || nz == 0 {
        return Err(invalid_geometry("nx, ny, and nz must be positive"));
    }

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

    let (mut mesh, vertex_points) = build_mesh(3, &vertices, &cells, CellType::Hexahedron)?;

    let mut labels = LabelSet::new();
    for k in 0..=nz {
        for j in 0..=ny {
            for i in 0..=nx {
                let idx = k * (nx + 1) * (ny + 1) + j * (nx + 1) + i;
                let point = vertex_points[idx];
                if i == 0 {
                    labels.set_label(point, BOUNDARY_X_MIN, 1);
                }
                if i == nx {
                    labels.set_label(point, BOUNDARY_X_MAX, 1);
                }
                if j == 0 {
                    labels.set_label(point, BOUNDARY_Y_MIN, 1);
                }
                if j == ny {
                    labels.set_label(point, BOUNDARY_Y_MAX, 1);
                }
                if k == 0 {
                    labels.set_label(point, BOUNDARY_Z_MIN, 1);
                }
                if k == nz {
                    labels.set_label(point, BOUNDARY_Z_MAX, 1);
                }
            }
        }
    }
    mesh.labels = Some(labels);

    let periodic = if options.periodic.x || options.periodic.y || options.periodic.z {
        let mut eq = PointEquivalence::new();
        if options.periodic.x {
            for k in 0..=nz {
                for j in 0..=ny {
                    let left_idx = k * (nx + 1) * (ny + 1) + j * (nx + 1);
                    let right_idx = left_idx + nx;
                    eq.add_equivalence(vertex_points[left_idx], vertex_points[right_idx]);
                }
            }
        }
        if options.periodic.y {
            for k in 0..=nz {
                for i in 0..=nx {
                    let bottom_idx = k * (nx + 1) * (ny + 1) + i;
                    let top_idx = k * (nx + 1) * (ny + 1) + ny * (nx + 1) + i;
                    eq.add_equivalence(vertex_points[bottom_idx], vertex_points[top_idx]);
                }
            }
        }
        if options.periodic.z {
            let slab = (nx + 1) * (ny + 1);
            for j in 0..=ny {
                for i in 0..=nx {
                    let bottom_idx = j * (nx + 1) + i;
                    let top_idx = nz * slab + j * (nx + 1) + i;
                    eq.add_equivalence(vertex_points[bottom_idx], vertex_points[top_idx]);
                }
            }
        }
        Some(eq)
    } else {
        None
    };

    Ok(GeneratedMesh { mesh, periodic })
}
