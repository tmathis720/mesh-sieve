//! Basic mesh generators and external mesh-generator integrations.

use crate::data::atlas::Atlas;
use crate::data::coordinates::Coordinates;
use crate::data::mixed_section::MixedSectionStore;
use crate::data::section::Section;
use crate::data::storage::VecStorage;
use crate::io::MeshData;
use crate::mesh_error::MeshSieveError;
use crate::mesh_generation::{
    MeshGenerationOptions, Periodicity, hex_mesh, interval_mesh, quad_mesh,
};
use crate::topology::cell_type::CellType;
use crate::topology::labels::LabelSet;
use crate::topology::point::PointId;
use crate::topology::sieve::sieve_trait::Sieve;
use crate::topology::sieve::{InMemorySieve, MutableSieve};
use std::collections::BTreeMap;
#[cfg(any(
    feature = "triangle-support",
    feature = "tetgen-support",
    feature = "gmsh-support"
))]
use std::fs::{self, File};
#[cfg(any(
    feature = "triangle-support",
    feature = "tetgen-support",
    feature = "gmsh-support"
))]
use std::io::{BufRead, BufReader, Write};
#[cfg(any(
    feature = "triangle-support",
    feature = "tetgen-support",
    feature = "gmsh-support"
))]
use std::path::{Path, PathBuf};
#[cfg(any(
    feature = "triangle-support",
    feature = "tetgen-support",
    feature = "gmsh-support"
))]
use std::process::Command;

#[derive(Clone, Copy, Debug)]
pub enum StructuredCellType {
    Triangle,
    Quadrilateral,
    Hexahedron,
}

#[derive(Clone, Debug, Default)]
pub struct MeshGenOptions {
    pub labels: Option<LabelSet>,
}

pub type MeshGenResult = Result<
    MeshData<InMemorySieve<PointId, ()>, f64, VecStorage<f64>, VecStorage<CellType>>,
    MeshSieveError,
>;

pub trait ExternalMeshGenerator {
    fn generate(&self) -> MeshGenResult;
}
pub trait ExternalRemesher {
    fn remesh(
        &self,
        input: &MeshData<InMemorySieve<PointId, ()>, f64, VecStorage<f64>, VecStorage<CellType>>,
    ) -> MeshGenResult;
}

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
        for &vidx in vertices {
            let vpoint = *vertex_points.get(vidx).ok_or_else(|| {
                invalid_geometry(format!("cell {cell_idx} references missing vertex {vidx}"))
            })?;
            sieve.add_arrow(cell_points[cell_idx], vpoint, ());
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

#[cfg(any(feature = "triangle-support", feature = "tetgen-support"))]
fn generated_vertex_point(vertex_index: usize) -> Result<PointId, MeshSieveError> {
    PointId::new(vertex_index as u64 + 1)
}

#[cfg(any(feature = "triangle-support", feature = "tetgen-support"))]
fn generated_cell_point(vertex_count: usize, cell_index: usize) -> Result<PointId, MeshSieveError> {
    PointId::new((vertex_count + cell_index) as u64 + 1)
}

#[cfg(any(feature = "triangle-support", feature = "tetgen-support"))]
fn merge_labels(
    mesh: &mut MeshData<InMemorySieve<PointId, ()>, f64, VecStorage<f64>, VecStorage<CellType>>,
    labels: LabelSet,
) {
    if labels.is_empty() {
        return;
    }
    match mesh.labels.as_mut() {
        Some(existing) => {
            for (name, point, value) in labels.iter() {
                existing.set_label(point, &name, value);
            }
        }
        None => mesh.labels = Some(labels),
    }
}

pub fn structured_box_1d(nx: usize, min: f64, max: f64, options: MeshGenOptions) -> MeshGenResult {
    let mut out = interval_mesh(
        nx,
        min,
        max,
        MeshGenerationOptions {
            periodic: Periodicity::none(),
        },
    )?
    .mesh;
    if options.labels.is_some() {
        out.labels = options.labels;
    }
    Ok(out)
}
pub fn structured_box_2d(
    nx: usize,
    ny: usize,
    min: [f64; 2],
    max: [f64; 2],
    cell_type: StructuredCellType,
    options: MeshGenOptions,
) -> MeshGenResult {
    match cell_type {
        StructuredCellType::Triangle => {
            if nx == 0 || ny == 0 {
                return Err(invalid_geometry("nx and ny must be positive"));
            }
            let dx = (max[0] - min[0]) / nx as f64;
            let dy = (max[1] - min[1]) / ny as f64;
            let mut vertices = Vec::new();
            for j in 0..=ny {
                for i in 0..=nx {
                    vertices.push(vec![min[0] + dx * i as f64, min[1] + dy * j as f64]);
                }
            }
            let mut cells = Vec::new();
            let rs = nx + 1;
            for j in 0..ny {
                for i in 0..nx {
                    let v0 = j * rs + i;
                    let v1 = v0 + 1;
                    let v3 = v0 + rs;
                    let v2 = v3 + 1;
                    cells.push(vec![v0, v1, v2]);
                    cells.push(vec![v0, v2, v3]);
                }
            }
            build_mesh(2, &vertices, &cells, CellType::Triangle, options.labels)
        }
        StructuredCellType::Quadrilateral => {
            let mut out = quad_mesh(nx, ny, min, max, MeshGenerationOptions::default())?.mesh;
            if options.labels.is_some() {
                out.labels = options.labels;
            }
            Ok(out)
        }
        StructuredCellType::Hexahedron => {
            Err(invalid_geometry("hex elements are not valid for 2D meshes"))
        }
    }
}
pub fn structured_box_3d(
    nx: usize,
    ny: usize,
    nz: usize,
    min: [f64; 3],
    max: [f64; 3],
    cell_type: StructuredCellType,
    options: MeshGenOptions,
) -> MeshGenResult {
    match cell_type {
        StructuredCellType::Hexahedron => {
            let mut out = hex_mesh(nx, ny, nz, min, max, MeshGenerationOptions::default())?.mesh;
            if options.labels.is_some() {
                out.labels = options.labels;
            }
            Ok(out)
        }
        _ => Err(invalid_geometry(
            "triangle/quadrilateral elements are not valid for 3D box meshes",
        )),
    }
}

pub fn reference_cell(cell_type: CellType, options: MeshGenOptions) -> MeshGenResult {
    match cell_type {
        CellType::Segment => build_mesh(
            1,
            &[vec![0.0], vec![1.0]],
            &[vec![0, 1]],
            CellType::Segment,
            options.labels,
        ),
        CellType::Triangle => build_mesh(
            2,
            &[vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]],
            &[vec![0, 1, 2]],
            CellType::Triangle,
            options.labels,
        ),
        CellType::Quadrilateral => build_mesh(
            2,
            &[
                vec![0.0, 0.0],
                vec![1.0, 0.0],
                vec![1.0, 1.0],
                vec![0.0, 1.0],
            ],
            &[vec![0, 1, 2, 3]],
            CellType::Quadrilateral,
            options.labels,
        ),
        CellType::Tetrahedron => build_mesh(
            3,
            &[
                vec![0.0, 0.0, 0.0],
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ],
            &[vec![0, 1, 2, 3]],
            CellType::Tetrahedron,
            options.labels,
        ),
        CellType::Hexahedron => structured_box_3d(
            1,
            1,
            1,
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            StructuredCellType::Hexahedron,
            options,
        ),
        _ => Err(invalid_geometry("unsupported reference cell type")),
    }
}

pub fn sphere_shell(
    radius: f64,
    n_lat: usize,
    n_lon: usize,
    options: MeshGenOptions,
) -> MeshGenResult {
    /* unchanged behavior simplified */
    if radius <= 0.0 || n_lat < 2 || n_lon < 3 {
        return Err(invalid_geometry("invalid sphere shell parameters"));
    }
    let mut vertices = vec![vec![0.0, 0.0, radius]];
    let mut rings = Vec::new();
    for lat in 1..n_lat {
        let th = std::f64::consts::PI * (lat as f64) / (n_lat as f64);
        let mut ring = Vec::new();
        for lon in 0..n_lon {
            let ph = std::f64::consts::TAU * (lon as f64) / (n_lon as f64);
            ring.push(vertices.len());
            vertices.push(vec![
                radius * th.sin() * ph.cos(),
                radius * th.sin() * ph.sin(),
                radius * th.cos(),
            ]);
        }
        rings.push(ring);
    }
    let bottom = vertices.len();
    vertices.push(vec![0.0, 0.0, -radius]);
    let mut cells = Vec::new();
    if let Some(r) = rings.first() {
        for l in 0..n_lon {
            cells.push(vec![0, r[l], r[(l + 1) % n_lon]])
        }
    }
    for b in 0..rings.len().saturating_sub(1) {
        for l in 0..n_lon {
            let n = (l + 1) % n_lon;
            let a = &rings[b];
            let c = &rings[b + 1];
            cells.push(vec![a[l], c[l], c[n]]);
            cells.push(vec![a[l], c[n], a[n]]);
        }
    }
    if let Some(r) = rings.last() {
        for l in 0..n_lon {
            cells.push(vec![r[l], bottom, r[(l + 1) % n_lon]])
        }
    }
    build_mesh(3, &vertices, &cells, CellType::Triangle, options.labels)
}

/// Generate a quadrilateral surface mesh for a cylindrical shell.
pub fn cylinder_shell(
    radius: f64,
    height: f64,
    n_around: usize,
    n_height: usize,
    options: MeshGenOptions,
) -> MeshGenResult {
    if radius <= 0.0 || height <= 0.0 || n_around < 3 || n_height == 0 {
        return Err(invalid_geometry("invalid cylinder shell parameters"));
    }
    let mut vertices = Vec::new();
    for layer in 0..=n_height {
        let z = height * layer as f64 / n_height as f64;
        for i in 0..n_around {
            let theta = std::f64::consts::TAU * i as f64 / n_around as f64;
            vertices.push(vec![radius * theta.cos(), radius * theta.sin(), z]);
        }
    }
    let mut cells = Vec::new();
    for layer in 0..n_height {
        let base = layer * n_around;
        let top = (layer + 1) * n_around;
        for i in 0..n_around {
            let next = (i + 1) % n_around;
            cells.push(vec![base + i, base + next, top + next, top + i]);
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

/// Polygonal input for Triangle constrained Delaunay triangulation.
#[derive(Clone, Debug, Default)]
pub struct TriangleInput {
    /// Planar vertex coordinates. Indices used by segments, holes, and regions are zero-based.
    pub vertices: Vec<[f64; 2]>,
    /// Optional per-vertex boundary markers written to Triangle `.poly` input and restored as
    /// `triangle:vertex_marker` labels when present in `.node` output. Missing entries default to 0.
    pub vertex_markers: Vec<i32>,
    /// Constrained segments as zero-based vertex index pairs.
    pub segments: Vec<[usize; 2]>,
    /// Optional per-segment boundary markers written to `.poly` input. Missing entries default to 0.
    pub segment_markers: Vec<i32>,
    /// Hole seed points in Triangle `.poly` syntax.
    pub holes: Vec<[f64; 2]>,
    /// Region seed points with attributes and optional maximum area.
    pub regions: Vec<TriangleRegion>,
}

/// Region metadata for Triangle `.poly` inputs.
#[derive(Clone, Copy, Debug)]
pub struct TriangleRegion {
    /// Interior point identifying the region.
    pub point: [f64; 2],
    /// Region attribute restored as `triangle:region` labels on output cells.
    pub attribute: i32,
    /// Optional maximum area for this region.
    pub max_area: Option<f64>,
}

/// Runtime options for the Triangle command-line backend.
#[derive(Clone, Debug)]
pub struct TriangleOptions {
    /// Executable name or path. Defaults to `triangle`.
    pub executable: String,
    /// Optional minimum angle passed with `-q{angle}`.
    pub min_angle: Option<f64>,
    /// Optional maximum triangle area passed with `-a{area}`.
    pub max_area: Option<f64>,
    /// Preserve temporary input/output files for debugging.
    pub keep_files: bool,
    /// Additional raw command-line flags.
    pub extra_args: Vec<String>,
}

impl Default for TriangleOptions {
    fn default() -> Self {
        Self {
            executable: "triangle".into(),
            min_angle: None,
            max_area: None,
            keep_files: false,
            extra_args: Vec::new(),
        }
    }
}

/// Piecewise-linear complex input for TetGen tetrahedralization.
#[derive(Clone, Debug, Default)]
pub struct TetGenInput {
    /// Spatial vertex coordinates. Facets use zero-based vertex indices.
    pub vertices: Vec<[f64; 3]>,
    /// Optional per-vertex boundary markers restored as `tetgen:vertex_marker` labels when present.
    pub vertex_markers: Vec<i32>,
    /// Boundary facets, each represented by one polygon with at least three vertices.
    pub facets: Vec<Vec<usize>>,
    /// Optional per-facet boundary markers written to `.poly` input. Missing entries default to 0.
    pub facet_markers: Vec<i32>,
    /// Hole seed points in TetGen `.poly` syntax.
    pub holes: Vec<[f64; 3]>,
    /// Region seed points with attributes and optional maximum volume.
    pub regions: Vec<TetGenRegion>,
}

/// Region metadata for TetGen `.poly` inputs.
#[derive(Clone, Copy, Debug)]
pub struct TetGenRegion {
    /// Interior point identifying the region.
    pub point: [f64; 3],
    /// Region attribute restored as `tetgen:region` labels on output cells.
    pub attribute: i32,
    /// Optional maximum volume for this region.
    pub max_volume: Option<f64>,
}

/// Runtime options for the TetGen command-line backend.
#[derive(Clone, Debug)]
pub struct TetGenOptions {
    /// Executable name or path. Defaults to `tetgen`.
    pub executable: String,
    /// Optional quality bound passed with `-q{value}`.
    pub quality: Option<f64>,
    /// Optional maximum tetrahedron volume passed with `-a{volume}`.
    pub max_volume: Option<f64>,
    /// Preserve temporary input/output files for debugging.
    pub keep_files: bool,
    /// Additional raw command-line flags.
    pub extra_args: Vec<String>,
}

impl Default for TetGenOptions {
    fn default() -> Self {
        Self {
            executable: "tetgen".into(),
            quality: None,
            max_volume: None,
            keep_files: false,
            extra_args: Vec::new(),
        }
    }
}

/// Gmsh geometry script input for DMPLEX-like mesh creation.
#[derive(Clone, Debug)]
pub struct GmshInput {
    /// Complete `.geo` script contents.
    pub geo: String,
    /// Topological dimension to generate (`2` or `3`).
    pub dimension: usize,
}

/// Runtime options for the Gmsh command-line backend.
#[derive(Clone, Debug)]
pub struct GmshOptions {
    /// Executable name or path. Defaults to `gmsh`.
    pub executable: String,
    /// Output format passed with `-format`. Defaults to `msh4`.
    pub format: String,
    /// Preserve temporary input/output files for debugging.
    pub keep_files: bool,
    /// Additional raw command-line flags.
    pub extra_args: Vec<String>,
}

impl Default for GmshOptions {
    fn default() -> Self {
        Self {
            executable: "gmsh".into(),
            format: "msh4".into(),
            keep_files: false,
            extra_args: Vec::new(),
        }
    }
}

#[cfg(any(
    feature = "triangle-support",
    feature = "tetgen-support",
    feature = "gmsh-support"
))]
struct TempRunDir {
    path: PathBuf,
    keep: bool,
}

#[cfg(any(
    feature = "triangle-support",
    feature = "tetgen-support",
    feature = "gmsh-support"
))]
impl TempRunDir {
    fn new(prefix: &str, keep: bool) -> Result<Self, MeshSieveError> {
        let mut path = std::env::temp_dir();
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|err| invalid_geometry(format!("system clock before UNIX epoch: {err}")))?
            .as_nanos();
        path.push(format!(
            "mesh-sieve-{prefix}-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&path)?;
        Ok(Self { path, keep })
    }
}

#[cfg(any(
    feature = "triangle-support",
    feature = "tetgen-support",
    feature = "gmsh-support"
))]
impl Drop for TempRunDir {
    fn drop(&mut self) {
        if !self.keep {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}

#[cfg(any(
    feature = "triangle-support",
    feature = "tetgen-support",
    feature = "gmsh-support"
))]
fn run_command(mut command: Command, name: &str) -> Result<(), MeshSieveError> {
    let output = command.output().map_err(|err| {
        invalid_geometry(format!(
            "failed to execute {name}; ensure the executable is installed and on PATH: {err}"
        ))
    })?;
    if !output.status.success() {
        return Err(invalid_geometry(format!(
            "{name} exited with status {}\nstdout:\n{}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )));
    }
    Ok(())
}

#[cfg(any(feature = "triangle-support", feature = "tetgen-support"))]
fn read_non_comment_line(reader: &mut impl BufRead) -> Result<String, MeshSieveError> {
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            return Err(invalid_geometry("unexpected end of mesh-generator output"));
        }
        let trimmed = line.split('#').next().unwrap_or_default().trim();
        if !trimmed.is_empty() {
            return Ok(trimmed.to_string());
        }
    }
}

#[cfg(any(feature = "triangle-support", feature = "tetgen-support"))]
fn parse_usize(raw: &str, context: &str) -> Result<usize, MeshSieveError> {
    raw.parse::<usize>()
        .map_err(|_| invalid_geometry(format!("invalid integer in {context}: {raw}")))
}

#[cfg(any(feature = "triangle-support", feature = "tetgen-support"))]
fn parse_f64(raw: &str, context: &str) -> Result<f64, MeshSieveError> {
    raw.parse::<f64>()
        .map_err(|_| invalid_geometry(format!("invalid coordinate in {context}: {raw}")))
}

#[cfg(feature = "triangle-support")]
fn write_triangle_poly(path: &Path, input: &TriangleInput) -> Result<(), MeshSieveError> {
    if input.vertices.len() < 3 {
        return Err(invalid_geometry(
            "Triangle requires at least three vertices",
        ));
    }
    for (seg_idx, segment) in input.segments.iter().enumerate() {
        for &idx in segment {
            if idx >= input.vertices.len() {
                return Err(invalid_geometry(format!(
                    "Triangle segment {seg_idx} references missing vertex {idx}"
                )));
            }
        }
    }
    let mut file = File::create(path)?;
    let has_vertex_markers = input.vertex_markers.iter().any(|marker| *marker != 0);
    writeln!(
        file,
        "{} 2 0 {}",
        input.vertices.len(),
        usize::from(has_vertex_markers)
    )?;
    for (idx, [x, y]) in input.vertices.iter().enumerate() {
        if has_vertex_markers {
            let marker = input.vertex_markers.get(idx).copied().unwrap_or_default();
            writeln!(file, "{} {x} {y} {marker}", idx + 1)?;
        } else {
            writeln!(file, "{} {x} {y}", idx + 1)?;
        }
    }
    let has_segment_markers = input.segment_markers.iter().any(|marker| *marker != 0);
    writeln!(
        file,
        "{} {}",
        input.segments.len(),
        usize::from(has_segment_markers)
    )?;
    for (idx, [a, b]) in input.segments.iter().enumerate() {
        let marker = input.segment_markers.get(idx).copied().unwrap_or_default();
        if has_segment_markers {
            writeln!(file, "{} {} {} {marker}", idx + 1, a + 1, b + 1)?;
        } else {
            writeln!(file, "{} {} {}", idx + 1, a + 1, b + 1)?;
        }
    }
    writeln!(file, "{}", input.holes.len())?;
    for (idx, [x, y]) in input.holes.iter().enumerate() {
        writeln!(file, "{} {x} {y}", idx + 1)?;
    }
    writeln!(file, "{}", input.regions.len())?;
    for (idx, region) in input.regions.iter().enumerate() {
        let [x, y] = region.point;
        if let Some(area) = region.max_area {
            writeln!(file, "{} {x} {y} {} {area}", idx + 1, region.attribute)?;
        } else {
            writeln!(file, "{} {x} {y} {}", idx + 1, region.attribute)?;
        }
    }
    Ok(())
}

#[cfg(feature = "triangle-support")]
fn read_triangle_output(prefix: &Path) -> MeshGenResult {
    let (nodes, node_markers) = read_triangle_nodes(&prefix.with_extension("1.node"))?;
    let (cells, cell_attrs) = read_triangle_elements(&prefix.with_extension("1.ele"), 3)?;
    let mut mesh = build_mesh(2, &nodes, &cells, CellType::Triangle, None)?;
    let mut labels = LabelSet::new();
    for (idx, marker) in node_markers.into_iter().enumerate() {
        if marker != 0 {
            labels.set_label(
                generated_vertex_point(idx)?,
                "triangle:vertex_marker",
                marker,
            );
            labels.set_label(generated_vertex_point(idx)?, "boundary", marker);
        }
    }
    for (idx, attrs) in cell_attrs.into_iter().enumerate() {
        if let Some(region) = attrs.first().copied() {
            labels.set_label(
                generated_cell_point(nodes.len(), idx)?,
                "triangle:region",
                region,
            );
        }
    }
    merge_labels(&mut mesh, labels);
    Ok(mesh)
}

#[cfg(feature = "triangle-support")]
fn read_triangle_nodes(path: &Path) -> Result<(Vec<Vec<f64>>, Vec<i32>), MeshSieveError> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let header = read_non_comment_line(&mut reader)?;
    let parts: Vec<_> = header.split_whitespace().collect();
    if parts.len() < 2 || parse_usize(parts[1], "Triangle .node header")? != 2 {
        return Err(invalid_geometry(
            "Triangle .node output must be two-dimensional",
        ));
    }
    let count = parse_usize(parts[0], "Triangle .node header")?;
    let has_marker = parts.get(3).is_some_and(|raw| *raw != "0");
    let mut out = Vec::with_capacity(count);
    let mut markers = Vec::with_capacity(count);
    for _ in 0..count {
        let line = read_non_comment_line(&mut reader)?;
        let parts: Vec<_> = line.split_whitespace().collect();
        if parts.len() < 3 {
            return Err(invalid_geometry("malformed Triangle .node row"));
        }
        out.push(vec![
            parse_f64(parts[1], "Triangle .node row")?,
            parse_f64(parts[2], "Triangle .node row")?,
        ]);
        markers.push(if has_marker && parts.len() > 3 {
            parts[3].parse::<i32>().map_err(|_| {
                invalid_geometry(format!(
                    "invalid marker in Triangle .node row: {}",
                    parts[3]
                ))
            })?
        } else {
            0
        });
    }
    Ok((out, markers))
}

#[cfg(feature = "triangle-support")]
fn read_triangle_elements(
    path: &Path,
    expected_nodes: usize,
) -> Result<(Vec<Vec<usize>>, Vec<Vec<i32>>), MeshSieveError> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let header = read_non_comment_line(&mut reader)?;
    let parts: Vec<_> = header.split_whitespace().collect();
    if parts.len() < 2 || parse_usize(parts[1], "Triangle .ele header")? != expected_nodes {
        return Err(invalid_geometry(
            "Triangle .ele output has unsupported element order",
        ));
    }
    let count = parse_usize(parts[0], "Triangle .ele header")?;
    let attribute_count = parts
        .get(2)
        .map_or(Ok(0), |raw| parse_usize(raw, "Triangle .ele header"))?;
    let mut out = Vec::with_capacity(count);
    let mut attrs = Vec::with_capacity(count);
    for _ in 0..count {
        let line = read_non_comment_line(&mut reader)?;
        let parts: Vec<_> = line.split_whitespace().collect();
        if parts.len() < expected_nodes + 1 {
            return Err(invalid_geometry("malformed Triangle .ele row"));
        }
        let mut cell = Vec::with_capacity(expected_nodes);
        for raw in &parts[1..=expected_nodes] {
            let one_based = parse_usize(raw, "Triangle .ele row")?;
            cell.push(
                one_based
                    .checked_sub(1)
                    .ok_or_else(|| invalid_geometry("Triangle wrote node id 0"))?,
            );
        }
        let mut row_attrs = Vec::with_capacity(attribute_count);
        for raw in parts.iter().skip(expected_nodes + 1).take(attribute_count) {
            row_attrs.push(raw.parse::<i32>().map_err(|_| {
                invalid_geometry(format!("invalid attribute in Triangle .ele row: {raw}"))
            })?);
        }
        out.push(cell);
        attrs.push(row_attrs);
    }
    Ok((out, attrs))
}

/// Generate a 2-D constrained triangulation by invoking Triangle.
#[cfg(feature = "triangle-support")]
pub fn generate_with_triangle(input: &TriangleInput, options: &TriangleOptions) -> MeshGenResult {
    let tmp = TempRunDir::new("triangle", options.keep_files)?;
    let prefix = tmp.path.join("domain");
    let poly_path = prefix.with_extension("poly");
    write_triangle_poly(&poly_path, input)?;
    let mut args = vec!["-p".to_string(), "-Q".to_string()];
    if let Some(angle) = options.min_angle {
        args.push(format!("-q{angle}"));
    }
    if let Some(area) = options.max_area {
        args.push(format!("-a{area}"));
    }
    args.extend(options.extra_args.clone());
    args.push(poly_path.to_string_lossy().into_owned());
    let mut command = Command::new(&options.executable);
    command.args(&args).current_dir(&tmp.path);
    run_command(command, "Triangle")?;
    read_triangle_output(&prefix)
}

#[cfg(not(feature = "triangle-support"))]
pub fn generate_with_triangle(_input: &TriangleInput, _options: &TriangleOptions) -> MeshGenResult {
    Err(invalid_geometry("triangle-support feature not enabled"))
}

#[cfg(feature = "triangle-support")]
pub fn generate_with_triangle_adapter(generator: &dyn ExternalMeshGenerator) -> MeshGenResult {
    generator.generate()
}
#[cfg(not(feature = "triangle-support"))]
pub fn generate_with_triangle_adapter(_generator: &dyn ExternalMeshGenerator) -> MeshGenResult {
    Err(invalid_geometry("triangle-support feature not enabled"))
}

#[cfg(feature = "tetgen-support")]
fn write_tetgen_poly(path: &Path, input: &TetGenInput) -> Result<(), MeshSieveError> {
    if input.vertices.len() < 4 {
        return Err(invalid_geometry("TetGen requires at least four vertices"));
    }
    let mut file = File::create(path)?;
    let has_vertex_markers = input.vertex_markers.iter().any(|marker| *marker != 0);
    writeln!(
        file,
        "{} 3 0 {}",
        input.vertices.len(),
        usize::from(has_vertex_markers)
    )?;
    for (idx, [x, y, z]) in input.vertices.iter().enumerate() {
        if has_vertex_markers {
            let marker = input.vertex_markers.get(idx).copied().unwrap_or_default();
            writeln!(file, "{} {x} {y} {z} {marker}", idx + 1)?;
        } else {
            writeln!(file, "{} {x} {y} {z}", idx + 1)?;
        }
    }
    let has_facet_markers = input.facet_markers.iter().any(|marker| *marker != 0);
    writeln!(
        file,
        "{} {}",
        input.facets.len(),
        usize::from(has_facet_markers)
    )?;
    for (facet_idx, facet) in input.facets.iter().enumerate() {
        if facet.len() < 3 {
            return Err(invalid_geometry(format!(
                "TetGen facet {facet_idx} has fewer than three vertices"
            )));
        }
        for &idx in facet {
            if idx >= input.vertices.len() {
                return Err(invalid_geometry(format!(
                    "TetGen facet {facet_idx} references missing vertex {idx}"
                )));
            }
        }
        if has_facet_markers {
            let marker = input
                .facet_markers
                .get(facet_idx)
                .copied()
                .unwrap_or_default();
            writeln!(file, "1 0 {marker}")?;
        } else {
            writeln!(file, "1 0")?;
        }
        write!(file, "{}", facet.len())?;
        for idx in facet {
            write!(file, " {}", idx + 1)?;
        }
        writeln!(file)?;
    }
    writeln!(file, "{}", input.holes.len())?;
    for (idx, [x, y, z]) in input.holes.iter().enumerate() {
        writeln!(file, "{} {x} {y} {z}", idx + 1)?;
    }
    writeln!(file, "{}", input.regions.len())?;
    for (idx, region) in input.regions.iter().enumerate() {
        let [x, y, z] = region.point;
        if let Some(volume) = region.max_volume {
            writeln!(
                file,
                "{} {x} {y} {z} {} {volume}",
                idx + 1,
                region.attribute
            )?;
        } else {
            writeln!(file, "{} {x} {y} {z} {}", idx + 1, region.attribute)?;
        }
    }
    Ok(())
}

#[cfg(feature = "tetgen-support")]
fn read_tetgen_output(prefix: &Path) -> MeshGenResult {
    let (nodes, node_markers) = read_tetgen_nodes(&prefix.with_extension("1.node"))?;
    let (cells, cell_attrs) = read_tetgen_elements(&prefix.with_extension("1.ele"))?;
    let mut mesh = build_mesh(3, &nodes, &cells, CellType::Tetrahedron, None)?;
    let mut labels = LabelSet::new();
    for (idx, marker) in node_markers.into_iter().enumerate() {
        if marker != 0 {
            labels.set_label(generated_vertex_point(idx)?, "tetgen:vertex_marker", marker);
            labels.set_label(generated_vertex_point(idx)?, "boundary", marker);
        }
    }
    for (idx, attrs) in cell_attrs.into_iter().enumerate() {
        if let Some(region) = attrs.first().copied() {
            labels.set_label(
                generated_cell_point(nodes.len(), idx)?,
                "tetgen:region",
                region,
            );
        }
    }
    merge_labels(&mut mesh, labels);
    Ok(mesh)
}

#[cfg(feature = "tetgen-support")]
fn read_tetgen_nodes(path: &Path) -> Result<(Vec<Vec<f64>>, Vec<i32>), MeshSieveError> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let header = read_non_comment_line(&mut reader)?;
    let parts: Vec<_> = header.split_whitespace().collect();
    if parts.len() < 2 || parse_usize(parts[1], "TetGen .node header")? != 3 {
        return Err(invalid_geometry(
            "TetGen .node output must be three-dimensional",
        ));
    }
    let count = parse_usize(parts[0], "TetGen .node header")?;
    let has_marker = parts.get(3).is_some_and(|raw| *raw != "0");
    let mut out = Vec::with_capacity(count);
    let mut markers = Vec::with_capacity(count);
    for _ in 0..count {
        let line = read_non_comment_line(&mut reader)?;
        let parts: Vec<_> = line.split_whitespace().collect();
        if parts.len() < 4 {
            return Err(invalid_geometry("malformed TetGen .node row"));
        }
        out.push(vec![
            parse_f64(parts[1], "TetGen .node row")?,
            parse_f64(parts[2], "TetGen .node row")?,
            parse_f64(parts[3], "TetGen .node row")?,
        ]);
        markers.push(if has_marker && parts.len() > 4 {
            parts[4].parse::<i32>().map_err(|_| {
                invalid_geometry(format!("invalid marker in TetGen .node row: {}", parts[4]))
            })?
        } else {
            0
        });
    }
    Ok((out, markers))
}

#[cfg(feature = "tetgen-support")]
fn read_tetgen_elements(path: &Path) -> Result<(Vec<Vec<usize>>, Vec<Vec<i32>>), MeshSieveError> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let header = read_non_comment_line(&mut reader)?;
    let parts: Vec<_> = header.split_whitespace().collect();
    if parts.len() < 2 || parse_usize(parts[1], "TetGen .ele header")? != 4 {
        return Err(invalid_geometry(
            "TetGen .ele output has unsupported element order",
        ));
    }
    let count = parse_usize(parts[0], "TetGen .ele header")?;
    let attribute_count = parts
        .get(2)
        .map_or(Ok(0), |raw| parse_usize(raw, "TetGen .ele header"))?;
    let mut out = Vec::with_capacity(count);
    let mut attrs = Vec::with_capacity(count);
    for _ in 0..count {
        let line = read_non_comment_line(&mut reader)?;
        let parts: Vec<_> = line.split_whitespace().collect();
        if parts.len() < 5 {
            return Err(invalid_geometry("malformed TetGen .ele row"));
        }
        let mut cell = Vec::with_capacity(4);
        for raw in &parts[1..=4] {
            let one_based = parse_usize(raw, "TetGen .ele row")?;
            cell.push(
                one_based
                    .checked_sub(1)
                    .ok_or_else(|| invalid_geometry("TetGen wrote node id 0"))?,
            );
        }
        let mut row_attrs = Vec::with_capacity(attribute_count);
        for raw in parts.iter().skip(5).take(attribute_count) {
            row_attrs.push(raw.parse::<i32>().map_err(|_| {
                invalid_geometry(format!("invalid attribute in TetGen .ele row: {raw}"))
            })?);
        }
        out.push(cell);
        attrs.push(row_attrs);
    }
    Ok((out, attrs))
}

/// Generate a 3-D tetrahedralization by invoking TetGen.
#[cfg(feature = "tetgen-support")]
pub fn generate_with_tetgen(input: &TetGenInput, options: &TetGenOptions) -> MeshGenResult {
    let tmp = TempRunDir::new("tetgen", options.keep_files)?;
    let prefix = tmp.path.join("domain");
    let poly_path = prefix.with_extension("poly");
    write_tetgen_poly(&poly_path, input)?;
    let mut args = vec!["-p".to_string(), "-Q".to_string()];
    if let Some(quality) = options.quality {
        args.push(format!("-q{quality}"));
    }
    if let Some(volume) = options.max_volume {
        args.push(format!("-a{volume}"));
    }
    args.extend(options.extra_args.clone());
    args.push(poly_path.to_string_lossy().into_owned());
    let mut command = Command::new(&options.executable);
    command.args(&args).current_dir(&tmp.path);
    run_command(command, "TetGen")?;
    read_tetgen_output(&prefix)
}

#[cfg(not(feature = "tetgen-support"))]
pub fn generate_with_tetgen(_input: &TetGenInput, _options: &TetGenOptions) -> MeshGenResult {
    Err(invalid_geometry("tetgen-support feature not enabled"))
}

#[cfg(feature = "tetgen-support")]
pub fn generate_with_tetgen_adapter(generator: &dyn ExternalMeshGenerator) -> MeshGenResult {
    generator.generate()
}
#[cfg(not(feature = "tetgen-support"))]
pub fn generate_with_tetgen_adapter(_generator: &dyn ExternalMeshGenerator) -> MeshGenResult {
    Err(invalid_geometry("tetgen-support feature not enabled"))
}

#[cfg(feature = "gmsh-support")]
fn gmsh_to_plain(
    mesh: MeshData<crate::topology::sieve::MeshSieve, f64, VecStorage<f64>, VecStorage<CellType>>,
) -> MeshGenResult {
    let mut sieve = InMemorySieve::<PointId, ()>::default();
    for point in mesh.sieve.points_sorted() {
        MutableSieve::add_point(&mut sieve, point);
    }
    for src in mesh.sieve.base_points() {
        for dst in mesh.sieve.cone_points(src) {
            sieve.add_arrow(src, dst, ());
        }
    }
    sieve.sort_adjacency();
    Ok(MeshData {
        sieve,
        coordinates: mesh.coordinates,
        sections: mesh.sections,
        mixed_sections: mesh.mixed_sections,
        labels: mesh.labels,
        cell_types: mesh.cell_types,
        discretization: mesh.discretization,
    })
}

#[cfg(feature = "gmsh-support")]
fn read_gmsh_mesh(path: &Path) -> MeshGenResult {
    use crate::io::SieveSectionReader;
    let file = File::open(path)?;
    let mesh = crate::io::gmsh::GmshReader::default().read(file)?;
    gmsh_to_plain(mesh)
}

#[cfg(feature = "gmsh-support")]
fn write_plain_gmsh_v2(
    path: &Path,
    mesh: &MeshData<InMemorySieve<PointId, ()>, f64, VecStorage<f64>, VecStorage<CellType>>,
) -> Result<(), MeshSieveError> {
    let coords = mesh
        .coordinates
        .as_ref()
        .ok_or_else(|| invalid_geometry("Gmsh remeshing requires coordinates"))?;
    let cell_types = mesh
        .cell_types
        .as_ref()
        .ok_or_else(|| invalid_geometry("Gmsh remeshing requires cell types"))?;
    let node_ids: Vec<PointId> = coords.section().atlas().points().collect();
    let mut element_ids = Vec::new();
    for point in cell_types.atlas().points() {
        let ty = cell_types.try_restrict(point)?[0];
        if ty != CellType::Vertex {
            element_ids.push(point);
        }
    }
    let mut file = File::create(path)?;
    writeln!(file, "$MeshFormat\n2.2 0 8\n$EndMeshFormat")?;
    writeln!(file, "$Nodes")?;
    writeln!(file, "{}", node_ids.len())?;
    for point in &node_ids {
        let xyz = coords.section().try_restrict(*point)?;
        let x = xyz.first().copied().unwrap_or(0.0);
        let y = xyz.get(1).copied().unwrap_or(0.0);
        let z = xyz.get(2).copied().unwrap_or(0.0);
        writeln!(file, "{} {x} {y} {z}", point.get())?;
    }
    writeln!(file, "$EndNodes")?;
    writeln!(file, "$Elements")?;
    writeln!(file, "{}", element_ids.len())?;
    for point in element_ids {
        let ty = cell_types.try_restrict(point)?[0];
        let elem_type = match ty {
            CellType::Segment => 1,
            CellType::Triangle => 2,
            CellType::Quadrilateral => 3,
            CellType::Tetrahedron => 4,
            CellType::Hexahedron => 5,
            CellType::Prism => 6,
            CellType::Pyramid => 7,
            _ => continue,
        };
        let (physical, entity) = mesh
            .labels
            .as_ref()
            .map(|labels| {
                (
                    labels
                        .get_label(point, "gmsh:physical")
                        .or_else(|| labels.get_label(point, "region"))
                        .unwrap_or_default(),
                    labels.get_label(point, "gmsh:entity").unwrap_or_default(),
                )
            })
            .unwrap_or_default();
        if physical != 0 || entity != 0 {
            write!(file, "{} {} 2 {physical} {entity}", point.get(), elem_type)?;
        } else {
            write!(file, "{} {} 0", point.get(), elem_type)?;
        }
        for node in mesh.sieve.cone_points(point) {
            write!(file, " {}", node.get())?;
        }
        writeln!(file)?;
    }
    writeln!(file, "$EndElements")?;
    Ok(())
}

/// Generate a mesh by invoking Gmsh on a `.geo` script and importing the result.
#[cfg(feature = "gmsh-support")]
pub fn generate_with_gmsh(input: &GmshInput, options: &GmshOptions) -> MeshGenResult {
    if input.dimension != 2 && input.dimension != 3 {
        return Err(invalid_geometry("Gmsh generation dimension must be 2 or 3"));
    }
    let tmp = TempRunDir::new("gmsh", options.keep_files)?;
    let geo_path = tmp.path.join("domain.geo");
    let out_path = tmp.path.join("domain.msh");
    fs::write(&geo_path, &input.geo)?;
    let mut command = Command::new(&options.executable);
    command
        .arg(format!("-{}", input.dimension))
        .arg(&geo_path)
        .arg("-format")
        .arg(&options.format)
        .arg("-o")
        .arg(&out_path)
        .args(&options.extra_args)
        .current_dir(&tmp.path);
    run_command(command, "Gmsh")?;
    read_gmsh_mesh(&out_path)
}

#[cfg(not(feature = "gmsh-support"))]
pub fn generate_with_gmsh(_input: &GmshInput, _options: &GmshOptions) -> MeshGenResult {
    Err(invalid_geometry("gmsh-support feature not enabled"))
}

/// Remesh an existing mesh by round-tripping through Gmsh `.msh` files.
#[cfg(feature = "gmsh-support")]
pub fn remesh_with_gmsh(
    input: &MeshData<InMemorySieve<PointId, ()>, f64, VecStorage<f64>, VecStorage<CellType>>,
    dimension: usize,
    options: &GmshOptions,
) -> MeshGenResult {
    if dimension != 2 && dimension != 3 {
        return Err(invalid_geometry("Gmsh remeshing dimension must be 2 or 3"));
    }
    let tmp = TempRunDir::new("gmsh-remesh", options.keep_files)?;
    let in_path = tmp.path.join("input.msh");
    let out_path = tmp.path.join("remeshed.msh");
    write_plain_gmsh_v2(&in_path, input)?;
    let mut command = Command::new(&options.executable);
    command
        .arg(format!("-{}", dimension))
        .arg(&in_path)
        .arg("-format")
        .arg(&options.format)
        .arg("-o")
        .arg(&out_path)
        .args(&options.extra_args)
        .current_dir(&tmp.path);
    run_command(command, "Gmsh")?;
    read_gmsh_mesh(&out_path)
}

#[cfg(not(feature = "gmsh-support"))]
pub fn remesh_with_gmsh(
    _input: &MeshData<InMemorySieve<PointId, ()>, f64, VecStorage<f64>, VecStorage<CellType>>,
    _dimension: usize,
    _options: &GmshOptions,
) -> MeshGenResult {
    Err(invalid_geometry("gmsh-support feature not enabled"))
}

#[cfg(feature = "gmsh-support")]
pub fn remesh_with_gmsh_adapter(
    remesher: &dyn ExternalRemesher,
    input: &MeshData<InMemorySieve<PointId, ()>, f64, VecStorage<f64>, VecStorage<CellType>>,
) -> MeshGenResult {
    remesher.remesh(input)
}
#[cfg(not(feature = "gmsh-support"))]
pub fn remesh_with_gmsh_adapter(
    _remesher: &dyn ExternalRemesher,
    _input: &MeshData<InMemorySieve<PointId, ()>, f64, VecStorage<f64>, VecStorage<CellType>>,
) -> MeshGenResult {
    Err(invalid_geometry("gmsh-support feature not enabled"))
}
