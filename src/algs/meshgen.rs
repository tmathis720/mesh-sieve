//! Basic mesh generators and external-generator hooks.

use crate::data::atlas::Atlas;
use crate::data::coordinates::Coordinates;
use crate::data::mixed_section::MixedSectionStore;
use crate::data::section::Section;
use crate::data::storage::VecStorage;
use crate::io::MeshData;
use crate::mesh_error::MeshSieveError;
use crate::mesh_generation::{MeshGenerationOptions, Periodicity, hex_mesh, interval_mesh, quad_mesh};
use crate::topology::cell_type::CellType;
use crate::topology::labels::LabelSet;
use crate::topology::point::PointId;
use crate::topology::sieve::sieve_trait::Sieve;
use crate::topology::sieve::{InMemorySieve, MutableSieve};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug)]
pub enum StructuredCellType { Triangle, Quadrilateral, Hexahedron }

#[derive(Clone, Debug, Default)]
pub struct MeshGenOptions { pub labels: Option<LabelSet> }

type MeshGenResult = Result<MeshData<InMemorySieve<PointId, ()>, f64, VecStorage<f64>, VecStorage<CellType>>, MeshSieveError>;

pub trait ExternalMeshGenerator { fn generate(&self) -> MeshGenResult; }
pub trait ExternalRemesher { fn remesh(&self, input: &MeshData<InMemorySieve<PointId, ()>, f64, VecStorage<f64>, VecStorage<CellType>>) -> MeshGenResult; }

fn invalid_geometry(message: impl Into<String>) -> MeshSieveError { MeshSieveError::InvalidGeometry(message.into()) }

fn build_mesh(dimension: usize, vertex_coords: &[Vec<f64>], cells: &[Vec<usize>], cell_type: CellType, labels: Option<LabelSet>) -> MeshGenResult {
    if dimension == 0 { return Err(invalid_geometry("dimension must be non-zero")); }
    let mut sieve = InMemorySieve::<PointId, ()>::default();
    let mut next_id = 1u64;
    let mut vertex_points = Vec::with_capacity(vertex_coords.len());
    for _ in 0..vertex_coords.len() { let pid = PointId::new(next_id)?; next_id += 1; MutableSieve::add_point(&mut sieve, pid); vertex_points.push(pid); }
    let mut cell_points = Vec::with_capacity(cells.len());
    for _ in 0..cells.len() { let pid = PointId::new(next_id)?; next_id += 1; MutableSieve::add_point(&mut sieve, pid); cell_points.push(pid); }
    for (cell_idx, vertices) in cells.iter().enumerate() {
        for &vidx in vertices {
            let vpoint = *vertex_points.get(vidx).ok_or_else(|| invalid_geometry(format!("cell {cell_idx} references missing vertex {vidx}")))?;
            sieve.add_arrow(cell_points[cell_idx], vpoint, ());
        }
    }
    sieve.sort_adjacency();
    let mut coord_atlas = Atlas::default();
    for &p in &vertex_points { coord_atlas.try_insert(p, dimension)?; }
    let mut coords = Coordinates::<f64, VecStorage<f64>>::try_new(dimension, dimension, coord_atlas)?;
    for (p, coord) in vertex_points.iter().zip(vertex_coords.iter()) { coords.section_mut().try_set(*p, coord)?; }
    let mut cell_atlas = Atlas::default();
    for &p in vertex_points.iter().chain(cell_points.iter()) { cell_atlas.try_insert(p, 1)?; }
    let mut cell_types = Section::<CellType, VecStorage<CellType>>::new(cell_atlas);
    for &p in &vertex_points { cell_types.try_set(p, &[CellType::Vertex])?; }
    for &p in &cell_points { cell_types.try_set(p, &[cell_type])?; }
    Ok(MeshData { sieve, coordinates: Some(coords), sections: BTreeMap::new(), mixed_sections: MixedSectionStore::default(), labels, cell_types: Some(cell_types), discretization: None })
}

pub fn structured_box_1d(nx: usize, min: f64, max: f64, options: MeshGenOptions) -> MeshGenResult {
    let mut out = interval_mesh(nx, min, max, MeshGenerationOptions { periodic: Periodicity::none() })?.mesh;
    if options.labels.is_some() { out.labels = options.labels; }
    Ok(out)
}
pub fn structured_box_2d(nx: usize, ny: usize, min: [f64; 2], max: [f64; 2], cell_type: StructuredCellType, options: MeshGenOptions) -> MeshGenResult {
    match cell_type {
        StructuredCellType::Triangle => {
            if nx == 0 || ny == 0 { return Err(invalid_geometry("nx and ny must be positive")); }
            let dx = (max[0]-min[0])/nx as f64; let dy = (max[1]-min[1])/ny as f64;
            let mut vertices = Vec::new();
            for j in 0..=ny { for i in 0..=nx { vertices.push(vec![min[0]+dx*i as f64,min[1]+dy*j as f64]); } }
            let mut cells=Vec::new(); let rs=nx+1;
            for j in 0..ny { for i in 0..nx { let v0=j*rs+i; let v1=v0+1; let v3=v0+rs; let v2=v3+1; cells.push(vec![v0,v1,v2]); cells.push(vec![v0,v2,v3]); } }
            build_mesh(2,&vertices,&cells,CellType::Triangle,options.labels)
        }
        StructuredCellType::Quadrilateral => { let mut out = quad_mesh(nx, ny, min, max, MeshGenerationOptions::default())?.mesh; if options.labels.is_some(){out.labels=options.labels;} Ok(out) }
        StructuredCellType::Hexahedron => Err(invalid_geometry("hex elements are not valid for 2D meshes")),
    }
}
pub fn structured_box_3d(nx: usize, ny: usize, nz: usize, min: [f64; 3], max: [f64; 3], cell_type: StructuredCellType, options: MeshGenOptions) -> MeshGenResult {
    match cell_type { StructuredCellType::Hexahedron => { let mut out = hex_mesh(nx,ny,nz,min,max,MeshGenerationOptions::default())?.mesh; if options.labels.is_some(){out.labels=options.labels;} Ok(out) }, _ => Err(invalid_geometry("triangle/quadrilateral elements are not valid for 3D box meshes")) }
}

pub fn reference_cell(cell_type: CellType, options: MeshGenOptions) -> MeshGenResult { match cell_type {
    CellType::Segment => build_mesh(1,&[vec![0.0],vec![1.0]],&[vec![0,1]],CellType::Segment,options.labels),
    CellType::Triangle => build_mesh(2,&[vec![0.0,0.0],vec![1.0,0.0],vec![0.0,1.0]],&[vec![0,1,2]],CellType::Triangle,options.labels),
    CellType::Quadrilateral => build_mesh(2,&[vec![0.0,0.0],vec![1.0,0.0],vec![1.0,1.0],vec![0.0,1.0]],&[vec![0,1,2,3]],CellType::Quadrilateral,options.labels),
    CellType::Tetrahedron => build_mesh(3,&[vec![0.0,0.0,0.0],vec![1.0,0.0,0.0],vec![0.0,1.0,0.0],vec![0.0,0.0,1.0]],&[vec![0,1,2,3]],CellType::Tetrahedron,options.labels),
    CellType::Hexahedron => structured_box_3d(1,1,1,[0.0,0.0,0.0],[1.0,1.0,1.0],StructuredCellType::Hexahedron,options),
    _ => Err(invalid_geometry("unsupported reference cell type")),
}}

pub fn sphere_shell(radius:f64,n_lat:usize,n_lon:usize,options:MeshGenOptions)->MeshGenResult { /* unchanged behavior simplified */
    if radius<=0.0 || n_lat<2 || n_lon<3 { return Err(invalid_geometry("invalid sphere shell parameters")); }
    let mut vertices=vec![vec![0.0,0.0,radius]]; let mut rings=Vec::new();
    for lat in 1..n_lat { let th=std::f64::consts::PI*(lat as f64)/(n_lat as f64); let mut ring=Vec::new(); for lon in 0..n_lon { let ph=std::f64::consts::TAU*(lon as f64)/(n_lon as f64); ring.push(vertices.len()); vertices.push(vec![radius*th.sin()*ph.cos(),radius*th.sin()*ph.sin(),radius*th.cos()]); } rings.push(ring); }
    let bottom=vertices.len(); vertices.push(vec![0.0,0.0,-radius]); let mut cells=Vec::new();
    if let Some(r)=rings.first(){for l in 0..n_lon{cells.push(vec![0,r[l],r[(l+1)%n_lon]])}}
    for b in 0..rings.len().saturating_sub(1){for l in 0..n_lon{let n=(l+1)%n_lon; let a=&rings[b]; let c=&rings[b+1]; cells.push(vec![a[l],c[l],c[n]]); cells.push(vec![a[l],c[n],a[n]]);}}
    if let Some(r)=rings.last(){for l in 0..n_lon{cells.push(vec![r[l],bottom,r[(l+1)%n_lon]])}}
    build_mesh(3,&vertices,&cells,CellType::Triangle,options.labels)
}

#[cfg(feature="triangle-support")]
pub fn generate_with_triangle(generator: &dyn ExternalMeshGenerator) -> MeshGenResult { generator.generate() }
#[cfg(not(feature="triangle-support"))]
pub fn generate_with_triangle(_generator: &dyn ExternalMeshGenerator) -> MeshGenResult { Err(invalid_geometry("triangle-support feature not enabled")) }

#[cfg(feature="tetgen-support")]
pub fn generate_with_tetgen(generator: &dyn ExternalMeshGenerator) -> MeshGenResult { generator.generate() }
#[cfg(not(feature="tetgen-support"))]
pub fn generate_with_tetgen(_generator: &dyn ExternalMeshGenerator) -> MeshGenResult { Err(invalid_geometry("tetgen-support feature not enabled")) }

#[cfg(feature="gmsh-support")]
pub fn remesh_with_gmsh(remesher: &dyn ExternalRemesher, input: &MeshData<InMemorySieve<PointId, ()>, f64, VecStorage<f64>, VecStorage<CellType>>) -> MeshGenResult { remesher.remesh(input) }
#[cfg(not(feature="gmsh-support"))]
pub fn remesh_with_gmsh(_remesher: &dyn ExternalRemesher, _input: &MeshData<InMemorySieve<PointId, ()>, f64, VecStorage<f64>, VecStorage<CellType>>) -> MeshGenResult { Err(invalid_geometry("gmsh-support feature not enabled")) }
