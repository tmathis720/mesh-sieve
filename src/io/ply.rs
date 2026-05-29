//! ASCII PLY mesh reader.
//!
//! Supports the common polygonal PLY surface subset used by PETSc/DMPlex:
//! `vertex` elements with `x y z` properties and `face` elements whose first
//! value is the vertex count followed by zero-based vertex indices. Faces with
//! three and four vertices become triangle and quadrilateral cells; larger faces
//! are imported as `CellType::Polygon(n)`.

use crate::data::atlas::Atlas;
use crate::data::coordinates::Coordinates;
use crate::data::section::Section;
use crate::data::storage::VecStorage;
use crate::io::{MeshData, SieveSectionReader};
use crate::mesh_error::MeshSieveError;
use crate::topology::cell_type::CellType;
use crate::topology::point::PointId;
use crate::topology::sieve::{MeshSieve, MutableSieve, Sieve};
use std::io::Read;

/// Reader for ASCII PLY polygon meshes.
#[derive(Debug, Default, Clone)]
pub struct PlyReader;

impl SieveSectionReader for PlyReader {
    type Sieve = MeshSieve;
    type Value = f64;
    type Storage = VecStorage<f64>;
    type CellStorage = VecStorage<CellType>;

    fn read<R: Read>(
        &self,
        mut reader: R,
    ) -> Result<MeshData<Self::Sieve, Self::Value, Self::Storage, Self::CellStorage>, MeshSieveError>
    {
        let mut text = String::new();
        reader.read_to_string(&mut text)?;
        parse_ascii_ply(&text)
    }
}

fn parse_ascii_ply(
    text: &str,
) -> Result<MeshData<MeshSieve, f64, VecStorage<f64>, VecStorage<CellType>>, MeshSieveError> {
    let mut lines = text.lines();
    if lines.next().map(str::trim) != Some("ply") {
        return Err(MeshSieveError::MeshIoParse(
            "PLY header must start with `ply`".into(),
        ));
    }
    let mut vertex_count = None;
    let mut face_count = None;
    let mut ascii = false;
    for line in &mut lines {
        let line = line.trim();
        if line == "end_header" {
            break;
        }
        let parts: Vec<_> = line.split_whitespace().collect();
        match parts.as_slice() {
            ["format", "ascii", ..] => ascii = true,
            ["element", "vertex", n] => vertex_count = Some(parse_usize(n, "vertex count")?),
            ["element", "face", n] => face_count = Some(parse_usize(n, "face count")?),
            _ => {}
        }
    }
    if !ascii {
        return Err(MeshSieveError::MeshIoParse(
            "only ASCII PLY is supported".into(),
        ));
    }
    let vertex_count = vertex_count
        .ok_or_else(|| MeshSieveError::MeshIoParse("missing PLY vertex element".into()))?;
    let face_count = face_count.unwrap_or(0);

    let mut vertices = Vec::with_capacity(vertex_count);
    for idx in 0..vertex_count {
        let line = lines
            .next()
            .ok_or_else(|| MeshSieveError::MeshIoParse(format!("missing PLY vertex {idx}")))?;
        let vals: Vec<f64> = line
            .split_whitespace()
            .take(3)
            .map(|v| v.parse::<f64>())
            .collect::<Result<_, _>>()
            .map_err(|_| MeshSieveError::MeshIoParse(format!("invalid PLY vertex line: {line}")))?;
        if vals.len() < 3 {
            return Err(MeshSieveError::MeshIoParse(format!(
                "PLY vertex {idx} has fewer than 3 coordinates"
            )));
        }
        vertices.push([vals[0], vals[1], vals[2]]);
    }

    let mut faces = Vec::with_capacity(face_count);
    for idx in 0..face_count {
        let line = lines
            .next()
            .ok_or_else(|| MeshSieveError::MeshIoParse(format!("missing PLY face {idx}")))?;
        let ints: Vec<usize> = line
            .split_whitespace()
            .map(|v| v.parse::<usize>())
            .collect::<Result<_, _>>()
            .map_err(|_| MeshSieveError::MeshIoParse(format!("invalid PLY face line: {line}")))?;
        let (&n, rest) = ints
            .split_first()
            .ok_or_else(|| MeshSieveError::MeshIoParse("empty PLY face line".into()))?;
        if rest.len() < n {
            return Err(MeshSieveError::MeshIoParse(format!(
                "PLY face {idx} has {n} vertices but only {} indices",
                rest.len()
            )));
        }
        let conn = rest[..n]
            .iter()
            .map(|zero| {
                if *zero >= vertex_count {
                    return Err(MeshSieveError::MeshIoParse(format!(
                        "PLY face {idx} references vertex {zero} outside 0..{vertex_count}"
                    )));
                }
                PointId::new((*zero as u64) + 1)
            })
            .collect::<Result<Vec<_>, _>>()?;
        faces.push(conn);
    }
    build_mesh(vertices, faces)
}

fn parse_usize(token: &str, what: &str) -> Result<usize, MeshSieveError> {
    token
        .parse::<usize>()
        .map_err(|_| MeshSieveError::MeshIoParse(format!("invalid PLY {what}: {token}")))
}

pub(crate) fn build_mesh(
    vertices: Vec<[f64; 3]>,
    faces: Vec<Vec<PointId>>,
) -> Result<MeshData<MeshSieve, f64, VecStorage<f64>, VecStorage<CellType>>, MeshSieveError> {
    let mut sieve = MeshSieve::default();
    let mut coord_atlas = Atlas::default();
    for i in 0..vertices.len() {
        let p = PointId::new((i as u64) + 1)?;
        MutableSieve::add_point(&mut sieve, p);
        coord_atlas.try_insert(p, 3)?;
    }
    let mut cell_atlas = Atlas::default();
    let cell_offset = vertices.len() as u64 + 1;
    for (i, face) in faces.iter().enumerate() {
        let cell = PointId::new(cell_offset + i as u64)?;
        MutableSieve::add_point(&mut sieve, cell);
        for vertex in face {
            Sieve::add_arrow(&mut sieve, cell, *vertex, ());
        }
        cell_atlas.try_insert(cell, 1)?;
    }
    let mut coordinates = Coordinates::try_new(2, 3, coord_atlas)?;
    for (i, xyz) in vertices.iter().enumerate() {
        coordinates
            .section_mut()
            .try_set(PointId::new((i as u64) + 1)?, xyz)?;
    }
    let mut cell_types = Section::<CellType, VecStorage<CellType>>::new(cell_atlas);
    for (i, face) in faces.iter().enumerate() {
        let cell = PointId::new(cell_offset + i as u64)?;
        let cell_type = match face.len() {
            1 => CellType::Vertex,
            2 => CellType::Segment,
            3 => CellType::Triangle,
            4 => CellType::Quadrilateral,
            n if n <= u8::MAX as usize => CellType::Polygon(n as u8),
            n => {
                return Err(MeshSieveError::MeshIoParse(format!(
                    "PLY polygon with {n} vertices is too large"
                )));
            }
        };
        cell_types.try_set(cell, &[cell_type])?;
    }
    let mut mesh = MeshData::new(sieve);
    mesh.coordinates = Some(coordinates);
    mesh.cell_types = Some(cell_types);
    Ok(mesh)
}
