//! Gmsh `.msh` reader.
//!
//! # Supported format
//! - ASCII `.msh` version **2.2**.
//! - Element types: 1 (line), 2 (triangle), 3 (quad), 4 (tet), 5 (hex),
//!   6 (prism), 7 (pyramid), 15 (point).
//!
//! # Limitations
//! - Binary files are not supported.
//! - `.msh` v4.x (block-based) is not supported.
//! - Higher-order elements are not supported.
//! - Element tags are ignored (no physical groups or boundary markers yet).
//! - Coordinates are always stored as 3D `(x, y, z)` tuples.

use crate::data::atlas::Atlas;
use crate::data::coordinates::Coordinates;
use crate::data::storage::VecStorage;
use crate::io::{MeshData, SieveSectionReader};
use crate::mesh_error::MeshSieveError;
use crate::topology::point::PointId;
use crate::topology::sieve::{InMemorySieve, MutableSieve, Sieve};
use std::collections::BTreeMap;
use std::io::Read;

/// Gmsh `.msh` reader for ASCII v2.2 meshes.
#[derive(Debug, Default, Clone)]
pub struct GmshReader;

impl GmshReader {
    fn parse_version(line: &str) -> Result<&str, MeshSieveError> {
        let mut parts = line.split_whitespace();
        let version = parts
            .next()
            .ok_or_else(|| MeshSieveError::MeshIoParse("missing mesh format version".into()))?;
        let file_type = parts
            .next()
            .ok_or_else(|| MeshSieveError::MeshIoParse("missing mesh format type".into()))?;
        if file_type != "0" {
            return Err(MeshSieveError::MeshIoParse(
                "binary .msh files are not supported".into(),
            ));
        }
        Ok(version)
    }

    fn element_node_count(elem_type: u32) -> Option<usize> {
        match elem_type {
            1 => Some(2),  // line
            2 => Some(3),  // triangle
            3 => Some(4),  // quad
            4 => Some(4),  // tet
            5 => Some(8),  // hex
            6 => Some(6),  // prism
            7 => Some(5),  // pyramid
            15 => Some(1), // point
            _ => None,
        }
    }

    fn parse_point_id(raw: &str) -> Result<PointId, MeshSieveError> {
        let raw = raw
            .parse::<u64>()
            .map_err(|_| MeshSieveError::MeshIoParse(format!("invalid point id: {raw}")))?;
        PointId::new(raw)
    }

    fn parse_coord(raw: &str) -> Result<f64, MeshSieveError> {
        raw.parse::<f64>()
            .map_err(|_| MeshSieveError::MeshIoParse(format!("invalid coordinate: {raw}")))
    }
}

impl SieveSectionReader for GmshReader {
    type Sieve = InMemorySieve<PointId, ()>;
    type Value = f64;
    type Storage = VecStorage<f64>;

    fn read<R: Read>(
        &self,
        mut reader: R,
    ) -> Result<MeshData<Self::Sieve, Self::Value, Self::Storage>, MeshSieveError> {
        let mut contents = String::new();
        reader.read_to_string(&mut contents)?;
        let mut lines = contents.lines().peekable();

        let mut version: Option<String> = None;
        let mut nodes: Vec<(PointId, [f64; 3])> = Vec::new();
        let mut elements: Vec<(PointId, Vec<PointId>)> = Vec::new();

        while let Some(line) = lines.next() {
            match line.trim() {
                "$MeshFormat" => {
                    let format_line = lines
                        .next()
                        .ok_or_else(|| MeshSieveError::MeshIoParse("missing MeshFormat".into()))?;
                    let parsed = Self::parse_version(format_line)?;
                    version = Some(parsed.to_string());
                    let end = lines.next().ok_or_else(|| {
                        MeshSieveError::MeshIoParse("missing EndMeshFormat".into())
                    })?;
                    if end.trim() != "$EndMeshFormat" {
                        return Err(MeshSieveError::MeshIoParse("missing $EndMeshFormat".into()));
                    }
                }
                "$Nodes" => {
                    let count_line = lines
                        .next()
                        .ok_or_else(|| MeshSieveError::MeshIoParse("missing node count".into()))?;
                    let node_count = count_line.trim().parse::<usize>().map_err(|_| {
                        MeshSieveError::MeshIoParse(format!("invalid node count: {count_line}"))
                    })?;
                    for _ in 0..node_count {
                        let node_line = lines.next().ok_or_else(|| {
                            MeshSieveError::MeshIoParse("unexpected end of node list".into())
                        })?;
                        let mut parts = node_line.split_whitespace();
                        let id_str = parts
                            .next()
                            .ok_or_else(|| MeshSieveError::MeshIoParse("missing node id".into()))?;
                        let point = Self::parse_point_id(id_str)?;
                        let x = Self::parse_coord(parts.next().ok_or_else(|| {
                            MeshSieveError::MeshIoParse("missing x coordinate".into())
                        })?)?;
                        let y = Self::parse_coord(parts.next().ok_or_else(|| {
                            MeshSieveError::MeshIoParse("missing y coordinate".into())
                        })?)?;
                        let z = Self::parse_coord(parts.next().ok_or_else(|| {
                            MeshSieveError::MeshIoParse("missing z coordinate".into())
                        })?)?;
                        nodes.push((point, [x, y, z]));
                    }
                    let end = lines
                        .next()
                        .ok_or_else(|| MeshSieveError::MeshIoParse("missing EndNodes".into()))?;
                    if end.trim() != "$EndNodes" {
                        return Err(MeshSieveError::MeshIoParse("missing $EndNodes".into()));
                    }
                }
                "$Elements" => {
                    let count_line = lines.next().ok_or_else(|| {
                        MeshSieveError::MeshIoParse("missing element count".into())
                    })?;
                    let elem_count = count_line.trim().parse::<usize>().map_err(|_| {
                        MeshSieveError::MeshIoParse(format!("invalid element count: {count_line}"))
                    })?;
                    for _ in 0..elem_count {
                        let elem_line = lines.next().ok_or_else(|| {
                            MeshSieveError::MeshIoParse("unexpected end of element list".into())
                        })?;
                        let mut parts = elem_line.split_whitespace();
                        let elem_id = Self::parse_point_id(parts.next().ok_or_else(|| {
                            MeshSieveError::MeshIoParse("missing element id".into())
                        })?)?;
                        let elem_type = parts
                            .next()
                            .ok_or_else(|| {
                                MeshSieveError::MeshIoParse("missing element type".into())
                            })?
                            .parse::<u32>()
                            .map_err(|_| {
                                MeshSieveError::MeshIoParse("invalid element type".into())
                            })?;
                        let node_count = Self::element_node_count(elem_type).ok_or_else(|| {
                            MeshSieveError::MeshIoParse(format!(
                                "unsupported element type: {elem_type}"
                            ))
                        })?;
                        let num_tags = parts
                            .next()
                            .ok_or_else(|| {
                                MeshSieveError::MeshIoParse("missing element tag count".into())
                            })?
                            .parse::<usize>()
                            .map_err(|_| {
                                MeshSieveError::MeshIoParse("invalid element tag count".into())
                            })?;
                        for _ in 0..num_tags {
                            parts.next();
                        }
                        let mut conn = Vec::with_capacity(node_count);
                        for _ in 0..node_count {
                            let node_id = parts.next().ok_or_else(|| {
                                MeshSieveError::MeshIoParse("missing element node id".into())
                            })?;
                            conn.push(Self::parse_point_id(node_id)?);
                        }
                        elements.push((elem_id, conn));
                    }
                    let end = lines
                        .next()
                        .ok_or_else(|| MeshSieveError::MeshIoParse("missing EndElements".into()))?;
                    if end.trim() != "$EndElements" {
                        return Err(MeshSieveError::MeshIoParse("missing $EndElements".into()));
                    }
                }
                _ => {
                    // ignore other sections
                }
            }
        }

        let version = version.unwrap_or_else(|| "2.2".to_string());
        if version != "2.2" {
            return Err(MeshSieveError::MeshIoParse(format!(
                "unsupported gmsh version: {version}"
            )));
        }

        let mut sieve = InMemorySieve::default();
        for (node, _) in &nodes {
            sieve.add_point(*node);
        }
        for (cell, conn) in &elements {
            sieve.add_point(*cell);
            for node in conn {
                Sieve::add_arrow(&mut sieve, *cell, *node, ());
            }
        }

        let mut atlas = Atlas::default();
        for (node, _) in &nodes {
            atlas.try_insert(*node, 3)?;
        }
        let mut coords = Coordinates::try_new(3, atlas)?;
        for (node, xyz) in &nodes {
            coords.section_mut().try_set(*node, xyz)?;
        }

        Ok(MeshData {
            sieve,
            coordinates: Some(coords),
            sections: BTreeMap::new(),
        })
    }
}
