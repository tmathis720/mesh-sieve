//! Gmsh `.msh` reader.
//!
//! # Supported format
//! - ASCII `.msh` version **2.2** and block-based Gmsh **4.x**.
//! - Element types: linear and common higher-order variants for lines,
//!   triangles, quads, tets, hexes, prisms, pyramids, and points.
//!
//! # Limitations
//! - Gmsh **4.x binary** files using 32-bit integers, 64-bit sizes, and 64-bit floats are supported.
//! - Gmsh 4.x physical names and entity physical tags are captured as labels.
//! - Higher-order elements are supported, stored as base cell types with
//!   higher-order geometry stored in the coordinates.
//! - Element tags are captured as labels (`gmsh:physical`, `gmsh:entity`,
//!   and `gmsh:tagN` for additional tags).
//! - Coordinates preserve the inferred mesh dimension (1D/2D/3D).
//! - Element IDs are remapped when they collide with node IDs.
//!
//! # Mixed-element meshes
//! The reader supports mixed-element meshes (for example, triangles and quads in
//! the same surface mesh). You can inspect `cell_types` to determine per-element
//! topology, and optionally enable mixed-dimension validation.
//!
//! ```rust
//! # use mesh_sieve::io::gmsh::{GmshReadOptions, GmshReader};
//! # use mesh_sieve::topology::cell_type::CellType;
//! # fn demo() -> Result<(), mesh_sieve::mesh_error::MeshSieveError> {
//! let msh = r#"$MeshFormat
//! 2.2 0 8
//! $EndMeshFormat
//! $Nodes
//! 5
//! 1 0 0 0
//! 2 1 0 0
//! 3 1 1 0
//! 4 0 1 0
//! 5 2 0 0
//! $EndNodes
//! $Elements
//! 2
//! 10 2 0 1 2 3
//! 11 3 0 2 3 4 5
//! $EndElements
//! "#;
//! let reader = GmshReader::default();
//! let mesh = reader.read_with_options(
//!     msh.as_bytes(),
//!     GmshReadOptions {
//!         validate_mixed_dimensions: true,
//!         ..Default::default()
//!     },
//! )?;
//! let cell_types = mesh.cell_types.as_ref().expect("cell types");
//! assert_eq!(cell_types.try_restrict(mesh_sieve::topology::point::PointId::new(10)?)?[0], CellType::Triangle);
//! assert_eq!(cell_types.try_restrict(mesh_sieve::topology::point::PointId::new(11)?)?[0], CellType::Quadrilateral);
//! # Ok(())
//! # }
//! ```

use crate::data::atlas::Atlas;
use crate::data::coordinates::{Coordinates, HighOrderCoordinates};
use crate::data::mixed_section::{MixedSectionStore, ScalarType, TaggedSection};
use crate::data::section::Section;
use crate::data::storage::VecStorage;
use crate::geometry::quality::validate_cell_geometry;
use crate::io::{MeshData, SieveSectionReader, SieveSectionWriter};
use crate::mesh_error::MeshSieveError;
use crate::topology::cell_type::CellType;
use crate::topology::labels::LabelSet;
use crate::topology::point::PointId;
use crate::topology::sieve::{MeshSieve, MutableSieve, Sieve};
use crate::topology::validation::{TopologyValidationOptions, validate_sieve_topology};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::{Read, Write};
use std::str::FromStr;

/// Gmsh `.msh` reader for ASCII v2.2 and ASCII/binary v4.x meshes.
#[derive(Debug, Default, Clone)]
pub struct GmshReader;

#[derive(Debug)]
struct ElementRecord {
    id: PointId,
    conn: Vec<PointId>,
    cell_type: CellType,
    tags: Vec<i32>,
    entity_dim: Option<u8>,
    entity_tag: Option<i32>,
    element_type: u32,
}

/// Optional settings for Gmsh import.
#[derive(Debug, Default, Clone, Copy)]
pub struct GmshReadOptions {
    /// When enabled, validate cell geometry and reject inverted or degenerate elements.
    pub check_geometry: bool,
    /// When enabled, validate topology (cone sizes, duplicate arrows, closure consistency).
    pub validate_topology: bool,
    /// When enabled, reject meshes containing elements with mixed topological dimensions.
    pub validate_mixed_dimensions: bool,
}

impl GmshReader {
    fn parse_format(line: &str) -> Result<(String, u32, usize), MeshSieveError> {
        let mut parts = line.split_whitespace();
        let version = parts
            .next()
            .ok_or_else(|| MeshSieveError::MeshIoParse("missing mesh format version".into()))?
            .to_string();
        let file_type = parts
            .next()
            .ok_or_else(|| MeshSieveError::MeshIoParse("missing mesh format type".into()))?
            .parse::<u32>()
            .map_err(|_| MeshSieveError::MeshIoParse("invalid mesh format type".into()))?;
        let data_size = parts
            .next()
            .ok_or_else(|| MeshSieveError::MeshIoParse("missing mesh format data size".into()))?
            .parse::<usize>()
            .map_err(|_| MeshSieveError::MeshIoParse("invalid mesh format data size".into()))?;
        Ok((version, file_type, data_size))
    }

    fn element_node_count(elem_type: u32) -> Option<usize> {
        match elem_type {
            1 => Some(2),   // line
            8 => Some(3),   // line3
            2 => Some(3),   // triangle
            9 => Some(6),   // triangle6
            20 => Some(9),  // triangle9
            21 => Some(10), // triangle10
            22 => Some(12), // triangle12
            23 => Some(15), // triangle15
            3 => Some(4),   // quad
            10 => Some(9),  // quad9
            16 => Some(8),  // quad8
            4 => Some(4),   // tet
            11 => Some(10), // tet10
            5 => Some(8),   // hex
            12 => Some(27), // hex27
            17 => Some(20), // hex20
            6 => Some(6),   // prism
            13 => Some(18), // prism18
            18 => Some(15), // prism15
            7 => Some(5),   // pyramid
            14 => Some(14), // pyramid14
            19 => Some(13), // pyramid13
            15 => Some(1),  // point
            _ => None,
        }
    }

    fn element_cell_type(elem_type: u32) -> Option<CellType> {
        match elem_type {
            1 => Some(CellType::Segment),
            8 => Some(CellType::Segment),
            2 => Some(CellType::Triangle),
            9 | 20 | 21 | 22 | 23 => Some(CellType::Triangle),
            3 => Some(CellType::Quadrilateral),
            10 | 16 => Some(CellType::Quadrilateral),
            4 => Some(CellType::Tetrahedron),
            11 => Some(CellType::Tetrahedron),
            5 => Some(CellType::Hexahedron),
            12 | 17 => Some(CellType::Hexahedron),
            6 => Some(CellType::Prism),
            13 | 18 => Some(CellType::Prism),
            7 => Some(CellType::Pyramid),
            14 | 19 => Some(CellType::Pyramid),
            15 => Some(CellType::Vertex),
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

    fn parse_string_tag(raw: &str) -> String {
        raw.trim().trim_matches('"').to_string()
    }

    fn parse_scalar_type(tags: &[String]) -> ScalarType {
        tags.iter()
            .find_map(|tag| tag.strip_prefix("mesh_sieve:type="))
            .and_then(ScalarType::parse)
            .unwrap_or(ScalarType::F64)
    }

    fn section_from_entries<T: Clone + Default>(
        entries: Vec<(PointId, Vec<T>)>,
        num_components: usize,
    ) -> Result<Section<T, VecStorage<T>>, MeshSieveError> {
        let mut atlas = Atlas::default();
        for (point, values) in &entries {
            if values.len() != num_components {
                return Err(MeshSieveError::MeshIoParse(format!(
                    "entry length mismatch for {point:?}"
                )));
            }
            atlas.try_insert(*point, num_components)?;
        }
        let mut section = Section::new(atlas);
        for (point, values) in entries {
            section.try_set(point, &values)?;
        }
        Ok(section)
    }

    fn parse_data_block<'a>(
        lines: &mut std::iter::Peekable<std::str::Lines<'a>>,
        block_name: &str,
    ) -> Result<(String, ScalarType, usize, Vec<(PointId, Vec<String>)>), MeshSieveError> {
        let string_count_line = lines.next().ok_or_else(|| {
            MeshSieveError::MeshIoParse(format!("missing {block_name} string tag count"))
        })?;
        let string_count = string_count_line.trim().parse::<usize>().map_err(|_| {
            MeshSieveError::MeshIoParse(format!("invalid {block_name} string tag count"))
        })?;
        let mut string_tags = Vec::with_capacity(string_count);
        for _ in 0..string_count {
            let line = lines.next().ok_or_else(|| {
                MeshSieveError::MeshIoParse(format!("missing {block_name} string tag"))
            })?;
            string_tags.push(Self::parse_string_tag(line));
        }
        let scalar_type = Self::parse_scalar_type(&string_tags);
        let name = string_tags
            .first()
            .cloned()
            .unwrap_or_else(|| format!("gmsh:{block_name}"));

        let real_count_line = lines.next().ok_or_else(|| {
            MeshSieveError::MeshIoParse(format!("missing {block_name} real tag count"))
        })?;
        let real_count = real_count_line.trim().parse::<usize>().map_err(|_| {
            MeshSieveError::MeshIoParse(format!("invalid {block_name} real tag count"))
        })?;
        for _ in 0..real_count {
            lines.next().ok_or_else(|| {
                MeshSieveError::MeshIoParse(format!("missing {block_name} real tag"))
            })?;
        }

        let int_count_line = lines.next().ok_or_else(|| {
            MeshSieveError::MeshIoParse(format!("missing {block_name} integer tag count"))
        })?;
        let int_count = int_count_line.trim().parse::<usize>().map_err(|_| {
            MeshSieveError::MeshIoParse(format!("invalid {block_name} integer tag count"))
        })?;
        let mut int_tags = Vec::with_capacity(int_count);
        for _ in 0..int_count {
            let tag_line = lines.next().ok_or_else(|| {
                MeshSieveError::MeshIoParse(format!("missing {block_name} integer tag"))
            })?;
            let tag = tag_line.trim().parse::<i64>().map_err(|_| {
                MeshSieveError::MeshIoParse(format!("invalid {block_name} integer tag"))
            })?;
            int_tags.push(tag);
        }
        if int_tags.len() < 3 {
            return Err(MeshSieveError::MeshIoParse(format!(
                "{block_name} expects at least 3 integer tags"
            )));
        }
        let num_components = usize::try_from(int_tags[1]).map_err(|_| {
            MeshSieveError::MeshIoParse(format!("invalid {block_name} component count"))
        })?;
        let num_entries = usize::try_from(int_tags[2]).map_err(|_| {
            MeshSieveError::MeshIoParse(format!("invalid {block_name} entry count"))
        })?;
        let mut entries = Vec::with_capacity(num_entries);
        for _ in 0..num_entries {
            let data_line = lines.next().ok_or_else(|| {
                MeshSieveError::MeshIoParse(format!("missing {block_name} entry"))
            })?;
            let mut parts = data_line.split_whitespace();
            let id_str = parts.next().ok_or_else(|| {
                MeshSieveError::MeshIoParse(format!("missing {block_name} entry id"))
            })?;
            let point = Self::parse_point_id(id_str)?;
            let mut values = Vec::with_capacity(num_components);
            for _ in 0..num_components {
                let value_str = parts.next().ok_or_else(|| {
                    MeshSieveError::MeshIoParse(format!("missing {block_name} value"))
                })?;
                values.push(value_str.to_string());
            }
            entries.push((point, values));
        }
        let end = lines
            .next()
            .ok_or_else(|| MeshSieveError::MeshIoParse(format!("missing End{block_name}")))?;
        if end.trim() != format!("$End{block_name}") {
            return Err(MeshSieveError::MeshIoParse(format!(
                "missing $End{block_name}"
            )));
        }

        Ok((name, scalar_type, num_components, entries))
    }

    fn build_tagged_section(
        scalar_type: ScalarType,
        entries: Vec<(PointId, Vec<String>)>,
        num_components: usize,
        block_name: &str,
    ) -> Result<TaggedSection, MeshSieveError> {
        match scalar_type {
            ScalarType::F64 => {
                let parsed = entries
                    .into_iter()
                    .map(|(p, vals)| {
                        let parsed = vals
                            .into_iter()
                            .map(|v| {
                                f64::from_str(&v).map_err(|_| {
                                    MeshSieveError::MeshIoParse(format!(
                                        "invalid {block_name} value: {v}"
                                    ))
                                })
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        Ok((p, parsed))
                    })
                    .collect::<Result<Vec<_>, MeshSieveError>>()?;
                Ok(TaggedSection::F64(Self::section_from_entries(
                    parsed,
                    num_components,
                )?))
            }
            ScalarType::F32 => {
                let parsed = entries
                    .into_iter()
                    .map(|(p, vals)| {
                        let parsed = vals
                            .into_iter()
                            .map(|v| {
                                f32::from_str(&v).map_err(|_| {
                                    MeshSieveError::MeshIoParse(format!(
                                        "invalid {block_name} value: {v}"
                                    ))
                                })
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        Ok((p, parsed))
                    })
                    .collect::<Result<Vec<_>, MeshSieveError>>()?;
                Ok(TaggedSection::F32(Self::section_from_entries(
                    parsed,
                    num_components,
                )?))
            }
            ScalarType::I32 => {
                let parsed = entries
                    .into_iter()
                    .map(|(p, vals)| {
                        let parsed = vals
                            .into_iter()
                            .map(|v| {
                                i32::from_str(&v).map_err(|_| {
                                    MeshSieveError::MeshIoParse(format!(
                                        "invalid {block_name} value: {v}"
                                    ))
                                })
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        Ok((p, parsed))
                    })
                    .collect::<Result<Vec<_>, MeshSieveError>>()?;
                Ok(TaggedSection::I32(Self::section_from_entries(
                    parsed,
                    num_components,
                )?))
            }
            ScalarType::I64 => {
                let parsed = entries
                    .into_iter()
                    .map(|(p, vals)| {
                        let parsed = vals
                            .into_iter()
                            .map(|v| {
                                i64::from_str(&v).map_err(|_| {
                                    MeshSieveError::MeshIoParse(format!(
                                        "invalid {block_name} value: {v}"
                                    ))
                                })
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        Ok((p, parsed))
                    })
                    .collect::<Result<Vec<_>, MeshSieveError>>()?;
                Ok(TaggedSection::I64(Self::section_from_entries(
                    parsed,
                    num_components,
                )?))
            }
            ScalarType::U32 => {
                let parsed = entries
                    .into_iter()
                    .map(|(p, vals)| {
                        let parsed = vals
                            .into_iter()
                            .map(|v| {
                                u32::from_str(&v).map_err(|_| {
                                    MeshSieveError::MeshIoParse(format!(
                                        "invalid {block_name} value: {v}"
                                    ))
                                })
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        Ok((p, parsed))
                    })
                    .collect::<Result<Vec<_>, MeshSieveError>>()?;
                Ok(TaggedSection::U32(Self::section_from_entries(
                    parsed,
                    num_components,
                )?))
            }
            ScalarType::U64 => {
                let parsed = entries
                    .into_iter()
                    .map(|(p, vals)| {
                        let parsed = vals
                            .into_iter()
                            .map(|v| {
                                u64::from_str(&v).map_err(|_| {
                                    MeshSieveError::MeshIoParse(format!(
                                        "invalid {block_name} value: {v}"
                                    ))
                                })
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        Ok((p, parsed))
                    })
                    .collect::<Result<Vec<_>, MeshSieveError>>()?;
                Ok(TaggedSection::U64(Self::section_from_entries(
                    parsed,
                    num_components,
                )?))
            }
        }
    }

    fn is_v4_version(version: &str) -> bool {
        version.starts_with("4.")
    }

    fn is_binary_v4(bytes: &[u8]) -> Result<bool, MeshSieveError> {
        let text =
            std::str::from_utf8(bytes.get(..bytes.len().min(256)).unwrap_or(bytes)).unwrap_or("");
        let mut lines = text.lines();
        while let Some(line) = lines.next() {
            if line.trim() == "$MeshFormat" {
                let format = lines
                    .next()
                    .ok_or_else(|| MeshSieveError::MeshIoParse("missing MeshFormat line".into()))?;
                let (version, file_type, _) = Self::parse_format(format)?;
                return Ok(Self::is_v4_version(&version) && file_type == 1);
            }
        }
        Ok(false)
    }

    fn parse_u64_token(raw: &str, what: &str) -> Result<u64, MeshSieveError> {
        raw.parse::<u64>()
            .map_err(|_| MeshSieveError::MeshIoParse(format!("invalid {what}: {raw}")))
    }

    fn parse_i32_token(raw: &str, what: &str) -> Result<i32, MeshSieveError> {
        raw.parse::<i32>()
            .map_err(|_| MeshSieveError::MeshIoParse(format!("invalid {what}: {raw}")))
    }

    fn skip_physical_names<'a>(
        lines: &mut std::iter::Peekable<std::str::Lines<'a>>,
    ) -> Result<(), MeshSieveError> {
        let count = lines
            .next()
            .ok_or_else(|| MeshSieveError::MeshIoParse("missing PhysicalNames count".into()))?
            .trim()
            .parse::<usize>()
            .map_err(|_| MeshSieveError::MeshIoParse("invalid PhysicalNames count".into()))?;
        for _ in 0..count {
            lines.next().ok_or_else(|| {
                MeshSieveError::MeshIoParse("unexpected end of PhysicalNames".into())
            })?;
        }
        let end = lines
            .next()
            .ok_or_else(|| MeshSieveError::MeshIoParse("missing EndPhysicalNames".into()))?;
        if end.trim() != "$EndPhysicalNames" {
            return Err(MeshSieveError::MeshIoParse(
                "missing $EndPhysicalNames".into(),
            ));
        }
        Ok(())
    }

    fn parse_entities_ascii<'a>(
        lines: &mut std::iter::Peekable<std::str::Lines<'a>>,
    ) -> Result<HashMap<(u8, i32), Vec<i32>>, MeshSieveError> {
        let header = lines
            .next()
            .ok_or_else(|| MeshSieveError::MeshIoParse("missing Entities header".into()))?;
        let counts = header
            .split_whitespace()
            .map(|v| v.parse::<usize>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| MeshSieveError::MeshIoParse("invalid Entities header".into()))?;
        if counts.len() != 4 {
            return Err(MeshSieveError::MeshIoParse(
                "invalid Entities header".into(),
            ));
        }
        let mut out = HashMap::new();
        for (dim, count) in counts.into_iter().enumerate() {
            for _ in 0..count {
                let line = lines.next().ok_or_else(|| {
                    MeshSieveError::MeshIoParse("unexpected end of Entities".into())
                })?;
                let parts: Vec<_> = line.split_whitespace().collect();
                let base = if dim == 0 { 4 } else { 7 };
                if parts.len() <= base {
                    return Err(MeshSieveError::MeshIoParse("invalid Entities entry".into()));
                }
                let tag = Self::parse_i32_token(parts[0], "entity tag")?;
                let nphys = parts[base].parse::<usize>().map_err(|_| {
                    MeshSieveError::MeshIoParse("invalid entity physical count".into())
                })?;
                if parts.len() < base + 1 + nphys {
                    return Err(MeshSieveError::MeshIoParse(
                        "truncated entity physical tags".into(),
                    ));
                }
                let physical = parts[base + 1..base + 1 + nphys]
                    .iter()
                    .map(|v| Self::parse_i32_token(v, "entity physical tag"))
                    .collect::<Result<Vec<_>, _>>()?;
                if !physical.is_empty() {
                    out.insert((dim as u8, tag), physical);
                }
            }
        }
        let end = lines
            .next()
            .ok_or_else(|| MeshSieveError::MeshIoParse("missing EndEntities".into()))?;
        if end.trim() != "$EndEntities" {
            return Err(MeshSieveError::MeshIoParse("missing $EndEntities".into()));
        }
        Ok(out)
    }

    fn parse_v4_nodes_ascii<'a>(
        header_line: &str,
        lines: &mut std::iter::Peekable<std::str::Lines<'a>>,
        nodes: &mut Vec<(PointId, [f64; 3])>,
    ) -> Result<(), MeshSieveError> {
        let header: Vec<_> = header_line.split_whitespace().collect();
        if header.len() < 2 {
            return Err(MeshSieveError::MeshIoParse(
                "invalid v4 Nodes header".into(),
            ));
        }
        let num_blocks = header[0]
            .parse::<usize>()
            .map_err(|_| MeshSieveError::MeshIoParse("invalid node block count".into()))?;
        for _ in 0..num_blocks {
            let block = lines.next().ok_or_else(|| {
                MeshSieveError::MeshIoParse("unexpected end of node blocks".into())
            })?;
            let parts: Vec<_> = block.split_whitespace().collect();
            if parts.len() != 4 {
                return Err(MeshSieveError::MeshIoParse(
                    "invalid node block header".into(),
                ));
            }
            let parametric = parts[2] != "0";
            let count = parts[3]
                .parse::<usize>()
                .map_err(|_| MeshSieveError::MeshIoParse("invalid node block size".into()))?;
            let mut ids = Vec::with_capacity(count);
            while ids.len() < count {
                let line = lines.next().ok_or_else(|| {
                    MeshSieveError::MeshIoParse("unexpected end of node tags".into())
                })?;
                for raw in line.split_whitespace() {
                    ids.push(PointId::new(Self::parse_u64_token(raw, "node tag")?)?);
                    if ids.len() == count {
                        break;
                    }
                }
            }
            for id in ids {
                let line = lines.next().ok_or_else(|| {
                    MeshSieveError::MeshIoParse("unexpected end of node coordinates".into())
                })?;
                let vals: Vec<_> = line.split_whitespace().collect();
                if vals.len() < 3 {
                    return Err(MeshSieveError::MeshIoParse(
                        "invalid node coordinate".into(),
                    ));
                }
                nodes.push((
                    id,
                    [
                        Self::parse_coord(vals[0])?,
                        Self::parse_coord(vals[1])?,
                        Self::parse_coord(vals[2])?,
                    ],
                ));
                if parametric {
                    // Parametric coordinates are intentionally skipped; geometric coordinates are canonical.
                }
            }
        }
        let end = lines
            .next()
            .ok_or_else(|| MeshSieveError::MeshIoParse("missing EndNodes".into()))?;
        if end.trim() != "$EndNodes" {
            return Err(MeshSieveError::MeshIoParse("missing $EndNodes".into()));
        }
        Ok(())
    }

    fn parse_v4_elements_ascii<'a>(
        header_line: &str,
        lines: &mut std::iter::Peekable<std::str::Lines<'a>>,
        entity_physical: &HashMap<(u8, i32), Vec<i32>>,
        elements: &mut Vec<ElementRecord>,
    ) -> Result<(), MeshSieveError> {
        let header: Vec<_> = header_line.split_whitespace().collect();
        if header.len() < 2 {
            return Err(MeshSieveError::MeshIoParse(
                "invalid v4 Elements header".into(),
            ));
        }
        let num_blocks = header[0]
            .parse::<usize>()
            .map_err(|_| MeshSieveError::MeshIoParse("invalid element block count".into()))?;
        for _ in 0..num_blocks {
            let block = lines.next().ok_or_else(|| {
                MeshSieveError::MeshIoParse("unexpected end of element blocks".into())
            })?;
            let parts: Vec<_> = block.split_whitespace().collect();
            if parts.len() != 4 {
                return Err(MeshSieveError::MeshIoParse(
                    "invalid element block header".into(),
                ));
            }
            let entity_dim = parts[0].parse::<u8>().map_err(|_| {
                MeshSieveError::MeshIoParse("invalid element entity dimension".into())
            })?;
            let entity_tag = Self::parse_i32_token(parts[1], "element entity tag")?;
            let elem_type = parts[2]
                .parse::<u32>()
                .map_err(|_| MeshSieveError::MeshIoParse("invalid element type".into()))?;
            let count = parts[3]
                .parse::<usize>()
                .map_err(|_| MeshSieveError::MeshIoParse("invalid element block size".into()))?;
            let cell_type = Self::element_cell_type(elem_type).ok_or_else(|| {
                MeshSieveError::MeshIoParse(format!("unsupported element type: {elem_type}"))
            })?;
            let expected = Self::element_node_count(elem_type).ok_or_else(|| {
                MeshSieveError::MeshIoParse(format!(
                    "unknown node count for element type: {elem_type}"
                ))
            })?;
            for _ in 0..count {
                let line = lines.next().ok_or_else(|| {
                    MeshSieveError::MeshIoParse("unexpected end of elements".into())
                })?;
                let parts: Vec<_> = line.split_whitespace().collect();
                if parts.len() != expected + 1 {
                    return Err(MeshSieveError::MeshIoParse(
                        "invalid v4 element entry".into(),
                    ));
                }
                let elem_id = PointId::new(Self::parse_u64_token(parts[0], "element tag")?)?;
                let conn = parts[1..]
                    .iter()
                    .map(|raw| PointId::new(Self::parse_u64_token(raw, "element node tag")?))
                    .collect::<Result<Vec<_>, _>>()?;
                let physical = entity_physical
                    .get(&(entity_dim, entity_tag))
                    .cloned()
                    .unwrap_or_default();
                let mut tags = vec![physical.first().copied().unwrap_or(0), entity_tag];
                tags.extend(physical.into_iter().skip(1));
                elements.push(ElementRecord {
                    id: elem_id,
                    conn,
                    cell_type,
                    tags,
                    entity_dim: Some(entity_dim),
                    entity_tag: Some(entity_tag),
                    element_type: elem_type,
                });
            }
        }
        let end = lines
            .next()
            .ok_or_else(|| MeshSieveError::MeshIoParse("missing EndElements".into()))?;
        if end.trim() != "$EndElements" {
            return Err(MeshSieveError::MeshIoParse("missing $EndElements".into()));
        }
        Ok(())
    }

    fn parse_v4_binary(
        bytes: &[u8],
    ) -> Result<
        (
            Vec<(PointId, [f64; 3])>,
            Vec<ElementRecord>,
            MixedSectionStore,
            HashSet<String>,
        ),
        MeshSieveError,
    > {
        let mut cursor = BinaryCursor::new(bytes);
        let mut nodes = Vec::new();
        let mut elements = Vec::new();
        let mut entity_physical: HashMap<(u8, i32), Vec<i32>> = HashMap::new();

        while let Some(line) = cursor.next_line_opt()? {
            match line.trim() {
                "$MeshFormat" => {
                    let format = cursor.next_line()?;
                    let (version, file_type, data_size) = Self::parse_format(&format)?;
                    if !Self::is_v4_version(&version) || file_type != 1 || data_size != 8 {
                        return Err(MeshSieveError::MeshIoParse(
                            "only Gmsh v4 binary files with 8-byte data are supported".into(),
                        ));
                    }
                    let one = cursor.read_i32()?;
                    if one != 1 {
                        return Err(MeshSieveError::MeshIoParse(
                            "big-endian Gmsh binary files are not supported".into(),
                        ));
                    }
                    cursor.consume_line_end();
                    cursor.expect_line("$EndMeshFormat")?;
                }
                "$Entities" => {
                    entity_physical = Self::parse_entities_binary(&mut cursor)?;
                }
                "$PhysicalNames" => {
                    cursor.skip_ascii_section("$EndPhysicalNames")?;
                }
                "$Nodes" => {
                    let num_blocks = cursor.read_u64()? as usize;
                    let _num_nodes = cursor.read_u64()?;
                    let _min_tag = cursor.read_u64()?;
                    let _max_tag = cursor.read_u64()?;
                    for _ in 0..num_blocks {
                        let entity_dim = cursor.read_i32()?;
                        let _entity_tag = cursor.read_i32()?;
                        let parametric = cursor.read_i32()? != 0;
                        let count = cursor.read_u64()? as usize;
                        let mut ids = Vec::with_capacity(count);
                        for _ in 0..count {
                            ids.push(PointId::new(cursor.read_u64()?)?);
                        }
                        for id in ids {
                            let x = cursor.read_f64()?;
                            let y = cursor.read_f64()?;
                            let z = cursor.read_f64()?;
                            if parametric {
                                for _ in 0..entity_dim.max(0) {
                                    let _ = cursor.read_f64()?;
                                }
                            }
                            nodes.push((id, [x, y, z]));
                        }
                    }
                    cursor.consume_line_end();
                    cursor.expect_line("$EndNodes")?;
                }
                "$Elements" => {
                    let num_blocks = cursor.read_u64()? as usize;
                    let _num_elements = cursor.read_u64()?;
                    let _min_tag = cursor.read_u64()?;
                    let _max_tag = cursor.read_u64()?;
                    for _ in 0..num_blocks {
                        let entity_dim = cursor.read_i32()? as u8;
                        let entity_tag = cursor.read_i32()?;
                        let elem_type = cursor.read_i32()? as u32;
                        let count = cursor.read_u64()? as usize;
                        let cell_type = Self::element_cell_type(elem_type).ok_or_else(|| {
                            MeshSieveError::MeshIoParse(format!(
                                "unsupported element type: {elem_type}"
                            ))
                        })?;
                        let expected = Self::element_node_count(elem_type).ok_or_else(|| {
                            MeshSieveError::MeshIoParse(format!(
                                "unknown node count for element type: {elem_type}"
                            ))
                        })?;
                        for _ in 0..count {
                            let elem_id = PointId::new(cursor.read_u64()?)?;
                            let mut conn = Vec::with_capacity(expected);
                            for _ in 0..expected {
                                conn.push(PointId::new(cursor.read_u64()?)?);
                            }
                            let physical = entity_physical
                                .get(&(entity_dim, entity_tag))
                                .cloned()
                                .unwrap_or_default();
                            let mut tags = vec![physical.first().copied().unwrap_or(0), entity_tag];
                            tags.extend(physical.into_iter().skip(1));
                            elements.push(ElementRecord {
                                id: elem_id,
                                conn,
                                cell_type,
                                tags,
                                entity_dim: Some(entity_dim),
                                entity_tag: Some(entity_tag),
                                element_type: elem_type,
                            });
                        }
                    }
                    cursor.consume_line_end();
                    cursor.expect_line("$EndElements")?;
                }
                _ => {}
            }
        }
        Ok((
            nodes,
            elements,
            MixedSectionStore::default(),
            HashSet::new(),
        ))
    }

    fn parse_entities_binary(
        cursor: &mut BinaryCursor<'_>,
    ) -> Result<HashMap<(u8, i32), Vec<i32>>, MeshSieveError> {
        let counts = [
            cursor.read_u64()? as usize,
            cursor.read_u64()? as usize,
            cursor.read_u64()? as usize,
            cursor.read_u64()? as usize,
        ];
        let mut out = HashMap::new();
        for (dim, count) in counts.into_iter().enumerate() {
            for _ in 0..count {
                let tag = cursor.read_i32()?;
                let bounds = if dim == 0 { 3 } else { 6 };
                for _ in 0..bounds {
                    let _ = cursor.read_f64()?;
                }
                let nphys = cursor.read_u64()? as usize;
                let mut physical = Vec::with_capacity(nphys);
                for _ in 0..nphys {
                    physical.push(cursor.read_i32()?);
                }
                let nbound = cursor.read_u64()? as usize;
                for _ in 0..nbound {
                    let _ = cursor.read_i32()?;
                }
                if !physical.is_empty() {
                    out.insert((dim as u8, tag), physical);
                }
            }
        }
        cursor.consume_line_end();
        cursor.expect_line("$EndEntities")?;
        Ok(out)
    }

    /// Read mesh data with explicit import options.
    pub fn read_with_options<R: Read>(
        &self,
        reader: R,
        options: GmshReadOptions,
    ) -> Result<MeshData<MeshSieve, f64, VecStorage<f64>, VecStorage<CellType>>, MeshSieveError>
    {
        let mut reader = reader;
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes)?;
        if Self::is_binary_v4(&bytes)? {
            let (nodes, elements, mixed_sections, element_data_names) =
                Self::parse_v4_binary(&bytes)?;
            return Self::build_mesh(nodes, elements, mixed_sections, element_data_names, options);
        }
        let contents = String::from_utf8(bytes).map_err(|err| {
            MeshSieveError::MeshIoParse(format!("Gmsh ASCII is not UTF-8: {err}"))
        })?;
        let mut lines = contents.lines().peekable();

        let mut version: Option<String> = None;
        let mut nodes: Vec<(PointId, [f64; 3])> = Vec::new();
        let mut elements: Vec<ElementRecord> = Vec::new();
        let mut mixed_sections = MixedSectionStore::default();
        let mut element_data_names: HashSet<String> = HashSet::new();
        let mut entity_physical: HashMap<(u8, i32), Vec<i32>> = HashMap::new();

        while let Some(line) = lines.next() {
            match line.trim() {
                "$MeshFormat" => {
                    let format_line = lines
                        .next()
                        .ok_or_else(|| MeshSieveError::MeshIoParse("missing MeshFormat".into()))?;
                    let (parsed_version, file_type, _) = Self::parse_format(format_line)?;
                    if file_type != 0 {
                        return Err(MeshSieveError::MeshIoParse(
                            "binary .msh files require Gmsh 4.x and must be read as bytes".into(),
                        ));
                    }
                    version = Some(parsed_version);
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
                    if version.as_deref().is_some_and(Self::is_v4_version) {
                        Self::parse_v4_nodes_ascii(count_line, &mut lines, &mut nodes)?;
                        continue;
                    }
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
                    if version.as_deref().is_some_and(Self::is_v4_version) {
                        Self::parse_v4_elements_ascii(
                            count_line,
                            &mut lines,
                            &entity_physical,
                            &mut elements,
                        )?;
                        continue;
                    }
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
                        let cell_type = Self::element_cell_type(elem_type).ok_or_else(|| {
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
                        let mut tags = Vec::with_capacity(num_tags);
                        for _ in 0..num_tags {
                            let tag = parts.next().ok_or_else(|| {
                                MeshSieveError::MeshIoParse("missing element tag".into())
                            })?;
                            let tag = tag.parse::<i32>().map_err(|_| {
                                MeshSieveError::MeshIoParse("invalid element tag".into())
                            })?;
                            tags.push(tag);
                        }
                        let mut conn = Vec::new();
                        for node_id in parts {
                            conn.push(Self::parse_point_id(node_id)?);
                        }
                        let min_nodes = Self::element_min_nodes(cell_type);
                        if conn.len() < min_nodes {
                            return Err(MeshSieveError::MeshIoParse(format!(
                                "element {elem_id:?} has {found} nodes, expected at least {min_nodes}",
                                found = conn.len()
                            )));
                        }
                        if let Some(expected) = Self::element_node_count(elem_type)
                            && conn.len() != expected {
                                return Err(MeshSieveError::MeshIoParse(format!(
                                    "element {elem_id:?} expected {expected} nodes, got {found}",
                                    found = conn.len()
                                )));
                            }
                        elements.push(ElementRecord {
                            id: elem_id,
                            conn,
                            cell_type,
                            tags,
                            entity_dim: None,
                            entity_tag: None,
                            element_type: elem_type,
                        });
                    }
                    let end = lines
                        .next()
                        .ok_or_else(|| MeshSieveError::MeshIoParse("missing EndElements".into()))?;
                    if end.trim() != "$EndElements" {
                        return Err(MeshSieveError::MeshIoParse("missing $EndElements".into()));
                    }
                }
                "$NodeData" => {
                    let (name, scalar_type, num_components, entries) =
                        Self::parse_data_block(&mut lines, "NodeData")?;
                    let tagged = Self::build_tagged_section(
                        scalar_type,
                        entries,
                        num_components,
                        "NodeData",
                    )?;
                    mixed_sections.insert_tagged(name, tagged);
                }
                "$ElementData" => {
                    let (name, scalar_type, num_components, entries) =
                        Self::parse_data_block(&mut lines, "ElementData")?;
                    let tagged = Self::build_tagged_section(
                        scalar_type,
                        entries,
                        num_components,
                        "ElementData",
                    )?;
                    element_data_names.insert(name.clone());
                    mixed_sections.insert_tagged(name, tagged);
                }
                "$PhysicalNames" => {
                    Self::skip_physical_names(&mut lines)?;
                }
                "$Entities" => {
                    entity_physical = Self::parse_entities_ascii(&mut lines)?;
                }
                _ => {
                    // ignore other sections
                }
            }
        }

        let version = version.unwrap_or_else(|| "2.2".to_string());
        if version != "2.2" && !Self::is_v4_version(&version) {
            return Err(MeshSieveError::MeshIoParse(format!(
                "unsupported gmsh version: {version}"
            )));
        }

        Self::build_mesh(nodes, elements, mixed_sections, element_data_names, options)
    }

    fn build_mesh(
        nodes: Vec<(PointId, [f64; 3])>,
        mut elements: Vec<ElementRecord>,
        mut mixed_sections: MixedSectionStore,
        element_data_names: HashSet<String>,
        options: GmshReadOptions,
    ) -> Result<MeshData<MeshSieve, f64, VecStorage<f64>, VecStorage<CellType>>, MeshSieveError>
    {
        let mesh_dimension = Self::mesh_dimension(&elements, &nodes);
        let coord_dimension = mesh_dimension.max(1).min(3);

        if options.validate_mixed_dimensions {
            Self::validate_element_dimensions(&elements)?;
        }

        Self::remap_elements_if_needed(&mut elements, &mut mixed_sections, &element_data_names)?;

        let node_lookup: HashMap<PointId, [f64; 3]> = nodes.iter().cloned().collect();
        let mut vertex_nodes = HashSet::new();
        for element in &elements {
            for node in &element.conn {
                vertex_nodes.insert(*node);
            }
        }
        if elements.is_empty() {
            for (node, _) in &nodes {
                vertex_nodes.insert(*node);
            }
        }

        let mut sieve = MeshSieve::default();
        let mut seen_arrows = if options.validate_topology {
            Some(HashSet::new())
        } else {
            None
        };
        for node in &vertex_nodes {
            MutableSieve::add_point(&mut sieve, *node);
        }
        for element in &elements {
            MutableSieve::add_point(&mut sieve, element.id);
            for node in &element.conn {
                if let Some(ref mut seen) = seen_arrows
                    && !seen.insert((element.id, *node)) {
                        return Err(MeshSieveError::DuplicateArrow {
                            src: element.id,
                            dst: *node,
                        });
                    }
                Sieve::add_arrow(&mut sieve, element.id, *node, ());
            }
        }

        let mut atlas = Atlas::default();
        for node in &vertex_nodes {
            atlas.try_insert(*node, coord_dimension)?;
        }
        let mut coords = Coordinates::try_new(mesh_dimension, coord_dimension, atlas)?;
        for node in &vertex_nodes {
            if let Some(xyz) = node_lookup.get(node) {
                coords
                    .section_mut()
                    .try_set(*node, &xyz[..coord_dimension])?;
            }
        }

        let mut high_order_entries: Vec<(PointId, Vec<f64>)> = Vec::new();
        for element in &elements {
            let min_nodes = Self::element_min_nodes(element.cell_type);
            let extra_nodes = element.conn.iter().skip(min_nodes);
            let mut data = Vec::new();
            for node in extra_nodes {
                let xyz = node_lookup.get(node).ok_or_else(|| {
                    MeshSieveError::MeshIoParse(format!("missing node coordinates for {node:?}"))
                })?;
                data.extend_from_slice(&xyz[..coord_dimension]);
            }
            if !data.is_empty() {
                high_order_entries.push((element.id, data));
            }
        }
        if !high_order_entries.is_empty() {
            let mut ho_atlas = Atlas::default();
            for (cell, data) in &high_order_entries {
                ho_atlas.try_insert(*cell, data.len())?;
            }
            let mut ho_section = Section::<f64, VecStorage<f64>>::new(ho_atlas);
            for (cell, data) in high_order_entries {
                ho_section.try_set(cell, &data)?;
            }
            let high_order = HighOrderCoordinates::from_section(coord_dimension, ho_section)?;
            coords.set_high_order(high_order)?;
        }

        if options.check_geometry {
            for element in &elements {
                let min_nodes = Self::element_min_nodes(element.cell_type);
                if let Err(err) =
                    validate_cell_geometry(element.cell_type, &element.conn[..min_nodes], &coords)
                {
                    return Err(MeshSieveError::InvalidGeometry(format!(
                        "element {id:?}: {err}",
                        id = element.id
                    )));
                }
            }
        }

        let mut cell_atlas = Atlas::default();
        for element in &elements {
            cell_atlas.try_insert(element.id, 1)?;
        }
        let mut cell_types = Section::<CellType, VecStorage<CellType>>::new(cell_atlas);
        for element in &elements {
            cell_types.try_set(element.id, &[element.cell_type])?;
        }

        if options.validate_topology {
            validate_sieve_topology(&sieve, &cell_types, TopologyValidationOptions::all())?;
        }

        let mut labels = LabelSet::new();
        let mut has_labels = false;
        for element in &elements {
            labels.set_label(element.id, "gmsh:element_type", element.element_type as i32);
            has_labels = true;
            if let Some(dim) = element.entity_dim {
                labels.set_label(element.id, "gmsh:entity_dim", i32::from(dim));
                has_labels = true;
            }
            if let Some(tag) = element.entity_tag {
                labels.set_label(element.id, "gmsh:entity", tag);
                has_labels = true;
            }
            if let Some(tag) = element.tags.first() {
                labels.set_label(element.id, "gmsh:physical", *tag);
                has_labels = true;
            }
            if let Some(tag) = element.tags.get(1) {
                labels.set_label(element.id, "gmsh:entity", *tag);
                has_labels = true;
            }
            for (index, tag) in element.tags.iter().enumerate().skip(2) {
                let label_name = format!("gmsh:tag{}", index + 1);
                labels.set_label(element.id, &label_name, *tag);
                has_labels = true;
            }
        }

        Ok(MeshData {
            sieve,
            coordinates: Some(coords),
            sections: BTreeMap::new(),
            mixed_sections,
            labels: has_labels.then_some(labels),
            cell_types: (!elements.is_empty()).then_some(cell_types),
            discretization: None,
        })
    }

    fn element_min_nodes(cell_type: CellType) -> usize {
        match cell_type {
            CellType::Vertex => 1,
            CellType::Segment => 2,
            CellType::Triangle => 3,
            CellType::Quadrilateral => 4,
            CellType::Tetrahedron => 4,
            CellType::Hexahedron => 8,
            CellType::Prism => 6,
            CellType::Pyramid => 5,
            CellType::Polygon(n) => usize::from(n.max(1)),
            CellType::Simplex(d) => match d {
                0 => 1,
                1 => 2,
                2 => 3,
                3 => 4,
                _ => 4,
            },
            CellType::Polyhedron => 4,
        }
    }

    fn remap_elements_if_needed(
        elements: &mut [ElementRecord],
        mixed_sections: &mut MixedSectionStore,
        element_data_names: &HashSet<String>,
    ) -> Result<(), MeshSieveError> {
        if elements.is_empty() {
            return Ok(());
        }

        let mut vertex_nodes = HashSet::new();
        for element in elements.iter() {
            for node in &element.conn {
                vertex_nodes.insert(*node);
            }
        }
        if vertex_nodes.is_empty() {
            return Ok(());
        }

        let needs_remap = elements
            .iter()
            .any(|element| vertex_nodes.contains(&element.id));
        if !needs_remap {
            return Ok(());
        }

        let max_node_id = vertex_nodes
            .iter()
            .map(|point| point.get())
            .max()
            .unwrap_or(0);
        let mut next_id = max_node_id
            .checked_add(1)
            .ok_or(MeshSieveError::InvalidPointId)?;
        let mut id_map = HashMap::with_capacity(elements.len());
        for element in elements.iter_mut() {
            let old_id = element.id;
            let new_id = PointId::new(next_id)?;
            next_id = next_id
                .checked_add(1)
                .ok_or(MeshSieveError::InvalidPointId)?;
            id_map.insert(old_id, new_id);
            element.id = new_id;
        }

        if !element_data_names.is_empty() {
            for (name, section) in mixed_sections.iter_mut() {
                if element_data_names.contains(name) {
                    Self::remap_tagged_section(name, section, &id_map)?;
                }
            }
        }

        Ok(())
    }

    fn remap_tagged_section(
        name: &str,
        section: &mut TaggedSection,
        id_map: &HashMap<PointId, PointId>,
    ) -> Result<(), MeshSieveError> {
        match section {
            TaggedSection::F64(sec) => {
                *sec = Self::remap_section(name, sec, id_map)?;
            }
            TaggedSection::F32(sec) => {
                *sec = Self::remap_section(name, sec, id_map)?;
            }
            TaggedSection::I32(sec) => {
                *sec = Self::remap_section(name, sec, id_map)?;
            }
            TaggedSection::I64(sec) => {
                *sec = Self::remap_section(name, sec, id_map)?;
            }
            TaggedSection::U32(sec) => {
                *sec = Self::remap_section(name, sec, id_map)?;
            }
            TaggedSection::U64(sec) => {
                *sec = Self::remap_section(name, sec, id_map)?;
            }
        }
        Ok(())
    }

    fn remap_section<T: Clone + Default>(
        name: &str,
        section: &Section<T, VecStorage<T>>,
        id_map: &HashMap<PointId, PointId>,
    ) -> Result<Section<T, VecStorage<T>>, MeshSieveError> {
        let mut entries = Vec::new();
        for (point, slice) in section.iter() {
            let remapped = id_map.get(&point).copied().ok_or_else(|| {
                MeshSieveError::MeshIoParse(format!(
                    "{name} references unknown element id {point:?}"
                ))
            })?;
            entries.push((remapped, slice.to_vec()));
        }

        let mut atlas = Atlas::default();
        for (point, values) in &entries {
            atlas.try_insert(*point, values.len())?;
        }
        let mut remapped = Section::<T, VecStorage<T>>::new(atlas);
        for (point, values) in entries {
            remapped.try_set(point, &values)?;
        }
        Ok(remapped)
    }

    fn mesh_dimension(elements: &[ElementRecord], nodes: &[(PointId, [f64; 3])]) -> usize {
        let element_dim = elements
            .iter()
            .map(|element| element.cell_type.dimension())
            .max()
            .unwrap_or(0) as usize;
        if element_dim > 0 {
            return element_dim.min(3);
        }
        let mut has_y = false;
        let mut has_z = false;
        for (_, [_, y, z]) in nodes {
            if *y != 0.0 {
                has_y = true;
            }
            if *z != 0.0 {
                has_z = true;
            }
        }
        if has_z {
            3
        } else if has_y {
            2
        } else {
            1
        }
    }

    fn validate_element_dimensions(elements: &[ElementRecord]) -> Result<(), MeshSieveError> {
        let mut dimension: Option<u8> = None;
        let mut seen = BTreeMap::<u8, HashSet<CellType>>::new();
        for element in elements {
            let dim = element.cell_type.dimension();
            seen.entry(dim).or_default().insert(element.cell_type);
            if let Some(existing) = dimension {
                if existing != dim {
                    let summary = seen
                        .iter()
                        .map(|(d, types)| {
                            let mut list: Vec<_> = types.iter().map(|t| format!("{t:?}")).collect();
                            list.sort();
                            format!("{d}D: [{}]", list.join(", "))
                        })
                        .collect::<Vec<_>>()
                        .join("; ");
                    return Err(MeshSieveError::InvalidGeometry(format!(
                        "mixed element dimensions detected ({summary})"
                    )));
                }
            } else {
                dimension = Some(dim);
            }
        }
        Ok(())
    }
}

struct BinaryCursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> BinaryCursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn next_line_opt(&mut self) -> Result<Option<String>, MeshSieveError> {
        if self.pos >= self.bytes.len() {
            return Ok(None);
        }
        Ok(Some(self.next_line()?))
    }

    fn next_line(&mut self) -> Result<String, MeshSieveError> {
        if self.pos >= self.bytes.len() {
            return Err(MeshSieveError::MeshIoParse("unexpected end of file".into()));
        }
        let start = self.pos;
        while self.pos < self.bytes.len() && self.bytes[self.pos] != b'\n' {
            self.pos += 1;
        }
        let end = self.pos;
        if self.pos < self.bytes.len() && self.bytes[self.pos] == b'\n' {
            self.pos += 1;
        }
        let line = &self.bytes[start..end];
        let line = if line.ends_with(b"\r") {
            &line[..line.len() - 1]
        } else {
            line
        };
        String::from_utf8(line.to_vec())
            .map_err(|err| MeshSieveError::MeshIoParse(format!("invalid ASCII line: {err}")))
    }

    fn expect_line(&mut self, expected: &str) -> Result<(), MeshSieveError> {
        let line = self.next_line()?;
        if line.trim() != expected {
            return Err(MeshSieveError::MeshIoParse(format!(
                "expected {expected}, found {line}"
            )));
        }
        Ok(())
    }

    fn consume_line_end(&mut self) {
        while self.pos < self.bytes.len()
            && (self.bytes[self.pos] == b'\n' || self.bytes[self.pos] == b'\r')
        {
            self.pos += 1;
        }
    }

    fn read_exact<const N: usize>(&mut self) -> Result<[u8; N], MeshSieveError> {
        if self.pos + N > self.bytes.len() {
            return Err(MeshSieveError::MeshIoParse(
                "unexpected end of binary data".into(),
            ));
        }
        let mut out = [0u8; N];
        out.copy_from_slice(&self.bytes[self.pos..self.pos + N]);
        self.pos += N;
        Ok(out)
    }

    fn read_i32(&mut self) -> Result<i32, MeshSieveError> {
        Ok(i32::from_le_bytes(self.read_exact()?))
    }

    fn read_u64(&mut self) -> Result<u64, MeshSieveError> {
        Ok(u64::from_le_bytes(self.read_exact()?))
    }

    fn read_f64(&mut self) -> Result<f64, MeshSieveError> {
        Ok(f64::from_le_bytes(self.read_exact()?))
    }

    fn skip_ascii_section(&mut self, end: &str) -> Result<(), MeshSieveError> {
        loop {
            let line = self.next_line()?;
            if line.trim() == end {
                return Ok(());
            }
        }
    }
}

/// Gmsh `.msh` writer for ASCII v2.2 meshes.
#[derive(Debug, Default, Clone)]
pub struct GmshWriter;

impl GmshWriter {
    /// Write a block-based ASCII Gmsh 4.1 file.
    ///
    /// The legacy [`SieveSectionWriter`] implementation intentionally remains
    /// ASCII v2.2 for backwards compatibility; call this method when a modern
    /// v4.x interchange file is desired.
    pub fn write_v4_ascii<W>(
        &self,
        mut writer: W,
        mesh: &MeshData<MeshSieve, f64, VecStorage<f64>, VecStorage<CellType>>,
    ) -> Result<(), MeshSieveError>
    where
        W: Write,
    {
        let coords = mesh.coordinates.as_ref().ok_or_else(|| {
            MeshSieveError::MeshIoParse("Gmsh writer requires coordinates".into())
        })?;
        let coord_dim = coords.embedding_dimension().max(1).min(3);
        let cell_types = mesh
            .cell_types
            .as_ref()
            .ok_or_else(|| MeshSieveError::MeshIoParse("Gmsh writer requires cell types".into()))?;
        let labels = mesh.labels.as_ref();
        let node_ids: Vec<PointId> = coords.section().atlas().points().collect();
        let element_ids: Vec<PointId> = cell_types.atlas().points().collect();

        writeln!(writer, "$MeshFormat")?;
        writeln!(writer, "4.1 0 8")?;
        writeln!(writer, "$EndMeshFormat")?;

        let mut entity_physical: BTreeMap<(u8, i32), i32> = BTreeMap::new();
        let mut element_blocks: BTreeMap<(u8, i32, u32), Vec<(PointId, Vec<PointId>)>> =
            BTreeMap::new();
        for element in &element_ids {
            let cell_type = cell_types.try_restrict(*element)?[0];
            let conn: Vec<PointId> = mesh.sieve.cone_points(*element).collect();
            let elem_type = Self::gmsh_element_type(cell_type, conn.len()).ok_or_else(|| {
                MeshSieveError::MeshIoParse(format!(
                    "unsupported cell type {:?} with {} nodes",
                    cell_type,
                    conn.len()
                ))
            })?;
            let dim = cell_type.dimension();
            let entity = labels
                .and_then(|l| l.get_label(*element, "gmsh:entity"))
                .unwrap_or(i32::from(dim) + 1);
            let physical = labels
                .and_then(|l| l.get_label(*element, "gmsh:physical"))
                .unwrap_or(0);
            if physical != 0 {
                entity_physical.entry((dim, entity)).or_insert(physical);
            }
            element_blocks
                .entry((dim, entity, elem_type))
                .or_default()
                .push((*element, conn));
        }

        writeln!(writer, "$Entities")?;
        let point_entities = 0usize;
        let curve_entities = entity_physical.keys().filter(|(d, _)| *d == 1).count();
        let surface_entities = entity_physical.keys().filter(|(d, _)| *d == 2).count();
        let volume_entities = entity_physical.keys().filter(|(d, _)| *d == 3).count();
        writeln!(
            writer,
            "{point_entities} {curve_entities} {surface_entities} {volume_entities}"
        )?;
        for dim in 1..=3u8 {
            for ((entity_dim, entity), physical) in
                entity_physical.iter().filter(|((d, _), _)| *d == dim)
            {
                let _ = entity_dim;
                writeln!(writer, "{entity} 0 0 0 0 0 0 1 {physical} 0")?;
            }
        }
        writeln!(writer, "$EndEntities")?;

        writeln!(writer, "$Nodes")?;
        let min_node = node_ids.iter().map(PointId::get).min().unwrap_or(0);
        let max_node = node_ids.iter().map(PointId::get).max().unwrap_or(0);
        writeln!(writer, "1 {} {min_node} {max_node}", node_ids.len())?;
        writeln!(writer, "0 1 0 {}", node_ids.len())?;
        for node in &node_ids {
            writeln!(writer, "{}", node.get())?;
        }
        for node in &node_ids {
            let xyz = coords.section().try_restrict(*node)?;
            let (x, y, z) = match coord_dim {
                1 => (xyz[0], 0.0, 0.0),
                2 => (xyz[0], xyz[1], 0.0),
                _ => (xyz[0], xyz[1], xyz[2]),
            };
            writeln!(writer, "{x} {y} {z}")?;
        }
        writeln!(writer, "$EndNodes")?;

        writeln!(writer, "$Elements")?;
        let min_elem = element_ids.iter().map(PointId::get).min().unwrap_or(0);
        let max_elem = element_ids.iter().map(PointId::get).max().unwrap_or(0);
        writeln!(
            writer,
            "{} {} {min_elem} {max_elem}",
            element_blocks.len(),
            element_ids.len()
        )?;
        for ((dim, entity, elem_type), entries) in element_blocks {
            writeln!(writer, "{dim} {entity} {elem_type} {}", entries.len())?;
            for (element, conn) in entries {
                write!(writer, "{}", element.get())?;
                for node in conn {
                    write!(writer, " {}", node.get())?;
                }
                writeln!(writer)?;
            }
        }
        writeln!(writer, "$EndElements")?;
        Ok(())
    }

    fn section_is_subset(tagged: &TaggedSection, atlas: &Atlas) -> bool {
        tagged.atlas().points().all(|point| atlas.contains(point))
    }

    fn write_data_block<W, T>(
        mut writer: W,
        block_name: &str,
        name: &str,
        scalar_type: ScalarType,
        section: &Section<T, VecStorage<T>>,
    ) -> Result<(), MeshSieveError>
    where
        W: Write,
        T: std::fmt::Display,
    {
        if section.atlas().points().next().is_none() {
            return Ok(());
        }
        let mut iter = section.iter();
        let first = iter
            .next()
            .ok_or_else(|| MeshSieveError::MeshIoParse("missing section data".into()))?;
        let component_count = first.1.len();
        let entry_count = section.atlas().points().count();

        writeln!(writer, "${block_name}")?;
        writeln!(writer, "2")?;
        writeln!(writer, "\"{name}\"")?;
        writeln!(writer, "\"mesh_sieve:type={}\"", scalar_type.as_str())?;
        writeln!(writer, "1")?;
        writeln!(writer, "0.0")?;
        writeln!(writer, "3")?;
        writeln!(writer, "0")?;
        writeln!(writer, "{component_count}")?;
        writeln!(writer, "{entry_count}")?;
        writeln!(writer, "{}{}", first.0.get(), format_slice(first.1))?;
        for (point, values) in iter {
            writeln!(writer, "{}{}", point.get(), format_slice(values))?;
        }
        writeln!(writer, "$End{block_name}")?;
        Ok(())
    }

    fn gmsh_element_type(cell_type: CellType, node_count: usize) -> Option<u32> {
        match (cell_type, node_count) {
            (CellType::Vertex, 1) => Some(15),
            (CellType::Segment, 2) => Some(1),
            (CellType::Segment, 3) => Some(8),
            (CellType::Triangle, 3) => Some(2),
            (CellType::Triangle, 6) => Some(9),
            (CellType::Triangle, 9) => Some(20),
            (CellType::Triangle, 10) => Some(21),
            (CellType::Triangle, 12) => Some(22),
            (CellType::Triangle, 15) => Some(23),
            (CellType::Quadrilateral, 4) => Some(3),
            (CellType::Quadrilateral, 8) => Some(16),
            (CellType::Quadrilateral, 9) => Some(10),
            (CellType::Tetrahedron, 4) => Some(4),
            (CellType::Tetrahedron, 10) => Some(11),
            (CellType::Hexahedron, 8) => Some(5),
            (CellType::Hexahedron, 20) => Some(17),
            (CellType::Hexahedron, 27) => Some(12),
            (CellType::Prism, 6) => Some(6),
            (CellType::Prism, 15) => Some(18),
            (CellType::Prism, 18) => Some(13),
            (CellType::Pyramid, 5) => Some(7),
            (CellType::Pyramid, 13) => Some(19),
            (CellType::Pyramid, 14) => Some(14),
            (CellType::Simplex(0), 1) => Some(15),
            (CellType::Simplex(1), 2) => Some(1),
            (CellType::Simplex(2), 3) => Some(2),
            (CellType::Simplex(3), 4) => Some(4),
            (CellType::Polygon(3), 3) => Some(2),
            (CellType::Polygon(4), 4) => Some(3),
            (CellType::Polyhedron, 4) => Some(4),
            _ => None,
        }
    }

    fn collect_tags(labels: &LabelSet) -> HashMap<PointId, Vec<i32>> {
        let mut tagged: HashMap<PointId, (Option<i32>, Option<i32>, BTreeMap<usize, i32>)> =
            HashMap::new();
        for (name, point, value) in labels.iter() {
            let entry = tagged.entry(point).or_insert((None, None, BTreeMap::new()));
            match name {
                "gmsh:physical" => entry.0 = Some(value),
                "gmsh:entity" => entry.1 = Some(value),
                _ => {
                    if let Some(suffix) = name.strip_prefix("gmsh:tag")
                        && let Ok(index) = suffix.parse::<usize>() {
                            entry.2.insert(index, value);
                        }
                }
            }
        }

        tagged
            .into_iter()
            .map(|(point, (physical, entity, extras))| {
                let mut tags = Vec::new();
                if let Some(value) = physical {
                    tags.push(value);
                }
                if let Some(value) = entity {
                    if tags.is_empty() {
                        tags.push(0);
                    }
                    tags.push(value);
                }
                for (index, value) in extras {
                    while tags.len() + 1 < index {
                        tags.push(0);
                    }
                    tags.push(value);
                }
                (point, tags)
            })
            .collect()
    }
}

fn format_slice<T: std::fmt::Display>(values: &[T]) -> String {
    let mut out = String::new();
    for value in values {
        out.push(' ');
        out.push_str(&value.to_string());
    }
    out
}

impl SieveSectionReader for GmshReader {
    type Sieve = MeshSieve;
    type Value = f64;
    type Storage = VecStorage<f64>;
    type CellStorage = VecStorage<CellType>;

    fn read<R: Read>(
        &self,
        mut reader: R,
    ) -> Result<MeshData<Self::Sieve, Self::Value, Self::Storage, Self::CellStorage>, MeshSieveError>
    {
        self.read_with_options(&mut reader, GmshReadOptions::default())
    }
}

impl SieveSectionWriter for GmshWriter {
    type Sieve = MeshSieve;
    type Value = f64;
    type Storage = VecStorage<f64>;
    type CellStorage = VecStorage<CellType>;

    fn write<W: Write>(
        &self,
        mut writer: W,
        mesh: &MeshData<Self::Sieve, Self::Value, Self::Storage, Self::CellStorage>,
    ) -> Result<(), MeshSieveError> {
        let coords = mesh.coordinates.as_ref().ok_or_else(|| {
            MeshSieveError::MeshIoParse("Gmsh writer requires coordinates".into())
        })?;
        let coord_dim = coords.dimension();
        if coord_dim == 0 || coord_dim > 3 {
            return Err(MeshSieveError::MeshIoParse(format!(
                "unsupported coordinate dimension: {coord_dim}"
            )));
        }

        let cell_types = mesh
            .cell_types
            .as_ref()
            .ok_or_else(|| MeshSieveError::MeshIoParse("Gmsh writer requires cell types".into()))?;

        let tag_map = mesh.labels.as_ref().map(GmshWriter::collect_tags);

        writeln!(writer, "$MeshFormat")?;
        writeln!(writer, "2.2 0 8")?;
        writeln!(writer, "$EndMeshFormat")?;

        let node_ids: Vec<PointId> = coords.section().atlas().points().collect();
        writeln!(writer, "$Nodes")?;
        writeln!(writer, "{}", node_ids.len())?;
        for node in &node_ids {
            let xyz = coords.section().try_restrict(*node)?;
            let (x, y, z) = match coord_dim {
                1 => (xyz[0], 0.0, 0.0),
                2 => (xyz[0], xyz[1], 0.0),
                _ => (xyz[0], xyz[1], xyz[2]),
            };
            writeln!(writer, "{} {} {} {}", node.get(), x, y, z)?;
        }
        writeln!(writer, "$EndNodes")?;

        let element_ids: Vec<PointId> = cell_types.atlas().points().collect();
        writeln!(writer, "$Elements")?;
        writeln!(writer, "{}", element_ids.len())?;
        for element in &element_ids {
            let cell_slice = cell_types.try_restrict(*element)?;
            let cell_type = *cell_slice.first().ok_or_else(|| {
                MeshSieveError::MeshIoParse(format!("missing cell type for element {element:?}"))
            })?;
            let conn: Vec<PointId> = mesh.sieve.cone_points(*element).collect();
            let elem_type = Self::gmsh_element_type(cell_type, conn.len()).ok_or_else(|| {
                MeshSieveError::MeshIoParse(format!(
                    "unsupported cell type {:?} with {} nodes",
                    cell_type,
                    conn.len()
                ))
            })?;
            let tags = tag_map
                .as_ref()
                .and_then(|map| map.get(element))
                .cloned()
                .unwrap_or_default();
            write!(writer, "{} {} {}", element.get(), elem_type, tags.len())?;
            for tag in &tags {
                write!(writer, " {tag}")?;
            }
            for node in &conn {
                write!(writer, " {}", node.get())?;
            }
            writeln!(writer)?;
        }
        writeln!(writer, "$EndElements")?;

        if !mesh.mixed_sections.is_empty() {
            let coord_atlas = coords.section().atlas();
            let cell_atlas = cell_types.atlas();
            for (name, tagged) in mesh.mixed_sections.iter() {
                let write_node_data = Self::section_is_subset(tagged, coord_atlas)
                    || !Self::section_is_subset(tagged, cell_atlas);
                match tagged {
                    TaggedSection::F64(section) => {
                        let block_name = if write_node_data {
                            "NodeData"
                        } else {
                            "ElementData"
                        };
                        Self::write_data_block(
                            &mut writer,
                            block_name,
                            name,
                            ScalarType::F64,
                            section,
                        )?;
                    }
                    TaggedSection::F32(section) => {
                        let block_name = if write_node_data {
                            "NodeData"
                        } else {
                            "ElementData"
                        };
                        Self::write_data_block(
                            &mut writer,
                            block_name,
                            name,
                            ScalarType::F32,
                            section,
                        )?;
                    }
                    TaggedSection::I32(section) => {
                        let block_name = if write_node_data {
                            "NodeData"
                        } else {
                            "ElementData"
                        };
                        Self::write_data_block(
                            &mut writer,
                            block_name,
                            name,
                            ScalarType::I32,
                            section,
                        )?;
                    }
                    TaggedSection::I64(section) => {
                        let block_name = if write_node_data {
                            "NodeData"
                        } else {
                            "ElementData"
                        };
                        Self::write_data_block(
                            &mut writer,
                            block_name,
                            name,
                            ScalarType::I64,
                            section,
                        )?;
                    }
                    TaggedSection::U32(section) => {
                        let block_name = if write_node_data {
                            "NodeData"
                        } else {
                            "ElementData"
                        };
                        Self::write_data_block(
                            &mut writer,
                            block_name,
                            name,
                            ScalarType::U32,
                            section,
                        )?;
                    }
                    TaggedSection::U64(section) => {
                        let block_name = if write_node_data {
                            "NodeData"
                        } else {
                            "ElementData"
                        };
                        Self::write_data_block(
                            &mut writer,
                            block_name,
                            name,
                            ScalarType::U64,
                            section,
                        )?;
                    }
                }
            }
        }

        Ok(())
    }
}
