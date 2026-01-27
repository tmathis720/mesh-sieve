//! Gmsh `.msh` reader.
//!
//! # Supported format
//! - ASCII `.msh` version **2.2**.
//! - Element types: linear and common higher-order variants for lines,
//!   triangles, quads, tets, hexes, prisms, pyramids, and points.
//!
//! # Limitations
//! - Binary files are not supported.
//! - `.msh` v4.x (block-based) is not supported.
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
use crate::topology::sieve::{InMemorySieve, MutableSieve, Sieve};
use crate::topology::validation::{TopologyValidationOptions, validate_sieve_topology};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::{Read, Write};
use std::str::FromStr;

/// Gmsh `.msh` reader for ASCII v2.2 meshes.
#[derive(Debug, Default, Clone)]
pub struct GmshReader;

#[derive(Debug)]
struct ElementRecord {
    id: PointId,
    conn: Vec<PointId>,
    cell_type: CellType,
    tags: Vec<i32>,
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

    /// Read mesh data with explicit import options.
    pub fn read_with_options<R: Read>(
        &self,
        reader: R,
        options: GmshReadOptions,
    ) -> Result<
        MeshData<InMemorySieve<PointId, ()>, f64, VecStorage<f64>, VecStorage<CellType>>,
        MeshSieveError,
    > {
        let mut reader = reader;
        let mut contents = String::new();
        reader.read_to_string(&mut contents)?;
        let mut lines = contents.lines().peekable();

        let mut version: Option<String> = None;
        let mut nodes: Vec<(PointId, [f64; 3])> = Vec::new();
        let mut elements: Vec<ElementRecord> = Vec::new();
        let mut mixed_sections = MixedSectionStore::default();
        let mut element_data_names: HashSet<String> = HashSet::new();

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
                        while let Some(node_id) = parts.next() {
                            conn.push(Self::parse_point_id(node_id)?);
                        }
                        let min_nodes = Self::element_min_nodes(cell_type);
                        if conn.len() < min_nodes {
                            return Err(MeshSieveError::MeshIoParse(format!(
                                "element {elem_id:?} has {found} nodes, expected at least {min_nodes}",
                                found = conn.len()
                            )));
                        }
                        if let Some(expected) = Self::element_node_count(elem_type) {
                            if conn.len() != expected {
                                return Err(MeshSieveError::MeshIoParse(format!(
                                    "element {elem_id:?} expected {expected} nodes, got {found}",
                                    found = conn.len()
                                )));
                            }
                        }
                        elements.push(ElementRecord {
                            id: elem_id,
                            conn,
                            cell_type,
                            tags,
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

        let mesh_dimension = Self::mesh_dimension(&elements, &nodes);
        let coord_dimension = mesh_dimension.max(1).min(3);

        if options.validate_mixed_dimensions {
            Self::validate_element_dimensions(&elements)?;
        }

        Self::remap_elements_if_needed(
            &mut elements,
            &mut mixed_sections,
            &element_data_names,
        )?;

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

        let mut sieve = InMemorySieve::default();
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
                if let Some(ref mut seen) = seen_arrows {
                    if !seen.insert((element.id, *node)) {
                        return Err(MeshSieveError::DuplicateArrow {
                            src: element.id,
                            dst: *node,
                        });
                    }
                }
                Sieve::add_arrow(&mut sieve, element.id, *node, ());
            }
        }

        let mut atlas = Atlas::default();
        for node in &vertex_nodes {
            atlas.try_insert(*node, coord_dimension)?;
        }
        let mut coords = Coordinates::try_new(coord_dimension, atlas)?;
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
            if let Some(tag) = element.tags.get(0) {
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

/// Gmsh `.msh` writer for ASCII v2.2 meshes.
#[derive(Debug, Default, Clone)]
pub struct GmshWriter;

impl GmshWriter {
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
                    if let Some(suffix) = name.strip_prefix("gmsh:tag") {
                        if let Ok(index) = suffix.parse::<usize>() {
                            entry.2.insert(index, value);
                        }
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
    type Sieve = InMemorySieve<PointId, ()>;
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
    type Sieve = InMemorySieve<PointId, ()>;
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
                MeshSieveError::MeshIoParse(format!("missing cell type for element {:?}", element))
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
                write!(writer, " {}", tag)?;
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
