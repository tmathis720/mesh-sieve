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
//! - Higher-order elements are supported, but are stored as the base cell type.
//! - Element tags are captured as labels (`gmsh:physical`, `gmsh:entity`,
//!   and `gmsh:tagN` for additional tags).
//! - Coordinates preserve the inferred mesh dimension (1D/2D/3D).

use crate::data::atlas::Atlas;
use crate::data::coordinates::Coordinates;
use crate::data::section::Section;
use crate::data::storage::VecStorage;
use crate::geometry::quality::validate_cell_geometry;
use crate::io::{MeshData, SieveSectionReader, SieveSectionWriter};
use crate::mesh_error::MeshSieveError;
use crate::topology::cell_type::CellType;
use crate::topology::labels::LabelSet;
use crate::topology::point::PointId;
use crate::topology::sieve::{InMemorySieve, MutableSieve, Sieve};
use std::collections::{BTreeMap, HashMap};
use std::io::{Read, Write};

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

        let mut sieve = InMemorySieve::default();
        for (node, _) in &nodes {
            MutableSieve::add_point(&mut sieve, *node);
        }
        for element in &elements {
            MutableSieve::add_point(&mut sieve, element.id);
            for node in &element.conn {
                Sieve::add_arrow(&mut sieve, element.id, *node, ());
            }
        }

        let mut atlas = Atlas::default();
        for (node, _) in &nodes {
            atlas.try_insert(*node, coord_dimension)?;
        }
        let mut coords = Coordinates::try_new(coord_dimension, atlas)?;
        for (node, xyz) in &nodes {
            coords
                .section_mut()
                .try_set(*node, &xyz[..coord_dimension])?;
        }

        if options.check_geometry {
            for element in &elements {
                if let Err(err) = validate_cell_geometry(element.cell_type, &element.conn, &coords)
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
            labels: has_labels.then_some(labels),
            cell_types: (!elements.is_empty()).then_some(cell_types),
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
}

/// Gmsh `.msh` writer for ASCII v2.2 meshes.
#[derive(Debug, Default, Clone)]
pub struct GmshWriter;

impl GmshWriter {
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

        Ok(())
    }
}
