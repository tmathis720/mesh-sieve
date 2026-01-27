//! Exodus mesh reader/writer.

use crate::data::coordinates::Coordinates;
use crate::data::section::Section;
use crate::data::storage::VecStorage;
use crate::io::{MeshData, SieveSectionReader, SieveSectionWriter};
use crate::mesh_error::MeshSieveError;
use crate::topology::cell_type::CellType;
use crate::topology::labels::LabelSet;
use crate::topology::point::PointId;
use crate::topology::sieve::{InMemorySieve, MutableSieve, Sieve};
use crate::topology::validation::{TopologyValidationOptions, validate_sieve_topology};
use crate::data::atlas::Atlas;
use std::collections::HashSet;
use std::io::{Read, Write};

/// Simple Exodus ASCII reader.
#[derive(Debug, Default, Clone)]
pub struct ExodusReader;

/// Simple Exodus ASCII writer.
#[derive(Debug, Default, Clone)]
pub struct ExodusWriter;

/// Optional settings for Exodus import.
#[derive(Debug, Default, Clone, Copy)]
pub struct ExodusReadOptions {
    /// When enabled, validate topology (cone sizes, duplicate arrows, closure consistency).
    pub validate_topology: bool,
}

fn cell_type_to_token(cell_type: CellType) -> String {
    match cell_type {
        CellType::Vertex => "Vertex".to_string(),
        CellType::Segment => "Segment".to_string(),
        CellType::Triangle => "Triangle".to_string(),
        CellType::Quadrilateral => "Quadrilateral".to_string(),
        CellType::Tetrahedron => "Tetrahedron".to_string(),
        CellType::Hexahedron => "Hexahedron".to_string(),
        CellType::Prism => "Prism".to_string(),
        CellType::Pyramid => "Pyramid".to_string(),
        CellType::Polygon(sides) => format!("Polygon:{sides}"),
        CellType::Simplex(dim) => format!("Simplex:{dim}"),
        CellType::Polyhedron => "Polyhedron".to_string(),
    }
}

fn token_to_cell_type(token: &str) -> Result<CellType, MeshSieveError> {
    match token {
        "Vertex" => Ok(CellType::Vertex),
        "Segment" => Ok(CellType::Segment),
        "Triangle" => Ok(CellType::Triangle),
        "Quadrilateral" => Ok(CellType::Quadrilateral),
        "Tetrahedron" => Ok(CellType::Tetrahedron),
        "Hexahedron" => Ok(CellType::Hexahedron),
        "Prism" => Ok(CellType::Prism),
        "Pyramid" => Ok(CellType::Pyramid),
        "Polyhedron" => Ok(CellType::Polyhedron),
        _ => {
            if let Some(value) = token.strip_prefix("Polygon:") {
                let sides = value.parse::<u8>().map_err(|_| {
                    MeshSieveError::MeshIoParse(format!("invalid polygon token: {token}"))
                })?;
                return Ok(CellType::Polygon(sides));
            }
            if let Some(value) = token.strip_prefix("Simplex:") {
                let dim = value.parse::<u8>().map_err(|_| {
                    MeshSieveError::MeshIoParse(format!("invalid simplex token: {token}"))
                })?;
                return Ok(CellType::Simplex(dim));
            }
            Err(MeshSieveError::MeshIoParse(format!(
                "unknown cell type token: {token}"
            )))
        }
    }
}

impl SieveSectionReader for ExodusReader {
    type Sieve = InMemorySieve<PointId, ()>;
    type Value = f64;
    type Storage = VecStorage<f64>;
    type CellStorage = VecStorage<CellType>;

    fn read<R: Read>(
        &self,
        reader: R,
    ) -> Result<MeshData<Self::Sieve, Self::Value, Self::Storage, Self::CellStorage>, MeshSieveError>
    {
        self.read_with_options(reader, ExodusReadOptions::default())
    }
}

impl ExodusReader {
    /// Read mesh data with explicit import options.
    pub fn read_with_options<R: Read>(
        &self,
        mut reader: R,
        options: ExodusReadOptions,
    ) -> Result<MeshData<InMemorySieve<PointId, ()>, f64, VecStorage<f64>, VecStorage<CellType>>, MeshSieveError>
    {
        let mut contents = String::new();
        reader.read_to_string(&mut contents)?;
        let mut lines = contents.lines();

        let header = lines
            .next()
            .ok_or_else(|| MeshSieveError::MeshIoParse("missing exodus header".into()))?;
        if header.trim() != "EXODUS" {
            return Err(MeshSieveError::MeshIoParse(
                "invalid exodus header".into(),
            ));
        }

        let dim_line = lines
            .next()
            .ok_or_else(|| MeshSieveError::MeshIoParse("missing dimension line".into()))?;
        let mut dim_parts = dim_line.split_whitespace();
        if dim_parts.next() != Some("DIM") {
            return Err(MeshSieveError::MeshIoParse(
                "missing DIM declaration".into(),
            ));
        }
        let coord_dim: usize = dim_parts
            .next()
            .ok_or_else(|| MeshSieveError::MeshIoParse("missing dimension value".into()))?
            .parse()
            .map_err(|_| MeshSieveError::MeshIoParse("invalid dimension".into()))?;
        if coord_dim == 0 || coord_dim > 3 {
            return Err(MeshSieveError::MeshIoParse(format!(
                "unsupported coordinate dimension: {coord_dim}"
            )));
        }

        let nodes_line = lines
            .next()
            .ok_or_else(|| MeshSieveError::MeshIoParse("missing nodes header".into()))?;
        let mut nodes_parts = nodes_line.split_whitespace();
        if nodes_parts.next() != Some("NODES") {
            return Err(MeshSieveError::MeshIoParse(
                "missing NODES section".into(),
            ));
        }
        let node_count: usize = nodes_parts
            .next()
            .ok_or_else(|| MeshSieveError::MeshIoParse("missing node count".into()))?
            .parse()
            .map_err(|_| MeshSieveError::MeshIoParse("invalid node count".into()))?;

        let mut atlas = Atlas::default();
        let mut nodes: Vec<PointId> = Vec::with_capacity(node_count);
        let mut coords_data: Vec<(PointId, Vec<f64>)> = Vec::with_capacity(node_count);
        for _ in 0..node_count {
            let line = lines
                .next()
                .ok_or_else(|| MeshSieveError::MeshIoParse("missing node entry".into()))?;
            let mut parts = line.split_whitespace();
            let id = parts
                .next()
                .ok_or_else(|| MeshSieveError::MeshIoParse("missing node id".into()))?
                .parse::<u64>()
                .map_err(|_| MeshSieveError::MeshIoParse("invalid node id".into()))?;
            let point = PointId::new(id).map_err(|_| {
                MeshSieveError::MeshIoParse(format!("invalid node id {id}"))
            })?;
            let mut values = Vec::with_capacity(coord_dim);
            for _ in 0..coord_dim {
                let value = parts
                    .next()
                    .ok_or_else(|| MeshSieveError::MeshIoParse("missing coordinate".into()))?
                    .parse::<f64>()
                    .map_err(|_| MeshSieveError::MeshIoParse("invalid coordinate".into()))?;
                values.push(value);
            }
            atlas.try_insert(point, coord_dim)?;
            nodes.push(point);
            coords_data.push((point, values));
        }

        let elements_line = lines
            .next()
            .ok_or_else(|| MeshSieveError::MeshIoParse("missing elements header".into()))?;
        let mut elements_parts = elements_line.split_whitespace();
        if elements_parts.next() != Some("ELEMENTS") {
            return Err(MeshSieveError::MeshIoParse(
                "missing ELEMENTS section".into(),
            ));
        }
        let element_count: usize = elements_parts
            .next()
            .ok_or_else(|| MeshSieveError::MeshIoParse("missing element count".into()))?
            .parse()
            .map_err(|_| MeshSieveError::MeshIoParse("invalid element count".into()))?;

        let mut sieve = InMemorySieve::default();
        let mut seen_arrows = if options.validate_topology {
            Some(HashSet::new())
        } else {
            None
        };
        for node in &nodes {
            MutableSieve::add_point(&mut sieve, *node);
        }

        let mut cell_types = Section::<CellType, VecStorage<CellType>>::new(Atlas::default());
        let mut label_set = LabelSet::new();
        let mut has_labels = false;

        for _ in 0..element_count {
            let line = lines
                .next()
                .ok_or_else(|| MeshSieveError::MeshIoParse("missing element entry".into()))?;
            let mut parts = line.split_whitespace();
            let id = parts
                .next()
                .ok_or_else(|| MeshSieveError::MeshIoParse("missing element id".into()))?
                .parse::<u64>()
                .map_err(|_| MeshSieveError::MeshIoParse("invalid element id".into()))?;
            let point = PointId::new(id).map_err(|_| {
                MeshSieveError::MeshIoParse(format!("invalid element id {id}"))
            })?;
            let cell_token = parts
                .next()
                .ok_or_else(|| MeshSieveError::MeshIoParse("missing cell type".into()))?;
            let cell_type = token_to_cell_type(cell_token)?;
            let node_count = parts
                .next()
                .ok_or_else(|| MeshSieveError::MeshIoParse("missing node count".into()))?
                .parse::<usize>()
                .map_err(|_| MeshSieveError::MeshIoParse("invalid node count".into()))?;
            let mut conn = Vec::with_capacity(node_count);
            for _ in 0..node_count {
                let node_id = parts
                    .next()
                    .ok_or_else(|| MeshSieveError::MeshIoParse("missing element node".into()))?
                    .parse::<u64>()
                    .map_err(|_| MeshSieveError::MeshIoParse("invalid element node".into()))?;
                let node_point = PointId::new(node_id).map_err(|_| {
                    MeshSieveError::MeshIoParse(format!("invalid node id {node_id}"))
                })?;
                conn.push(node_point);
            }

            MutableSieve::add_point(&mut sieve, point);
            for node in conn {
                if let Some(ref mut seen) = seen_arrows {
                    if !seen.insert((point, node)) {
                        return Err(MeshSieveError::DuplicateArrow { src: point, dst: node });
                    }
                }
                Sieve::add_arrow(&mut sieve, point, node, ());
            }
            cell_types.try_add_point(point, 1)?;
            cell_types.try_set(point, &[cell_type])?;
        }

        if let Some(labels_line) = lines.next() {
            let mut label_parts = labels_line.split_whitespace();
            if label_parts.next() == Some("LABELS") {
                let label_count: usize = label_parts
                    .next()
                    .ok_or_else(|| MeshSieveError::MeshIoParse("missing label count".into()))?
                    .parse()
                    .map_err(|_| MeshSieveError::MeshIoParse("invalid label count".into()))?;
                for _ in 0..label_count {
                    let line = lines
                        .next()
                        .ok_or_else(|| MeshSieveError::MeshIoParse("missing label entry".into()))?;
                    let mut parts = line.split_whitespace();
                    let id = parts
                        .next()
                        .ok_or_else(|| MeshSieveError::MeshIoParse("missing label id".into()))?
                        .parse::<u64>()
                        .map_err(|_| MeshSieveError::MeshIoParse("invalid label id".into()))?;
                    let point = PointId::new(id).map_err(|_| {
                        MeshSieveError::MeshIoParse(format!("invalid label id {id}"))
                    })?;
                    let name = parts
                        .next()
                        .ok_or_else(|| MeshSieveError::MeshIoParse("missing label name".into()))?;
                    let value = parts
                        .next()
                        .ok_or_else(|| MeshSieveError::MeshIoParse("missing label value".into()))?
                        .parse::<i32>()
                        .map_err(|_| MeshSieveError::MeshIoParse("invalid label value".into()))?;
                    label_set.set_label(point, name, value);
                    has_labels = true;
                }
            }
        }

        if options.validate_topology {
            validate_sieve_topology(&sieve, &cell_types, TopologyValidationOptions::all())?;
        }

        let mut coords = Coordinates::try_new(coord_dim, coord_dim, atlas)?;
        for (point, values) in coords_data {
            coords.section_mut().try_set(point, &values)?;
        }

        Ok(MeshData {
            sieve,
            coordinates: Some(coords),
            sections: Default::default(),
            mixed_sections: Default::default(),
            labels: has_labels.then_some(label_set),
            cell_types: if element_count > 0 {
                Some(cell_types)
            } else {
                None
            },
            discretization: None,
        })
    }
}

impl SieveSectionWriter for ExodusWriter {
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
            MeshSieveError::MeshIoParse("Exodus writer requires coordinates".into())
        })?;
        let coord_dim = coords.dimension();
        if coord_dim == 0 || coord_dim > 3 {
            return Err(MeshSieveError::MeshIoParse(format!(
                "unsupported coordinate dimension: {coord_dim}"
            )));
        }
        let cell_types = mesh.cell_types.as_ref().ok_or_else(|| {
            MeshSieveError::MeshIoParse("Exodus writer requires cell types".into())
        })?;

        let node_ids: Vec<PointId> = coords.section().atlas().points().collect();
        writeln!(writer, "EXODUS")?;
        writeln!(writer, "DIM {coord_dim}")?;
        writeln!(writer, "NODES {}", node_ids.len())?;
        for node in &node_ids {
            let slice = coords.section().try_restrict(*node)?;
            write!(writer, "{}", node.get())?;
            for value in slice.iter().take(coord_dim) {
                write!(writer, " {value}")?;
            }
            writeln!(writer)?;
        }

        let element_ids: Vec<PointId> = cell_types.atlas().points().collect();
        writeln!(writer, "ELEMENTS {}", element_ids.len())?;
        for element in &element_ids {
            let cell_slice = cell_types.try_restrict(*element)?;
            let cell_type = *cell_slice.first().ok_or_else(|| {
                MeshSieveError::MeshIoParse(format!("missing cell type for {element:?}"))
            })?;
            let conn: Vec<PointId> = mesh.sieve.cone_points(*element).collect();
            let token = cell_type_to_token(cell_type);
            write!(writer, "{} {} {}", element.get(), token, conn.len())?;
            for node in conn {
                write!(writer, " {}", node.get())?;
            }
            writeln!(writer)?;
        }

        let label_entries = mesh
            .labels
            .as_ref()
            .map(|labels| labels.iter().collect::<Vec<_>>())
            .unwrap_or_default();
        writeln!(writer, "LABELS {}", label_entries.len())?;
        for (name, point, value) in label_entries {
            writeln!(writer, "{} {} {}", point.get(), name, value)?;
        }

        writeln!(writer, "END")?;
        Ok(())
    }
}
