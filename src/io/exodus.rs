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
use hdf5::File;
use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

/// Legacy Exodus ASCII reader.
#[derive(Debug, Default, Clone)]
pub struct ExodusReader;

/// Legacy Exodus ASCII writer.
#[derive(Debug, Default, Clone)]
pub struct ExodusWriter;

/// Exodus II (HDF5) reader.
#[derive(Debug, Default, Clone)]
pub struct ExodusIiReader;

/// Exodus II (HDF5) writer.
#[derive(Debug, Default, Clone)]
pub struct ExodusIiWriter;

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

fn read_i64_dataset(dataset: &hdf5::Dataset) -> Result<Vec<i64>, MeshSieveError> {
    if let Ok(values) = dataset.read_raw::<i64>() {
        return Ok(values);
    }
    let values: Vec<i32> = dataset.read_raw()?;
    Ok(values.into_iter().map(|value| value as i64).collect())
}

fn read_f64_dataset(dataset: &hdf5::Dataset) -> Result<Vec<f64>, MeshSieveError> {
    if let Ok(values) = dataset.read_raw::<f64>() {
        return Ok(values);
    }
    let values: Vec<f32> = dataset.read_raw()?;
    Ok(values.into_iter().map(|value| value as f64).collect())
}

fn read_i64_dataset_optional(file: &File, name: &str) -> Option<Vec<i64>> {
    file.dataset(name).ok().and_then(|dataset| read_i64_dataset(&dataset).ok())
}

fn read_i32_scalar_optional(file: &File, name: &str) -> Option<i32> {
    let dataset = file.dataset(name).ok()?;
    if let Ok(values) = dataset.read_raw::<i32>() {
        return values.first().copied();
    }
    dataset
        .read_raw::<i64>()
        .ok()
        .and_then(|values| values.first().copied().map(|value| value as i32))
}

fn read_string_attr_optional(dataset: &hdf5::Dataset, name: &str) -> Option<String> {
    let attr = dataset.attr(name).ok()?;
    if let Ok(value) = attr.read_scalar::<String>() {
        return Some(value.trim_matches('\0').trim().to_string());
    }
    let raw: Vec<u8> = attr.read_raw().ok()?;
    let value = String::from_utf8_lossy(&raw).trim_matches('\0').trim().to_string();
    (!value.is_empty()).then_some(value)
}

fn exodus_elem_type_to_cell_type(
    elem_type: &str,
    nodes_per_elem: usize,
) -> Result<CellType, MeshSieveError> {
    let token = elem_type.trim().to_ascii_uppercase();
    let token = token.replace([' ', '\t'], "");
    let match_type = match token.as_str() {
        "POINT" | "NODE" => Some((CellType::Vertex, 1)),
        "BAR2" | "BAR" | "LINE2" | "LINE" | "EDGE2" | "EDGE" => Some((CellType::Segment, 2)),
        "TRI3" | "TRIANGLE" | "TRI" => Some((CellType::Triangle, 3)),
        "QUAD4" | "QUAD" | "SHELL4" => Some((CellType::Quadrilateral, 4)),
        "TET4" | "TETRA" | "TET" => Some((CellType::Tetrahedron, 4)),
        "HEX8" | "HEX" => Some((CellType::Hexahedron, 8)),
        "WEDGE6" | "WEDGE" | "PRISM" => Some((CellType::Prism, 6)),
        "PYRAMID5" | "PYRAMID" | "PYR" => Some((CellType::Pyramid, 5)),
        _ => None,
    };

    if let Some((cell_type, expected)) = match_type {
        if nodes_per_elem == expected {
            return Ok(cell_type);
        }
        return Err(MeshSieveError::MeshIoParse(format!(
            "unsupported Exodus II element type {elem_type} with {nodes_per_elem} nodes"
        )));
    }

    Err(MeshSieveError::MeshIoParse(format!(
        "unsupported Exodus II element type {elem_type}"
    )))
}

fn cell_type_to_exodus_elem_type(cell_type: CellType) -> Option<(&'static str, usize)> {
    match cell_type {
        CellType::Vertex => Some(("POINT", 1)),
        CellType::Segment => Some(("BAR2", 2)),
        CellType::Triangle => Some(("TRI3", 3)),
        CellType::Quadrilateral => Some(("QUAD4", 4)),
        CellType::Tetrahedron => Some(("TET4", 4)),
        CellType::Hexahedron => Some(("HEX8", 8)),
        CellType::Prism => Some(("WEDGE6", 6)),
        CellType::Pyramid => Some(("PYRAMID5", 5)),
        _ => None,
    }
}

fn temp_exodus_path() -> PathBuf {
    let mut path = std::env::temp_dir();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|v| v.as_nanos())
        .unwrap_or(0);
    let pid = std::process::id();
    path.push(format!("mesh_sieve_exodus_{pid}_{nanos}.e2"));
    path
}

fn write_temp_exodus(bytes: &[u8]) -> Result<PathBuf, MeshSieveError> {
    let path = temp_exodus_path();
    fs::write(&path, bytes)?;
    Ok(path)
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

impl SieveSectionReader for ExodusIiReader {
    type Sieve = InMemorySieve<PointId, ()>;
    type Value = f64;
    type Storage = VecStorage<f64>;
    type CellStorage = VecStorage<CellType>;

    fn read<R: Read>(
        &self,
        mut reader: R,
    ) -> Result<MeshData<Self::Sieve, Self::Value, Self::Storage, Self::CellStorage>, MeshSieveError>
    {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes)?;
        let temp_path = write_temp_exodus(&bytes)?;
        let file = File::open(&temp_path)
            .map_err(|err| MeshSieveError::MeshIoParse(format!("Exodus II open error: {err}")))?;
        let result = read_exodus_ii_from_hdf5(&file);
        let _ = fs::remove_file(&temp_path);
        result
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

fn read_exodus_ii_from_hdf5(
    file: &File,
) -> Result<
    MeshData<InMemorySieve<PointId, ()>, f64, VecStorage<f64>, VecStorage<CellType>>,
    MeshSieveError,
> {
    let coord_dim_hint = read_i32_scalar_optional(file, "num_dim").map(|value| value as usize);
    let coord_dataset = file.dataset("coord");

    let (coord_dim, coord_arrays) = if let Ok(dataset) = coord_dataset {
        let shape = dataset.shape();
        if shape.len() != 2 {
            return Err(MeshSieveError::MeshIoParse(
                "Exodus II coord dataset must be 2D".into(),
            ));
        }
        let values = read_f64_dataset(&dataset)?;
        let (dim, nodes, transposed) = if shape[0] <= 3 {
            (shape[0] as usize, shape[1] as usize, false)
        } else {
            (shape[1] as usize, shape[0] as usize, true)
        };
        let mut arrays = vec![vec![0.0; nodes]; dim];
        if transposed {
            for node_idx in 0..nodes {
                for dim_idx in 0..dim {
                    arrays[dim_idx][node_idx] = values[node_idx * dim + dim_idx];
                }
            }
        } else {
            for dim_idx in 0..dim {
                for node_idx in 0..nodes {
                    arrays[dim_idx][node_idx] = values[dim_idx * nodes + node_idx];
                }
            }
        }
        (coord_dim_hint.unwrap_or(dim), arrays)
    } else {
        let mut arrays = Vec::new();
        if let Ok(dataset) = file.dataset("coordx") {
            arrays.push(read_f64_dataset(&dataset)?);
        }
        if let Ok(dataset) = file.dataset("coordy") {
            arrays.push(read_f64_dataset(&dataset)?);
        }
        if let Ok(dataset) = file.dataset("coordz") {
            arrays.push(read_f64_dataset(&dataset)?);
        }
        if arrays.is_empty() {
            return Err(MeshSieveError::MeshIoParse(
                "Exodus II missing coordinate datasets".into(),
            ));
        }
        let inferred = coord_dim_hint.unwrap_or(arrays.len());
        (inferred, arrays)
    };

    if coord_dim == 0 || coord_dim > 3 {
        return Err(MeshSieveError::MeshIoParse(format!(
            "unsupported coordinate dimension: {coord_dim}"
        )));
    }

    let node_count = coord_arrays
        .first()
        .map(|values| values.len())
        .unwrap_or(0);
    for values in &coord_arrays {
        if values.len() != node_count {
            return Err(MeshSieveError::MeshIoParse(
                "Exodus II coordinate arrays must match in length".into(),
            ));
        }
    }

    let node_ids_raw = read_i64_dataset_optional(file, "node_num_map")
        .or_else(|| read_i64_dataset_optional(file, "node_id_map"))
        .or_else(|| read_i64_dataset_optional(file, "node_map"));
    let node_ids: Vec<PointId> = if let Some(values) = node_ids_raw {
        if values.len() != node_count {
            return Err(MeshSieveError::MeshIoParse(
                "node map length does not match coordinate length".into(),
            ));
        }
        values
            .into_iter()
            .map(|value| PointId::new(value as u64))
            .collect::<Result<Vec<_>, _>>()?
    } else {
        (1..=node_count as u64)
            .map(PointId::new)
            .collect::<Result<Vec<_>, _>>()?
    };

    let mut coord_atlas = Atlas::default();
    for point in &node_ids {
        coord_atlas.try_insert(*point, coord_dim)?;
    }
    let mut coords = Coordinates::try_new(coord_dim, coord_dim, coord_atlas)?;
    for (idx, point) in node_ids.iter().enumerate() {
        let mut values = Vec::with_capacity(coord_dim);
        for dim_idx in 0..coord_dim {
            let value = coord_arrays
                .get(dim_idx)
                .and_then(|array| array.get(idx))
                .copied()
                .unwrap_or(0.0);
            values.push(value);
        }
        coords.section_mut().try_set(*point, &values)?;
    }

    let mut sieve = InMemorySieve::default();
    for point in &node_ids {
        MutableSieve::add_point(&mut sieve, *point);
    }

    let mut cell_types = Section::<CellType, VecStorage<CellType>>::new(Atlas::default());
    let mut labels = LabelSet::new();
    let mut has_labels = false;

    let member_names = file
        .member_names()
        .map_err(|err| MeshSieveError::MeshIoParse(format!("Exodus II metadata error: {err}")))?;
    let mut connect_names: Vec<(usize, String)> = member_names
        .iter()
        .filter_map(|name| {
            name.strip_prefix("connect")
                .and_then(|suffix| suffix.parse::<usize>().ok())
                .map(|idx| (idx, name.clone()))
        })
        .collect();
    connect_names.sort_by_key(|(idx, _)| *idx);

    let block_ids_raw = read_i64_dataset_optional(file, "elem_blk_id")
        .or_else(|| read_i64_dataset_optional(file, "eb_prop1"));
    let mut block_ids: Vec<i64> = block_ids_raw.unwrap_or_default();
    if block_ids.len() < connect_names.len() {
        block_ids.extend((block_ids.len() + 1..=connect_names.len()).map(|v| v as i64));
    }

    let total_elems: usize = connect_names
        .iter()
        .filter_map(|(_, name)| file.dataset(name).ok())
        .map(|dataset| {
            dataset
                .shape()
                .get(0)
                .copied()
                .unwrap_or(0) as usize
        })
        .sum();

    let elem_ids_raw = read_i64_dataset_optional(file, "elem_num_map")
        .or_else(|| read_i64_dataset_optional(file, "elem_id_map"));
    let mut elem_ids: Vec<i64> = elem_ids_raw.unwrap_or_else(|| {
        (1..=total_elems as i64).collect()
    });
    if elem_ids.len() < total_elems {
        elem_ids.extend((elem_ids.len() as i64 + 1..=total_elems as i64));
    }

    let mut elem_offset = 0usize;
    for (block_order, (_idx, name)) in connect_names.iter().enumerate() {
        let dataset = file.dataset(name).map_err(|err| {
            MeshSieveError::MeshIoParse(format!("missing connectivity {name}: {err}"))
        })?;
        let shape = dataset.shape();
        if shape.len() != 2 {
            return Err(MeshSieveError::MeshIoParse(format!(
                "connectivity dataset {name} must be 2D"
            )));
        }
        let elems_in_block = shape[0] as usize;
        let nodes_per_elem = shape[1] as usize;
        let values = read_i64_dataset(&dataset)?;
        let elem_type = read_string_attr_optional(&dataset, "elem_type")
            .unwrap_or_else(|| "UNKNOWN".to_string());
        let cell_type = exodus_elem_type_to_cell_type(&elem_type, nodes_per_elem)?;
        let block_id = *block_ids
            .get(block_order)
            .unwrap_or(&(block_order as i64 + 1));
        let label_name = format!("exodus:block:{block_id}");

        for elem_idx in 0..elems_in_block {
            let raw_id = *elem_ids
                .get(elem_offset + elem_idx)
                .unwrap_or(&(elem_offset as i64 + elem_idx as i64 + 1));
            let elem_point = PointId::new(raw_id as u64)?;
            MutableSieve::add_point(&mut sieve, elem_point);
            for node_idx in 0..nodes_per_elem {
                let offset = elem_idx * nodes_per_elem + node_idx;
                let node_raw = values
                    .get(offset)
                    .copied()
                    .ok_or_else(|| {
                        MeshSieveError::MeshIoParse(format!(
                            "connectivity dataset {name} is incomplete"
                        ))
                    })?;
                let node_point = PointId::new(node_raw as u64)?;
                Sieve::add_arrow(&mut sieve, elem_point, node_point, ());
            }
            cell_types.try_add_point(elem_point, 1)?;
            cell_types.try_set(elem_point, &[cell_type])?;
            labels.set_label(elem_point, &label_name, 1);
            has_labels = true;
        }
        elem_offset += elems_in_block;
    }

    let node_set_names: Vec<(usize, String)> = member_names
        .iter()
        .filter_map(|name| {
            name.strip_prefix("node_ns")
                .and_then(|suffix| suffix.parse::<usize>().ok())
                .map(|idx| (idx, name.clone()))
        })
        .collect();
    let node_set_ids = read_i64_dataset_optional(file, "node_set_ids")
        .or_else(|| read_i64_dataset_optional(file, "ns_prop1"))
        .or_else(|| read_i64_dataset_optional(file, "node_set_id"))
        .unwrap_or_default();
    for (idx, name) in node_set_names {
        let dataset = file.dataset(&name)?;
        let nodes = read_i64_dataset(&dataset)?;
        let set_id = node_set_ids
            .get(idx - 1)
            .copied()
            .unwrap_or(idx as i64);
        let label_name = format!("exodus:node_set:{set_id}");
        for raw_id in nodes {
            let point = PointId::new(raw_id as u64)?;
            labels.set_label(point, &label_name, 1);
            has_labels = true;
        }
    }

    let side_set_names: Vec<(usize, String)> = member_names
        .iter()
        .filter_map(|name| {
            name.strip_prefix("elem_ss")
                .and_then(|suffix| suffix.parse::<usize>().ok())
                .map(|idx| (idx, name.clone()))
        })
        .collect();
    let side_set_ids = read_i64_dataset_optional(file, "side_set_ids")
        .or_else(|| read_i64_dataset_optional(file, "ss_prop1"))
        .or_else(|| read_i64_dataset_optional(file, "side_set_id"))
        .unwrap_or_default();
    for (idx, name) in side_set_names {
        let elem_dataset = file.dataset(&name)?;
        let elem_ids = read_i64_dataset(&elem_dataset)?;
        let side_dataset_name = format!("side_ss{idx}");
        let side_ids = file
            .dataset(&side_dataset_name)
            .ok()
            .and_then(|dataset| read_i64_dataset(&dataset).ok())
            .unwrap_or_default();
        let set_id = side_set_ids
            .get(idx - 1)
            .copied()
            .unwrap_or(idx as i64);
        let label_name = format!("exodus:side_set:{set_id}");
        for (entry_idx, raw_id) in elem_ids.iter().enumerate() {
            let point = PointId::new(*raw_id as u64)?;
            let side_value = side_ids
                .get(entry_idx)
                .copied()
                .unwrap_or(1) as i32;
            labels.set_label(point, &label_name, side_value);
            has_labels = true;
        }
    }

    let labels = has_labels.then_some(labels);
    let cell_types = if cell_types.atlas().points().next().is_some() {
        Some(cell_types)
    } else {
        None
    };

    Ok(MeshData {
        sieve,
        coordinates: Some(coords),
        sections: BTreeMap::new(),
        mixed_sections: Default::default(),
        labels,
        cell_types,
        discretization: None,
    })
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

impl SieveSectionWriter for ExodusIiWriter {
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
            MeshSieveError::MeshIoParse("Exodus II writer requires coordinates".into())
        })?;
        let coord_dim = coords.embedding_dimension();
        if coord_dim == 0 || coord_dim > 3 {
            return Err(MeshSieveError::MeshIoParse(format!(
                "unsupported coordinate dimension: {coord_dim}"
            )));
        }
        let cell_types = mesh.cell_types.as_ref().ok_or_else(|| {
            MeshSieveError::MeshIoParse("Exodus II writer requires cell types".into())
        })?;

        let node_ids: Vec<PointId> = coords.section().atlas().points().collect();
        let mut coord_values = vec![vec![0.0f64; node_ids.len()]; coord_dim];
        for (idx, node) in node_ids.iter().enumerate() {
            let slice = coords.section().try_restrict(*node)?;
            for dim_idx in 0..coord_dim {
                coord_values[dim_idx][idx] = *slice.get(dim_idx).unwrap_or(&0.0);
            }
        }

        let mut block_map: BTreeMap<i64, Vec<PointId>> = BTreeMap::new();
        if let Some(labels) = &mesh.labels {
            for (name, point, _) in labels.iter() {
                if let Some(id) = name.strip_prefix("exodus:block:") {
                    if let Ok(block_id) = id.parse::<i64>() {
                        block_map.entry(block_id).or_default().push(point);
                    }
                }
            }
        }
        let all_cells: Vec<PointId> = cell_types.atlas().points().collect();
        if block_map.is_empty() {
            block_map.insert(1, all_cells.clone());
        } else {
            let assigned: HashSet<PointId> =
                block_map.values().flat_map(|v| v.iter().copied()).collect();
            let unassigned: Vec<PointId> = all_cells
                .iter()
                .copied()
                .filter(|cell| !assigned.contains(cell))
                .collect();
            if !unassigned.is_empty() {
                let fallback_id = block_map.keys().max().copied().unwrap_or(0) + 1;
                block_map.insert(fallback_id, unassigned);
            }
        }

        let mut block_defs = Vec::new();
        for (block_id, mut cells) in block_map {
            cells.sort_unstable();
            let mut block_cell_type: Option<CellType> = None;
            for cell in &cells {
                let cell_type = cell_types.try_restrict(*cell)?[0];
                if let Some(existing) = block_cell_type {
                    if existing != cell_type {
                        return Err(MeshSieveError::MeshIoParse(format!(
                            "block {block_id} mixes cell types"
                        )));
                    }
                } else {
                    block_cell_type = Some(cell_type);
                }
            }
            let Some(cell_type) = block_cell_type else {
                continue;
            };
            block_defs.push((block_id, cell_type, cells));
        }

        let temp_path = temp_exodus_path();
        let file = File::create(&temp_path)
            .map_err(|err| MeshSieveError::MeshIoParse(format!("Exodus II create error: {err}")))?;

        file.new_dataset::<i32>()
            .shape(1)
            .create("num_dim")?
            .write(&[coord_dim as i32])?;

        if coord_dim > 0 {
            file.new_dataset::<f64>()
                .shape(node_ids.len())
                .create("coordx")?
                .write(&coord_values[0])?;
        }
        if coord_dim > 1 {
            file.new_dataset::<f64>()
                .shape(node_ids.len())
                .create("coordy")?
                .write(&coord_values[1])?;
        }
        if coord_dim > 2 {
            file.new_dataset::<f64>()
                .shape(node_ids.len())
                .create("coordz")?
                .write(&coord_values[2])?;
        }

        let node_map: Vec<i64> = node_ids.iter().map(|id| id.get() as i64).collect();
        file.new_dataset::<i64>()
            .shape(node_map.len())
            .create("node_num_map")?
            .write(&node_map)?;

        let mut elem_ids = Vec::new();
        let mut block_ids = Vec::new();
        let mut block_connectivity: Vec<(String, Vec<i64>, usize, String)> = Vec::new();

        for (idx, (block_id, cell_type, cells)) in block_defs.into_iter().enumerate() {
            let (elem_type, nodes_per_elem) = cell_type_to_exodus_elem_type(cell_type)
                .ok_or_else(|| {
                    MeshSieveError::MeshIoParse(format!(
                        "unsupported cell type {cell_type:?} for Exodus II"
                    ))
                })?;
            block_ids.push(block_id);
            let mut connectivity = Vec::with_capacity(cells.len() * nodes_per_elem);
            for cell in &cells {
                elem_ids.push(cell.get() as i64);
                let cone: Vec<PointId> = mesh.sieve.cone_points(*cell).collect();
                if cone.len() != nodes_per_elem {
                    return Err(MeshSieveError::MeshIoParse(format!(
                        "cell {cell:?} expected {nodes_per_elem} nodes, found {}",
                        cone.len()
                    )));
                }
                for node in cone {
                    connectivity.push(node.get() as i64);
                }
            }
            let dataset_name = format!("connect{}", idx + 1);
            block_connectivity.push((dataset_name, connectivity, nodes_per_elem, elem_type.to_string()));
        }

        file.new_dataset::<i64>()
            .shape(block_ids.len())
            .create("elem_blk_id")?
            .write(&block_ids)?;
        file.new_dataset::<i64>()
            .shape(elem_ids.len())
            .create("elem_num_map")?
            .write(&elem_ids)?;

        for (name, connectivity, nodes_per_elem, elem_type) in block_connectivity {
            let dataset = file
                .new_dataset::<i64>()
                .shape((connectivity.len() / nodes_per_elem, nodes_per_elem))
                .create(&name)?;
            dataset.write(&connectivity)?;
            if let Ok(attr) = dataset.new_attr::<String>().create("elem_type") {
                let _ = attr.write_scalar(&elem_type);
            }
        }

        if let Some(labels) = &mesh.labels {
            let mut node_sets: BTreeMap<i64, Vec<i64>> = BTreeMap::new();
            let mut side_sets: BTreeMap<i64, Vec<(i64, i64)>> = BTreeMap::new();
            for (name, point, value) in labels.iter() {
                if let Some(id) = name.strip_prefix("exodus:node_set:") {
                    if let Ok(set_id) = id.parse::<i64>() {
                        node_sets.entry(set_id).or_default().push(point.get() as i64);
                    }
                }
                if let Some(id) = name.strip_prefix("exodus:side_set:") {
                    if let Ok(set_id) = id.parse::<i64>() {
                        let side = if value > 0 { value } else { 1 } as i64;
                        side_sets
                            .entry(set_id)
                            .or_default()
                            .push((point.get() as i64, side));
                    }
                }
            }

            if !node_sets.is_empty() {
                let ids: Vec<i64> = node_sets.keys().copied().collect();
                file.new_dataset::<i64>()
                    .shape(ids.len())
                    .create("node_set_ids")?
                    .write(&ids)?;
                for (idx, (_set_id, nodes)) in node_sets.into_iter().enumerate() {
                    let dataset_name = format!("node_ns{}", idx + 1);
                    file.new_dataset::<i64>()
                        .shape(nodes.len())
                        .create(&dataset_name)?
                        .write(&nodes)?;
                }
            }

            if !side_sets.is_empty() {
                let ids: Vec<i64> = side_sets.keys().copied().collect();
                file.new_dataset::<i64>()
                    .shape(ids.len())
                    .create("side_set_ids")?
                    .write(&ids)?;
                for (idx, (_set_id, entries)) in side_sets.into_iter().enumerate() {
                    let elem_name = format!("elem_ss{}", idx + 1);
                    let side_name = format!("side_ss{}", idx + 1);
                    let elems: Vec<i64> = entries.iter().map(|(elem, _)| *elem).collect();
                    let sides: Vec<i64> = entries.iter().map(|(_, side)| *side).collect();
                    file.new_dataset::<i64>()
                        .shape(elems.len())
                        .create(&elem_name)?
                        .write(&elems)?;
                    file.new_dataset::<i64>()
                        .shape(sides.len())
                        .create(&side_name)?
                        .write(&sides)?;
                }
            }
        }

        let bytes = fs::read(&temp_path)?;
        writer.write_all(&bytes)?;
        let _ = fs::remove_file(&temp_path);
        Ok(())
    }
}
