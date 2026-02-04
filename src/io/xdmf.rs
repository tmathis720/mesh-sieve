//! XDMF (XML) reader/writer for unstructured grids.
//!
//! This implementation writes inline XML `DataItem` payloads (for small meshes)
//! and can optionally emit HDF5-backed `DataItem`s with a companion HDF5 file.
//! Mesh-sieve metadata is preserved via `Attribute` arrays.

use crate::data::atlas::Atlas;
use crate::data::coordinates::Coordinates;
use crate::data::section::Section;
use crate::data::storage::VecStorage;
use crate::io::hdf5::{
    DATASET_CELL_IDS, DATASET_COORDINATES, DATASET_EMBED_DIM, DATASET_MIXED, DATASET_POINT_IDS,
    DATASET_SECTION_IDS, DATASET_SECTION_VALUES, DATASET_TOPO_DIM, GROUP_GEOMETRY, GROUP_LABELS,
    GROUP_SECTIONS, GROUP_TOPOLOGY, write_mesh_to_hdf5,
};
use crate::io::{MeshData, SieveSectionReader, SieveSectionWriter};
use crate::mesh_error::MeshSieveError;
use crate::topology::cell_type::CellType;
use crate::topology::labels::LabelSet;
use crate::topology::point::PointId;
use crate::topology::sieve::{InMemorySieve, Sieve};
use hdf5::File;
use roxmltree::Document;
use std::collections::{BTreeMap, HashMap};
use std::io::{Read, Write};
use std::path::PathBuf;

const ATTR_POINT_IDS: &str = "mesh_sieve:point_ids";
const ATTR_CELL_IDS: &str = "mesh_sieve:cell_ids";
const ATTR_COORD_TOPO_DIM: &str = "mesh_sieve:coords:topo_dim";
const ATTR_COORD_EMBED_DIM: &str = "mesh_sieve:coords:embed_dim";
const ATTR_SECTION_PREFIX: &str = "mesh_sieve:section:";
const ATTR_LABEL_PREFIX: &str = "mesh_sieve:label:";

#[derive(Debug, Default, Clone)]
pub struct XdmfReader;

#[derive(Debug, Clone)]
pub struct XdmfWriter {
    hdf5_path: Option<PathBuf>,
    inline_threshold: usize,
}

impl Default for XdmfWriter {
    fn default() -> Self {
        Self {
            hdf5_path: None,
            inline_threshold: 1024,
        }
    }
}

impl XdmfWriter {
    /// Create a writer that emits HDF5-backed DataItems to the given path.
    pub fn with_hdf5_path<P: Into<PathBuf>>(path: P) -> Self {
        Self {
            hdf5_path: Some(path.into()),
            ..Self::default()
        }
    }

    /// Adjust the inline XML threshold (number of scalar values).
    pub fn with_inline_threshold(mut self, threshold: usize) -> Self {
        self.inline_threshold = threshold;
        self
    }

    fn hdf5_reference(&self) -> Option<String> {
        self.hdf5_path.as_ref().map(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .map(|name| name.to_string())
                .unwrap_or_else(|| path.to_string_lossy().to_string())
        })
    }

    fn should_inline(&self, value_count: usize) -> bool {
        self.hdf5_path.is_none() || value_count <= self.inline_threshold
    }
}

impl XdmfWriter {
    fn mixed_cell_type(cell_type: CellType) -> Option<(i64, usize)> {
        match cell_type {
            CellType::Vertex => Some((1, 1)),
            CellType::Segment => Some((2, 2)),
            CellType::Triangle => Some((3, 3)),
            CellType::Quadrilateral => Some((4, 4)),
            CellType::Tetrahedron => Some((5, 4)),
            CellType::Hexahedron => Some((6, 8)),
            CellType::Prism => Some((7, 6)),
            CellType::Pyramid => Some((8, 5)),
            _ => None,
        }
    }

    fn write_data_item<W: Write>(
        writer: &mut W,
        indent: &str,
        dimensions: &str,
        number_type: &str,
        values: &[String],
    ) -> Result<(), MeshSieveError> {
        writeln!(
            writer,
            "{indent}<DataItem Dimensions=\"{dimensions}\" NumberType=\"{number_type}\" Format=\"XML\">"
        )?;
        write!(writer, "{indent}  ")?;
        for (idx, value) in values.iter().enumerate() {
            if idx > 0 {
                write!(writer, " ")?;
            }
            write!(writer, "{value}")?;
        }
        writeln!(writer)?;
        writeln!(writer, "{indent}</DataItem>")?;
        Ok(())
    }

    fn write_data_item_hdf<W: Write>(
        writer: &mut W,
        indent: &str,
        dimensions: &str,
        number_type: &str,
        reference: &str,
    ) -> Result<(), MeshSieveError> {
        writeln!(
            writer,
            "{indent}<DataItem Dimensions=\"{dimensions}\" NumberType=\"{number_type}\" Format=\"HDF\">"
        )?;
        writeln!(writer, "{indent}  {reference}")?;
        writeln!(writer, "{indent}</DataItem>")?;
        Ok(())
    }
}

impl SieveSectionWriter for XdmfWriter {
    type Sieve = InMemorySieve<PointId, ()>;
    type Value = f64;
    type Storage = VecStorage<f64>;
    type CellStorage = VecStorage<CellType>;

    fn write<W: Write>(
        &self,
        mut writer: W,
        mesh: &MeshData<Self::Sieve, Self::Value, Self::Storage, Self::CellStorage>,
    ) -> Result<(), MeshSieveError> {
        let coords = mesh
            .coordinates
            .as_ref()
            .ok_or_else(|| MeshSieveError::MeshIoParse("XDMF requires coordinates".into()))?;
        let cell_types = mesh
            .cell_types
            .as_ref()
            .ok_or_else(|| MeshSieveError::MeshIoParse("XDMF requires cell types".into()))?;

        let point_ids: Vec<PointId> = coords.section().atlas().points().collect();
        let mut point_index = HashMap::new();
        for (idx, point) in point_ids.iter().enumerate() {
            point_index.insert(*point, idx);
        }

        let cell_ids: Vec<PointId> = cell_types.atlas().points().collect();
        let mut mixed = Vec::new();
        for cell in &cell_ids {
            let cell_type = cell_types.try_restrict(*cell)?[0];
            let (code, count) = Self::mixed_cell_type(cell_type).ok_or_else(|| {
                MeshSieveError::MeshIoParse(format!(
                    "unsupported XDMF cell type {cell_type:?}"
                ))
            })?;
            let cone: Vec<PointId> = mesh.sieve.cone_points(*cell).collect();
            if cone.len() != count {
                return Err(MeshSieveError::MeshIoParse(format!(
                    "XDMF cell {cell:?} expected {count} vertices, found {}",
                    cone.len()
                )));
            }
            mixed.push(code.to_string());
            for point in cone {
                let idx = point_index.get(&point).ok_or_else(|| {
                    MeshSieveError::MeshIoParse(format!(
                        "missing point {point:?} in coordinates"
                    ))
                })?;
                mixed.push(idx.to_string());
            }
        }

        let mut coord_values = Vec::with_capacity(point_ids.len() * 3);
        for point in &point_ids {
            let slice = coords.try_restrict(*point)?;
            let mut values = [0.0f64; 3];
            for (idx, value) in slice.iter().enumerate().take(3) {
                values[idx] = *value;
            }
            coord_values.extend(values.iter().map(|v| v.to_string()));
        }

        let mut value_count = coord_values.len() + mixed.len() + point_ids.len() + cell_ids.len();
        for section in mesh.sections.values() {
            let ids_len = section.atlas().points().count();
            let values_len: usize = section.iter().map(|(_, data)| data.len()).sum();
            value_count += ids_len + values_len;
        }
        if let Some(labels) = &mesh.labels {
            let mut label_count = 0usize;
            for (_name, _point, _value) in labels.iter() {
                label_count += 2;
            }
            value_count += label_count;
        }

        let use_hdf = self.hdf5_path.is_some() && !self.should_inline(value_count);
        let hdf_ref = self.hdf5_reference();
        if use_hdf {
            let path = self
                .hdf5_path
                .as_ref()
                .ok_or_else(|| MeshSieveError::MeshIoParse("missing HDF5 path".into()))?;
            let file = File::create(path).map_err(|err| {
                MeshSieveError::MeshIoParse(format!("HDF5 create error: {err}"))
            })?;
            write_mesh_to_hdf5(&file, mesh)?;
        }

        writeln!(writer, "<?xml version=\"1.0\" ?>")?;
        writeln!(writer, "<Xdmf Version=\"3.0\">")?;
        writeln!(writer, "  <Domain>")?;
        writeln!(
            writer,
            "    <Grid Name=\"mesh\" GridType=\"Uniform\">"
        )?;
        writeln!(
            writer,
            "      <Topology TopologyType=\"Mixed\" NumberOfElements=\"{}\">",
            cell_ids.len()
        )?;
        if use_hdf {
            let reference = format!(
                "{}:/{GROUP_TOPOLOGY}/{DATASET_MIXED}",
                hdf_ref.as_deref().unwrap_or(\"mesh.h5\")
            );
            Self::write_data_item_hdf(
                &mut writer,
                "        ",
                &mixed.len().to_string(),
                "Int",
                &reference,
            )?;
        } else {
            Self::write_data_item(
                &mut writer,
                "        ",
                &mixed.len().to_string(),
                "Int",
                &mixed,
            )?;
        }
        writeln!(writer, "      </Topology>")?;
        writeln!(writer, "      <Geometry GeometryType=\"XYZ\">")?;
        if use_hdf {
            let reference = format!(
                "{}:/{GROUP_GEOMETRY}/{DATASET_COORDINATES}",
                hdf_ref.as_deref().unwrap_or(\"mesh.h5\")
            );
            Self::write_data_item_hdf(
                &mut writer,
                "        ",
                &format!("{} 3", point_ids.len()),
                "Float",
                &reference,
            )?;
        } else {
            Self::write_data_item(
                &mut writer,
                "        ",
                &format!("{} 3", point_ids.len()),
                "Float",
                &coord_values,
            )?;
        }
        writeln!(writer, "      </Geometry>")?;

        writeln!(
            writer,
            "      <Attribute Name=\"{ATTR_POINT_IDS}\" AttributeType=\"Scalar\" Center=\"Node\">"
        )?;
        let point_id_values: Vec<String> =
            point_ids.iter().map(|p| p.get().to_string()).collect();
        if use_hdf {
            let reference = format!(
                "{}:/{GROUP_GEOMETRY}/{DATASET_POINT_IDS}",
                hdf_ref.as_deref().unwrap_or(\"mesh.h5\")
            );
            Self::write_data_item_hdf(
                &mut writer,
                "        ",
                &point_ids.len().to_string(),
                "Int",
                &reference,
            )?;
        } else {
            Self::write_data_item(
                &mut writer,
                "        ",
                &point_ids.len().to_string(),
                "Int",
                &point_id_values,
            )?;
        }
        writeln!(writer, "      </Attribute>")?;

        writeln!(
            writer,
            "      <Attribute Name=\"{ATTR_CELL_IDS}\" AttributeType=\"Scalar\" Center=\"Cell\">"
        )?;
        let cell_id_values: Vec<String> =
            cell_ids.iter().map(|c| c.get().to_string()).collect();
        if use_hdf {
            let reference = format!(
                "{}:/{GROUP_TOPOLOGY}/{DATASET_CELL_IDS}",
                hdf_ref.as_deref().unwrap_or(\"mesh.h5\")
            );
            Self::write_data_item_hdf(
                &mut writer,
                "        ",
                &cell_ids.len().to_string(),
                "Int",
                &reference,
            )?;
        } else {
            Self::write_data_item(
                &mut writer,
                "        ",
                &cell_ids.len().to_string(),
                "Int",
                &cell_id_values,
            )?;
        }
        writeln!(writer, "      </Attribute>")?;

        writeln!(
            writer,
            "      <Attribute Name=\"{ATTR_COORD_TOPO_DIM}\" AttributeType=\"Scalar\" Center=\"Grid\">"
        )?;
        if use_hdf {
            let reference = format!(
                "{}:/{GROUP_GEOMETRY}/{DATASET_TOPO_DIM}",
                hdf_ref.as_deref().unwrap_or(\"mesh.h5\")
            );
            Self::write_data_item_hdf(&mut writer, "        ", "1", "Int", &reference)?;
        } else {
            Self::write_data_item(
                &mut writer,
                "        ",
                "1",
                "Int",
                &[coords.topological_dimension().to_string()],
            )?;
        }
        writeln!(writer, "      </Attribute>")?;

        writeln!(
            writer,
            "      <Attribute Name=\"{ATTR_COORD_EMBED_DIM}\" AttributeType=\"Scalar\" Center=\"Grid\">"
        )?;
        if use_hdf {
            let reference = format!(
                "{}:/{GROUP_GEOMETRY}/{DATASET_EMBED_DIM}",
                hdf_ref.as_deref().unwrap_or(\"mesh.h5\")
            );
            Self::write_data_item_hdf(&mut writer, "        ", "1", "Int", &reference)?;
        } else {
            Self::write_data_item(
                &mut writer,
                "        ",
                "1",
                "Int",
                &[coords.embedding_dimension().to_string()],
            )?;
        }
        writeln!(writer, "      </Attribute>")?;

        for (name, section) in &mesh.sections {
            let ids: Vec<String> = section
                .atlas()
                .points()
                .map(|p| p.get().to_string())
                .collect();
            let mut values = Vec::new();
            let mut num_components = 0usize;
            for (_point, data) in section.iter() {
                if num_components == 0 {
                    num_components = data.len();
                }
                values.extend(data.iter().map(|v| v.to_string()));
            }
            writeln!(
                writer,
                "      <Attribute Name=\"{ATTR_SECTION_PREFIX}{name}:ids\" AttributeType=\"Scalar\" Center=\"Node\">"
            )?;
            if use_hdf {
                let reference = format!(
                    "{}:/{GROUP_SECTIONS}/{name}/{DATASET_SECTION_IDS}",
                    hdf_ref.as_deref().unwrap_or(\"mesh.h5\")
                );
                Self::write_data_item_hdf(
                    &mut writer,
                    "        ",
                    &ids.len().to_string(),
                    "Int",
                    &reference,
                )?;
            } else {
                Self::write_data_item(
                    &mut writer,
                    "        ",
                    &ids.len().to_string(),
                    "Int",
                    &ids,
                )?;
            }
            writeln!(writer, "      </Attribute>")?;

            writeln!(
                writer,
                "      <Attribute Name=\"{ATTR_SECTION_PREFIX}{name}:values\" AttributeType=\"Scalar\" Center=\"Node\">"
            )?;
            let tuples = if num_components == 0 {
                0
            } else {
                values.len() / num_components
            };
            let dims = if num_components == 0 {
                "0".to_string()
            } else {
                format!("{tuples} {num_components}")
            };
            if use_hdf {
                let reference = format!(
                    "{}:/{GROUP_SECTIONS}/{name}/{DATASET_SECTION_VALUES}",
                    hdf_ref.as_deref().unwrap_or(\"mesh.h5\")
                );
                Self::write_data_item_hdf(&mut writer, "        ", &dims, "Float", &reference)?;
            } else {
                Self::write_data_item(&mut writer, "        ", &dims, "Float", &values)?;
            }
            writeln!(writer, "      </Attribute>")?;
        }

        if let Some(labels) = &mesh.labels {
            let mut grouped: BTreeMap<String, Vec<(PointId, i32)>> = BTreeMap::new();
            for (name, point, value) in labels.iter() {
                grouped
                    .entry(name.to_string())
                    .or_default()
                    .push((point, value));
            }
            for (name, entries) in grouped {
                let ids: Vec<String> =
                    entries.iter().map(|(p, _)| p.get().to_string()).collect();
                let values: Vec<String> =
                    entries.iter().map(|(_, v)| v.to_string()).collect();
                writeln!(
                    writer,
                    "      <Attribute Name=\"{ATTR_LABEL_PREFIX}{name}:ids\" AttributeType=\"Scalar\" Center=\"Node\">"
                )?;
                if use_hdf {
                    let reference = format!(
                        "{}:/{GROUP_LABELS}/{name}/{DATASET_SECTION_IDS}",
                        hdf_ref.as_deref().unwrap_or(\"mesh.h5\")
                    );
                    Self::write_data_item_hdf(
                        &mut writer,
                        "        ",
                        &ids.len().to_string(),
                        "Int",
                        &reference,
                    )?;
                } else {
                    Self::write_data_item(
                        &mut writer,
                        "        ",
                        &ids.len().to_string(),
                        "Int",
                        &ids,
                    )?;
                }
                writeln!(writer, "      </Attribute>")?;
                writeln!(
                    writer,
                    "      <Attribute Name=\"{ATTR_LABEL_PREFIX}{name}:values\" AttributeType=\"Scalar\" Center=\"Node\">"
                )?;
                if use_hdf {
                    let reference = format!(
                        "{}:/{GROUP_LABELS}/{name}/{DATASET_SECTION_VALUES}",
                        hdf_ref.as_deref().unwrap_or(\"mesh.h5\")
                    );
                    Self::write_data_item_hdf(
                        &mut writer,
                        "        ",
                        &values.len().to_string(),
                        "Int",
                        &reference,
                    )?;
                } else {
                    Self::write_data_item(
                        &mut writer,
                        "        ",
                        &values.len().to_string(),
                        "Int",
                        &values,
                    )?;
                }
                writeln!(writer, "      </Attribute>")?;
            }
        }

        writeln!(writer, "    </Grid>")?;
        writeln!(writer, "  </Domain>")?;
        writeln!(writer, "</Xdmf>")?;
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct DataItem {
    dimensions: Vec<usize>,
    values: Vec<String>,
    format: DataFormat,
}

#[derive(Clone, Debug)]
enum DataFormat {
    Xml,
    Hdf5(Hdf5Ref),
}

#[derive(Clone, Debug)]
struct Hdf5Ref {
    file: String,
    dataset: String,
}

impl DataItem {
    fn values_as_i64(&self) -> Result<Vec<i64>, MeshSieveError> {
        match &self.format {
            DataFormat::Xml => self
                .values
                .iter()
                .map(|v| {
                    v.parse::<i64>()
                        .map_err(|_| MeshSieveError::MeshIoParse(format!("invalid int value {v}")))
                })
                .collect(),
            DataFormat::Hdf5(hdf_ref) => read_hdf_dataset_i64(hdf_ref),
        }
    }

    fn values_as_f64(&self) -> Result<Vec<f64>, MeshSieveError> {
        match &self.format {
            DataFormat::Xml => self
                .values
                .iter()
                .map(|v| {
                    v.parse::<f64>()
                        .map_err(|_| MeshSieveError::MeshIoParse(format!("invalid float value {v}")))
                })
                .collect(),
            DataFormat::Hdf5(hdf_ref) => read_hdf_dataset_f64(hdf_ref),
        }
    }

    fn effective_dimensions(&self) -> Result<Vec<usize>, MeshSieveError> {
        if !self.dimensions.is_empty() {
            return Ok(self.dimensions.clone());
        }
        match &self.format {
            DataFormat::Xml => Ok(self.dimensions.clone()),
            DataFormat::Hdf5(hdf_ref) => {
                let file = File::open(&hdf_ref.file).map_err(|err| {
                    MeshSieveError::MeshIoParse(format!("HDF5 open error: {err}"))
                })?;
                let dataset = file.dataset(&hdf_ref.dataset).map_err(|err| {
                    MeshSieveError::MeshIoParse(format!(
                        "HDF5 dataset {} error: {err}",
                        hdf_ref.dataset
                    ))
                })?;
                Ok(dataset.shape())
            }
        }
    }
}

fn read_hdf_dataset_i64(hdf_ref: &Hdf5Ref) -> Result<Vec<i64>, MeshSieveError> {
    let file = File::open(&hdf_ref.file)
        .map_err(|err| MeshSieveError::MeshIoParse(format!("HDF5 open error: {err}")))?;
    let dataset = file.dataset(&hdf_ref.dataset).map_err(|err| {
        MeshSieveError::MeshIoParse(format!("HDF5 dataset {} error: {err}", hdf_ref.dataset))
    })?;
    if let Ok(values) = dataset.read_raw::<i64>() {
        return Ok(values);
    }
    let values = dataset
        .read_raw::<i32>()
        .map_err(|err| MeshSieveError::MeshIoParse(format!("HDF5 int read error: {err}")))?;
    Ok(values.into_iter().map(|v| v as i64).collect())
}

fn read_hdf_dataset_f64(hdf_ref: &Hdf5Ref) -> Result<Vec<f64>, MeshSieveError> {
    let file = File::open(&hdf_ref.file)
        .map_err(|err| MeshSieveError::MeshIoParse(format!("HDF5 open error: {err}")))?;
    let dataset = file.dataset(&hdf_ref.dataset).map_err(|err| {
        MeshSieveError::MeshIoParse(format!("HDF5 dataset {} error: {err}", hdf_ref.dataset))
    })?;
    if let Ok(values) = dataset.read_raw::<f64>() {
        return Ok(values);
    }
    let values = dataset
        .read_raw::<f32>()
        .map_err(|err| MeshSieveError::MeshIoParse(format!("HDF5 float read error: {err}")))?;
    Ok(values.into_iter().map(|v| v as f64).collect())
}

impl XdmfReader {
    fn parse_dimensions(raw: Option<&str>) -> Vec<usize> {
        raw.unwrap_or("")
            .split_whitespace()
            .filter_map(|v| v.parse::<usize>().ok())
            .collect()
    }

    fn parse_data_item(node: roxmltree::Node) -> Result<DataItem, MeshSieveError> {
        let dimensions = Self::parse_dimensions(node.attribute("Dimensions"));
        let format = node
            .attribute("Format")
            .unwrap_or("XML")
            .trim()
            .to_ascii_uppercase();
        if format == "HDF" {
            let text = node.text().unwrap_or("").trim();
            let mut parts = text.splitn(2, ':');
            let file = parts
                .next()
                .filter(|v| !v.is_empty())
                .ok_or_else(|| MeshSieveError::MeshIoParse("HDF DataItem missing file".into()))?;
            let dataset = parts
                .next()
                .filter(|v| !v.is_empty())
                .ok_or_else(|| MeshSieveError::MeshIoParse("HDF DataItem missing dataset".into()))?;
            let dataset = if dataset.starts_with('/') {
                dataset.to_string()
            } else {
                format!("/{dataset}")
            };
            Ok(DataItem {
                dimensions,
                values: Vec::new(),
                format: DataFormat::Hdf5(Hdf5Ref {
                    file: file.to_string(),
                    dataset,
                }),
            })
        } else {
            let text = node
                .text()
                .unwrap_or("")
                .split_whitespace()
                .map(|v| v.to_string())
                .collect::<Vec<_>>();
            Ok(DataItem {
                dimensions,
                values: text,
                format: DataFormat::Xml,
            })
        }
    }

    fn mixed_cell_type(code: i64) -> Result<(CellType, usize), MeshSieveError> {
        match code {
            1 => Ok((CellType::Vertex, 1)),
            2 => Ok((CellType::Segment, 2)),
            3 => Ok((CellType::Triangle, 3)),
            4 => Ok((CellType::Quadrilateral, 4)),
            5 => Ok((CellType::Tetrahedron, 4)),
            6 => Ok((CellType::Hexahedron, 8)),
            7 => Ok((CellType::Prism, 6)),
            8 => Ok((CellType::Pyramid, 5)),
            _ => Err(MeshSieveError::MeshIoParse(format!(
                "unsupported XDMF mixed cell code {code}"
            ))),
        }
    }
}

impl SieveSectionReader for XdmfReader {
    type Sieve = InMemorySieve<PointId, ()>;
    type Value = f64;
    type Storage = VecStorage<f64>;
    type CellStorage = VecStorage<CellType>;

    fn read<R: Read>(
        &self,
        mut reader: R,
    ) -> Result<MeshData<Self::Sieve, Self::Value, Self::Storage, Self::CellStorage>, MeshSieveError>
    {
        let mut input = String::new();
        reader.read_to_string(&mut input)?;
        let doc = Document::parse(&input)
            .map_err(|err| MeshSieveError::MeshIoParse(format!("XML parse error: {err}")))?;

        let grid = doc
            .descendants()
            .find(|n| n.has_tag_name("Grid"))
            .ok_or_else(|| MeshSieveError::MeshIoParse("missing Grid".into()))?;

        let topology = grid
            .descendants()
            .find(|n| n.has_tag_name("Topology"))
            .ok_or_else(|| MeshSieveError::MeshIoParse("missing Topology".into()))?;
        let topology_data = topology
            .descendants()
            .find(|n| n.has_tag_name("DataItem"))
            .ok_or_else(|| MeshSieveError::MeshIoParse("missing Topology DataItem".into()))?;
        let topology_item = Self::parse_data_item(topology_data)?;

        let geometry = grid
            .descendants()
            .find(|n| n.has_tag_name("Geometry"))
            .ok_or_else(|| MeshSieveError::MeshIoParse("missing Geometry".into()))?;
        let geometry_data = geometry
            .descendants()
            .find(|n| n.has_tag_name("DataItem"))
            .ok_or_else(|| MeshSieveError::MeshIoParse("missing Geometry DataItem".into()))?;
        let geometry_item = Self::parse_data_item(geometry_data)?;

        let mut attributes = HashMap::new();
        for attr in grid.descendants().filter(|n| n.has_tag_name("Attribute")) {
            if let Some(name) = attr.attribute("Name") {
                if let Some(data) = attr.descendants().find(|n| n.has_tag_name("DataItem")) {
                    let item = Self::parse_data_item(data)?;
                    attributes.insert(name.to_string(), item);
                }
            }
        }

        let geometry_dims = geometry_item.effective_dimensions()?;
        let num_points = geometry_dims.get(0).copied().unwrap_or(0);
        let point_ids = if let Some(item) = attributes.get(ATTR_POINT_IDS) {
            item.values_as_i64()?
                .into_iter()
                .map(|v| PointId::new(v as u64))
                .collect::<Result<Vec<_>, _>>()?
        } else {
            (0..num_points)
                .map(|idx| PointId::new((idx + 1) as u64))
                .collect::<Result<Vec<_>, _>>()?
        };

        let cell_ids = if let Some(item) = attributes.get(ATTR_CELL_IDS) {
            item.values_as_i64()?
                .into_iter()
                .map(|v| PointId::new(v as u64))
                .collect::<Result<Vec<_>, _>>()?
        } else {
            let num_cells = topology.attribute("NumberOfElements").and_then(|v| v.parse().ok());
            let count = num_cells.unwrap_or(0);
            let start = point_ids.len() as u64 + 1;
            (0..count)
                .map(|idx| PointId::new(start + idx as u64))
                .collect::<Result<Vec<_>, _>>()?
        };

        let topo_dim = attributes
            .get(ATTR_COORD_TOPO_DIM)
            .and_then(|item| item.values.get(0))
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(3);
        let embed_dim = attributes
            .get(ATTR_COORD_EMBED_DIM)
            .and_then(|item| item.values.get(0))
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(3);

        let mut coord_atlas = Atlas::default();
        for point in &point_ids {
            coord_atlas.try_insert(*point, embed_dim)?;
        }
        let mut coords = Coordinates::try_new(topo_dim, embed_dim, coord_atlas)?;
        let coord_values = geometry_item.values_as_f64()?;
        for (idx, point) in point_ids.iter().enumerate() {
            let offset = idx * 3;
            let slice = &coord_values[offset..offset + 3];
            coords
                .try_restrict_mut(*point)?
                .copy_from_slice(&slice[..embed_dim.min(3)]);
        }

        let mut sieve = InMemorySieve::<PointId, ()>::default();
        let mixed_values = topology_item.values_as_i64()?;
        let mut idx = 0usize;
        let mut cell_types = Vec::new();
        let mut connectivity = Vec::new();
        while idx < mixed_values.len() {
            let code = mixed_values[idx];
            idx += 1;
            let (cell_type, count) = Self::mixed_cell_type(code)?;
            let mut cell_conn = Vec::with_capacity(count);
            for _ in 0..count {
                let point_idx = mixed_values
                    .get(idx)
                    .ok_or_else(|| MeshSieveError::MeshIoParse("mixed connectivity underflow".into()))?;
                idx += 1;
                cell_conn.push(*point_idx as usize);
            }
            cell_types.push(cell_type);
            connectivity.push(cell_conn);
        }
        for (cell_idx, cell) in cell_ids.iter().enumerate() {
            let conn = &connectivity[cell_idx];
            for &point_idx in conn {
                let point = point_ids
                    .get(point_idx)
                    .ok_or_else(|| MeshSieveError::MeshIoParse("point index out of range".into()))?;
                sieve.add_arrow(*cell, *point, ());
            }
        }

        let mut cell_atlas = Atlas::default();
        for cell in &cell_ids {
            cell_atlas.try_insert(*cell, 1)?;
        }
        let mut cell_section = Section::<CellType, VecStorage<CellType>>::new(cell_atlas);
        for (cell_idx, cell) in cell_ids.iter().enumerate() {
            let cell_type = cell_types
                .get(cell_idx)
                .copied()
                .ok_or_else(|| MeshSieveError::MeshIoParse("cell type index out of range".into()))?;
            cell_section.try_set(*cell, &[cell_type])?;
        }

        let mut sections = BTreeMap::new();
        let mut labels = LabelSet::new();
        for (name, item) in &attributes {
            if let Some(rest) = name.strip_prefix(ATTR_SECTION_PREFIX) {
                if let Some(section_name) = rest.strip_suffix(":ids") {
                    let values_item = attributes
                        .get(&format!("{ATTR_SECTION_PREFIX}{section_name}:values"))
                        .ok_or_else(|| {
                            MeshSieveError::MeshIoParse(format!(
                                "missing values for section {section_name}"
                            ))
                        })?;
                    let ids = item.values_as_i64()?;
                    let values = values_item.values_as_f64()?;
                    let value_dims = values_item.effective_dimensions()?;
                    let num_components = value_dims.get(1).copied().unwrap_or(1);
                    let mut atlas = Atlas::default();
                    for raw_id in &ids {
                        let point = PointId::new(*raw_id as u64)?;
                        atlas.try_insert(point, num_components)?;
                    }
                    let mut section = Section::<f64, VecStorage<f64>>::new(atlas);
                    for (idx, raw_id) in ids.iter().enumerate() {
                        let point = PointId::new(*raw_id as u64)?;
                        let start = idx * num_components;
                        let end = start + num_components;
                        section.try_set(point, &values[start..end])?;
                    }
                    sections.insert(section_name.to_string(), section);
                }
            }
            if let Some(rest) = name.strip_prefix(ATTR_LABEL_PREFIX) {
                if let Some(label_name) = rest.strip_suffix(":ids") {
                    let values_item = attributes
                        .get(&format!("{ATTR_LABEL_PREFIX}{label_name}:values"))
                        .ok_or_else(|| {
                            MeshSieveError::MeshIoParse(format!(
                                "missing label values for {label_name}"
                            ))
                        })?;
                    let ids = item.values_as_i64()?;
                    let values = values_item.values_as_i64()?;
                    for (raw_id, value) in ids.iter().zip(values.iter()) {
                        let point = PointId::new(*raw_id as u64)?;
                        labels.set_label(point, label_name, *value as i32);
                    }
                }
            }
        }

        let labels = if labels.is_empty() { None } else { Some(labels) };

        Ok(MeshData {
            sieve,
            coordinates: Some(coords),
            sections,
            mixed_sections: Default::default(),
            labels,
            cell_types: Some(cell_section),
            discretization: None,
        })
    }
}
