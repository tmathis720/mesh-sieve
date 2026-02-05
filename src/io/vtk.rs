//! Legacy VTK (`.vtk`) reader/writer for unstructured grids.
//!
//! This implementation targets ASCII legacy VTK files with an
//! `UNSTRUCTURED_GRID` dataset and preserves mesh-sieve metadata via
//! `FIELD` data arrays.

use crate::data::atlas::Atlas;
use crate::data::coordinates::Coordinates;
use crate::data::section::Section;
use crate::data::storage::VecStorage;
use crate::io::{MeshData, SieveSectionReader, SieveSectionWriter};
use crate::mesh_error::MeshSieveError;
use crate::topology::cell_type::CellType;
use crate::topology::labels::LabelSet;
use crate::topology::point::PointId;
use crate::topology::sieve::{InMemorySieve, Sieve};
use std::collections::{BTreeMap, HashMap};
use std::io::{Read, Write};

const FIELD_POINT_IDS: &str = "mesh_sieve:point_ids";
const FIELD_CELL_IDS: &str = "mesh_sieve:cell_ids";
const FIELD_COORD_TOPO_DIM: &str = "mesh_sieve:coords:topo_dim";
const FIELD_COORD_EMBED_DIM: &str = "mesh_sieve:coords:embed_dim";
const FIELD_SECTION_PREFIX: &str = "mesh_sieve:section:";
const FIELD_LABEL_PREFIX: &str = "mesh_sieve:label:";

#[derive(Debug, Default, Clone)]
pub struct VtkReader;

#[derive(Debug, Default, Clone)]
pub struct VtkWriter;

impl VtkWriter {
    fn vtk_cell_type(cell_type: CellType) -> Option<i32> {
        match cell_type {
            CellType::Vertex => Some(1),
            CellType::Segment => Some(3),
            CellType::Triangle => Some(5),
            CellType::Quadrilateral => Some(9),
            CellType::Tetrahedron => Some(10),
            CellType::Hexahedron => Some(12),
            CellType::Prism => Some(13),
            CellType::Pyramid => Some(14),
            _ => None,
        }
    }

    fn write_field_array<W: Write>(
        writer: &mut W,
        name: &str,
        num_components: usize,
        num_tuples: usize,
        data_type: &str,
        values: &[String],
    ) -> Result<(), MeshSieveError> {
        writeln!(writer, "{name} {num_components} {num_tuples} {data_type}")?;
        let mut line_len = 0usize;
        for value in values {
            if line_len + value.len() + 1 > 70 {
                writeln!(writer)?;
                line_len = 0;
            }
            if line_len > 0 {
                write!(writer, " ")?;
                line_len += 1;
            }
            write!(writer, "{value}")?;
            line_len += value.len();
        }
        writeln!(writer)?;
        Ok(())
    }
}

impl SieveSectionWriter for VtkWriter {
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
            .ok_or_else(|| MeshSieveError::MeshIoParse("VTK requires coordinates".into()))?;
        let cell_types = mesh
            .cell_types
            .as_ref()
            .ok_or_else(|| MeshSieveError::MeshIoParse("VTK requires cell types".into()))?;

        let point_ids: Vec<PointId> = coords.section().atlas().points().collect();
        let mut point_index = HashMap::new();
        for (idx, point) in point_ids.iter().enumerate() {
            point_index.insert(*point, idx);
        }

        let cell_ids: Vec<PointId> = cell_types.atlas().points().collect();
        let mut vtk_cell_types = Vec::with_capacity(cell_ids.len());
        let mut connectivity = Vec::new();
        let mut total_size = 0usize;
        for cell in &cell_ids {
            let cell_type = cell_types.try_restrict(*cell)?[0];
            let vtk_type = Self::vtk_cell_type(cell_type).ok_or_else(|| {
                MeshSieveError::MeshIoParse(format!("unsupported VTK cell type {cell_type:?}"))
            })?;
            vtk_cell_types.push(vtk_type);
            let cone: Vec<PointId> = mesh.sieve.cone_points(*cell).collect();
            total_size += cone.len() + 1;
            connectivity.push(cone);
        }

        writeln!(writer, "# vtk DataFile Version 3.0")?;
        writeln!(writer, "mesh-sieve")?;
        writeln!(writer, "ASCII")?;
        writeln!(writer, "DATASET UNSTRUCTURED_GRID")?;
        writeln!(writer, "POINTS {} float", point_ids.len())?;
        for point in &point_ids {
            let coords_slice = coords.try_restrict(*point)?;
            let mut values = [0.0f64; 3];
            for (idx, value) in coords_slice.iter().enumerate().take(3) {
                values[idx] = *value;
            }
            writeln!(writer, "{} {} {}", values[0], values[1], values[2])?;
        }

        writeln!(writer, "CELLS {} {}", cell_ids.len(), total_size)?;
        for cone in &connectivity {
            write!(writer, "{}", cone.len())?;
            for point in cone {
                let idx = point_index.get(point).ok_or_else(|| {
                    MeshSieveError::MeshIoParse(format!("missing point {point:?} in coordinates"))
                })?;
                write!(writer, " {idx}")?;
            }
            writeln!(writer)?;
        }

        writeln!(writer, "CELL_TYPES {}", vtk_cell_types.len())?;
        for vtk_type in &vtk_cell_types {
            writeln!(writer, "{vtk_type}")?;
        }

        let mut fields: Vec<(String, usize, usize, String, Vec<String>)> = Vec::new();
        fields.push((
            FIELD_POINT_IDS.to_string(),
            1,
            point_ids.len(),
            "long".to_string(),
            point_ids.iter().map(|p| p.get().to_string()).collect(),
        ));
        fields.push((
            FIELD_CELL_IDS.to_string(),
            1,
            cell_ids.len(),
            "long".to_string(),
            cell_ids.iter().map(|c| c.get().to_string()).collect(),
        ));
        fields.push((
            FIELD_COORD_TOPO_DIM.to_string(),
            1,
            1,
            "int".to_string(),
            vec![coords.topological_dimension().to_string()],
        ));
        fields.push((
            FIELD_COORD_EMBED_DIM.to_string(),
            1,
            1,
            "int".to_string(),
            vec![coords.embedding_dimension().to_string()],
        ));

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
                for value in data {
                    values.push(value.to_string());
                }
            }
            fields.push((
                format!("{FIELD_SECTION_PREFIX}{name}:ids"),
                1,
                ids.len(),
                "long".to_string(),
                ids,
            ));
            fields.push((
                format!("{FIELD_SECTION_PREFIX}{name}:values"),
                num_components.max(1),
                values.len() / num_components.max(1),
                "double".to_string(),
                values,
            ));
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
                let ids: Vec<String> = entries.iter().map(|(p, _)| p.get().to_string()).collect();
                let values: Vec<String> = entries.iter().map(|(_, v)| v.to_string()).collect();
                fields.push((
                    format!("{FIELD_LABEL_PREFIX}{name}:ids"),
                    1,
                    ids.len(),
                    "long".to_string(),
                    ids,
                ));
                fields.push((
                    format!("{FIELD_LABEL_PREFIX}{name}:values"),
                    1,
                    values.len(),
                    "int".to_string(),
                    values,
                ));
            }
        }

        if !fields.is_empty() {
            writeln!(writer, "FIELD FieldData {}", fields.len())?;
            for (name, components, tuples, data_type, values) in fields {
                Self::write_field_array(
                    &mut writer,
                    &name,
                    components,
                    tuples,
                    &data_type,
                    &values,
                )?;
            }
        }

        Ok(())
    }
}

#[derive(Debug)]
struct FieldArray {
    components: usize,
    tuples: usize,
    values: Vec<String>,
    data_type: String,
}

impl FieldArray {
    fn values_as_i64(&self) -> Result<Vec<i64>, MeshSieveError> {
        self.values
            .iter()
            .map(|v| {
                v.parse::<i64>()
                    .map_err(|_| MeshSieveError::MeshIoParse(format!("invalid int value {v}")))
            })
            .collect()
    }

    fn values_as_f64(&self) -> Result<Vec<f64>, MeshSieveError> {
        self.values
            .iter()
            .map(|v| {
                v.parse::<f64>()
                    .map_err(|_| MeshSieveError::MeshIoParse(format!("invalid float value {v}")))
            })
            .collect()
    }
}

impl VtkReader {
    fn vtk_cell_type(cell_type: i32) -> Result<CellType, MeshSieveError> {
        match cell_type {
            1 => Ok(CellType::Vertex),
            3 => Ok(CellType::Segment),
            5 => Ok(CellType::Triangle),
            9 => Ok(CellType::Quadrilateral),
            10 => Ok(CellType::Tetrahedron),
            12 => Ok(CellType::Hexahedron),
            13 => Ok(CellType::Prism),
            14 => Ok(CellType::Pyramid),
            _ => Err(MeshSieveError::MeshIoParse(format!(
                "unsupported VTK cell type {cell_type}"
            ))),
        }
    }

    fn parse_fields(
        tokens: &mut impl Iterator<Item = String>,
    ) -> Result<HashMap<String, FieldArray>, MeshSieveError> {
        let mut fields = HashMap::new();
        while let Some(token) = tokens.next() {
            match token.as_str() {
                "POINT_DATA" | "CELL_DATA" => {
                    tokens.next();
                    if let Some(next) = tokens.next() {
                        if next != "FIELD" {
                            return Err(MeshSieveError::MeshIoParse(
                                "expected FIELD after POINT_DATA/CELL_DATA".into(),
                            ));
                        }
                    } else {
                        break;
                    }
                    let _field_name = tokens.next();
                    let num_arrays: usize = tokens
                        .next()
                        .ok_or_else(|| MeshSieveError::MeshIoParse("missing field count".into()))?
                        .parse()
                        .map_err(|_| MeshSieveError::MeshIoParse("invalid field count".into()))?;
                    for _ in 0..num_arrays {
                        let name = tokens.next().ok_or_else(|| {
                            MeshSieveError::MeshIoParse("missing field name".into())
                        })?;
                        let components: usize = tokens
                            .next()
                            .ok_or_else(|| {
                                MeshSieveError::MeshIoParse("missing field components".into())
                            })?
                            .parse()
                            .map_err(|_| {
                                MeshSieveError::MeshIoParse("invalid field components".into())
                            })?;
                        let tuples: usize = tokens
                            .next()
                            .ok_or_else(|| {
                                MeshSieveError::MeshIoParse("missing field tuples".into())
                            })?
                            .parse()
                            .map_err(|_| {
                                MeshSieveError::MeshIoParse("invalid field tuples".into())
                            })?;
                        let data_type = tokens.next().ok_or_else(|| {
                            MeshSieveError::MeshIoParse("missing field type".into())
                        })?;
                        let total = components * tuples;
                        let mut values = Vec::with_capacity(total);
                        for _ in 0..total {
                            values.push(tokens.next().ok_or_else(|| {
                                MeshSieveError::MeshIoParse("missing field data values".into())
                            })?);
                        }
                        fields.insert(
                            name,
                            FieldArray {
                                components,
                                tuples,
                                values,
                                data_type,
                            },
                        );
                    }
                }
                "FIELD" => {
                    let _field_name = tokens.next();
                    let num_arrays: usize = tokens
                        .next()
                        .ok_or_else(|| MeshSieveError::MeshIoParse("missing field count".into()))?
                        .parse()
                        .map_err(|_| MeshSieveError::MeshIoParse("invalid field count".into()))?;
                    for _ in 0..num_arrays {
                        let name = tokens.next().ok_or_else(|| {
                            MeshSieveError::MeshIoParse("missing field name".into())
                        })?;
                        let components: usize = tokens
                            .next()
                            .ok_or_else(|| {
                                MeshSieveError::MeshIoParse("missing field components".into())
                            })?
                            .parse()
                            .map_err(|_| {
                                MeshSieveError::MeshIoParse("invalid field components".into())
                            })?;
                        let tuples: usize = tokens
                            .next()
                            .ok_or_else(|| {
                                MeshSieveError::MeshIoParse("missing field tuples".into())
                            })?
                            .parse()
                            .map_err(|_| {
                                MeshSieveError::MeshIoParse("invalid field tuples".into())
                            })?;
                        let data_type = tokens.next().ok_or_else(|| {
                            MeshSieveError::MeshIoParse("missing field type".into())
                        })?;
                        let total = components * tuples;
                        let mut values = Vec::with_capacity(total);
                        for _ in 0..total {
                            values.push(tokens.next().ok_or_else(|| {
                                MeshSieveError::MeshIoParse("missing field data values".into())
                            })?);
                        }
                        fields.insert(
                            name,
                            FieldArray {
                                components,
                                tuples,
                                values,
                                data_type,
                            },
                        );
                    }
                }
                _ => {
                    return Err(MeshSieveError::MeshIoParse(format!(
                        "unexpected token {token}"
                    )));
                }
            }
        }
        Ok(fields)
    }
}

impl SieveSectionReader for VtkReader {
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
        let mut lines = input.lines();
        let _version = lines.next();
        let _comment = lines.next();
        let format = lines
            .next()
            .ok_or_else(|| MeshSieveError::MeshIoParse("missing ASCII line".into()))?;
        if format.trim() != "ASCII" {
            return Err(MeshSieveError::MeshIoParse(
                "VTK ASCII format required".into(),
            ));
        }
        let dataset = lines
            .next()
            .ok_or_else(|| MeshSieveError::MeshIoParse("missing DATASET line".into()))?;
        if !dataset.trim().ends_with("UNSTRUCTURED_GRID") {
            return Err(MeshSieveError::MeshIoParse(
                "VTK UNSTRUCTURED_GRID required".into(),
            ));
        }

        let remaining: String = lines.collect::<Vec<_>>().join("\n");
        let mut tokens = remaining
            .split_whitespace()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .into_iter();

        let points_token = tokens
            .next()
            .ok_or_else(|| MeshSieveError::MeshIoParse("missing POINTS".into()))?;
        if points_token != "POINTS" {
            return Err(MeshSieveError::MeshIoParse(
                "expected POINTS section".into(),
            ));
        }
        let num_points: usize = tokens
            .next()
            .ok_or_else(|| MeshSieveError::MeshIoParse("missing point count".into()))?
            .parse()
            .map_err(|_| MeshSieveError::MeshIoParse("invalid point count".into()))?;
        let _point_type = tokens
            .next()
            .ok_or_else(|| MeshSieveError::MeshIoParse("missing point type".into()))?;
        let mut point_coords = Vec::with_capacity(num_points * 3);
        for _ in 0..num_points * 3 {
            let value = tokens
                .next()
                .ok_or_else(|| MeshSieveError::MeshIoParse("missing point value".into()))?;
            let coord = value
                .parse::<f64>()
                .map_err(|_| MeshSieveError::MeshIoParse("invalid point value".into()))?;
            point_coords.push(coord);
        }

        let cells_token = tokens
            .next()
            .ok_or_else(|| MeshSieveError::MeshIoParse("missing CELLS".into()))?;
        if cells_token != "CELLS" {
            return Err(MeshSieveError::MeshIoParse("expected CELLS section".into()));
        }
        let num_cells: usize = tokens
            .next()
            .ok_or_else(|| MeshSieveError::MeshIoParse("missing cell count".into()))?
            .parse()
            .map_err(|_| MeshSieveError::MeshIoParse("invalid cell count".into()))?;
        let _total_size = tokens
            .next()
            .ok_or_else(|| MeshSieveError::MeshIoParse("missing cell size".into()))?;
        let mut connectivity = Vec::with_capacity(num_cells);
        for _ in 0..num_cells {
            let count: usize = tokens
                .next()
                .ok_or_else(|| MeshSieveError::MeshIoParse("missing cell size".into()))?
                .parse()
                .map_err(|_| MeshSieveError::MeshIoParse("invalid cell size".into()))?;
            let mut cell_conn = Vec::with_capacity(count);
            for _ in 0..count {
                let idx: usize = tokens
                    .next()
                    .ok_or_else(|| MeshSieveError::MeshIoParse("missing cell index".into()))?
                    .parse()
                    .map_err(|_| MeshSieveError::MeshIoParse("invalid cell index".into()))?;
                cell_conn.push(idx);
            }
            connectivity.push(cell_conn);
        }

        let cell_types_token = tokens
            .next()
            .ok_or_else(|| MeshSieveError::MeshIoParse("missing CELL_TYPES".into()))?;
        if cell_types_token != "CELL_TYPES" {
            return Err(MeshSieveError::MeshIoParse(
                "expected CELL_TYPES section".into(),
            ));
        }
        let cell_types_count: usize = tokens
            .next()
            .ok_or_else(|| MeshSieveError::MeshIoParse("missing cell types count".into()))?
            .parse()
            .map_err(|_| MeshSieveError::MeshIoParse("invalid cell types count".into()))?;
        let mut vtk_cell_types = Vec::with_capacity(cell_types_count);
        for _ in 0..cell_types_count {
            let value = tokens
                .next()
                .ok_or_else(|| MeshSieveError::MeshIoParse("missing cell type".into()))?;
            let cell_type = value
                .parse::<i32>()
                .map_err(|_| MeshSieveError::MeshIoParse("invalid cell type".into()))?;
            vtk_cell_types.push(cell_type);
        }

        let fields = if tokens.len() > 0 {
            Self::parse_fields(&mut tokens)?
        } else {
            HashMap::new()
        };

        let point_ids = if let Some(field) = fields.get(FIELD_POINT_IDS) {
            field
                .values_as_i64()?
                .into_iter()
                .map(|v| PointId::new(v as u64))
                .collect::<Result<Vec<_>, _>>()?
        } else {
            (0..num_points)
                .map(|idx| PointId::new((idx + 1) as u64))
                .collect::<Result<Vec<_>, _>>()?
        };

        let cell_ids = if let Some(field) = fields.get(FIELD_CELL_IDS) {
            field
                .values_as_i64()?
                .into_iter()
                .map(|v| PointId::new(v as u64))
                .collect::<Result<Vec<_>, _>>()?
        } else {
            let start = point_ids.len() as u64 + 1;
            (0..num_cells)
                .map(|idx| PointId::new(start + idx as u64))
                .collect::<Result<Vec<_>, _>>()?
        };

        let topo_dim = fields
            .get(FIELD_COORD_TOPO_DIM)
            .and_then(|field| field.values.get(0))
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(3);
        let embed_dim = fields
            .get(FIELD_COORD_EMBED_DIM)
            .and_then(|field| field.values.get(0))
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(3);

        let mut coord_atlas = Atlas::default();
        for point in &point_ids {
            coord_atlas.try_insert(*point, embed_dim)?;
        }
        let mut coords = Coordinates::try_new(topo_dim, embed_dim, coord_atlas)?;
        for (idx, point) in point_ids.iter().enumerate() {
            let offset = idx * 3;
            let slice = &point_coords[offset..offset + 3];
            coords
                .try_restrict_mut(*point)?
                .copy_from_slice(&slice[..embed_dim.min(3)]);
        }

        let mut sieve = InMemorySieve::<PointId, ()>::default();
        for (cell_idx, cell) in cell_ids.iter().enumerate() {
            let conn = &connectivity[cell_idx];
            for &point_idx in conn {
                let point = point_ids.get(point_idx).ok_or_else(|| {
                    MeshSieveError::MeshIoParse("point index out of range".into())
                })?;
                sieve.add_arrow(*cell, *point, ());
            }
        }

        let mut cell_atlas = Atlas::default();
        for cell in &cell_ids {
            cell_atlas.try_insert(*cell, 1)?;
        }
        let mut cell_section = Section::<CellType, VecStorage<CellType>>::new(cell_atlas);
        for (cell_idx, cell) in cell_ids.iter().enumerate() {
            let cell_type = Self::vtk_cell_type(vtk_cell_types[cell_idx])?;
            cell_section.try_set(*cell, &[cell_type])?;
        }

        let mut sections = BTreeMap::new();
        let mut labels = LabelSet::new();

        for (name, field) in &fields {
            if let Some(rest) = name.strip_prefix(FIELD_SECTION_PREFIX) {
                if let Some(section_name) = rest.strip_suffix(":ids") {
                    let values_field = fields
                        .get(&format!("{FIELD_SECTION_PREFIX}{section_name}:values"))
                        .ok_or_else(|| {
                            MeshSieveError::MeshIoParse(format!(
                                "missing values for section {section_name}"
                            ))
                        })?;
                    let ids = field.values_as_i64()?;
                    let values = values_field.values_as_f64()?;
                    let num_components = values_field.components;
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
            if let Some(rest) = name.strip_prefix(FIELD_LABEL_PREFIX) {
                if let Some(label_name) = rest.strip_suffix(":ids") {
                    let values_field = fields
                        .get(&format!("{FIELD_LABEL_PREFIX}{label_name}:values"))
                        .ok_or_else(|| {
                            MeshSieveError::MeshIoParse(format!(
                                "missing label values for {label_name}"
                            ))
                        })?;
                    let ids = field.values_as_i64()?;
                    let values = values_field.values_as_i64()?;
                    for (point_raw, value) in ids.iter().zip(values.iter()) {
                        let point = PointId::new(*point_raw as u64)?;
                        labels.set_label(point, label_name, *value as i32);
                    }
                }
            }
        }

        let labels = if labels.is_empty() {
            None
        } else {
            Some(labels)
        };

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
