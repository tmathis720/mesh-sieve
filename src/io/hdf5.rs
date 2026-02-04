//! HDF5-based mesh reader/writer.

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
use hdf5::File;
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

pub(crate) const GROUP_TOPOLOGY: &str = "topology";
pub(crate) const GROUP_GEOMETRY: &str = "geometry";
pub(crate) const GROUP_SECTIONS: &str = "sections";
pub(crate) const GROUP_LABELS: &str = "labels";

pub(crate) const DATASET_POINT_IDS: &str = "point_ids";
pub(crate) const DATASET_CELL_IDS: &str = "cell_ids";
pub(crate) const DATASET_CELL_TYPES: &str = "cell_types";
pub(crate) const DATASET_CELL_OFFSETS: &str = "cell_offsets";
pub(crate) const DATASET_CELLS: &str = "cells";
pub(crate) const DATASET_MIXED: &str = "mixed";
pub(crate) const DATASET_COORDINATES: &str = "coordinates";
pub(crate) const DATASET_TOPO_DIM: &str = "topological_dimension";
pub(crate) const DATASET_EMBED_DIM: &str = "embedding_dimension";
pub(crate) const DATASET_SECTION_IDS: &str = "ids";
pub(crate) const DATASET_SECTION_VALUES: &str = "values";
pub(crate) const DATASET_SECTION_COMPONENTS: &str = "num_components";

/// HDF5 reader.
#[derive(Debug, Default, Clone)]
pub struct Hdf5Reader;

/// HDF5 writer.
#[derive(Debug, Default, Clone)]
pub struct Hdf5Writer;

impl SieveSectionReader for Hdf5Reader {
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
        let temp_path = write_temp_file(&bytes)?;
        let file = File::open(&temp_path)
            .map_err(|err| MeshSieveError::MeshIoParse(format!("HDF5 open error: {err}")))?;
        let result = read_mesh_from_hdf5(&file);
        let _ = fs::remove_file(&temp_path);
        result
    }
}

impl SieveSectionWriter for Hdf5Writer {
    type Sieve = InMemorySieve<PointId, ()>;
    type Value = f64;
    type Storage = VecStorage<f64>;
    type CellStorage = VecStorage<CellType>;

    fn write<W: Write>(
        &self,
        mut writer: W,
        mesh: &MeshData<Self::Sieve, Self::Value, Self::Storage, Self::CellStorage>,
    ) -> Result<(), MeshSieveError> {
        let temp_path = temp_hdf5_path();
        let file = File::create(&temp_path)
            .map_err(|err| MeshSieveError::MeshIoParse(format!("HDF5 create error: {err}")))?;
        write_mesh_to_hdf5(&file, mesh)?;
        let bytes = fs::read(&temp_path)?;
        writer.write_all(&bytes)?;
        let _ = fs::remove_file(&temp_path);
        Ok(())
    }
}

fn write_temp_file(bytes: &[u8]) -> Result<PathBuf, MeshSieveError> {
    let path = temp_hdf5_path();
    fs::write(&path, bytes)?;
    Ok(path)
}

fn temp_hdf5_path() -> PathBuf {
    let mut path = std::env::temp_dir();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|v| v.as_nanos())
        .unwrap_or(0);
    let pid = std::process::id();
    path.push(format!("mesh_sieve_{pid}_{nanos}.h5"));
    path
}

fn cell_type_to_dmplex(cell_type: CellType) -> Result<i32, MeshSieveError> {
    match cell_type {
        CellType::Vertex => Ok(0),
        CellType::Segment => Ok(1),
        CellType::Triangle => Ok(2),
        CellType::Quadrilateral => Ok(3),
        CellType::Tetrahedron => Ok(4),
        CellType::Hexahedron => Ok(5),
        CellType::Prism => Ok(6),
        CellType::Pyramid => Ok(7),
        _ => Err(MeshSieveError::MeshIoParse(format!(
            "unsupported cell type for HDF5: {cell_type:?}"
        ))),
    }
}

fn dmplex_to_cell_type(code: i32) -> Result<CellType, MeshSieveError> {
    match code {
        0 => Ok(CellType::Vertex),
        1 => Ok(CellType::Segment),
        2 => Ok(CellType::Triangle),
        3 => Ok(CellType::Quadrilateral),
        4 => Ok(CellType::Tetrahedron),
        5 => Ok(CellType::Hexahedron),
        6 => Ok(CellType::Prism),
        7 => Ok(CellType::Pyramid),
        _ => Err(MeshSieveError::MeshIoParse(format!(
            "unknown DMPlex cell type code {code}"
        ))),
    }
}

pub(crate) fn write_mesh_to_hdf5<S>(
    file: &File,
    mesh: &MeshData<S, f64, VecStorage<f64>, VecStorage<CellType>>,
) -> Result<(), MeshSieveError>
where
    S: Sieve<Point = PointId>,
{
    let coords = mesh
        .coordinates
        .as_ref()
        .ok_or_else(|| MeshSieveError::MeshIoParse("HDF5 requires coordinates".into()))?;
    let cell_types = mesh
        .cell_types
        .as_ref()
        .ok_or_else(|| MeshSieveError::MeshIoParse("HDF5 requires cell types".into()))?;

    let point_ids: Vec<PointId> = coords.section().atlas().points().collect();
    let cell_ids: Vec<PointId> = cell_types.atlas().points().collect();
    let mut point_index = std::collections::HashMap::new();
    for (idx, point) in point_ids.iter().enumerate() {
        point_index.insert(*point, idx);
    }

    let topo_group = file
        .create_group(GROUP_TOPOLOGY)
        .map_err(|err| MeshSieveError::MeshIoParse(format!("HDF5 create topology: {err}")))?;
    let geom_group = file
        .create_group(GROUP_GEOMETRY)
        .map_err(|err| MeshSieveError::MeshIoParse(format!("HDF5 create geometry: {err}")))?;
    let sections_group = file
        .create_group(GROUP_SECTIONS)
        .map_err(|err| MeshSieveError::MeshIoParse(format!("HDF5 create sections: {err}")))?;
    let labels_group = file
        .create_group(GROUP_LABELS)
        .map_err(|err| MeshSieveError::MeshIoParse(format!("HDF5 create labels: {err}")))?;

    let point_id_values: Vec<i64> = point_ids.iter().map(|p| p.get() as i64).collect();
    let cell_id_values: Vec<i64> = cell_ids.iter().map(|c| c.get() as i64).collect();

    geom_group
        .new_dataset::<i64>()
        .shape(point_id_values.len())
        .create(DATASET_POINT_IDS)?
        .write(&point_id_values)?;
    topo_group
        .new_dataset::<i64>()
        .shape(cell_id_values.len())
        .create(DATASET_CELL_IDS)?
        .write(&cell_id_values)?;
    topo_group
        .new_dataset::<i64>()
        .shape(point_id_values.len())
        .create(DATASET_POINT_IDS)?
        .write(&point_id_values)?;

    let topo_dim = coords.topological_dimension() as i32;
    let embed_dim = coords.embedding_dimension() as i32;
    geom_group
        .new_dataset::<i32>()
        .shape(1)
        .create(DATASET_TOPO_DIM)?
        .write(&[topo_dim])?;
    geom_group
        .new_dataset::<i32>()
        .shape(1)
        .create(DATASET_EMBED_DIM)?
        .write(&[embed_dim])?;

    let mut coord_values = Vec::with_capacity(point_ids.len() * 3);
    for point in &point_ids {
        let slice = coords.try_restrict(*point)?;
        let mut values = [0.0f64; 3];
        for (idx, value) in slice.iter().enumerate().take(3) {
            values[idx] = *value;
        }
        coord_values.extend(values);
    }
    geom_group
        .new_dataset::<f64>()
        .shape((point_ids.len(), 3))
        .create(DATASET_COORDINATES)?
        .write(&coord_values)?;

    let mut cell_type_values = Vec::with_capacity(cell_ids.len());
    let mut offsets = Vec::with_capacity(cell_ids.len() + 1);
    let mut cells = Vec::new();
    let mut mixed = Vec::new();
    offsets.push(0i64);
    for cell in &cell_ids {
        let cell_type = cell_types.try_restrict(*cell)?[0];
        cell_type_values.push(cell_type_to_dmplex(cell_type)?);
        let cone: Vec<PointId> = mesh.sieve.cone_points(*cell).collect();
        for point in &cone {
            cells.push(point.get() as i64);
        }
        if let Some((code, count)) = dmplex_to_xdmf(cell_type) {
            mixed.push(code);
            if cone.len() != count {
                return Err(MeshSieveError::MeshIoParse(format!(
                    "cell {cell:?} expected {count} vertices, found {}",
                    cone.len()
                )));
            }
            for point in &cone {
                let idx = point_index.get(point).ok_or_else(|| {
                    MeshSieveError::MeshIoParse(format!("missing point {point:?}"))
                })?;
                mixed.push(*idx as i64);
            }
        }
        offsets.push(cells.len() as i64);
    }

    topo_group
        .new_dataset::<i32>()
        .shape(cell_type_values.len())
        .create(DATASET_CELL_TYPES)?
        .write(&cell_type_values)?;
    topo_group
        .new_dataset::<i64>()
        .shape(offsets.len())
        .create(DATASET_CELL_OFFSETS)?
        .write(&offsets)?;
    topo_group
        .new_dataset::<i64>()
        .shape(cells.len())
        .create(DATASET_CELLS)?
        .write(&cells)?;
    if !mixed.is_empty() {
        topo_group
            .new_dataset::<i64>()
            .shape(mixed.len())
            .create(DATASET_MIXED)?
            .write(&mixed)?;
    }

    for (name, section) in &mesh.sections {
        let group = sections_group
            .create_group(name)
            .map_err(|err| MeshSieveError::MeshIoParse(format!("section {name}: {err}")))?;
        let ids: Vec<i64> = section
            .atlas()
            .points()
            .map(|p| p.get() as i64)
            .collect();
        let mut values = Vec::new();
        let mut num_components = 0usize;
        for (_point, data) in section.iter() {
            if num_components == 0 {
                num_components = data.len();
            }
            values.extend_from_slice(data);
        }
        group
            .new_dataset::<i64>()
            .shape(ids.len())
            .create(DATASET_SECTION_IDS)?
            .write(&ids)?;
        let dims = if num_components == 0 {
            (0usize, 0usize)
        } else {
            (ids.len(), num_components)
        };
        group
            .new_dataset::<f64>()
            .shape(dims)
            .create(DATASET_SECTION_VALUES)?
            .write(&values)?;
        group
            .new_dataset::<i32>()
            .shape(1)
            .create(DATASET_SECTION_COMPONENTS)?
            .write(&[num_components as i32])?;
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
            let group = labels_group
                .create_group(&name)
                .map_err(|err| MeshSieveError::MeshIoParse(format!("label {name}: {err}")))?;
            let ids: Vec<i64> = entries.iter().map(|(p, _)| p.get() as i64).collect();
            let values: Vec<i32> = entries.iter().map(|(_, v)| *v).collect();
            group
                .new_dataset::<i64>()
                .shape(ids.len())
                .create(DATASET_SECTION_IDS)?
                .write(&ids)?;
            group
                .new_dataset::<i32>()
                .shape(values.len())
                .create(DATASET_SECTION_VALUES)?
                .write(&values)?;
        }
    }

    Ok(())
}

pub(crate) fn read_mesh_from_hdf5(
    file: &File,
) -> Result<
    MeshData<InMemorySieve<PointId, ()>, f64, VecStorage<f64>, VecStorage<CellType>>,
    MeshSieveError,
> {
    let topo_group = file.group(GROUP_TOPOLOGY).map_err(|err| {
        MeshSieveError::MeshIoParse(format!("HDF5 missing topology group: {err}"))
    })?;
    let geom_group = file.group(GROUP_GEOMETRY).map_err(|err| {
        MeshSieveError::MeshIoParse(format!("HDF5 missing geometry group: {err}"))
    })?;

    let point_ids: Vec<i64> = geom_group.dataset(DATASET_POINT_IDS)?.read_raw()?;
    let cell_ids: Vec<i64> = topo_group.dataset(DATASET_CELL_IDS)?.read_raw()?;
    let topo_dim_values: Vec<i32> = geom_group.dataset(DATASET_TOPO_DIM)?.read_raw()?;
    let embed_dim_values: Vec<i32> = geom_group.dataset(DATASET_EMBED_DIM)?.read_raw()?;
    let topo_dim = topo_dim_values.first().copied().unwrap_or(3) as usize;
    let embed_dim = embed_dim_values.first().copied().unwrap_or(3) as usize;

    let coord_dataset = geom_group.dataset(DATASET_COORDINATES)?;
    let coord_values: Vec<f64> = coord_dataset.read_raw()?;
    let point_ids: Vec<PointId> = point_ids
        .into_iter()
        .map(|v| PointId::new(v as u64))
        .collect::<Result<Vec<_>, _>>()?;
    let cell_ids: Vec<PointId> = cell_ids
        .into_iter()
        .map(|v| PointId::new(v as u64))
        .collect::<Result<Vec<_>, _>>()?;

    let mut coord_atlas = Atlas::default();
    for point in &point_ids {
        coord_atlas.try_insert(*point, embed_dim)?;
    }
    let mut coords = Coordinates::try_new(topo_dim, embed_dim, coord_atlas)?;
    for (idx, point) in point_ids.iter().enumerate() {
        let offset = idx * 3;
        let slice = &coord_values[offset..offset + 3];
        coords
            .try_restrict_mut(*point)?
            .copy_from_slice(&slice[..embed_dim.min(3)]);
    }

    let offsets: Vec<i64> = topo_group.dataset(DATASET_CELL_OFFSETS)?.read_raw()?;
    let cells: Vec<i64> = topo_group.dataset(DATASET_CELLS)?.read_raw()?;
    let cell_type_codes: Vec<i32> = topo_group.dataset(DATASET_CELL_TYPES)?.read_raw()?;

    let mut sieve = InMemorySieve::<PointId, ()>::default();
    for (cell_idx, cell) in cell_ids.iter().enumerate() {
        let start = offsets
            .get(cell_idx)
            .copied()
            .unwrap_or(0) as usize;
        let end = offsets
            .get(cell_idx + 1)
            .copied()
            .unwrap_or(start as i64) as usize;
        for raw_id in &cells[start..end] {
            let point = PointId::new(*raw_id as u64)?;
            sieve.add_arrow(*cell, point, ());
        }
    }

    let mut cell_atlas = Atlas::default();
    for cell in &cell_ids {
        cell_atlas.try_insert(*cell, 1)?;
    }
    let mut cell_section = Section::<CellType, VecStorage<CellType>>::new(cell_atlas);
    for (cell_idx, cell) in cell_ids.iter().enumerate() {
        let code = cell_type_codes
            .get(cell_idx)
            .copied()
            .ok_or_else(|| MeshSieveError::MeshIoParse("cell type index out of range".into()))?;
        let cell_type = dmplex_to_cell_type(code)?;
        cell_section.try_set(*cell, &[cell_type])?;
    }

    let mut sections = BTreeMap::new();
    if let Ok(section_group) = file.group(GROUP_SECTIONS) {
        for name in section_group.member_names()? {
            let group = section_group.group(&name)?;
            let ids: Vec<i64> = group.dataset(DATASET_SECTION_IDS)?.read_raw()?;
            let values: Vec<f64> = group.dataset(DATASET_SECTION_VALUES)?.read_raw()?;
            let components: Vec<i32> = group.dataset(DATASET_SECTION_COMPONENTS)?.read_raw()?;
            let num_components = components.first().copied().unwrap_or(1) as usize;
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
                if end > values.len() {
                    return Err(MeshSieveError::MeshIoParse(format!(
                        "section {name} values underflow"
                    )));
                }
                section.try_set(point, &values[start..end])?;
            }
            sections.insert(name, section);
        }
    }

    let mut labels = LabelSet::new();
    if let Ok(label_group) = file.group(GROUP_LABELS) {
        for name in label_group.member_names()? {
            let group = label_group.group(&name)?;
            let ids: Vec<i64> = group.dataset(DATASET_SECTION_IDS)?.read_raw()?;
            let values: Vec<i32> = group.dataset(DATASET_SECTION_VALUES)?.read_raw()?;
            for (raw_id, value) in ids.iter().zip(values.iter()) {
                let point = PointId::new(*raw_id as u64)?;
                labels.set_label(point, &name, *value);
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

fn dmplex_to_xdmf(cell_type: CellType) -> Option<(i64, usize)> {
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
