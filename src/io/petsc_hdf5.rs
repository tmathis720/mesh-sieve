//! PETSc DMPlex HDF5 v3-compatible mesh, section, label, and vector I/O.
//!
//! This module intentionally lives alongside (rather than replacing)
//! [`crate::io::hdf5`].  The legacy module stores a compact mesh-sieve layout;
//! this module mirrors PETSc's modern DMPlex hierarchy closely enough for
//! mesh/section/vector data to be written, loaded independently, and
//! re-associated by the preserved point permutation/order.

use crate::algs::point_sf::PointSF;
use crate::data::atlas::Atlas;
use crate::data::section::Section;
use crate::data::storage::VecStorage;
use crate::io::{MeshData, SieveSectionReader, SieveSectionWriter};
use crate::mesh_error::MeshSieveError;
use crate::topology::cell_type::CellType;
use crate::topology::labels::LabelSet;
use crate::topology::point::PointId;
use crate::topology::sieve::strata::compute_strata;
use crate::topology::sieve::{MeshSieve, OrientedSieve, Sieve};
use hdf5::{File, Group};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

/// Compose load/redistribute/section SF maps for DMPlex-style migration workflows.
pub fn compose_migration_pipeline<C>(
    load_map: &PointSF<'_, C>,
    redistribute_map: &PointSF<'_, C>,
    section_map: &PointSF<'_, C>,
) -> PointSF<'static, C>
where
    C: crate::algs::communicator::Communicator + Sync,
{
    let topo = load_map.compose(redistribute_map);
    topo.compose(section_map)
}

/// PETSc DMPlex HDF5 storage version written by this module.
pub const DMPLEX_STORAGE_VERSION: i32 = 3;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PetscLoadMode {
    Strict,
    Permissive,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PetscLoadFilter {
    pub section_name_patterns: Vec<String>,
    pub label_subset: Option<(String, i32)>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PetscLoadOptions {
    pub mode: PetscLoadMode,
    pub filter: PetscLoadFilter,
}

impl Default for PetscLoadOptions {
    fn default() -> Self {
        Self {
            mode: PetscLoadMode::Strict,
            filter: PetscLoadFilter::default(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PetscProvenance {
    pub storage_version: i32,
    pub permutation_source: String,
    pub redistribution_map_id: Option<String>,
}

const GROUP_TOPOLOGIES: &str = "topologies";
const GROUP_TOPOLOGY: &str = "topology";
const GROUP_STRATA: &str = "strata";
const GROUP_LABELS: &str = "labels";
const GROUP_DMS: &str = "dms";
const GROUP_SECTION: &str = "section";
const GROUP_VECS: &str = "vecs";

const DATASET_VERSION: &str = "dmplex_storage_version";
const DATASET_PERMUTATION: &str = "permutation";
const DATASET_CONE_SIZES: &str = "cone_sizes";
const DATASET_CONES: &str = "cones";
const DATASET_ORIENTATIONS: &str = "orientations";
const DATASET_POINTS: &str = "points";
const DATASET_SECTION_POINTS: &str = "points";
const DATASET_SECTION_DOFS: &str = "dofs";
const DATASET_SECTION_OFFSETS: &str = "offsets";
const DATASET_CELL_TYPES: &str = "cell_types";
const DATASET_REDISTRIBUTION_MAP_ID: &str = "redistribution_map_id";

/// Options controlling DMPlex HDF5 path names.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PetscHdf5Options {
    /// Mesh name in `/topologies/<mesh>`.
    pub mesh_name: String,
    /// DM name in `/topologies/<mesh>/dms/<dm>`.
    pub dm_name: String,
}

impl Default for PetscHdf5Options {
    fn default() -> Self {
        Self {
            mesh_name: "plex".to_string(),
            dm_name: "dm".to_string(),
        }
    }
}

/// PETSc DMPlex HDF5 v3 reader.
#[derive(Clone, Debug, Default)]
pub struct PetscHdf5Reader {
    options: PetscHdf5Options,
}

/// PETSc DMPlex HDF5 v3 writer.
#[derive(Clone, Debug, Default)]
pub struct PetscHdf5Writer {
    options: PetscHdf5Options,
}

impl PetscHdf5Reader {
    /// Construct a reader using explicit mesh and DM names.
    pub fn new(mesh_name: impl Into<String>, dm_name: impl Into<String>) -> Self {
        Self {
            options: PetscHdf5Options {
                mesh_name: mesh_name.into(),
                dm_name: dm_name.into(),
            },
        }
    }

    /// Read only the global point permutation from an HDF5 file.
    pub fn read_order_from_file(
        file: &File,
        mesh_name: &str,
    ) -> Result<Vec<PointId>, MeshSieveError> {
        let topology = topology_group(file, mesh_name)?;
        read_permutation(&topology)
    }

    /// Read only sections/vectors from an HDF5 file and associate them with a saved order.
    pub fn read_sections_from_file(
        file: &File,
        mesh_name: &str,
        dm_name: &str,
    ) -> Result<BTreeMap<String, Section<f64, VecStorage<f64>>>, MeshSieveError> {
        let topo = file.group(&format!("/{GROUP_TOPOLOGIES}/{mesh_name}"))?;
        let dm = topo.group(&format!("{GROUP_DMS}/{dm_name}"))?;
        read_sections_with_options(&dm, None, &PetscLoadOptions::default())
    }
}

impl PetscHdf5Writer {
    /// Construct a writer using explicit mesh and DM names.
    pub fn new(mesh_name: impl Into<String>, dm_name: impl Into<String>) -> Self {
        Self {
            options: PetscHdf5Options {
                mesh_name: mesh_name.into(),
                dm_name: dm_name.into(),
            },
        }
    }
}

impl SieveSectionReader for PetscHdf5Reader {
    type Sieve = MeshSieve;
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
        let result =
            read_mesh_from_petsc_hdf5(&file, &self.options.mesh_name, &self.options.dm_name);
        let _ = fs::remove_file(&temp_path);
        result
    }
}

impl SieveSectionWriter for PetscHdf5Writer {
    type Sieve = MeshSieve;
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
        write_mesh_to_petsc_hdf5(&file, mesh, &self.options.mesh_name, &self.options.dm_name)?;
        drop(file);
        let bytes = fs::read(&temp_path)?;
        writer.write_all(&bytes)?;
        let _ = fs::remove_file(&temp_path);
        Ok(())
    }
}

/// Write `mesh` to a PETSc DMPlex HDF5 v3-style hierarchy.
pub fn write_mesh_to_petsc_hdf5(
    file: &File,
    mesh: &MeshData<MeshSieve, f64, VecStorage<f64>, VecStorage<CellType>>,
    mesh_name: &str,
    dm_name: &str,
) -> Result<(), MeshSieveError> {
    file.new_dataset::<i32>()
        .shape(1)
        .create(DATASET_VERSION)?
        .write(&[DMPLEX_STORAGE_VERSION])?;

    let topologies = create_or_open_group(file, GROUP_TOPOLOGIES)?;
    let topology_root = create_or_open_group(&topologies, mesh_name)?;
    let topology = create_or_open_group(&topology_root, GROUP_TOPOLOGY)?;

    let order = point_order(mesh)?;
    let order_raw: Vec<i64> = order.iter().map(|p| p.get() as i64).collect();
    topology
        .new_dataset::<i64>()
        .shape(order_raw.len())
        .create(DATASET_PERMUTATION)?
        .write(&order_raw)?;

    let index_by_point: HashMap<PointId, i64> = order
        .iter()
        .enumerate()
        .map(|(idx, point)| (*point, idx as i64))
        .collect();
    let strata = compute_strata(&mesh.sieve)?;
    let strata_group = create_or_open_group(&topology, GROUP_STRATA)?;
    let depth_count = strata
        .depth
        .values()
        .copied()
        .max()
        .map_or(0usize, |d| d as usize + 1);
    for depth in 0..depth_count {
        let points: Vec<PointId> = order
            .iter()
            .copied()
            .filter(|point| strata.depth.get(point).copied() == Some(depth as u32))
            .collect();
        write_stratum(&strata_group, depth, &points, &mesh.sieve, &index_by_point)?;
    }

    write_cell_types(&topology, mesh, &order)?;
    write_labels(&topology_root, mesh.labels.as_ref())?;
    write_dm(&topology_root, dm_name, &order, &mesh.sections)?;
    Ok(())
}

/// Read mesh, labels, sections, and vectors from a PETSc DMPlex HDF5 v3-style hierarchy.
pub fn read_mesh_from_petsc_hdf5(
    file: &File,
    mesh_name: &str,
    dm_name: &str,
) -> Result<MeshData<MeshSieve, f64, VecStorage<f64>, VecStorage<CellType>>, MeshSieveError> {
    Ok(read_mesh_from_petsc_hdf5_with_options(
        file,
        mesh_name,
        dm_name,
        &PetscLoadOptions::default(),
    )?
    .0)
}

pub fn read_mesh_from_petsc_hdf5_with_options(
    file: &File,
    mesh_name: &str,
    dm_name: &str,
    options: &PetscLoadOptions,
) -> Result<
    (
        MeshData<MeshSieve, f64, VecStorage<f64>, VecStorage<CellType>>,
        PetscProvenance,
    ),
    MeshSieveError,
> {
    let file_storage_version = file
        .dataset(DATASET_VERSION)
        .ok()
        .and_then(|d| d.read_raw::<i32>().ok())
        .and_then(|v| v.first().copied())
        .unwrap_or(DMPLEX_STORAGE_VERSION);
    if matches!(options.mode, PetscLoadMode::Strict) && !matches!(file_storage_version, 2 | 3) {
        return Err(MeshSieveError::MeshIoParse(format!(
            "unsupported DMPlex HDF5 storage version {file_storage_version}; expected one of [2, 3]"
        )));
    }

    let topology_root = file.group(&format!("/{GROUP_TOPOLOGIES}/{mesh_name}"))?;
    let topology = if let Ok(group) = topology_root.group(GROUP_TOPOLOGY) {
        group
    } else if let Ok(group) = topology_root.group("mesh") {
        group
    } else {
        topology_root.clone()
    };
    let order = read_permutation_checked(&topology, options.mode)?;
    let mut sieve = MeshSieve::default();
    read_strata(&topology, &order, &mut sieve)?;
    let cell_types = read_cell_types(&topology, &order)?;
    let labels = read_labels(&topology_root)?;
    let subset = options
        .filter
        .label_subset
        .as_ref()
        .and_then(|(name, value)| labels.as_ref().map(|l| l.stratum_points(name, *value)))
        .map(|v| v.into_iter().collect::<BTreeSet<_>>());
    let sections = match topology_root.group(&format!("{GROUP_DMS}/{dm_name}")) {
        Ok(dm) => read_sections_with_options(&dm, subset.as_ref(), options)?,
        Err(_) => BTreeMap::new(),
    };

    Ok((
        MeshData {
            sieve,
            coordinates: None,
            sections,
            mixed_sections: Default::default(),
            labels,
            cell_types,
            discretization: None,
        },
        PetscProvenance {
            storage_version: file_storage_version,
            permutation_source: format!(
                "/{GROUP_TOPOLOGIES}/{mesh_name}/{GROUP_TOPOLOGY}/{DATASET_PERMUTATION}"
            ),
            redistribution_map_id: file
                .dataset(DATASET_REDISTRIBUTION_MAP_ID)
                .ok()
                .and_then(|d| d.read_raw::<hdf5::types::VarLenUnicode>().ok())
                .and_then(|v| v.first().map(|s| s.as_str().to_owned())),
        },
    ))
}

fn point_order(
    mesh: &MeshData<MeshSieve, f64, VecStorage<f64>, VecStorage<CellType>>,
) -> Result<Vec<PointId>, MeshSieveError> {
    let mut points: Vec<PointId> = mesh
        .sieve
        .base_points()
        .chain(mesh.sieve.cap_points())
        .collect();
    if let Some(cell_types) = &mesh.cell_types {
        points.extend(cell_types.atlas().points());
    }
    for section in mesh.sections.values() {
        points.extend(section.atlas().points());
    }
    points.sort_unstable();
    points.dedup();

    let strata = compute_strata(&mesh.sieve)?;
    points.sort_unstable_by_key(|point| {
        (
            strata.depth.get(point).copied().unwrap_or(u32::MAX),
            point.get(),
        )
    });
    Ok(points)
}

fn write_stratum(
    strata_group: &Group,
    depth: usize,
    points: &[PointId],
    sieve: &MeshSieve,
    index_by_point: &HashMap<PointId, i64>,
) -> Result<(), MeshSieveError> {
    let group = create_or_open_group(strata_group, &depth.to_string())?;
    let mut cone_sizes = Vec::with_capacity(points.len());
    let mut cones = Vec::new();
    let mut orientations = Vec::new();
    for point in points {
        let cone: Vec<(PointId, i32)> = sieve.cone_o(*point).collect();
        cone_sizes.push(cone.len() as i32);
        for (cone_point, orientation) in cone {
            let cone_index = index_by_point.get(&cone_point).copied().ok_or_else(|| {
                MeshSieveError::MeshIoParse(format!(
                    "point {cone_point:?} missing from permutation"
                ))
            })?;
            cones.push(cone_index);
            orientations.push(orientation);
        }
    }
    group
        .new_dataset::<i32>()
        .shape(cone_sizes.len())
        .create(DATASET_CONE_SIZES)?
        .write(&cone_sizes)?;
    group
        .new_dataset::<i64>()
        .shape(cones.len())
        .create(DATASET_CONES)?
        .write(&cones)?;
    group
        .new_dataset::<i32>()
        .shape(orientations.len())
        .create(DATASET_ORIENTATIONS)?
        .write(&orientations)?;
    Ok(())
}

fn read_strata(
    topology: &Group,
    order: &[PointId],
    sieve: &mut MeshSieve,
) -> Result<(), MeshSieveError> {
    let strata_group = topology.group(GROUP_STRATA)?;
    let mut names = strata_group.member_names()?;
    names.sort_by_key(|name| name.parse::<usize>().unwrap_or(usize::MAX));
    let mut point_offset = 0usize;
    for name in names {
        let group = strata_group.group(&name)?;
        let cone_sizes: Vec<i32> = group.dataset(DATASET_CONE_SIZES)?.read_raw()?;
        let cones: Vec<i64> = group.dataset(DATASET_CONES)?.read_raw()?;
        let orientations: Vec<i32> = group.dataset(DATASET_ORIENTATIONS)?.read_raw()?;
        let mut cone_offset = 0usize;
        for cone_size in cone_sizes {
            let point = *order.get(point_offset).ok_or_else(|| {
                MeshSieveError::MeshIoParse("strata contain more points than permutation".into())
            })?;
            point_offset += 1;
            for local in 0..(cone_size as usize) {
                let idx = *cones
                    .get(cone_offset + local)
                    .ok_or_else(|| MeshSieveError::MeshIoParse("cone array underflow".into()))?
                    as usize;
                let dst = *order.get(idx).ok_or_else(|| {
                    MeshSieveError::MeshIoParse(format!("cone index {idx} outside permutation"))
                })?;
                let orientation = orientations.get(cone_offset + local).copied().unwrap_or(0);
                sieve.add_arrow_o(point, dst, (), orientation);
            }
            cone_offset += cone_size as usize;
        }
    }
    Ok(())
}

fn write_cell_types(
    topology: &Group,
    mesh: &MeshData<MeshSieve, f64, VecStorage<f64>, VecStorage<CellType>>,
    order: &[PointId],
) -> Result<(), MeshSieveError> {
    let Some(cell_types) = &mesh.cell_types else {
        return Ok(());
    };
    let mut values = Vec::with_capacity(order.len());
    for point in order {
        let code = match cell_types.try_restrict(*point) {
            Ok(slice) if !slice.is_empty() => cell_type_to_dmplex(slice[0])?,
            _ => -1,
        };
        values.push(code);
    }
    topology
        .new_dataset::<i32>()
        .shape(values.len())
        .create(DATASET_CELL_TYPES)?
        .write(&values)?;
    Ok(())
}

fn read_cell_types(
    topology: &Group,
    order: &[PointId],
) -> Result<Option<Section<CellType, VecStorage<CellType>>>, MeshSieveError> {
    let Ok(dataset) = topology.dataset(DATASET_CELL_TYPES) else {
        return Ok(None);
    };
    let values: Vec<i32> = dataset.read_raw()?;
    let mut atlas = Atlas::default();
    for (point, code) in order.iter().zip(values.iter()) {
        if *code >= 0 {
            atlas.try_insert(*point, 1)?;
        }
    }
    if atlas.is_empty() {
        return Ok(None);
    }
    let mut section = Section::<CellType, VecStorage<CellType>>::new(atlas);
    for (point, code) in order.iter().zip(values.iter()) {
        if *code >= 0 {
            section.try_set(*point, &[dmplex_to_cell_type(*code)?])?;
        }
    }
    Ok(Some(section))
}

fn write_labels(topology_root: &Group, labels: Option<&LabelSet>) -> Result<(), MeshSieveError> {
    let labels_group = create_or_open_group(topology_root, GROUP_LABELS)?;
    let Some(labels) = labels else {
        return Ok(());
    };

    let mut grouped: BTreeMap<String, BTreeMap<i32, Vec<i64>>> = BTreeMap::new();
    for (name, point, value) in labels.iter() {
        grouped
            .entry(name.to_string())
            .or_default()
            .entry(value)
            .or_default()
            .push(point.get() as i64);
    }

    for (name, by_value) in grouped {
        let group = create_or_open_group(&labels_group, &name)?;
        for (value, mut points) in by_value {
            points.sort_unstable();
            points.dedup();
            let value_group = create_or_open_group(&group, &value.to_string())?;
            value_group
                .new_dataset::<i64>()
                .shape(points.len())
                .create(DATASET_POINTS)?
                .write(&points)?;
        }
    }
    Ok(())
}

fn read_labels(topology_root: &Group) -> Result<Option<LabelSet>, MeshSieveError> {
    let Ok(labels_group) = topology_root.group(GROUP_LABELS) else {
        return Ok(None);
    };
    let mut labels = LabelSet::new();
    for name in labels_group.member_names()? {
        let group = labels_group.group(&name)?;
        for value_name in group.member_names()? {
            let value = value_name.parse::<i32>().map_err(|err| {
                MeshSieveError::MeshIoParse(format!("invalid label value {value_name}: {err}"))
            })?;
            let points: Vec<i64> = group
                .group(&value_name)?
                .dataset(DATASET_POINTS)?
                .read_raw()?;
            for raw in points {
                labels.set_label(PointId::new(raw as u64)?, &name, value);
            }
        }
    }
    Ok((!labels.is_empty()).then_some(labels))
}

fn write_dm(
    topology_root: &Group,
    dm_name: &str,
    order: &[PointId],
    sections: &BTreeMap<String, Section<f64, VecStorage<f64>>>,
) -> Result<(), MeshSieveError> {
    let dms = create_or_open_group(topology_root, GROUP_DMS)?;
    let dm = create_or_open_group(&dms, dm_name)?;
    let section_root = create_or_open_group(&dm, GROUP_SECTION)?;
    let vecs = create_or_open_group(&dm, GROUP_VECS)?;
    for (name, section) in sections {
        let group = create_or_open_group(&section_root, name)?;
        let mut section_points = Vec::new();
        let mut dofs = Vec::new();
        let mut offsets = Vec::new();
        let mut offset = 0i64;
        for point in order {
            if let Some((_off, len)) = section.atlas().get(*point) {
                section_points.push(point.get() as i64);
                dofs.push(len as i32);
                offsets.push(offset);
                offset += len as i64;
            }
        }
        group
            .new_dataset::<i64>()
            .shape(section_points.len())
            .create(DATASET_SECTION_POINTS)?
            .write(&section_points)?;
        group
            .new_dataset::<i32>()
            .shape(dofs.len())
            .create(DATASET_SECTION_DOFS)?
            .write(&dofs)?;
        group
            .new_dataset::<i64>()
            .shape(offsets.len())
            .create(DATASET_SECTION_OFFSETS)?
            .write(&offsets)?;

        let mut values = Vec::new();
        for raw in &section_points {
            values.extend_from_slice(section.try_restrict(PointId::new(*raw as u64)?)?);
        }
        vecs.new_dataset::<f64>()
            .shape(values.len())
            .create(name.as_str())?
            .write(&values)?;
    }
    Ok(())
}

fn read_sections_with_options(
    dm: &Group,
    subset: Option<&BTreeSet<PointId>>,
    options: &PetscLoadOptions,
) -> Result<BTreeMap<String, Section<f64, VecStorage<f64>>>, MeshSieveError> {
    let mut sections = BTreeMap::new();
    let section_root = dm.group(GROUP_SECTION)?;
    let vecs = dm.group(GROUP_VECS)?;
    for name in section_root.member_names()? {
        if !matches_any_pattern(&name, &options.filter.section_name_patterns) {
            continue;
        }
        let group = section_root.group(&name)?;
        let points: Vec<i64> = group.dataset(DATASET_SECTION_POINTS)?.read_raw()?;
        let dofs: Vec<i32> = group.dataset(DATASET_SECTION_DOFS)?.read_raw()?;
        let values: Vec<f64> = vecs.dataset(&name)?.read_raw()?;
        check_section_layout_compatible(&name, &points, &dofs, &values, options.mode)?;
        let mut atlas = Atlas::default();
        for (raw, dof) in points.iter().zip(dofs.iter()) {
            let point = PointId::new(*raw as u64)?;
            if *dof > 0 && subset.is_none_or(|s| s.contains(&point)) {
                atlas.try_insert(point, *dof as usize)?;
            }
        }
        let mut section = Section::<f64, VecStorage<f64>>::new(atlas);
        let mut offset = 0usize;
        for (raw, dof) in points.iter().zip(dofs.iter()) {
            if *dof > 0 {
                let point = PointId::new(*raw as u64)?;
                let end = offset + *dof as usize;
                if end > values.len() {
                    if matches!(options.mode, PetscLoadMode::Strict) {
                        return Err(MeshSieveError::MeshIoParse(format!(
                            "vector {name} shorter than section layout"
                        )));
                    }
                    break;
                }
                if subset.is_none_or(|s| s.contains(&point)) {
                    section.try_set(point, &values[offset..end])?;
                }
                offset = end;
            }
        }
        sections.insert(name, section);
    }
    Ok(sections)
}

fn check_section_layout_compatible(
    name: &str,
    points: &[i64],
    dofs: &[i32],
    values: &[f64],
    mode: PetscLoadMode,
) -> Result<(), MeshSieveError> {
    if points.len() != dofs.len() && matches!(mode, PetscLoadMode::Strict) {
        return Err(MeshSieveError::MeshIoParse(format!(
            "section {name} points/dofs mismatch: {} vs {}",
            points.len(),
            dofs.len()
        )));
    }
    let expected: usize = dofs.iter().filter(|d| **d > 0).map(|d| *d as usize).sum();
    if expected != values.len() && matches!(mode, PetscLoadMode::Strict) {
        return Err(MeshSieveError::MeshIoParse(format!(
            "section {name} vector length mismatch: expected {expected}, found {}",
            values.len()
        )));
    }
    Ok(())
}

fn matches_any_pattern(name: &str, patterns: &[String]) -> bool {
    if patterns.is_empty() {
        return true;
    }
    patterns.iter().any(|p| {
        if p == "*" {
            true
        } else if let Some(core) = p.strip_prefix('*').and_then(|s| s.strip_suffix('*')) {
            name.contains(core)
        } else if let Some(s) = p.strip_prefix('*') {
            name.ends_with(s)
        } else if let Some(s) = p.strip_suffix('*') {
            name.starts_with(s)
        } else {
            name == p
        }
    })
}

fn topology_group(file: &File, mesh_name: &str) -> Result<Group, MeshSieveError> {
    Ok(file.group(&format!("/{GROUP_TOPOLOGIES}/{mesh_name}/{GROUP_TOPOLOGY}"))?)
}

fn read_permutation(topology: &Group) -> Result<Vec<PointId>, MeshSieveError> {
    read_permutation_checked(topology, PetscLoadMode::Strict)
}

fn read_permutation_checked(
    topology: &Group,
    mode: PetscLoadMode,
) -> Result<Vec<PointId>, MeshSieveError> {
    let raw: Vec<i64> = topology.dataset(DATASET_PERMUTATION)?.read_raw()?;
    for (idx, value) in raw.iter().enumerate() {
        if *value <= 0 && matches!(mode, PetscLoadMode::Strict) {
            return Err(MeshSieveError::MeshIoParse(format!(
                "out-of-range permutation point tag at index {idx}: {value}"
            )));
        }
    }
    let points = raw
        .into_iter()
        .map(|value| PointId::new(value as u64))
        .collect::<Result<Vec<_>, _>>()?;
    let mut seen = BTreeSet::new();
    for (idx, point) in points.iter().enumerate() {
        if !seen.insert(*point) && matches!(mode, PetscLoadMode::Strict) {
            return Err(MeshSieveError::MeshIoParse(format!(
                "duplicate permutation point tag at index {idx}: {point:?}"
            )));
        }
    }
    Ok(points)
}

fn create_or_open_group(parent: &Group, name: &str) -> Result<Group, MeshSieveError> {
    match parent.group(name) {
        Ok(group) => Ok(group),
        Err(_) => Ok(parent.create_group(name)?),
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
    path.push(format!("mesh_sieve_petsc_{pid}_{nanos}.h5"));
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
            "unsupported DMPlex cell type: {cell_type:?}"
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
        8 => Ok(CellType::Polygon(0)),
        9 => Ok(CellType::Polyhedron),
        10 => Ok(CellType::Simplex(4)),
        11 => Ok(CellType::Simplex(5)),
        _ => Err(MeshSieveError::MeshIoParse(format!(
            "unknown DMPlex cell type code {code}"
        ))),
    }
}
