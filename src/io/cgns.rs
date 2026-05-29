//! CGNS/HDF5 mesh reader.
//!
//! The reader is feature-gated behind `cgns`. With the feature enabled it reads
//! an HDF5-backed CGNS subset covering unstructured zones, coordinate arrays,
//! common fixed-size and `MIXED` element sections, `ZoneBC_t` point labels, and
//! scalar/vector `FlowSolution_t` fields where their cardinality matches vertices
//! or imported cells.

#[cfg(feature = "cgns")]
use crate::data::atlas::Atlas;
#[cfg(feature = "cgns")]
use crate::data::coordinates::Coordinates;
#[cfg(feature = "cgns")]
use crate::data::section::Section;
use crate::data::storage::VecStorage;
use crate::io::{MeshData, SieveSectionReader};
use crate::mesh_error::MeshSieveError;
use crate::topology::cell_type::CellType;
#[cfg(feature = "cgns")]
use crate::topology::labels::LabelSet;
#[cfg(feature = "cgns")]
use crate::topology::point::PointId;
use crate::topology::sieve::MeshSieve;
#[cfg(feature = "cgns")]
use crate::topology::sieve::{MutableSieve, Sieve};
use std::io::Read;

/// CGNS reader entry point.
#[derive(Debug, Default, Clone)]
pub struct CgnsReader;

#[cfg(not(feature = "cgns"))]
impl SieveSectionReader for CgnsReader {
    type Sieve = MeshSieve;
    type Value = f64;
    type Storage = VecStorage<f64>;
    type CellStorage = VecStorage<CellType>;

    fn read<R: Read>(
        &self,
        _reader: R,
    ) -> Result<MeshData<Self::Sieve, Self::Value, Self::Storage, Self::CellStorage>, MeshSieveError>
    {
        Err(MeshSieveError::MeshIoParse(
            "CGNS support is not compiled in; rebuild mesh-sieve with `--features cgns`".into(),
        ))
    }
}

#[cfg(feature = "cgns")]
mod enabled {
    use super::*;
    use hdf5::{
        File, Group,
        types::{VarLenAscii, VarLenUnicode},
    };
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[derive(Debug)]
    struct ElementRecord {
        id: PointId,
        conn: Vec<PointId>,
        cell_type: CellType,
    }

    impl SieveSectionReader for CgnsReader {
        type Sieve = MeshSieve;
        type Value = f64;
        type Storage = VecStorage<f64>;
        type CellStorage = VecStorage<CellType>;

        fn read<R: Read>(
            &self,
            mut reader: R,
        ) -> Result<
            MeshData<Self::Sieve, Self::Value, Self::Storage, Self::CellStorage>,
            MeshSieveError,
        > {
            let mut bytes = Vec::new();
            reader.read_to_end(&mut bytes)?;
            let path = temp_hdf5_path();
            fs::write(&path, bytes)?;
            let file = File::open(&path).map_err(|err| {
                MeshSieveError::MeshIoParse(format!("CGNS/HDF5 open error: {err}"))
            })?;
            let result = read_file(&file);
            drop(file);
            let _ = fs::remove_file(&path);
            result
        }
    }

    fn read_file(
        file: &File,
    ) -> Result<MeshData<MeshSieve, f64, VecStorage<f64>, VecStorage<CellType>>, MeshSieveError>
    {
        let zones = collect_groups(file, |g| {
            group_label(g).as_deref() == Some("Zone_t")
                || g.name()
                    .rsplit('/')
                    .next()
                    .is_some_and(|n| n.starts_with("Zone"))
        })?;
        let zone = zones.first().ok_or_else(|| {
            MeshSieveError::MeshIoParse("CGNS file contains no Zone_t group".into())
        })?;
        let coords = read_coordinates(zone)?;
        let elements = read_elements(zone, coords.len())?;
        build_mesh(zone, coords, elements)
    }

    fn build_mesh(
        zone: &Group,
        coords_in: Vec<[f64; 3]>,
        elements: Vec<ElementRecord>,
    ) -> Result<MeshData<MeshSieve, f64, VecStorage<f64>, VecStorage<CellType>>, MeshSieveError>
    {
        let mut sieve = MeshSieve::default();
        let mut coord_atlas = Atlas::default();
        for i in 0..coords_in.len() {
            let p = PointId::new((i as u64) + 1)?;
            MutableSieve::add_point(&mut sieve, p);
            coord_atlas.try_insert(p, 3)?;
        }
        let mut cell_atlas = Atlas::default();
        for elem in &elements {
            MutableSieve::add_point(&mut sieve, elem.id);
            for vertex in &elem.conn {
                Sieve::add_arrow(&mut sieve, elem.id, *vertex, ());
            }
            cell_atlas.try_insert(elem.id, 1)?;
        }
        let mesh_dim = elements
            .iter()
            .map(|e| e.cell_type.dimension())
            .max()
            .unwrap_or(0);
        let mut coords = Coordinates::try_new(mesh_dim as usize, 3, coord_atlas)?;
        for (i, xyz) in coords_in.iter().enumerate() {
            coords
                .section_mut()
                .try_set(PointId::new((i as u64) + 1)?, xyz)?;
        }
        let mut cell_types = Section::<CellType, VecStorage<CellType>>::new(cell_atlas);
        for elem in &elements {
            cell_types.try_set(elem.id, &[elem.cell_type])?;
        }

        let mut labels = read_boundary_labels(zone)?;
        for elem in &elements {
            labels.set_label(elem.id, "cgns:cell", 1);
        }
        let sections = read_solution_sections(zone, coords_in.len(), &elements)?;

        let mut mesh = MeshData::new(sieve);
        mesh.coordinates = Some(coords);
        mesh.cell_types = Some(cell_types);
        mesh.labels = (!labels.is_empty()).then_some(labels);
        mesh.sections = sections;
        Ok(mesh)
    }

    fn read_coordinates(zone: &Group) -> Result<Vec<[f64; 3]>, MeshSieveError> {
        let coord_groups = collect_groups(zone, |g| {
            group_label(g).as_deref() == Some("GridCoordinates_t")
                || g.name().ends_with("GridCoordinates")
        })?;
        let group = coord_groups.first().ok_or_else(|| {
            MeshSieveError::MeshIoParse("CGNS zone contains no GridCoordinates_t group".into())
        })?;
        let x = read_dataset_f64_any(group, &["CoordinateX", "CoordinateR"])?;
        let y = read_dataset_f64_any(group, &["CoordinateY", "CoordinateTheta"])
            .unwrap_or_else(|_| vec![0.0; x.len()]);
        let z =
            read_dataset_f64_any(group, &["CoordinateZ"]).unwrap_or_else(|_| vec![0.0; x.len()]);
        if y.len() != x.len() || z.len() != x.len() {
            return Err(MeshSieveError::MeshIoParse(
                "CGNS coordinate arrays have different lengths".into(),
            ));
        }
        Ok((0..x.len()).map(|i| [x[i], y[i], z[i]]).collect())
    }

    fn read_elements(
        zone: &Group,
        vertex_count: usize,
    ) -> Result<Vec<ElementRecord>, MeshSieveError> {
        let groups = collect_groups(zone, |g| {
            group_label(g).as_deref() == Some("Elements_t")
                || g.dataset("ElementConnectivity").is_ok()
        })?;
        let mut records = Vec::new();
        let mut next_id = vertex_count as u64 + 1;
        for group in groups {
            let conn = match group.dataset("ElementConnectivity") {
                Ok(ds) => read_i64_dataset(&ds)?,
                Err(_) => continue,
            };
            let etype = read_element_type(&group)?;
            if etype == 20 {
                let mut i = 0;
                while i < conn.len() {
                    let code = conn[i] as i32;
                    i += 1;
                    let (cell_type, n) = cgns_element_type(code).ok_or_else(|| {
                        MeshSieveError::MeshIoParse(format!(
                            "unsupported CGNS MIXED element type code {code}"
                        ))
                    })?;
                    if i + n > conn.len() {
                        return Err(MeshSieveError::MeshIoParse(
                            "truncated CGNS MIXED connectivity".into(),
                        ));
                    }
                    let id = PointId::new(next_id)?;
                    next_id += 1;
                    let nodes = conn[i..i + n]
                        .iter()
                        .map(|v| PointId::new(*v as u64))
                        .collect::<Result<Vec<_>, _>>()?;
                    i += n;
                    records.push(ElementRecord {
                        id,
                        conn: nodes,
                        cell_type,
                    });
                }
            } else {
                let (cell_type, n) = cgns_element_type(etype).ok_or_else(|| {
                    MeshSieveError::MeshIoParse(format!(
                        "unsupported CGNS element type code {etype}"
                    ))
                })?;
                if n == 0 || conn.len() % n != 0 {
                    return Err(MeshSieveError::MeshIoParse(format!(
                        "CGNS connectivity length {} is not divisible by {n}",
                        conn.len()
                    )));
                }
                for chunk in conn.chunks(n) {
                    let id = PointId::new(next_id)?;
                    next_id += 1;
                    let nodes = chunk
                        .iter()
                        .map(|v| PointId::new(*v as u64))
                        .collect::<Result<Vec<_>, _>>()?;
                    records.push(ElementRecord {
                        id,
                        conn: nodes,
                        cell_type,
                    });
                }
            }
        }
        Ok(records)
    }

    fn read_boundary_labels(zone: &Group) -> Result<LabelSet, MeshSieveError> {
        let mut labels = LabelSet::new();
        let bcs = collect_groups(zone, |g| {
            group_label(g).as_deref() == Some("BC_t")
                || g.dataset("PointList").is_ok()
                || g.dataset("PointRange").is_ok()
        })?;
        for (idx, bc) in bcs.iter().enumerate() {
            let name = bc.name().rsplit('/').next().unwrap_or("BC").to_string();
            let value = i32::try_from(idx + 1).unwrap_or(i32::MAX);
            if let Ok(points) = read_dataset_i64_any(bc, &["PointList"]) {
                for p in points {
                    let point = PointId::new(p as u64)?;
                    labels.set_label(point, "cgns:bc", value);
                    labels.set_label(point, &format!("cgns:bc:{name}"), 1);
                }
            }
            if let Ok(range) = read_dataset_i64_any(bc, &["PointRange"]) {
                if range.len() >= 2 {
                    for raw in range[0]..=range[1] {
                        let point = PointId::new(raw as u64)?;
                        labels.set_label(point, "cgns:bc", value);
                        labels.set_label(point, &format!("cgns:bc:{name}"), 1);
                    }
                }
            }
        }
        Ok(labels)
    }

    fn read_solution_sections(
        zone: &Group,
        vertex_count: usize,
        elements: &[ElementRecord],
    ) -> Result<std::collections::BTreeMap<String, Section<f64, VecStorage<f64>>>, MeshSieveError>
    {
        let mut out = std::collections::BTreeMap::new();
        let solutions = collect_groups(zone, |g| {
            group_label(g).as_deref() == Some("FlowSolution_t") || g.name().contains("FlowSolution")
        })?;
        for sol in solutions {
            let sol_name = sol
                .name()
                .rsplit('/')
                .next()
                .unwrap_or("FlowSolution")
                .to_string();
            for name in sol.member_names()? {
                let Ok(ds) = sol.dataset(&name) else {
                    continue;
                };
                if dataset_label(&ds)
                    .as_deref()
                    .is_some_and(|l| l != "DataArray_t")
                    && !name.starts_with(|c: char| c.is_ascii_alphabetic())
                {
                    continue;
                }
                let values = read_f64_dataset(&ds)?;
                let (points, dof): (Vec<PointId>, usize) = if values.len() == vertex_count {
                    (
                        (1..=vertex_count)
                            .map(|i| PointId::new(i as u64))
                            .collect::<Result<_, _>>()?,
                        1,
                    )
                } else if !elements.is_empty() && values.len() == elements.len() {
                    (elements.iter().map(|e| e.id).collect(), 1)
                } else if vertex_count > 0 && values.len() % vertex_count == 0 {
                    (
                        (1..=vertex_count)
                            .map(|i| PointId::new(i as u64))
                            .collect::<Result<_, _>>()?,
                        values.len() / vertex_count,
                    )
                } else {
                    continue;
                };
                let mut atlas = Atlas::default();
                for p in &points {
                    atlas.try_insert(*p, dof)?;
                }
                let mut section = Section::<f64, VecStorage<f64>>::new(atlas);
                for (i, p) in points.iter().enumerate() {
                    section.try_set(*p, &values[i * dof..(i + 1) * dof])?;
                }
                out.insert(format!("{sol_name}/{name}"), section);
            }
        }
        Ok(out)
    }

    fn cgns_element_type(code: i32) -> Option<(CellType, usize)> {
        match code {
            2 => Some((CellType::Vertex, 1)),
            3 => Some((CellType::Segment, 2)),
            5 => Some((CellType::Triangle, 3)),
            7 => Some((CellType::Quadrilateral, 4)),
            10 => Some((CellType::Tetrahedron, 4)),
            12 => Some((CellType::Pyramid, 5)),
            14 => Some((CellType::Prism, 6)),
            17 => Some((CellType::Hexahedron, 8)),
            _ => None,
        }
    }

    fn read_element_type(group: &Group) -> Result<i32, MeshSieveError> {
        if let Ok(ds) = group.dataset("ElementType") {
            return Ok(read_i64_dataset(&ds)?.first().copied().unwrap_or(0) as i32);
        }
        if let Some(s) = string_attr(group, "ElementType") {
            return element_type_name_to_code(&s);
        }
        if let Ok(attr) = group.attr("ElementType") {
            if let Ok(v) = attr.read_scalar::<i32>() {
                return Ok(v);
            }
            if let Ok(v) = attr.read_scalar::<i64>() {
                return Ok(v as i32);
            }
        }
        Err(MeshSieveError::MeshIoParse(format!(
            "CGNS Elements_t group {} lacks ElementType",
            group.name()
        )))
    }

    fn element_type_name_to_code(name: &str) -> Result<i32, MeshSieveError> {
        match name.trim().to_ascii_uppercase().as_str() {
            "NODE" => Ok(2),
            "BAR_2" => Ok(3),
            "TRI_3" => Ok(5),
            "QUAD_4" => Ok(7),
            "TETRA_4" => Ok(10),
            "PYRA_5" => Ok(12),
            "PENTA_6" => Ok(14),
            "HEXA_8" => Ok(17),
            "MIXED" => Ok(20),
            _ => Err(MeshSieveError::MeshIoParse(format!(
                "unsupported CGNS ElementType {name}"
            ))),
        }
    }

    fn collect_groups<F>(root: &Group, pred: F) -> Result<Vec<Group>, MeshSieveError>
    where
        F: Fn(&Group) -> bool,
    {
        let mut out = Vec::new();
        collect_groups_rec(root, &pred, &mut out)?;
        Ok(out)
    }
    fn collect_groups_rec<F>(
        group: &Group,
        pred: &F,
        out: &mut Vec<Group>,
    ) -> Result<(), MeshSieveError>
    where
        F: Fn(&Group) -> bool,
    {
        if pred(group) {
            out.push(group.clone());
        }
        for name in group.member_names()? {
            if let Ok(child) = group.group(&name) {
                collect_groups_rec(&child, pred, out)?;
            }
        }
        Ok(())
    }
    fn group_label(group: &Group) -> Option<String> {
        string_attr_from_attr(group.attr("label").ok()?)
    }
    fn dataset_label(ds: &hdf5::Dataset) -> Option<String> {
        string_attr_from_attr(ds.attr("label").ok()?)
    }
    fn string_attr(group: &Group, name: &str) -> Option<String> {
        string_attr_from_attr(group.attr(name).ok()?)
    }
    fn string_attr_from_attr(attr: hdf5::Attribute) -> Option<String> {
        if let Ok(value) = attr.read_scalar::<VarLenUnicode>() {
            return Some(value.as_str().trim_matches('\0').trim().to_string());
        }
        if let Ok(value) = attr.read_scalar::<VarLenAscii>() {
            return Some(value.as_str().trim_matches('\0').trim().to_string());
        }
        None
    }
    fn read_dataset_f64_any(group: &Group, names: &[&str]) -> Result<Vec<f64>, MeshSieveError> {
        for name in names {
            if let Ok(ds) = group.dataset(name) {
                return read_f64_dataset(&ds);
            }
        }
        Err(MeshSieveError::MeshIoParse(format!(
            "missing dataset one of {names:?} in {}",
            group.name()
        )))
    }
    fn read_dataset_i64_any(group: &Group, names: &[&str]) -> Result<Vec<i64>, MeshSieveError> {
        for name in names {
            if let Ok(ds) = group.dataset(name) {
                return read_i64_dataset(&ds);
            }
        }
        Err(MeshSieveError::MeshIoParse(format!(
            "missing dataset one of {names:?} in {}",
            group.name()
        )))
    }
    fn read_i64_dataset(dataset: &hdf5::Dataset) -> Result<Vec<i64>, MeshSieveError> {
        if let Ok(v) = dataset.read_raw::<i64>() {
            return Ok(v);
        }
        if let Ok(v) = dataset.read_raw::<i32>() {
            return Ok(v.into_iter().map(i64::from).collect());
        }
        let v: Vec<u64> = dataset.read_raw()?;
        Ok(v.into_iter().map(|x| x as i64).collect())
    }
    fn read_f64_dataset(dataset: &hdf5::Dataset) -> Result<Vec<f64>, MeshSieveError> {
        if let Ok(v) = dataset.read_raw::<f64>() {
            return Ok(v);
        }
        let v: Vec<f32> = dataset.read_raw()?;
        Ok(v.into_iter().map(f64::from).collect())
    }
    fn temp_hdf5_path() -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|v| v.as_nanos())
            .unwrap_or(0);
        p.push(format!(
            "mesh_sieve_cgns_{}_{nanos}.cgns",
            std::process::id()
        ));
        p
    }
}
