//! Renumbering utilities for topology, sections, and labels.

use crate::data::atlas::Atlas;
use crate::data::coordinates::{Coordinates, HighOrderCoordinates};
use crate::data::mixed_section::{MixedSectionStore, TaggedSection};
use crate::data::section::Section;
use crate::data::storage::Storage;
use crate::io::MeshData;
use crate::mesh_error::MeshSieveError;
use crate::topology::cell_type::CellType;
use crate::topology::labels::LabelSet;
use crate::topology::point::PointId;
use crate::topology::sieve::strata::compute_strata;
use crate::topology::sieve::{InMemorySieve, MutableSieve, Sieve};
use std::collections::{BTreeMap, HashMap, HashSet};

/// Ordering options for stratified renumbering.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum StratifiedOrdering {
    /// Use depth strata so vertices (depth 0) appear first.
    VertexFirst,
    /// Use height strata so cells (height 0) appear first.
    CellFirst,
}

/// Compute a point permutation using sieve strata.
///
/// The returned vector is a permutation of all points in the sieve. For
/// [`StratifiedOrdering::VertexFirst`], points are ordered by increasing
/// depth (vertices first). For [`StratifiedOrdering::CellFirst`], points are
/// ordered by increasing height (cells first).
pub fn stratified_permutation<S>(
    sieve: &S,
    ordering: StratifiedOrdering,
) -> Result<Vec<PointId>, MeshSieveError>
where
    S: Sieve<Point = PointId>,
{
    let cache = compute_strata(sieve)?;
    match ordering {
        StratifiedOrdering::CellFirst => Ok(cache.chart_points),
        StratifiedOrdering::VertexFirst => {
            let max_depth = cache.depth.values().copied().max().unwrap_or(0) as usize;
            let mut strata = vec![Vec::new(); max_depth + 1];
            for (&p, &d) in &cache.depth {
                let idx = d as usize;
                strata[idx].push(p);
            }
            for lev in &mut strata {
                lev.sort_unstable();
            }
            let mut out = Vec::with_capacity(cache.depth.len());
            for lev in strata {
                out.extend(lev);
            }
            Ok(out)
        }
    }
}

/// Renumber points using a provided permutation.
///
/// The permutation is interpreted as the new order of existing points. The
/// first entry maps to `PointId(1)`, the next to `PointId(2)`, etc. The
/// topology, sections, and labels are remapped consistently.
pub fn renumber_points<S, V, St, CtSt>(
    mesh: &MeshData<S, V, St, CtSt>,
    permutation: &[PointId],
) -> Result<MeshData<InMemorySieve<PointId, S::Payload>, V, St, CtSt>, MeshSieveError>
where
    S: Sieve<Point = PointId>,
    S::Payload: Clone,
    V: Clone + Default,
    St: Storage<V> + Clone,
    CtSt: Storage<CellType> + Clone,
{
    let (old_to_new, new_to_old) = build_renumber_map(mesh, permutation)?;

    let mut sieve = InMemorySieve::default();
    for &old in &new_to_old {
        let new = old_to_new[&old];
        MutableSieve::add_point(&mut sieve, new);
    }
    for &old in &new_to_old {
        let new_src = old_to_new[&old];
        for (dst, payload) in mesh.sieve.cone(old) {
            let new_dst = old_to_new
                .get(&dst)
                .copied()
                .ok_or_else(|| {
                    MeshSieveError::InvalidPermutation(format!(
                        "missing destination {dst:?} for source {old:?}"
                    ))
                })?;
            sieve.add_arrow(new_src, new_dst, payload.clone());
        }
    }

    let coordinates = match &mesh.coordinates {
        Some(coords) => Some(remap_coordinates(coords, &old_to_new, &new_to_old)?),
        None => None,
    };

    let mut sections = BTreeMap::new();
    for (name, section) in &mesh.sections {
        sections.insert(
            name.clone(),
            remap_section(section, &old_to_new, &new_to_old)?,
        );
    }

    let mut mixed_sections = MixedSectionStore::default();
    for (name, section) in mesh.mixed_sections.iter() {
        mixed_sections.insert_tagged(
            name.clone(),
            remap_tagged_section(section, &old_to_new, &new_to_old)?,
        );
    }

    let labels = match &mesh.labels {
        Some(labels) => {
            let out = remap_labels(labels, &old_to_new)?;
            (!out.is_empty()).then_some(out)
        }
        None => None,
    };

    let cell_types = match &mesh.cell_types {
        Some(section) => Some(remap_section(section, &old_to_new, &new_to_old)?),
        None => None,
    };

    Ok(MeshData {
        sieve,
        coordinates,
        sections,
        mixed_sections,
        labels,
        cell_types,
        discretization: mesh.discretization.clone(),
    })
}

/// Renumber points using strata-based ordering.
pub fn renumber_points_stratified<S, V, St, CtSt>(
    mesh: &MeshData<S, V, St, CtSt>,
    ordering: StratifiedOrdering,
) -> Result<MeshData<InMemorySieve<PointId, S::Payload>, V, St, CtSt>, MeshSieveError>
where
    S: Sieve<Point = PointId>,
    S::Payload: Clone,
    V: Clone + Default,
    St: Storage<V> + Clone,
    CtSt: Storage<CellType> + Clone,
{
    let permutation = stratified_permutation(&mesh.sieve, ordering)?;
    renumber_points(mesh, &permutation)
}

fn build_renumber_map<S, V, St, CtSt>(
    mesh: &MeshData<S, V, St, CtSt>,
    permutation: &[PointId],
) -> Result<(HashMap<PointId, PointId>, Vec<PointId>), MeshSieveError>
where
    S: Sieve<Point = PointId>,
    St: Storage<V>,
    CtSt: Storage<CellType>,
{
    let mesh_points: Vec<PointId> = mesh.sieve.points().collect();
    let total = mesh_points.len();

    if permutation.len() != total {
        return Err(MeshSieveError::InvalidPermutation(format!(
            "expected {total} points, got {}",
            permutation.len()
        )));
    }

    let mut expected: HashSet<PointId> = mesh_points.into_iter().collect();
    let mut old_to_new = HashMap::with_capacity(permutation.len());
    let mut new_to_old = Vec::with_capacity(permutation.len());

    for (idx, &old) in permutation.iter().enumerate() {
        if !expected.remove(&old) {
            return Err(MeshSieveError::InvalidPermutation(format!(
                "duplicate or unknown point {old:?}"
            )));
        }
        let new = PointId::new((idx + 1) as u64)?;
        if old_to_new.insert(old, new).is_some() {
            return Err(MeshSieveError::InvalidPermutation(format!(
                "duplicate point {old:?}"
            )));
        }
        new_to_old.push(old);
    }

    if !expected.is_empty() {
        return Err(MeshSieveError::InvalidPermutation(format!(
            "missing points: {expected:?}"
        )));
    }

    Ok((old_to_new, new_to_old))
}

fn remap_section<V, S>(
    section: &Section<V, S>,
    old_to_new: &HashMap<PointId, PointId>,
    new_to_old: &[PointId],
) -> Result<Section<V, S>, MeshSieveError>
where
    V: Clone + Default,
    S: Storage<V> + Clone,
{
    let mut atlas = Atlas::default();
    for &old in new_to_old {
        if let Some((_, len)) = section.atlas().get(old) {
            let new = old_to_new[&old];
            atlas.try_insert(new, len)?;
        }
    }
    let mut out = Section::new(atlas);
    for &old in new_to_old {
        if section.atlas().contains(old) {
            let new = old_to_new[&old];
            let data = section.try_restrict(old)?;
            out.try_set(new, data)?;
        }
    }
    Ok(out)
}

fn remap_tagged_section(
    section: &TaggedSection,
    old_to_new: &HashMap<PointId, PointId>,
    new_to_old: &[PointId],
) -> Result<TaggedSection, MeshSieveError> {
    Ok(match section {
        TaggedSection::F64(sec) => TaggedSection::F64(remap_section(sec, old_to_new, new_to_old)?),
        TaggedSection::F32(sec) => TaggedSection::F32(remap_section(sec, old_to_new, new_to_old)?),
        TaggedSection::I32(sec) => TaggedSection::I32(remap_section(sec, old_to_new, new_to_old)?),
        TaggedSection::I64(sec) => TaggedSection::I64(remap_section(sec, old_to_new, new_to_old)?),
        TaggedSection::U32(sec) => TaggedSection::U32(remap_section(sec, old_to_new, new_to_old)?),
        TaggedSection::U64(sec) => TaggedSection::U64(remap_section(sec, old_to_new, new_to_old)?),
    })
}

fn remap_coordinates<V, S>(
    coords: &Coordinates<V, S>,
    old_to_new: &HashMap<PointId, PointId>,
    new_to_old: &[PointId],
) -> Result<Coordinates<V, S>, MeshSieveError>
where
    V: Clone + Default,
    S: Storage<V> + Clone,
{
    let section = remap_section(coords.section(), old_to_new, new_to_old)?;
    let mut out = Coordinates::from_section(coords.dimension(), section)?;
    if let Some(high_order) = coords.high_order() {
        let ho_section = remap_section(high_order.section(), old_to_new, new_to_old)?;
        let ho = HighOrderCoordinates::from_section(high_order.dimension(), ho_section)?;
        out.set_high_order(ho)?;
    }
    Ok(out)
}

fn remap_labels(
    labels: &LabelSet,
    old_to_new: &HashMap<PointId, PointId>,
) -> Result<LabelSet, MeshSieveError> {
    let mut out = LabelSet::new();
    for (name, point, value) in labels.iter() {
        let new_point = old_to_new.get(&point).copied().ok_or_else(|| {
            MeshSieveError::InvalidPermutation(format!("missing label point {point:?}"))
        })?;
        out.set_label(new_point, name, value);
    }
    Ok(out)
}
