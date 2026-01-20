//! Submesh extraction utilities.

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
use crate::topology::sieve::{InMemorySieve, MutableSieve, Sieve};
use std::collections::{BTreeMap, HashMap, HashSet};

/// Bidirectional mapping between parent and submesh point IDs.
#[derive(Debug, Clone)]
pub struct SubmeshMaps {
    pub parent_to_sub: HashMap<PointId, PointId>,
    pub sub_to_parent: Vec<PointId>,
}

/// Extract a submesh from labeled points and their closure.
///
/// The output mesh reindexes points to a compact `PointId` range and carries
/// over any sections/coordinates defined on the retained points.
pub fn extract_by_label<S, V, St, CtSt>(
    mesh: &MeshData<S, V, St, CtSt>,
    labels: &LabelSet,
    label_name: &str,
    label_value: i32,
) -> Result<
    (
        MeshData<InMemorySieve<PointId, S::Payload>, V, St, CtSt>,
        SubmeshMaps,
    ),
    MeshSieveError,
>
where
    S: Sieve<Point = PointId>,
    S::Payload: Clone,
    V: Clone + Default,
    St: Storage<V> + Clone,
    CtSt: Storage<CellType> + Clone,
{
    let seeds: Vec<PointId> = labels.points_with_label(label_name, label_value).collect();
    let mut points: HashSet<PointId> = HashSet::new();
    for p in mesh.sieve.closure_iter(seeds) {
        points.insert(p);
    }

    let mut parent_points: Vec<PointId> = points.into_iter().collect();
    parent_points.sort_unstable();

    let mut parent_to_sub = HashMap::with_capacity(parent_points.len());
    let mut sub_to_parent = Vec::with_capacity(parent_points.len());
    for (idx, parent) in parent_points.iter().enumerate() {
        let sub = PointId::new((idx + 1) as u64)?;
        parent_to_sub.insert(*parent, sub);
        sub_to_parent.push(*parent);
    }

    let mut sieve = InMemorySieve::default();
    for parent in &parent_points {
        let sub = parent_to_sub[parent];
        MutableSieve::add_point(&mut sieve, sub);
    }
    for parent in &parent_points {
        let sub_src = parent_to_sub[parent];
        for (dst, payload) in mesh.sieve.cone(*parent) {
            if let Some(&sub_dst) = parent_to_sub.get(&dst) {
                sieve.add_arrow(sub_src, sub_dst, payload.clone());
            }
        }
    }

    let coordinates = match &mesh.coordinates {
        Some(coords) => Some(transfer_coordinates(
            coords,
            &parent_to_sub,
            &parent_points,
        )?),
        None => None,
    };

    let mut sections = BTreeMap::new();
    for (name, section) in &mesh.sections {
        sections.insert(
            name.clone(),
            transfer_section(section, &parent_to_sub, &parent_points)?,
        );
    }

    let mut mixed_sections = MixedSectionStore::default();
    for (name, section) in mesh.mixed_sections.iter() {
        mixed_sections.insert_tagged(
            name.clone(),
            transfer_tagged_section(section, &parent_to_sub, &parent_points)?,
        );
    }

    let labels_out = remap_labels(labels, &parent_to_sub);
    let labels_out = (!labels_out.is_empty()).then_some(labels_out);

    let cell_types = match &mesh.cell_types {
        Some(section) => Some(transfer_section(section, &parent_to_sub, &parent_points)?),
        None => None,
    };

    Ok((
        MeshData {
            sieve,
            coordinates,
            sections,
            mixed_sections,
            labels: labels_out,
            cell_types,
        },
        SubmeshMaps {
            parent_to_sub,
            sub_to_parent,
        },
    ))
}

fn transfer_section<V, S>(
    section: &Section<V, S>,
    parent_to_sub: &HashMap<PointId, PointId>,
    parent_points: &[PointId],
) -> Result<Section<V, S>, MeshSieveError>
where
    V: Clone + Default,
    S: Storage<V> + Clone,
{
    let mut atlas = Atlas::default();
    for parent in parent_points {
        if let Some((_, len)) = section.atlas().get(*parent) {
            let sub = parent_to_sub[parent];
            atlas.try_insert(sub, len)?;
        }
    }
    let mut out = Section::new(atlas);
    for parent in parent_points {
        if section.atlas().contains(*parent) {
            let sub = parent_to_sub[parent];
            let data = section.try_restrict(*parent)?;
            out.try_set(sub, data)?;
        }
    }
    Ok(out)
}

fn transfer_tagged_section(
    section: &TaggedSection,
    parent_to_sub: &HashMap<PointId, PointId>,
    parent_points: &[PointId],
) -> Result<TaggedSection, MeshSieveError> {
    Ok(match section {
        TaggedSection::F64(sec) => {
            TaggedSection::F64(transfer_section(sec, parent_to_sub, parent_points)?)
        }
        TaggedSection::F32(sec) => {
            TaggedSection::F32(transfer_section(sec, parent_to_sub, parent_points)?)
        }
        TaggedSection::I32(sec) => {
            TaggedSection::I32(transfer_section(sec, parent_to_sub, parent_points)?)
        }
        TaggedSection::I64(sec) => {
            TaggedSection::I64(transfer_section(sec, parent_to_sub, parent_points)?)
        }
        TaggedSection::U32(sec) => {
            TaggedSection::U32(transfer_section(sec, parent_to_sub, parent_points)?)
        }
        TaggedSection::U64(sec) => {
            TaggedSection::U64(transfer_section(sec, parent_to_sub, parent_points)?)
        }
    })
}

fn transfer_coordinates<V, S>(
    coords: &Coordinates<V, S>,
    parent_to_sub: &HashMap<PointId, PointId>,
    parent_points: &[PointId],
) -> Result<Coordinates<V, S>, MeshSieveError>
where
    V: Clone + Default,
    S: Storage<V> + Clone,
{
    let section = transfer_section(coords.section(), parent_to_sub, parent_points)?;
    let mut out = Coordinates::from_section(coords.dimension(), section)?;
    if let Some(high_order) = coords.high_order() {
        let ho_section = transfer_section(high_order.section(), parent_to_sub, parent_points)?;
        let ho = HighOrderCoordinates::from_section(high_order.dimension(), ho_section)?;
        out.set_high_order(ho)?;
    }
    Ok(out)
}

fn remap_labels(labels: &LabelSet, parent_to_sub: &HashMap<PointId, PointId>) -> LabelSet {
    let mut out = LabelSet::new();
    for (name, point, value) in labels.iter() {
        if let Some(&sub) = parent_to_sub.get(&point) {
            out.set_label(sub, name, value);
        }
    }
    out
}
