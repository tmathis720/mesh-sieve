//! Field transfer utilities between meshes.
//!
//! Supports transferring [`Section`] data using refinement maps or nearest-neighbor
//! interpolation (points or cell centroids). Pointwise transfer via shared labels
//! is also available.

use crate::data::atlas::Atlas;
use crate::data::coordinates::Coordinates;
use crate::data::section::Section;
use crate::data::storage::Storage;
use crate::mesh_error::MeshSieveError;
use crate::topology::labels::LabelSet;
use crate::topology::point::PointId;
use crate::topology::refine::RefinementMap;
use crate::topology::sieve::Sieve;
use std::collections::{HashMap, HashSet};

/// Transfer a section using a refinement map (coarse → fine), copying coarse values
/// to all refined points.
pub fn transfer_section_by_refinement_map<V, S>(
    coarse: &Section<V, S>,
    refinement: &RefinementMap,
) -> Result<Section<V, S>, MeshSieveError>
where
    V: Clone + Default,
    S: Storage<V> + Clone,
{
    let mut atlas = Atlas::default();
    let mut seen = HashSet::new();

    for (coarse_point, fine_points) in refinement {
        let (_, len) = coarse
            .atlas()
            .get(*coarse_point)
            .ok_or(MeshSieveError::MissingSectionPoint(*coarse_point))?;
        for (fine_point, _) in fine_points {
            if !seen.insert(*fine_point) {
                return Err(MeshSieveError::DuplicateRefinementTarget { fine: *fine_point });
            }
            atlas.try_insert(*fine_point, len)?;
        }
    }

    let mut out = Section::new(atlas);
    for (coarse_point, fine_points) in refinement {
        let data = coarse.try_restrict(*coarse_point)?;
        for (fine_point, _) in fine_points {
            out.try_set(*fine_point, data)?;
        }
    }
    Ok(out)
}

/// Transfer a section between meshes using matching label values.
///
/// Each label value in `target_labels` for `label_name` maps to the same label
/// value in `source_labels`, and the corresponding source point data is copied.
pub fn transfer_section_by_shared_labels<V, S>(
    source: &Section<V, S>,
    source_labels: &LabelSet,
    target_labels: &LabelSet,
    label_name: &str,
) -> Result<Section<V, S>, MeshSieveError>
where
    V: Clone + Default,
    S: Storage<V> + Clone,
{
    let mut label_to_point = HashMap::new();
    for (name, point, value) in source_labels.iter() {
        if name == label_name {
            if label_to_point.insert(value, point).is_some() {
                return Err(MeshSieveError::InvalidGeometry(format!(
                    "duplicate source label value {value} for label '{label_name}'"
                )));
            }
        }
    }

    let mut atlas = Atlas::default();
    let mut transfers = Vec::new();
    for (name, target_point, value) in target_labels.iter() {
        if name != label_name {
            continue;
        }
        let source_point = label_to_point.get(&value).ok_or_else(|| {
            MeshSieveError::InvalidGeometry(format!(
                "missing source label value {value} for label '{label_name}'"
            ))
        })?;
        let (_, len) = source
            .atlas()
            .get(*source_point)
            .ok_or(MeshSieveError::MissingSectionPoint(*source_point))?;
        atlas.try_insert(target_point, len)?;
        transfers.push((target_point, *source_point));
    }

    let mut out = Section::new(atlas);
    for (target_point, source_point) in transfers {
        let data = source.try_restrict(source_point)?;
        out.try_set(target_point, data)?;
    }
    Ok(out)
}

/// Transfer a section by nearest-neighbor point interpolation using coordinates.
pub fn transfer_section_by_nearest_point<V, S, Cs>(
    source: &Section<V, S>,
    source_coords: &Coordinates<f64, Cs>,
    target_coords: &Coordinates<f64, Cs>,
) -> Result<Section<V, S>, MeshSieveError>
where
    V: Clone + Default,
    S: Storage<V> + Clone,
    Cs: Storage<f64>,
{
    let dimension = source_coords.dimension();
    if dimension != target_coords.dimension() {
        return Err(MeshSieveError::InvalidGeometry(format!(
            "coordinate dimension mismatch: source {dimension}, target {}",
            target_coords.dimension()
        )));
    }

    let source_points: Vec<PointId> = source.atlas().points().collect();
    if source_points.is_empty() {
        return Err(MeshSieveError::InvalidGeometry(
            "nearest-point transfer requires at least one source point".into(),
        ));
    }

    let mut transfers = Vec::new();
    for target_point in target_coords.section().atlas().points() {
        let target_pos = target_coords.try_restrict(target_point)?;
        let mut best = None;
        for source_point in &source_points {
            let source_pos = source_coords.try_restrict(*source_point)?;
            let dist = squared_distance(source_pos, target_pos);
            match best {
                Some((best_dist, _)) if dist >= best_dist => {}
                _ => best = Some((dist, *source_point)),
            }
        }
        let source_point = best
            .map(|(_, point)| point)
            .expect("source_points is non-empty");
        transfers.push((target_point, source_point));
    }

    let mut atlas = Atlas::default();
    for (target_point, source_point) in &transfers {
        let (_, len) = source
            .atlas()
            .get(*source_point)
            .ok_or(MeshSieveError::MissingSectionPoint(*source_point))?;
        atlas.try_insert(*target_point, len)?;
    }

    let mut out = Section::new(atlas);
    for (target_point, source_point) in transfers {
        let data = source.try_restrict(source_point)?;
        out.try_set(target_point, data)?;
    }
    Ok(out)
}

/// Transfer a section by nearest-neighbor cell centroids using coordinates.
///
/// Cell centroids are computed from points in the closure that have coordinates
/// available (typically vertices). If a cell point itself has coordinates, they
/// are used directly.
pub fn transfer_section_by_nearest_cell_centroid<V, S, Cs, Ss, Ts>(
    source: &Section<V, S>,
    source_sieve: &Ss,
    source_coords: &Coordinates<f64, Cs>,
    target_sieve: &Ts,
    target_coords: &Coordinates<f64, Cs>,
    target_cells: impl IntoIterator<Item = PointId>,
) -> Result<Section<V, S>, MeshSieveError>
where
    V: Clone + Default,
    S: Storage<V> + Clone,
    Cs: Storage<f64>,
    Ss: Sieve<Point = PointId>,
    Ts: Sieve<Point = PointId>,
{
    let dimension = source_coords.dimension();
    if dimension != target_coords.dimension() {
        return Err(MeshSieveError::InvalidGeometry(format!(
            "coordinate dimension mismatch: source {dimension}, target {}",
            target_coords.dimension()
        )));
    }

    let mut source_centroids = Vec::new();
    for point in source.atlas().points() {
        let centroid = centroid_from_closure(source_sieve, source_coords, point)?;
        source_centroids.push((point, centroid));
    }

    if source_centroids.is_empty() {
        return Err(MeshSieveError::InvalidGeometry(
            "nearest-cell transfer requires at least one source cell".into(),
        ));
    }

    let mut transfers = Vec::new();
    for target_cell in target_cells {
        let target_centroid = centroid_from_closure(target_sieve, target_coords, target_cell)?;
        let mut best = None;
        for (source_cell, source_centroid) in &source_centroids {
            let dist = squared_distance(&target_centroid, source_centroid);
            match best {
                Some((best_dist, _)) if dist >= best_dist => {}
                _ => best = Some((dist, *source_cell)),
            }
        }
        let source_cell = best
            .map(|(_, cell)| cell)
            .expect("source_centroids is non-empty");
        transfers.push((target_cell, source_cell));
    }

    let mut atlas = Atlas::default();
    for (target_cell, source_cell) in &transfers {
        let (_, len) = source
            .atlas()
            .get(*source_cell)
            .ok_or(MeshSieveError::MissingSectionPoint(*source_cell))?;
        atlas.try_insert(*target_cell, len)?;
    }

    let mut out = Section::new(atlas);
    for (target_cell, source_cell) in transfers {
        let data = source.try_restrict(source_cell)?;
        out.try_set(target_cell, data)?;
    }
    Ok(out)
}

fn squared_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(lhs, rhs)| {
            let diff = lhs - rhs;
            diff * diff
        })
        .sum()
}

fn centroid_from_closure<S, Cs>(
    sieve: &S,
    coords: &Coordinates<f64, Cs>,
    point: PointId,
) -> Result<Vec<f64>, MeshSieveError>
where
    S: Sieve<Point = PointId>,
    Cs: Storage<f64>,
{
    let dimension = coords.dimension();
    if coords.section().atlas().contains(point) {
        return Ok(coords.try_restrict(point)?.to_vec());
    }

    let mut sum = vec![0.0; dimension];
    let mut count = 0usize;
    for p in sieve.closure_iter([point]) {
        if coords.section().atlas().contains(p) {
            let slice = coords.try_restrict(p)?;
            for (idx, value) in slice.iter().enumerate() {
                sum[idx] += *value;
            }
            count += 1;
        }
    }

    if count == 0 {
        return Err(MeshSieveError::InvalidGeometry(format!(
            "no coordinate-bearing points found in closure of {point:?}"
        )));
    }

    let inv = 1.0 / count as f64;
    for value in &mut sum {
        *value *= inv;
    }
    Ok(sum)
}
