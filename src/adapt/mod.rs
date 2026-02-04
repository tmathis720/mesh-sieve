//! Quality-driven adaptivity with data transfer helpers.

use crate::data::atlas::Atlas;
use crate::data::coordinates::Coordinates;
use crate::data::refine::sieved_array::SievedArray;
use crate::data::section::Section;
use crate::data::storage::{Storage, VecStorage};
use crate::geometry::quality::{cell_quality_from_section, CellQuality};
use crate::mesh_error::MeshSieveError;
use crate::topology::arrow::Polarity;
use crate::topology::cell_type::CellType;
use crate::topology::coarsen::{coarsen_topology, CoarsenEntity, CoarsenedTopology};
use crate::topology::point::PointId;
use crate::topology::refine::{
    collapse_to_cell_vertices, refine_mesh_with_options, AnisotropicSplitHints, RefineOptions,
    RefinedMesh,
};
use crate::topology::sieve::Sieve;
use std::collections::HashMap;

/// Aggregated quality metrics used for adaptivity decisions.
#[derive(Clone, Copy, Debug)]
pub struct QualityMetrics {
    /// Ratio of the longest edge length to the shortest edge length.
    pub aspect_ratio: f64,
    /// Minimum corner angle (degrees).
    pub min_angle_deg: f64,
    /// Signed Jacobian (area for 2D, volume for 3D).
    pub jacobian_sign: f64,
    /// Absolute cell size (area/volume) derived from the Jacobian.
    pub cell_size: f64,
}

impl From<CellQuality> for QualityMetrics {
    fn from(value: CellQuality) -> Self {
        Self {
            aspect_ratio: value.aspect_ratio,
            min_angle_deg: value.min_angle_deg,
            jacobian_sign: value.jacobian_sign,
            cell_size: value.jacobian_sign.abs(),
        }
    }
}

/// Evaluate quality metrics for each cell in the mesh.
pub fn evaluate_quality_metrics<S, Cs>(
    sieve: &mut impl Sieve<Point = PointId>,
    cell_types: &Section<CellType, S>,
    coordinates: &Coordinates<f64, Cs>,
) -> Result<Vec<(PointId, QualityMetrics)>, MeshSieveError>
where
    S: Storage<CellType>,
    Cs: Storage<f64>,
{
    let collapsed = collapse_to_cell_vertices(sieve, cell_types)?;
    let mut metrics = Vec::new();
    for (cell, cell_slice) in cell_types.iter() {
        if cell_slice.len() != 1 {
            return Err(MeshSieveError::SliceLengthMismatch {
                point: cell,
                expected: 1,
                found: cell_slice.len(),
            });
        }
        let cell_type = cell_slice[0];
        let vertices: Vec<_> = collapsed.cone_points(cell).collect();
        let quality = cell_quality_from_section(cell_type, &vertices, coordinates)?;
        metrics.push((cell, QualityMetrics::from(quality)));
    }
    Ok(metrics)
}

/// Thresholds for quality-driven refinement and coarsening.
#[derive(Clone, Copy, Debug)]
pub struct QualityThresholds {
    /// Refine when `min_angle_deg` drops below this value.
    pub refine_min_angle_deg: f64,
    /// Refine when `aspect_ratio` exceeds this value.
    pub refine_max_aspect_ratio: f64,
    /// Refine when `cell_size` exceeds this value.
    pub refine_max_size: f64,
    /// Coarsen when `min_angle_deg` exceeds this value.
    pub coarsen_min_angle_deg: f64,
    /// Coarsen when `aspect_ratio` is below this value.
    pub coarsen_max_aspect_ratio: f64,
    /// Coarsen when `cell_size` drops below this value.
    pub coarsen_min_size: f64,
    /// When enabled, enforce geometry checks during refinement.
    pub check_geometry: bool,
}

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            refine_min_angle_deg: 20.0,
            refine_max_aspect_ratio: 4.0,
            refine_max_size: f64::INFINITY,
            coarsen_min_angle_deg: 30.0,
            coarsen_max_aspect_ratio: 2.0,
            coarsen_min_size: 0.0,
            check_geometry: false,
        }
    }
}

/// Symmetric metric tensor for anisotropic sizing (2D/3D).
#[derive(Clone, Copy, Debug)]
pub struct MetricTensor {
    /// Dimension of the metric (2 or 3).
    pub dimension: usize,
    /// Symmetric tensor entries stored as (m00, m11, m22, m01, m02, m12).
    pub data: [f64; 6],
}

impl MetricTensor {
    /// Construct a 2D metric tensor.
    pub fn new_2d(m00: f64, m11: f64, m01: f64) -> Self {
        Self {
            dimension: 2,
            data: [m00, m11, 0.0, m01, 0.0, 0.0],
        }
    }

    /// Construct a 3D metric tensor.
    pub fn new_3d(m00: f64, m11: f64, m22: f64, m01: f64, m02: f64, m12: f64) -> Self {
        Self {
            dimension: 3,
            data: [m00, m11, m22, m01, m02, m12],
        }
    }

    fn length_squared(&self, vector: &[f64]) -> Result<f64, MeshSieveError> {
        match self.dimension {
            2 => {
                if vector.len() < 2 {
                    return Err(MeshSieveError::InvalidGeometry(
                        "metric tensor expects 2D vectors".into(),
                    ));
                }
                let (x, y) = (vector[0], vector[1]);
                let [m00, m11, _, m01, _, _] = self.data;
                Ok(m00 * x * x + m11 * y * y + 2.0 * m01 * x * y)
            }
            3 => {
                if vector.len() < 3 {
                    return Err(MeshSieveError::InvalidGeometry(
                        "metric tensor expects 3D vectors".into(),
                    ));
                }
                let (x, y, z) = (vector[0], vector[1], vector[2]);
                let [m00, m11, m22, m01, m02, m12] = self.data;
                Ok(m00 * x * x
                    + m11 * y * y
                    + m22 * z * z
                    + 2.0 * (m01 * x * y + m02 * x * z + m12 * y * z))
            }
            _ => Err(MeshSieveError::InvalidGeometry(
                "metric tensor dimension must be 2 or 3".into(),
            )),
        }
    }
}

/// Metric-driven cell summary.
#[derive(Clone, Copy, Debug)]
pub struct MetricCellMetrics {
    /// Maximum metric edge length in the cell.
    pub max_edge_length: f64,
    /// Minimum metric edge length in the cell.
    pub min_edge_length: f64,
    /// Ratio of maximum to minimum metric edge lengths.
    pub anisotropy_ratio: f64,
}

/// Thresholds for metric-driven refinement and coarsening.
#[derive(Clone, Copy, Debug)]
pub struct MetricThresholds {
    /// Refine when the maximum metric edge length exceeds this value.
    pub refine_max_edge_length: f64,
    /// Coarsen when the maximum metric edge length drops below this value.
    pub coarsen_max_edge_length: f64,
    /// Split edges when their metric length exceeds this value.
    pub split_edge_length: f64,
    /// Split faces when their metric length exceeds this value.
    pub split_face_length: f64,
    /// When enabled, enforce geometry checks during refinement.
    pub check_geometry: bool,
}

impl Default for MetricThresholds {
    fn default() -> Self {
        Self {
            refine_max_edge_length: 1.2,
            coarsen_max_edge_length: 0.8,
            split_edge_length: 1.0,
            split_face_length: 1.0,
            check_geometry: false,
        }
    }
}

/// Edge/face split hints for anisotropic refinement.
#[derive(Clone, Debug)]
pub struct MetricSplitHint {
    /// Cell that owns the split hints.
    pub cell: PointId,
    /// Edges to split represented by vertex pairs.
    pub split_edges: Vec<[PointId; 2]>,
    /// Faces to split represented by ordered vertex loops.
    pub split_faces: Vec<Vec<PointId>>,
}

#[derive(Clone, Debug)]
struct MetricCellEvaluation {
    cell: PointId,
    metrics: MetricCellMetrics,
    edge_lengths: Vec<([PointId; 2], f64)>,
    face_lengths: Vec<(Vec<PointId>, f64)>,
}

/// Cells selected for refinement and coarsening.
#[derive(Clone, Debug, Default)]
pub struct AdaptivitySelection {
    /// Cells chosen for refinement.
    pub refine_cells: Vec<PointId>,
    /// Cells chosen for coarsening.
    pub coarsen_cells: Vec<PointId>,
}

/// Select cells for refinement and coarsening based on quality thresholds.
pub fn select_cells_for_adaptation(
    metrics: &[(PointId, QualityMetrics)],
    thresholds: QualityThresholds,
) -> AdaptivitySelection {
    let mut selection = AdaptivitySelection::default();
    for (cell, quality) in metrics {
        if quality.min_angle_deg < thresholds.refine_min_angle_deg
            || quality.aspect_ratio > thresholds.refine_max_aspect_ratio
            || quality.cell_size > thresholds.refine_max_size
        {
            selection.refine_cells.push(*cell);
        } else if quality.min_angle_deg >= thresholds.coarsen_min_angle_deg
            && quality.aspect_ratio <= thresholds.coarsen_max_aspect_ratio
            && quality.cell_size < thresholds.coarsen_min_size
        {
            selection.coarsen_cells.push(*cell);
        }
    }
    selection
}

fn polygon_edges(vertices: &[PointId]) -> Vec<[PointId; 2]> {
    if vertices.len() < 2 {
        return Vec::new();
    }
    let mut edges = Vec::with_capacity(vertices.len());
    for i in 0..vertices.len() {
        let next = (i + 1) % vertices.len();
        edges.push([vertices[i], vertices[next]]);
    }
    edges
}

fn cell_edges(cell_type: CellType, vertices: &[PointId]) -> Result<Vec<[PointId; 2]>, MeshSieveError> {
    let edges = match cell_type {
        CellType::Triangle => vec![
            [vertices[0], vertices[1]],
            [vertices[1], vertices[2]],
            [vertices[2], vertices[0]],
        ],
        CellType::Quadrilateral => vec![
            [vertices[0], vertices[1]],
            [vertices[1], vertices[2]],
            [vertices[2], vertices[3]],
            [vertices[3], vertices[0]],
        ],
        CellType::Polygon(_) => polygon_edges(vertices),
        CellType::Tetrahedron => vec![
            [vertices[0], vertices[1]],
            [vertices[1], vertices[2]],
            [vertices[2], vertices[0]],
            [vertices[0], vertices[3]],
            [vertices[1], vertices[3]],
            [vertices[2], vertices[3]],
        ],
        CellType::Hexahedron => vec![
            [vertices[0], vertices[1]],
            [vertices[1], vertices[2]],
            [vertices[2], vertices[3]],
            [vertices[3], vertices[0]],
            [vertices[4], vertices[5]],
            [vertices[5], vertices[6]],
            [vertices[6], vertices[7]],
            [vertices[7], vertices[4]],
            [vertices[0], vertices[4]],
            [vertices[1], vertices[5]],
            [vertices[2], vertices[6]],
            [vertices[3], vertices[7]],
        ],
        CellType::Prism => vec![
            [vertices[0], vertices[1]],
            [vertices[1], vertices[2]],
            [vertices[2], vertices[0]],
            [vertices[3], vertices[4]],
            [vertices[4], vertices[5]],
            [vertices[5], vertices[3]],
            [vertices[0], vertices[3]],
            [vertices[1], vertices[4]],
            [vertices[2], vertices[5]],
        ],
        CellType::Polyhedron => {
            return Err(MeshSieveError::InvalidGeometry(
                "metric evaluation does not support polyhedron edges".into(),
            ));
        }
        _ => {
            return Err(MeshSieveError::InvalidGeometry(format!(
                "unsupported cell type: {cell_type:?}"
            )));
        }
    };
    Ok(edges)
}

fn cell_faces(cell_type: CellType, vertices: &[PointId]) -> Result<Vec<Vec<PointId>>, MeshSieveError> {
    let faces = match cell_type {
        CellType::Tetrahedron => vec![
            vec![vertices[0], vertices[1], vertices[2]],
            vec![vertices[0], vertices[1], vertices[3]],
            vec![vertices[1], vertices[2], vertices[3]],
            vec![vertices[2], vertices[0], vertices[3]],
        ],
        CellType::Hexahedron => vec![
            vec![vertices[0], vertices[1], vertices[2], vertices[3]],
            vec![vertices[4], vertices[5], vertices[6], vertices[7]],
            vec![vertices[0], vertices[1], vertices[5], vertices[4]],
            vec![vertices[1], vertices[2], vertices[6], vertices[5]],
            vec![vertices[2], vertices[3], vertices[7], vertices[6]],
            vec![vertices[3], vertices[0], vertices[4], vertices[7]],
        ],
        CellType::Prism => vec![
            vec![vertices[0], vertices[1], vertices[2]],
            vec![vertices[3], vertices[4], vertices[5]],
            vec![vertices[0], vertices[1], vertices[4], vertices[3]],
            vec![vertices[1], vertices[2], vertices[5], vertices[4]],
            vec![vertices[2], vertices[0], vertices[3], vertices[5]],
        ],
        CellType::Polyhedron => {
            return Err(MeshSieveError::InvalidGeometry(
                "metric evaluation does not support polyhedron faces".into(),
            ));
        }
        _ => Vec::new(),
    };
    Ok(faces)
}

fn metric_edge_length<Cs>(
    tensor: MetricTensor,
    coords: &Coordinates<f64, Cs>,
    edge: [PointId; 2],
) -> Result<f64, MeshSieveError>
where
    Cs: Storage<f64>,
{
    let a = coords.try_restrict(edge[0])?;
    let b = coords.try_restrict(edge[1])?;
    if a.len() != b.len() {
        return Err(MeshSieveError::InvalidGeometry(
            "edge coordinate dimensions do not match".into(),
        ));
    }
    if a.len() != tensor.dimension {
        return Err(MeshSieveError::InvalidGeometry(
            "metric tensor dimension does not match coordinates".into(),
        ));
    }
    let vector: Vec<f64> = b.iter().zip(a.iter()).map(|(bv, av)| bv - av).collect();
    let length_sq = tensor.length_squared(&vector)?;
    Ok(length_sq.max(0.0).sqrt())
}

fn evaluate_metric_cells<S, Cs, Ms>(
    sieve: &mut impl Sieve<Point = PointId>,
    cell_types: &Section<CellType, S>,
    coordinates: &Coordinates<f64, Cs>,
    metrics: &Section<MetricTensor, Ms>,
) -> Result<Vec<MetricCellEvaluation>, MeshSieveError>
where
    S: Storage<CellType>,
    Cs: Storage<f64>,
    Ms: Storage<MetricTensor>,
{
    let collapsed = collapse_to_cell_vertices(sieve, cell_types)?;
    let mut evaluations = Vec::new();
    for (cell, cell_slice) in cell_types.iter() {
        if cell_slice.len() != 1 {
            return Err(MeshSieveError::SliceLengthMismatch {
                point: cell,
                expected: 1,
                found: cell_slice.len(),
            });
        }
        let tensor_slice = metrics.try_restrict(cell)?;
        if tensor_slice.len() != 1 {
            return Err(MeshSieveError::SliceLengthMismatch {
                point: cell,
                expected: 1,
                found: tensor_slice.len(),
            });
        }
        let tensor = tensor_slice[0];
        let cell_type = cell_slice[0];
        let vertices: Vec<_> = collapsed.cone_points(cell).collect();
        let edges = cell_edges(cell_type, &vertices)?;
        if edges.is_empty() {
            return Err(MeshSieveError::InvalidGeometry(format!(
                "cell {cell:?} has no edges for metric evaluation"
            )));
        }
        let mut edge_lengths = Vec::with_capacity(edges.len());
        for edge in edges {
            let length = metric_edge_length(tensor, coordinates, edge)?;
            edge_lengths.push((edge, length));
        }
        let mut max_edge = 0.0;
        let mut min_edge = f64::INFINITY;
        for (_, length) in &edge_lengths {
            if *length > max_edge {
                max_edge = *length;
            }
            if *length < min_edge {
                min_edge = *length;
            }
        }
        let anisotropy_ratio = if min_edge > 0.0 {
            max_edge / min_edge
        } else {
            f64::INFINITY
        };
        let face_lengths = cell_faces(cell_type, &vertices)?
            .into_iter()
            .map(|face_vertices| {
                let mut face_max = 0.0;
                for edge in polygon_edges(&face_vertices) {
                    let length = metric_edge_length(tensor, coordinates, edge)?;
                    if length > face_max {
                        face_max = length;
                    }
                }
                Ok((face_vertices, face_max))
            })
            .collect::<Result<Vec<_>, MeshSieveError>>()?;
        evaluations.push(MetricCellEvaluation {
            cell,
            metrics: MetricCellMetrics {
                max_edge_length: max_edge,
                min_edge_length: min_edge,
                anisotropy_ratio,
            },
            edge_lengths,
            face_lengths,
        });
    }
    Ok(evaluations)
}

/// Select cells for refinement and coarsening based on metric thresholds.
pub fn select_cells_for_metric_adaptation(
    metrics: &[(PointId, MetricCellMetrics)],
    thresholds: MetricThresholds,
) -> AdaptivitySelection {
    let mut selection = AdaptivitySelection::default();
    for (cell, metric) in metrics {
        if metric.max_edge_length > thresholds.refine_max_edge_length {
            selection.refine_cells.push(*cell);
        } else if metric.max_edge_length < thresholds.coarsen_max_edge_length {
            selection.coarsen_cells.push(*cell);
        }
    }
    selection
}

fn metric_split_hints(
    evaluations: &[MetricCellEvaluation],
    thresholds: MetricThresholds,
) -> Vec<MetricSplitHint> {
    let mut hints = Vec::new();
    for evaluation in evaluations {
        let split_edges: Vec<_> = evaluation
            .edge_lengths
            .iter()
            .filter(|(_, length)| *length > thresholds.split_edge_length)
            .map(|(edge, _)| *edge)
            .collect();
        let split_faces: Vec<_> = evaluation
            .face_lengths
            .iter()
            .filter(|(_, length)| *length > thresholds.split_face_length)
            .map(|(face, _)| face.clone())
            .collect();
        if !split_edges.is_empty() || !split_faces.is_empty() {
            hints.push(MetricSplitHint {
                cell: evaluation.cell,
                split_edges,
                split_faces,
            });
        }
    }
    hints
}

fn build_anisotropic_hints(hints: &[MetricSplitHint]) -> AnisotropicSplitHints {
    let mut edge_splits: HashMap<PointId, Vec<[PointId; 2]>> = HashMap::new();
    let mut face_splits: HashMap<PointId, Vec<Vec<PointId>>> = HashMap::new();
    for hint in hints {
        if !hint.split_edges.is_empty() {
            edge_splits.insert(hint.cell, hint.split_edges.clone());
        }
        if !hint.split_faces.is_empty() {
            face_splits.insert(hint.cell, hint.split_faces.clone());
        }
    }
    AnisotropicSplitHints {
        edge_splits,
        face_splits,
    }
}

/// Outcome of quality-driven adaptivity with data transfer.
#[derive(Clone, Debug)]
pub enum AdaptationAction<V> {
    /// The driver refined a subset of cells.
    Refined {
        /// Refined topology and transfer map.
        mesh: RefinedMesh,
        /// Refined data mapped onto the new cells.
        data: SievedArray<PointId, V>,
    },
    /// The driver coarsened a subset of cells.
    Coarsened {
        /// Coarsened topology and transfer map.
        mesh: CoarsenedTopology,
        /// Coarsened data assembled onto the coarse cells.
        data: SievedArray<PointId, V>,
    },
    /// No changes were requested.
    NoChange,
}

/// Summary of a quality-driven adaptivity pass.
#[derive(Clone, Debug)]
pub struct AdaptationResult<V> {
    /// Computed metrics for each cell.
    pub metrics: Vec<(PointId, QualityMetrics)>,
    /// Cells chosen for refinement.
    pub refine_cells: Vec<PointId>,
    /// Cells chosen for coarsening.
    pub coarsen_cells: Vec<PointId>,
    /// The action taken by the driver.
    pub action: AdaptationAction<V>,
}

/// Outcome of metric-driven adaptivity without data transfer.
#[derive(Clone, Debug)]
pub enum MetricAdaptationAction {
    /// The driver refined a subset of cells.
    Refined { mesh: RefinedMesh },
    /// The driver coarsened a subset of cells.
    Coarsened { mesh: CoarsenedTopology },
    /// No changes were requested.
    NoChange,
}

/// Summary of a metric-driven adaptivity pass.
#[derive(Clone, Debug)]
pub struct MetricAdaptationResult<V> {
    /// Computed metric summaries for each cell.
    pub metrics: Vec<(PointId, MetricCellMetrics)>,
    /// Cells chosen for refinement.
    pub refine_cells: Vec<PointId>,
    /// Cells chosen for coarsening.
    pub coarsen_cells: Vec<PointId>,
    /// Edge/face split hints derived from the metric field.
    pub split_hints: Vec<MetricSplitHint>,
    /// The action taken by the driver.
    pub action: MetricAdaptationAction,
    /// Optional transferred data (only for data-aware adaptation).
    pub data: Option<SievedArray<PointId, V>>,
}

fn subset_cell_types<S>(
    cell_types: &Section<CellType, S>,
    cells: &[PointId],
) -> Result<Section<CellType, VecStorage<CellType>>, MeshSieveError>
where
    S: Storage<CellType>,
{
    let mut atlas = Atlas::default();
    for cell in cells {
        atlas.try_insert(*cell, 1)?;
    }
    let mut section = Section::<CellType, VecStorage<CellType>>::new(atlas);
    for cell in cells {
        let slice = cell_types.try_restrict(*cell)?;
        if slice.len() != 1 {
            return Err(MeshSieveError::SliceLengthMismatch {
                point: *cell,
                expected: 1,
                found: slice.len(),
            });
        }
        section.try_set(*cell, slice)?;
    }
    Ok(section)
}

fn refined_data_atlas<V>(
    coarse: &SievedArray<PointId, V>,
    refinement: &[(PointId, Vec<(PointId, Polarity)>)],
) -> Result<Atlas, MeshSieveError> {
    let mut atlas = Atlas::default();
    for (coarse_cell, fine_cells) in refinement {
        let (_offset, len) = coarse
            .atlas()
            .get(*coarse_cell)
            .ok_or(MeshSieveError::SievedArrayPointNotInAtlas(*coarse_cell))?;
        for (fine_cell, _) in fine_cells {
            atlas.try_insert(*fine_cell, len)?;
        }
    }
    Ok(atlas)
}

fn refine_data_with_transfer<V>(
    coarse: &SievedArray<PointId, V>,
    refinement: &[(PointId, Vec<(PointId, Polarity)>)],
) -> Result<SievedArray<PointId, V>, MeshSieveError>
where
    V: Clone + Default,
{
    let atlas = refined_data_atlas(coarse, refinement)?;
    let mut fine = SievedArray::<PointId, V>::new(atlas);
    fine.try_refine_with_sifter(coarse, refinement)?;
    Ok(fine)
}

fn coarsened_data_atlas<V>(
    fine: &SievedArray<PointId, V>,
    transfer: &[(PointId, Vec<(PointId, Polarity)>)],
) -> Result<Atlas, MeshSieveError> {
    let mut atlas = Atlas::default();
    for (coarse_cell, fine_cells) in transfer {
        let first = fine_cells
            .first()
            .ok_or_else(|| MeshSieveError::InvalidGeometry("empty coarsen map".into()))?;
        let (_offset, len) = fine
            .atlas()
            .get(first.0)
            .ok_or(MeshSieveError::SievedArrayPointNotInAtlas(first.0))?;
        atlas.try_insert(*coarse_cell, len)?;
    }
    Ok(atlas)
}

fn assemble_with_sifter<V>(
    fine: &SievedArray<PointId, V>,
    coarse: &mut SievedArray<PointId, V>,
    transfer: &[(PointId, Vec<(PointId, Polarity)>)],
) -> Result<(), MeshSieveError>
where
    V: Clone
        + Default
        + num_traits::FromPrimitive
        + core::ops::AddAssign
        + core::ops::Div<Output = V>,
{
    for (coarse_cell, fine_cells) in transfer {
        let coarse_len = coarse.try_get(*coarse_cell)?.len();
        let mut accum = vec![V::default(); coarse_len];
        let mut count = 0;
        for (fine_cell, polarity) in fine_cells {
            let slice = fine.try_get(*fine_cell)?;
            if slice.len() != accum.len() {
                return Err(MeshSieveError::SievedArraySliceLengthMismatch {
                    point: *fine_cell,
                    expected: accum.len(),
                    found: slice.len(),
                });
            }
            let mut oriented = vec![V::default(); slice.len()];
            polarity.apply(slice, &mut oriented)?;
            for (acc, val) in accum.iter_mut().zip(oriented.iter()) {
                *acc += val.clone();
            }
            count += 1;
        }
        if count > 0 {
            let divisor: V = num_traits::FromPrimitive::from_usize(count)
                .ok_or(MeshSieveError::SievedArrayPrimitiveConversionFailure(count))?;
            for val in accum.iter_mut() {
                *val = val.clone() / divisor.clone();
            }
            coarse.try_set(*coarse_cell, &accum)?;
        }
    }
    Ok(())
}

fn coarsen_data_with_transfer<V>(
    fine: &SievedArray<PointId, V>,
    transfer: &[(PointId, Vec<(PointId, Polarity)>)],
) -> Result<SievedArray<PointId, V>, MeshSieveError>
where
    V: Clone
        + Default
        + num_traits::FromPrimitive
        + core::ops::AddAssign
        + core::ops::Div<Output = V>,
{
    let atlas = coarsened_data_atlas(fine, transfer)?;
    let mut coarse = SievedArray::<PointId, V>::new(atlas);
    assemble_with_sifter(fine, &mut coarse, transfer)?;
    Ok(coarse)
}

/// Apply quality-driven refinement/coarsening with data transfer.
///
/// When both refinement and coarsening are requested, refinement is applied
/// preferentially and coarsening is deferred to a later pass.
pub fn adapt_with_quality_and_transfer<S, Cs, V, C>(
    sieve: &mut impl Sieve<Point = PointId>,
    cell_types: &Section<CellType, S>,
    coordinates: &Coordinates<f64, Cs>,
    cell_data: &SievedArray<PointId, V>,
    coarsen_plan: C,
    thresholds: QualityThresholds,
) -> Result<AdaptationResult<V>, MeshSieveError>
where
    S: Storage<CellType>,
    Cs: Storage<f64>,
    V: Clone
        + Default
        + num_traits::FromPrimitive
        + core::ops::AddAssign
        + core::ops::Div<Output = V>,
    C: Fn(&[PointId]) -> Vec<CoarsenEntity>,
{
    let metrics = evaluate_quality_metrics(sieve, cell_types, coordinates)?;
    let selection = select_cells_for_adaptation(&metrics, thresholds);
    let refine_cells = selection.refine_cells.clone();
    let coarsen_cells = selection.coarsen_cells.clone();

    if !refine_cells.is_empty() {
        let refined_section = subset_cell_types(cell_types, &refine_cells)?;
        let refined = refine_mesh_with_options(
            sieve,
            &refined_section,
            Some(coordinates),
            RefineOptions {
                check_geometry: thresholds.check_geometry,
                anisotropic_splits: None,
            },
        )?;
        let refined_data = refine_data_with_transfer(cell_data, &refined.cell_refinement)?;
        return Ok(AdaptationResult {
            metrics,
            refine_cells,
            coarsen_cells,
            action: AdaptationAction::Refined {
                mesh: refined,
                data: refined_data,
            },
        });
    }

    if !coarsen_cells.is_empty() {
        let entities = coarsen_plan(&coarsen_cells);
        if entities.is_empty() {
            return Ok(AdaptationResult {
                metrics,
                refine_cells,
                coarsen_cells,
                action: AdaptationAction::NoChange,
            });
        }
        let coarsened = coarsen_topology(sieve, &entities)?;
        let coarsened_data = coarsen_data_with_transfer(cell_data, &coarsened.transfer_map)?;
        return Ok(AdaptationResult {
            metrics,
            refine_cells,
            coarsen_cells,
            action: AdaptationAction::Coarsened {
                mesh: coarsened,
                data: coarsened_data,
            },
        });
    }

    Ok(AdaptationResult {
        metrics,
        refine_cells,
        coarsen_cells,
        action: AdaptationAction::NoChange,
    })
}

fn filter_split_hints(hints: &[MetricSplitHint], cells: &[PointId]) -> Vec<MetricSplitHint> {
    let cell_set: std::collections::HashSet<_> = cells.iter().copied().collect();
    hints
        .iter()
        .filter(|hint| cell_set.contains(&hint.cell))
        .cloned()
        .collect()
}

/// Apply metric-driven refinement/coarsening without data transfer.
///
/// This mirrors DMPlex-style "adapt with metric" workflows by using a per-cell metric tensor
/// to decide refinement, coarsening, and anisotropic split hints.
pub fn adapt_with_metric<S, Cs, Ms, C>(
    sieve: &mut impl Sieve<Point = PointId>,
    cell_types: &Section<CellType, S>,
    coordinates: &Coordinates<f64, Cs>,
    metric: &Section<MetricTensor, Ms>,
    coarsen_plan: C,
    thresholds: MetricThresholds,
) -> Result<MetricAdaptationResult<()>, MeshSieveError>
where
    S: Storage<CellType>,
    Cs: Storage<f64>,
    Ms: Storage<MetricTensor>,
    C: Fn(&[MetricSplitHint]) -> Vec<CoarsenEntity>,
{
    let evaluations = evaluate_metric_cells(sieve, cell_types, coordinates, metric)?;
    let metrics: Vec<_> = evaluations
        .iter()
        .map(|eval| (eval.cell, eval.metrics))
        .collect();
    let split_hints = metric_split_hints(&evaluations, thresholds);
    let selection = select_cells_for_metric_adaptation(&metrics, thresholds);
    let refine_cells = selection.refine_cells.clone();
    let coarsen_cells = selection.coarsen_cells.clone();

    if !refine_cells.is_empty() {
        let refined_section = subset_cell_types(cell_types, &refine_cells)?;
        let refine_hints = filter_split_hints(&split_hints, &refine_cells);
        let anisotropic = build_anisotropic_hints(&refine_hints);
        let refined = refine_mesh_with_options(
            sieve,
            &refined_section,
            Some(coordinates),
            RefineOptions {
                check_geometry: thresholds.check_geometry,
                anisotropic_splits: if anisotropic.is_empty() {
                    None
                } else {
                    Some(anisotropic)
                },
            },
        )?;
        return Ok(MetricAdaptationResult {
            metrics,
            refine_cells,
            coarsen_cells,
            split_hints,
            action: MetricAdaptationAction::Refined { mesh: refined },
            data: None,
        });
    }

    if !coarsen_cells.is_empty() {
        let coarsen_hints = filter_split_hints(&split_hints, &coarsen_cells);
        let entities = coarsen_plan(&coarsen_hints);
        if entities.is_empty() {
            return Ok(MetricAdaptationResult {
                metrics,
                refine_cells,
                coarsen_cells,
                split_hints,
                action: MetricAdaptationAction::NoChange,
                data: None,
            });
        }
        let coarsened = coarsen_topology(sieve, &entities)?;
        return Ok(MetricAdaptationResult {
            metrics,
            refine_cells,
            coarsen_cells,
            split_hints,
            action: MetricAdaptationAction::Coarsened { mesh: coarsened },
            data: None,
        });
    }

    Ok(MetricAdaptationResult {
        metrics,
        refine_cells,
        coarsen_cells,
        split_hints,
        action: MetricAdaptationAction::NoChange,
        data: None,
    })
}

/// Apply metric-driven refinement/coarsening with data transfer.
///
/// When both refinement and coarsening are requested, refinement is applied
/// preferentially and coarsening is deferred to a later pass.
pub fn adapt_with_metric_and_transfer<S, Cs, Ms, V, C>(
    sieve: &mut impl Sieve<Point = PointId>,
    cell_types: &Section<CellType, S>,
    coordinates: &Coordinates<f64, Cs>,
    metric: &Section<MetricTensor, Ms>,
    cell_data: &SievedArray<PointId, V>,
    coarsen_plan: C,
    thresholds: MetricThresholds,
) -> Result<MetricAdaptationResult<V>, MeshSieveError>
where
    S: Storage<CellType>,
    Cs: Storage<f64>,
    Ms: Storage<MetricTensor>,
    V: Clone
        + Default
        + num_traits::FromPrimitive
        + core::ops::AddAssign
        + core::ops::Div<Output = V>,
    C: Fn(&[MetricSplitHint]) -> Vec<CoarsenEntity>,
{
    let evaluations = evaluate_metric_cells(sieve, cell_types, coordinates, metric)?;
    let metrics: Vec<_> = evaluations
        .iter()
        .map(|eval| (eval.cell, eval.metrics))
        .collect();
    let split_hints = metric_split_hints(&evaluations, thresholds);
    let selection = select_cells_for_metric_adaptation(&metrics, thresholds);
    let refine_cells = selection.refine_cells.clone();
    let coarsen_cells = selection.coarsen_cells.clone();

    if !refine_cells.is_empty() {
        let refined_section = subset_cell_types(cell_types, &refine_cells)?;
        let refine_hints = filter_split_hints(&split_hints, &refine_cells);
        let anisotropic = build_anisotropic_hints(&refine_hints);
        let refined = refine_mesh_with_options(
            sieve,
            &refined_section,
            Some(coordinates),
            RefineOptions {
                check_geometry: thresholds.check_geometry,
                anisotropic_splits: if anisotropic.is_empty() {
                    None
                } else {
                    Some(anisotropic)
                },
            },
        )?;
        let refined_data = refine_data_with_transfer(cell_data, &refined.cell_refinement)?;
        return Ok(MetricAdaptationResult {
            metrics,
            refine_cells,
            coarsen_cells,
            split_hints,
            action: MetricAdaptationAction::Refined { mesh: refined },
            data: Some(refined_data),
        });
    }

    if !coarsen_cells.is_empty() {
        let coarsen_hints = filter_split_hints(&split_hints, &coarsen_cells);
        let entities = coarsen_plan(&coarsen_hints);
        if entities.is_empty() {
            return Ok(MetricAdaptationResult {
                metrics,
                refine_cells,
                coarsen_cells,
                split_hints,
                action: MetricAdaptationAction::NoChange,
                data: None,
            });
        }
        let coarsened = coarsen_topology(sieve, &entities)?;
        let coarsened_data = coarsen_data_with_transfer(cell_data, &coarsened.transfer_map)?;
        return Ok(MetricAdaptationResult {
            metrics,
            refine_cells,
            coarsen_cells,
            split_hints,
            action: MetricAdaptationAction::Coarsened { mesh: coarsened },
            data: Some(coarsened_data),
        });
    }

    Ok(MetricAdaptationResult {
        metrics,
        refine_cells,
        coarsen_cells,
        split_hints,
        action: MetricAdaptationAction::NoChange,
        data: None,
    })
}
