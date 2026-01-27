//! Quality-driven adaptivity with data transfer helpers.

use crate::data::atlas::Atlas;
use crate::data::coordinates::Coordinates;
use crate::data::refine::delta::SliceDelta;
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
    collapse_to_cell_vertices, refine_mesh_with_options, RefineOptions, RefinedMesh,
};
use crate::topology::sieve::Sieve;

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
