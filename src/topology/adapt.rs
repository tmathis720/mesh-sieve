//! Adaptivity driver for refinement and coarsening decisions.

use crate::data::atlas::Atlas;
use crate::data::coordinates::Coordinates;
use crate::data::section::Section;
use crate::data::storage::{Storage, VecStorage};
use crate::mesh_error::MeshSieveError;
use crate::topology::cell_type::CellType;
use crate::topology::coarsen::{coarsen_topology, CoarsenEntity, CoarsenedTopology};
use crate::topology::point::PointId;
use crate::topology::refine::{refine_mesh_with_options, RefineOptions, RefinedMesh};
use crate::topology::sieve::Sieve;

/// Configuration for the adaptivity driver.
#[derive(Clone, Copy, Debug)]
pub struct AdaptivityOptions {
    /// Indicator threshold above which cells are refined.
    pub refine_threshold: f64,
    /// Indicator threshold below which cells are coarsened.
    pub coarsen_threshold: f64,
    /// When enabled, enforce geometry checks during refinement.
    pub check_geometry: bool,
}

impl Default for AdaptivityOptions {
    fn default() -> Self {
        Self {
            refine_threshold: 0.5,
            coarsen_threshold: 0.1,
            check_geometry: false,
        }
    }
}

/// Outcome of an adaptivity pass.
#[derive(Clone, Debug)]
pub enum AdaptivityAction {
    /// The driver refined a subset of cells.
    Refined(RefinedMesh),
    /// The driver coarsened a subset of cells.
    Coarsened(CoarsenedTopology),
    /// No changes were requested.
    NoChange,
}

/// Summary of a refinement/coarsening pass.
#[derive(Clone, Debug)]
pub struct AdaptivityResult {
    /// Cells chosen for refinement.
    pub refine_cells: Vec<PointId>,
    /// Cells chosen for coarsening.
    pub coarsen_cells: Vec<PointId>,
    /// The resulting action taken by the driver.
    pub action: AdaptivityAction,
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

/// Apply refinement or coarsening based on an indicator function.
///
/// When both refinement and coarsening are requested, refinement is applied
/// preferentially and coarsening is deferred to a later pass.
pub fn adapt_topology<S, Cs, F, C>(
    sieve: &mut impl Sieve<Point = PointId>,
    cell_types: &Section<CellType, S>,
    coordinates: Option<&Coordinates<f64, Cs>>,
    indicator: F,
    coarsen_plan: C,
    options: AdaptivityOptions,
) -> Result<AdaptivityResult, MeshSieveError>
where
    S: Storage<CellType>,
    Cs: Storage<f64>,
    F: Fn(PointId, CellType) -> f64,
    C: Fn(&[PointId]) -> Vec<CoarsenEntity>,
{
    let mut refine_cells = Vec::new();
    let mut coarsen_cells = Vec::new();

    for (cell, slice) in cell_types.iter() {
        if slice.len() != 1 {
            return Err(MeshSieveError::SliceLengthMismatch {
                point: cell,
                expected: 1,
                found: slice.len(),
            });
        }
        let value = indicator(cell, slice[0]);
        if value > options.refine_threshold {
            refine_cells.push(cell);
        } else if value < options.coarsen_threshold {
            coarsen_cells.push(cell);
        }
    }

    if !refine_cells.is_empty() {
        let refined_section = subset_cell_types(cell_types, &refine_cells)?;
        let refined = refine_mesh_with_options(
            sieve,
            &refined_section,
            coordinates,
            RefineOptions {
                check_geometry: options.check_geometry,
                anisotropic_splits: None,
            },
        )?;
        return Ok(AdaptivityResult {
            refine_cells,
            coarsen_cells,
            action: AdaptivityAction::Refined(refined),
        });
    }

    if !coarsen_cells.is_empty() {
        let entities = coarsen_plan(&coarsen_cells);
        if entities.is_empty() {
            return Ok(AdaptivityResult {
                refine_cells,
                coarsen_cells,
                action: AdaptivityAction::NoChange,
            });
        }
        let coarsened = coarsen_topology(sieve, &entities)?;
        return Ok(AdaptivityResult {
            refine_cells,
            coarsen_cells,
            action: AdaptivityAction::Coarsened(coarsened),
        });
    }

    Ok(AdaptivityResult {
        refine_cells,
        coarsen_cells,
        action: AdaptivityAction::NoChange,
    })
}
