//! DMPLEX-like high-level mesh data-management facade.
//!
//! [`MeshDM`] is the feature-parity entry point for users coming from PETSc
//! DMPLEX.  It does not replace the lower-level sieve, section, label,
//! coordinate, distribution, and discretization modules; instead it owns those
//! pieces together and provides a single orchestration surface for the common
//! "build topology → attach data → validate → distribute → number sections →
//! assemble vectors/matrices" workflow.

use std::collections::{BTreeMap, HashSet};

use crate::adapt::{
    AdaptationProvenanceMap, BoundaryRemeshingPolicy, FvStabilityThresholds,
    MetricAdaptationAction, MetricAdaptationOptions, MetricAdaptationResult,
    MetricRemeshingBackend, MetricSplitHint, MetricTensor, MetricThresholds,
    adapt_with_metric_policy, select_cells_from_fvm_diagnostics, transfer_cell_types_refinement,
    transfer_labels_refinement, transfer_ownership_refinement,
};
use crate::algs::communicator::Communicator;
use crate::algs::distribute::{
    CellPartitioner, DistributedMeshData, DistributionConfig, distribute_with_overlap,
};
use crate::algs::dual_graph::{DualGraph, build_dual};
use crate::algs::point_sf::PointSF;
use crate::algs::point_sf::{PointSfLeaf, RemotePoint};
use crate::algs::renumber::{StratifiedOrdering, stratified_permutation};
use crate::algs::submesh::{SubmeshMaps, SubmeshSelection, extract_by_label};
use crate::data::atlas::Atlas;
use crate::data::closure::{ClosureIndex, ClosureOrder, IdentitySectionSym, build_closure_index};
use crate::data::coordinates::Coordinates;
use crate::data::discretization::Discretization;
use crate::data::global_map::{LocalToGlobalMap, global_vector_for_map};
use crate::data::multi_section::constrained_section_from_label_specs;
use crate::data::section::Section;
use crate::data::storage::{Storage, VecStorage};
use crate::data::{ConstrainedSection, LabelConstraintSpec};
use crate::diagnostics::{
    FvmQualityDiagnostics, MeshCheckOptions, PrepareForSolveDiagnostics, PrepareForSolveOptions,
    PrepareForSolvePreallocationDiagnostic, PrepareForSolveStepDiagnostic, fvm_cell_diagnostics,
    fvm_quality_diagnostics, prepare_for_solve_prerequisites, run_mesh_checks,
};
use crate::geometry::fvm::build_fvm_face_metrics;
use crate::io::MeshData;
use crate::mesh_error::MeshSieveError;
use crate::mesh_graph::{AdjacencyWeighting, MeshGraph, cell_adjacency_graph_with_cells};
use crate::overlap::overlap::Overlap;
use crate::physics::fe::{ElementClosureData, extract_oriented_element_closure};
use crate::topology::cache::InvalidateCache;
use crate::topology::cell_type::CellType;
use crate::topology::labels::LabelSet;
use crate::topology::ownership::PointOwnership;
use crate::topology::point::PointId;
use crate::topology::refine::refine_mesh;
use crate::topology::sieve::strata::compute_strata;
use crate::topology::sieve::{MeshSieve, Sieve};

/// Options for a DMPLEX-like setup pipeline.
///
/// The refinement counters are intentionally high-level policy markers.  They
/// are stored with the DM so application-specific refinement hooks can inspect
/// them; generic topology refinement remains available in lower-level modules
/// because mesh-sieve supports several refinement representations.
#[derive(Clone, Debug)]
pub struct MeshDMOptions {
    /// Number of pre-distribution refinement passes requested by the user.
    pub pre_refine: usize,
    /// Whether the setup pipeline is expected to distribute the DM.
    pub distribute: bool,
    /// Number of ghost layers to create during distribution.
    pub distribute_overlap: usize,
    /// Number of post-distribution refinement passes requested by the user.
    pub post_refine: usize,
    /// Ensure coordinate data is present on local/ghost points after distribution.
    pub localize_coordinates: bool,
    /// Run orientation/symmetry checks when cell-type metadata is available.
    pub check_symmetry: bool,
    /// Run stratum/skeleton construction checks.
    pub check_skeleton: bool,
    /// Run face adjacency construction checks.
    pub check_faces: bool,
    /// Validate coordinate geometry metadata when coordinates are present.
    pub check_geometry: bool,
    /// Enable all mesh checks in a DMPLEX-like single switch.
    pub check_all: bool,
    /// Reorder section atlas entries by topology strata during setup.
    pub reorder_section: Option<StratifiedOrdering>,
    /// Balance ownership of partition-boundary points during distribution.
    pub balance_boundary_ownership: bool,
    /// Synchronize section data onto ghost points during distribution.
    pub synchronize_sections: bool,
}

impl Default for MeshDMOptions {
    fn default() -> Self {
        Self {
            pre_refine: 0,
            distribute: false,
            distribute_overlap: 1,
            post_refine: 0,
            localize_coordinates: false,
            check_symmetry: false,
            check_skeleton: false,
            check_faces: false,
            check_geometry: false,
            check_all: false,
            reorder_section: None,
            balance_boundary_ownership: false,
            synchronize_sections: true,
        }
    }
}

impl MeshDMOptions {
    /// Convert the distribution-related subset to the lower-level config type.
    pub fn distribution_config(&self) -> DistributionConfig {
        DistributionConfig {
            overlap_depth: self.distribute_overlap,
            synchronize_sections: self.synchronize_sections || self.localize_coordinates,
            balance_boundary_ownership: self.balance_boundary_ownership,
        }
    }
}

/// Builder for [`MeshDM`].
#[derive(Debug)]
pub struct MeshDMBuilder<V, St = VecStorage<V>, CtSt = VecStorage<CellType>>
where
    St: Storage<V>,
    CtSt: Storage<CellType>,
{
    mesh_data: MeshData<MeshSieve, V, St, CtSt>,
    options: MeshDMOptions,
}

impl<V, St, CtSt> MeshDMBuilder<V, St, CtSt>
where
    V: Clone + Default,
    St: Storage<V> + Clone,
    CtSt: Storage<CellType> + Clone,
{
    /// Start a builder from an existing oriented mesh topology.
    pub fn new(topology: MeshSieve) -> Self {
        Self {
            mesh_data: MeshData::new(topology),
            options: MeshDMOptions::default(),
        }
    }

    /// Start a builder from a full mesh-data container.
    pub fn from_mesh_data(mesh_data: MeshData<MeshSieve, V, St, CtSt>) -> Self {
        Self {
            mesh_data,
            options: MeshDMOptions::default(),
        }
    }

    /// Replace the full options object.
    pub fn options(mut self, options: MeshDMOptions) -> Self {
        self.options = options;
        self
    }

    /// Mutate options in-place.
    pub fn configure(mut self, f: impl FnOnce(&mut MeshDMOptions)) -> Self {
        f(&mut self.options);
        self
    }

    /// Attach coordinate metadata.
    pub fn coordinates(mut self, coordinates: Coordinates<V, St>) -> Self {
        self.mesh_data.coordinates = Some(coordinates);
        self
    }

    /// Attach labels.
    pub fn labels(mut self, labels: LabelSet) -> Self {
        self.mesh_data.labels = Some(labels);
        self
    }

    /// Attach cell types.
    pub fn cell_types(mut self, cell_types: Section<CellType, CtSt>) -> Self {
        self.mesh_data.cell_types = Some(cell_types);
        self
    }

    /// Attach discretization metadata.
    pub fn discretization(mut self, discretization: Discretization) -> Self {
        self.mesh_data.discretization = Some(discretization);
        self
    }

    /// Attach a named scalar section.
    pub fn section(mut self, name: impl Into<String>, section: Section<V, St>) -> Self {
        self.mesh_data.sections.insert(name.into(), section);
        self
    }

    /// Build the DM and run serial setup checks/reordering requested in options.
    pub fn build(self) -> Result<MeshDM<V, St, CtSt>, MeshSieveError> {
        let mut dm = MeshDM::from_mesh_data_with_options(self.mesh_data, self.options);
        dm.setup_serial()?;
        Ok(dm)
    }
}

/// Minimal vector object created by a [`MeshDM`] section.
#[derive(Clone, Debug, PartialEq)]
pub struct MeshVector<V> {
    /// Optional section name for solver diagnostics.
    pub section: Option<String>,
    /// Contiguous values in local or global numbering order.
    pub values: Vec<V>,
}

/// Matrix/preallocation graph derived from a DM adjacency graph and section map.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreallocationGraph {
    /// CSR offsets into [`Self::adjncy`].
    pub xadj: Vec<usize>,
    /// CSR adjacency by point index.
    pub adjncy: Vec<usize>,
    /// Point ordering represented by CSR rows.
    pub order: Vec<PointId>,
    /// Number of point-neighbor blocks for each row.
    pub row_nnz: Vec<usize>,
}

/// Label-driven point/submesh extraction policy for DM helper methods.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MeshDMLabelSelection {
    /// Label name to query.
    pub label_name: String,
    /// Integer label value to query.
    pub label_value: i32,
    /// Topological subset to retain from labeled seed points.
    pub topology: SubmeshSelection,
    /// Optional named section whose atlas filters the selected points.
    pub section: Option<String>,
}

impl MeshDMLabelSelection {
    /// Select a label value and its full downward closure.
    pub fn new(label_name: impl Into<String>, label_value: i32) -> Self {
        Self {
            label_name: label_name.into(),
            label_value,
            topology: SubmeshSelection::FullClosure,
            section: None,
        }
    }

    /// Override the topology subsetting policy.
    pub fn topology(mut self, topology: SubmeshSelection) -> Self {
        self.topology = topology;
        self
    }

    /// Retain only selected points present in a named section atlas.
    pub fn section(mut self, section: impl Into<String>) -> Self {
        self.section = Some(section.into());
        self
    }
}

/// DM slice built by label/topology extraction.
#[derive(Debug)]
pub struct MeshDMSubmesh<V, St = VecStorage<V>, CtSt = VecStorage<CellType>>
where
    St: Storage<V>,
    CtSt: Storage<CellType>,
{
    /// Extracted DM with compact point numbering.
    pub dm: MeshDM<V, St, CtSt>,
    /// Bidirectional parent/submesh point map.
    pub maps: SubmeshMaps,
}

/// Distribution state owned by a [`MeshDM`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MeshDMDistribution {
    /// Point owners indexed by `PointId - 1`.
    pub point_owners: Vec<usize>,
    /// Cell partition assignment used to create this local DM.
    pub cell_parts: Vec<usize>,
    /// Rank that owns this local DM state.
    pub rank: usize,
    /// Communicator size at distribution time.
    pub size: usize,
}

/// Strategy for transferring non-topological state after an adaptation pass.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MeshDMTransferStrategy {
    /// Keep only topology-level transfer information; user sections are untouched.
    #[default]
    TopologyOnly,
    /// Keep and remap labels where a direct point mapping is available.
    PreserveLabels,
    /// Keep and remap coordinates and labels where direct maps are available.
    PreserveCoordinatesAndLabels,
    /// Preserve all DM-owned metadata that has explicit provenance maps.
    PreserveAll,
}

/// Label-aware adaptation constraints for DM-level metric workflows.
#[derive(Clone, Debug, Default)]
pub struct MeshDMAdaptLabelPolicy {
    pub fixed_boundary_labels: Vec<(String, i32)>,
    pub protected_region_labels: Vec<(String, i32)>,
    pub relax_region_labels: Vec<(String, i32)>,
}

/// Controls for a DM-level metric adaptation pass.
#[derive(Clone, Debug)]
pub struct MeshDMMetricAdaptOptions {
    pub thresholds: MetricThresholds,
    pub labels: MeshDMAdaptLabelPolicy,
    pub transfer: MeshDMTransferStrategy,
    pub normalization: crate::adapt::MetricNormalizationControls,
    pub backend: MetricRemeshingBackend,
}

impl Default for MeshDMMetricAdaptOptions {
    fn default() -> Self {
        Self {
            thresholds: MetricThresholds::default(),
            labels: MeshDMAdaptLabelPolicy::default(),
            transfer: MeshDMTransferStrategy::TopologyOnly,
            normalization: crate::adapt::MetricNormalizationControls::default(),
            backend: MetricRemeshingBackend::Internal,
        }
    }
}

/// Structured diagnostics emitted by DM-level adaptation.
#[derive(Clone, Debug, Default)]
pub struct MeshDMAdaptDiagnostics {
    pub changed_cells: usize,
    pub preserved_labels: usize,
    pub failed_transfer_points: Vec<PointId>,
    pub split_hint_cells: usize,
    pub fvm_face_metrics_cached: usize,
    pub boundary_faces: usize,
    pub provenance_edges: usize,
    pub transferred_sections: usize,
    pub transferred_constraints: usize,
    pub transferred_overlap_points: usize,
}

/// Result of a DM-level metric adaptation call.
#[derive(Clone, Debug)]
pub struct MeshDMMetricAdaptResult {
    pub action: MetricAdaptationAction,
    pub metrics: Vec<(PointId, crate::adapt::MetricCellMetrics)>,
    pub diagnostics: MeshDMAdaptDiagnostics,
    pub fv_selection: Option<(Vec<PointId>, Vec<PointId>)>,
    pub provenance: AdaptationProvenanceMap,
}

#[derive(Clone, Debug, Default)]
pub struct MeshDMAdaptIterationOptions {
    pub max_iterations: usize,
    pub stop_when_below_threshold: bool,
}

#[derive(Clone, Debug, Default)]
pub struct MeshDMFvIterationReport {
    pub pass_index: usize,
    pub fvm_quality: Option<FvmQualityDiagnostics>,
    pub refine_candidates: usize,
    pub coarsen_candidates: usize,
}

/// Provenance SF maps tracking load/redistribute/section movement.
#[derive(Clone, Debug, Default)]
pub struct MeshDMProvenance<C: Communicator + Sync + 'static> {
    pub load_map: Option<PointSF<'static, C>>,
    pub redistribute_map: Option<PointSF<'static, C>>,
    pub section_map: Option<PointSF<'static, C>>,
    pub storage_version: Option<i32>,
    pub permutation_source: Option<String>,
    pub redistribution_map_id: Option<String>,
}

impl<St, CtSt> MeshDM<f64, St, CtSt>
where
    St: Storage<f64> + Clone,
    CtSt: Storage<CellType> + Clone,
{
    /// Adapt this DM from an attached per-cell metric section with label-aware constraints.
    pub fn adapt_with_attached_metric<C>(
        &mut self,
        metric_section_name: &str,
        metric: &Section<MetricTensor, VecStorage<MetricTensor>>,
        coarsen_plan: C,
        options: MeshDMMetricAdaptOptions,
    ) -> Result<MeshDMMetricAdaptResult, MeshSieveError>
    where
        C: Fn(&[MetricSplitHint]) -> Vec<crate::topology::coarsen::CoarsenEntity>,
    {
        let coords = self.mesh_data.coordinates.as_ref().ok_or_else(|| {
            MeshSieveError::MissingSectionName {
                name: "coordinates".into(),
            }
        })?;
        let cell_types = self.mesh_data.cell_types.as_ref().ok_or_else(|| {
            MeshSieveError::MissingSectionName {
                name: "cell_types".into(),
            }
        })?;
        let _ = metric_section_name;

        let boundary_policy = self.compose_boundary_policy(&options.labels);
        let result: MetricAdaptationResult<()> = adapt_with_metric_policy(
            &mut self.mesh_data.sieve,
            cell_types,
            coords,
            metric,
            self.mesh_data.labels.as_ref(),
            coarsen_plan,
            options.thresholds,
            MetricAdaptationOptions {
                boundary_policy,
                normalization: options.normalization,
                backend: options.backend,
            },
        )?;

        let provenance = result.provenance.clone();
        let diagnostics =
            self.transfer_after_adaptation(&result.action, &provenance, options.transfer)?;
        self.enforce_post_adaptation_fvm_contract()?;
        Ok(MeshDMMetricAdaptResult {
            action: result.action,
            metrics: result.metrics,
            diagnostics,
            fv_selection: None,
            provenance,
        })
    }

    pub fn adapt_until_fv_thresholds<C>(
        &mut self,
        metric_section_name: &str,
        metric: &Section<MetricTensor, VecStorage<MetricTensor>>,
        coarsen_plan: C,
        options: MeshDMMetricAdaptOptions,
        fv_thresholds: FvStabilityThresholds,
        iter: MeshDMAdaptIterationOptions,
    ) -> Result<Vec<MeshDMMetricAdaptResult>, MeshSieveError>
    where
        C: Fn(&[MetricSplitHint]) -> Vec<crate::topology::coarsen::CoarsenEntity> + Copy,
    {
        let mut history = Vec::new();
        for _ in 0..iter.max_iterations.max(1) {
            let mut pass = self.adapt_with_attached_metric(
                metric_section_name,
                metric,
                coarsen_plan,
                options.clone(),
            )?;
            let coords = self.mesh_data.coordinates.as_ref().ok_or_else(|| {
                MeshSieveError::MissingSectionName {
                    name: "coordinates".into(),
                }
            })?;
            let cell_types = self.mesh_data.cell_types.as_ref().ok_or_else(|| {
                MeshSieveError::MissingSectionName {
                    name: "cell_types".into(),
                }
            })?;
            let cell_diags = fvm_cell_diagnostics(&self.mesh_data.sieve, cell_types, coords)?;
            let selection = select_cells_from_fvm_diagnostics(&cell_diags, fv_thresholds);
            pass.fv_selection = Some((
                selection.refine_cells.clone(),
                selection.coarsen_cells.clone(),
            ));
            history.push(pass.clone());
            let stop_by_action = matches!(pass.action, MetricAdaptationAction::NoChange);
            let stop_by_threshold = iter.stop_when_below_threshold
                && selection.refine_cells.is_empty()
                && selection.coarsen_cells.is_empty();
            if stop_by_action || stop_by_threshold {
                break;
            }
        }
        Ok(history)
    }

    pub fn adapt_until_fv_thresholds_with_history<C>(
        &mut self,
        metric_section_name: &str,
        metric: &Section<MetricTensor, VecStorage<MetricTensor>>,
        coarsen_plan: C,
        options: MeshDMMetricAdaptOptions,
        fv_thresholds: FvStabilityThresholds,
        iter: MeshDMAdaptIterationOptions,
    ) -> Result<(Vec<MeshDMMetricAdaptResult>, Vec<MeshDMFvIterationReport>), MeshSieveError>
    where
        C: Fn(&[MetricSplitHint]) -> Vec<crate::topology::coarsen::CoarsenEntity> + Copy,
    {
        let mut history = Vec::new();
        let mut reports = Vec::new();
        for pass_idx in 0..iter.max_iterations.max(1) {
            let mut pass = self.adapt_with_attached_metric(
                metric_section_name,
                metric,
                coarsen_plan,
                options.clone(),
            )?;
            self.refresh_fv_geometry_caches()?;
            let coords = self.mesh_data.coordinates.as_ref().ok_or_else(|| {
                MeshSieveError::MissingSectionName {
                    name: "coordinates".into(),
                }
            })?;
            let cell_types = self.mesh_data.cell_types.as_ref().ok_or_else(|| {
                MeshSieveError::MissingSectionName {
                    name: "cell_types".into(),
                }
            })?;
            let quality = fvm_quality_diagnostics(&self.mesh_data.sieve, cell_types, coords).ok();
            let cell_diags = fvm_cell_diagnostics(&self.mesh_data.sieve, cell_types, coords)?;
            let selection = select_cells_from_fvm_diagnostics(&cell_diags, fv_thresholds);
            pass.fv_selection = Some((
                selection.refine_cells.clone(),
                selection.coarsen_cells.clone(),
            ));
            reports.push(MeshDMFvIterationReport {
                pass_index: pass_idx,
                fvm_quality: quality,
                refine_candidates: selection.refine_cells.len(),
                coarsen_candidates: selection.coarsen_cells.len(),
            });
            history.push(pass.clone());
            let stop_by_action = matches!(pass.action, MetricAdaptationAction::NoChange);
            let stop_by_threshold = iter.stop_when_below_threshold
                && selection.refine_cells.is_empty()
                && selection.coarsen_cells.is_empty();
            if stop_by_action || stop_by_threshold {
                break;
            }
        }
        Ok((history, reports))
    }

    /// Iteratively adapt until no action is taken or iteration budget is exhausted.
    pub fn adapt_with_attached_metric_iterative<C, F>(
        &mut self,
        metric_section_name: &str,
        metric: &Section<MetricTensor, VecStorage<MetricTensor>>,
        coarsen_plan: C,
        options: MeshDMMetricAdaptOptions,
        max_iterations: usize,
        stop: F,
    ) -> Result<Vec<MeshDMMetricAdaptResult>, MeshSieveError>
    where
        C: Fn(&[MetricSplitHint]) -> Vec<crate::topology::coarsen::CoarsenEntity> + Copy,
        F: Fn(&MeshDMMetricAdaptResult) -> bool,
    {
        let mut history = Vec::new();
        for _ in 0..max_iterations {
            let pass = self.adapt_with_attached_metric(
                metric_section_name,
                metric,
                coarsen_plan,
                options.clone(),
            )?;
            let done = matches!(pass.action, MetricAdaptationAction::NoChange) || stop(&pass);
            history.push(pass);
            if done {
                break;
            }
        }
        Ok(history)
    }

    fn compose_boundary_policy(&self, labels: &MeshDMAdaptLabelPolicy) -> BoundaryRemeshingPolicy {
        let mut strata = labels.fixed_boundary_labels.clone();
        strata.extend(labels.protected_region_labels.clone());
        if strata.is_empty() {
            BoundaryRemeshingPolicy::PreserveBoundary
        } else {
            BoundaryRemeshingPolicy::PreserveLabeled(strata)
        }
    }

    fn transfer_after_adaptation(
        &mut self,
        action: &MetricAdaptationAction,
        provenance: &AdaptationProvenanceMap,
        strategy: MeshDMTransferStrategy,
    ) -> Result<MeshDMAdaptDiagnostics, MeshSieveError> {
        let changed_cells = match action {
            MetricAdaptationAction::Refined { mesh } => mesh.cell_refinement.len(),
            MetricAdaptationAction::Coarsened { mesh } => mesh.transfer_map.len(),
            MetricAdaptationAction::NoChange => 0,
        };
        let preserved_labels = self
            .mesh_data
            .labels
            .as_ref()
            .map_or(0, |l| l.iter().count());
        let coords = self.mesh_data.coordinates.as_ref();
        let cell_types = self.mesh_data.cell_types.as_ref();
        let (fvm_face_metrics_cached, boundary_faces) = if let (Some(coords), Some(cell_types)) =
            (coords, cell_types)
        {
            let face_metrics = build_fvm_face_metrics(&self.mesh_data.sieve, cell_types, coords)?;
            let bfaces = face_metrics.iter().filter(|m| m.neighbor.is_none()).count();
            (face_metrics.len(), bfaces)
        } else {
            (0, 0)
        };
        let mut transferred_sections = 0;
        let mut transferred_constraints = 0;
        let mut transferred_overlap_points = 0;
        let failed_transfer_points = Vec::new();

        match action {
            MetricAdaptationAction::Refined { mesh } => {
                if matches!(
                    strategy,
                    MeshDMTransferStrategy::PreserveLabels
                        | MeshDMTransferStrategy::PreserveCoordinatesAndLabels
                        | MeshDMTransferStrategy::PreserveAll
                ) && let Some(labels) = self.mesh_data.labels.as_ref()
                {
                    self.mesh_data.labels =
                        Some(transfer_labels_refinement(labels, &mesh.cell_refinement));
                }
                if matches!(
                    strategy,
                    MeshDMTransferStrategy::PreserveCoordinatesAndLabels
                        | MeshDMTransferStrategy::PreserveAll
                ) && let Some(coords) = mesh.coordinates.clone()
                {
                    self.mesh_data.coordinates = Some(convert_coordinates_storage(coords)?);
                }
                if matches!(strategy, MeshDMTransferStrategy::PreserveAll) {
                    if let Some(cell_types) = self.mesh_data.cell_types.as_ref() {
                        self.mesh_data.cell_types = Some(convert_cell_type_section_storage(
                            transfer_cell_types_refinement(cell_types, &mesh.cell_refinement)?,
                        )?);
                    }
                    self.mesh_data.sections = transfer_dm_sections_refinement(
                        &self.mesh_data.sections,
                        &mesh.cell_refinement,
                    )?;
                    transferred_sections = self.mesh_data.sections.len();
                    transferred_constraints = mesh.hanging_constraints.constraints().len();
                    if let Some(ownership) = self.ownership.as_ref() {
                        let rank = self.distribution.as_ref().map_or(0, |d| d.rank);
                        self.ownership =
                            Some(transfer_ownership_refinement(ownership, mesh, rank)?);
                    }
                    if let Some(overlap) = self.overlap.as_ref() {
                        let (mapped, count) =
                            transfer_overlap_refinement(overlap, &mesh.cell_refinement)?;
                        self.overlap = Some(mapped);
                        transferred_overlap_points = count;
                    }
                }
                self.mesh_data.sieve = oriented_from_refined_sieve(&mesh.sieve);
            }
            MetricAdaptationAction::Coarsened { mesh } => {
                if matches!(strategy, MeshDMTransferStrategy::PreserveAll) {
                    self.mesh_data.sections = transfer_dm_sections_coarsening(
                        &self.mesh_data.sections,
                        &mesh.transfer_map,
                    )?;
                    transferred_sections = self.mesh_data.sections.len();
                }
                self.mesh_data.sieve = oriented_from_refined_sieve(&mesh.sieve);
            }
            MetricAdaptationAction::NoChange => {}
        }

        self.refresh_fv_geometry_caches()?;
        let diagnostics = MeshDMAdaptDiagnostics {
            changed_cells,
            preserved_labels,
            failed_transfer_points,
            split_hint_cells: changed_cells,
            fvm_face_metrics_cached,
            boundary_faces,
            provenance_edges: provenance
                .old_to_new
                .iter()
                .map(|(_, points)| points.len())
                .sum(),
            transferred_sections,
            transferred_constraints,
            transferred_overlap_points,
        };
        Ok(diagnostics)
    }

    /// Enforce DM post-adaptation FV readiness before any further assembly/diagnostics.
    ///
    /// Contract:
    /// 1. Invalidate stale topology-derived caches that may hold pre-adaptation face/cell state.
    /// 2. Recompute face-loop classification from the adapted topology.
    /// 3. Rebuild packed FV traversal buffers from the refreshed loop partition before assembly resumes.
    fn enforce_post_adaptation_fvm_contract(&mut self) -> Result<(), MeshSieveError> {
        self.mesh_data.sieve.invalidate_cache();
        let _ = compute_strata(&self.mesh_data.sieve)?;
        if let (Some(coords), Some(cell_types)) = (
            self.mesh_data.coordinates.as_ref(),
            self.mesh_data.cell_types.as_ref(),
        ) {
            let face_metrics = build_fvm_face_metrics(&self.mesh_data.sieve, cell_types, coords)?;
            let faces = face_metrics.iter().map(|m| m.face);
            let loops = crate::physics::fvm::classify_face_loops(&self.mesh_data.sieve, faces)?;
            let mut packed = crate::physics::fvm::FvmInputs::new(
                loops.internal.into_iter().chain(loops.boundary),
                Vec::new(),
                Vec::new(),
            );
            packed.build_packed_cache();
        }
        Ok(())
    }

    fn refresh_fv_geometry_caches(&mut self) -> Result<(), MeshSieveError> {
        let _ = compute_strata(&self.mesh_data.sieve)?;
        if let (Some(coords), Some(cell_types)) = (
            self.mesh_data.coordinates.as_ref(),
            self.mesh_data.cell_types.as_ref(),
        ) {
            let _ = build_fvm_face_metrics(&self.mesh_data.sieve, cell_types, coords)?;
        }
        Ok(())
    }
}

fn convert_section_storage<V, Src, Dst>(
    section: Section<V, Src>,
) -> Result<Section<V, Dst>, MeshSieveError>
where
    V: Clone + Default,
    Src: Storage<V>,
    Dst: Storage<V> + Clone,
{
    let atlas = section.atlas().clone();
    let mut out = Section::<V, Dst>::new(atlas);
    for (point, slice) in section.iter() {
        out.try_set(point, slice)?;
    }
    Ok(out)
}

fn convert_coordinates_storage<St>(
    coords: Coordinates<f64, VecStorage<f64>>,
) -> Result<Coordinates<f64, St>, MeshSieveError>
where
    St: Storage<f64> + Clone,
{
    let section = convert_section_storage::<f64, _, St>(coords.section().clone())?;
    Coordinates::from_section(
        coords.topological_dimension(),
        coords.embedding_dimension(),
        section,
    )
}

fn convert_cell_type_section_storage<CtSt>(
    section: Section<CellType, VecStorage<CellType>>,
) -> Result<Section<CellType, CtSt>, MeshSieveError>
where
    CtSt: Storage<CellType> + Clone,
{
    convert_section_storage::<CellType, _, CtSt>(section)
}

fn transfer_section_refinement_as_section<St>(
    section: &Section<f64, St>,
    refinement: &[(PointId, Vec<(PointId, crate::topology::arrow::Polarity)>)],
) -> Result<Section<f64, St>, MeshSieveError>
where
    St: Storage<f64> + Clone,
{
    let mut atlas = Atlas::default();
    let mut values: Vec<(PointId, Vec<f64>)> = Vec::new();
    for (old, new_points) in refinement {
        if let Ok(slice) = section.try_restrict(*old) {
            for (new_point, _) in new_points {
                atlas.try_insert(*new_point, slice.len())?;
                values.push((*new_point, slice.to_vec()));
            }
        }
    }
    let mut out = Section::<f64, St>::new(atlas);
    for (point, slice) in values {
        out.try_set(point, &slice)?;
    }
    Ok(out)
}

fn transfer_dm_sections_refinement<St>(
    sections: &BTreeMap<String, Section<f64, St>>,
    refinement: &[(PointId, Vec<(PointId, crate::topology::arrow::Polarity)>)],
) -> Result<BTreeMap<String, Section<f64, St>>, MeshSieveError>
where
    St: Storage<f64> + Clone,
{
    let mut out = BTreeMap::new();
    for (name, section) in sections {
        out.insert(
            name.clone(),
            transfer_section_refinement_as_section(section, refinement)?,
        );
    }
    Ok(out)
}

fn transfer_section_coarsening_as_section<St>(
    section: &Section<f64, St>,
    transfer: &[(PointId, Vec<(PointId, crate::topology::arrow::Polarity)>)],
) -> Result<Section<f64, St>, MeshSieveError>
where
    St: Storage<f64> + Clone,
{
    let mut atlas = Atlas::default();
    let mut values: Vec<(PointId, Vec<f64>)> = Vec::new();
    for (new_point, old_points) in transfer {
        let Some((first_old, _)) = old_points.first() else {
            continue;
        };
        if let Ok(first) = section.try_restrict(*first_old) {
            let mut accum = vec![0.0; first.len()];
            let mut count = 0.0;
            for (old, _) in old_points {
                if let Ok(slice) = section.try_restrict(*old)
                    && slice.len() == accum.len()
                {
                    for (a, v) in accum.iter_mut().zip(slice) {
                        *a += *v;
                    }
                    count += 1.0;
                }
            }
            if count > 0.0 {
                for a in &mut accum {
                    *a /= count;
                }
                atlas.try_insert(*new_point, accum.len())?;
                values.push((*new_point, accum));
            }
        }
    }
    let mut out = Section::<f64, St>::new(atlas);
    for (point, slice) in values {
        out.try_set(point, &slice)?;
    }
    Ok(out)
}

fn transfer_dm_sections_coarsening<St>(
    sections: &BTreeMap<String, Section<f64, St>>,
    transfer: &[(PointId, Vec<(PointId, crate::topology::arrow::Polarity)>)],
) -> Result<BTreeMap<String, Section<f64, St>>, MeshSieveError>
where
    St: Storage<f64> + Clone,
{
    let mut out = BTreeMap::new();
    for (name, section) in sections {
        out.insert(
            name.clone(),
            transfer_section_coarsening_as_section(section, transfer)?,
        );
    }
    Ok(out)
}

fn oriented_from_refined_sieve(refined: &impl Sieve<Point = PointId>) -> MeshSieve {
    let mut out = MeshSieve::default();
    for src in refined.base_points() {
        for dst in refined.cone_points(src) {
            out.add_arrow(src, dst, ());
        }
    }
    out
}

fn transfer_overlap_refinement(
    overlap: &Overlap,
    refinement: &[(PointId, Vec<(PointId, crate::topology::arrow::Polarity)>)],
) -> Result<(Overlap, usize), MeshSieveError> {
    let mut out = overlap.clone();
    let mut count = 0;
    for (old, new_points) in refinement {
        for rank in overlap.neighbor_ranks() {
            for (local, remote) in overlap.links_to(rank) {
                if local == *old {
                    for (new_point, _) in new_points {
                        match remote {
                            Some(remote_point) => {
                                out.try_add_link(*new_point, rank, remote_point)?
                            }
                            None => {
                                out.try_add_link_structural_one(*new_point, rank)?;
                            }
                        }
                        count += 1;
                    }
                }
            }
        }
    }
    Ok((out, count))
}

/// DMPLEX-like facade owning topology, coordinates, labels, sections,
/// discretization metadata, distribution state, and solver numbering maps.
#[derive(Debug)]
pub struct MeshDM<V, St = VecStorage<V>, CtSt = VecStorage<CellType>>
where
    St: Storage<V>,
    CtSt: Storage<CellType>,
{
    mesh_data: MeshData<MeshSieve, V, St, CtSt>,
    options: MeshDMOptions,
    ownership: Option<PointOwnership>,
    overlap: Option<Overlap>,
    global_sections: BTreeMap<String, LocalToGlobalMap>,
    distribution: Option<MeshDMDistribution>,
    provenance_maps: MeshDMProvenance<crate::algs::communicator::NoComm>,
}

impl<V, St, CtSt> MeshDM<V, St, CtSt>
where
    V: Clone + Default,
    St: Storage<V> + Clone,
    CtSt: Storage<CellType> + Clone,
{
    /// Create a builder from topology.
    pub fn builder(topology: MeshSieve) -> MeshDMBuilder<V, St, CtSt> {
        MeshDMBuilder::new(topology)
    }

    /// Wrap existing mesh data with default DM options.
    pub fn from_mesh_data(mesh_data: MeshData<MeshSieve, V, St, CtSt>) -> Self {
        Self::from_mesh_data_with_options(mesh_data, MeshDMOptions::default())
    }

    /// Wrap existing mesh data with explicit DM options.
    pub fn from_mesh_data_with_options(
        mesh_data: MeshData<MeshSieve, V, St, CtSt>,
        options: MeshDMOptions,
    ) -> Self {
        Self {
            mesh_data,
            options,
            ownership: None,
            overlap: None,
            global_sections: BTreeMap::new(),
            distribution: None,
            provenance_maps: MeshDMProvenance::default(),
        }
    }

    /// Consume the facade, returning the owned lower-level mesh data.
    pub fn into_mesh_data(self) -> MeshData<MeshSieve, V, St, CtSt> {
        self.mesh_data
    }

    /// Borrow the full lower-level mesh data container.
    pub fn mesh_data(&self) -> &MeshData<MeshSieve, V, St, CtSt> {
        &self.mesh_data
    }

    /// Borrow the topology.
    pub fn topology(&self) -> &MeshSieve {
        &self.mesh_data.sieve
    }

    /// Mutably borrow the topology.
    pub fn topology_mut(&mut self) -> &mut MeshSieve {
        &mut self.mesh_data.sieve
    }

    /// Borrow labels, if present.
    pub fn labels(&self) -> Option<&LabelSet> {
        self.mesh_data.labels.as_ref()
    }

    /// Borrow coordinates, if present.
    pub fn coordinates(&self) -> Option<&Coordinates<V, St>> {
        self.mesh_data.coordinates.as_ref()
    }

    /// Borrow a named local section.
    pub fn section(&self, name: &str) -> Option<&Section<V, St>> {
        self.mesh_data.sections.get(name)
    }

    /// Mutably borrow or create labels.
    pub fn labels_mut_or_insert(&mut self) -> &mut LabelSet {
        self.mesh_data.labels.get_or_insert_with(LabelSet::new)
    }

    /// Insert or replace a named local section.
    pub fn insert_section(
        &mut self,
        name: impl Into<String>,
        section: Section<V, St>,
    ) -> Option<Section<V, St>> {
        self.mesh_data.sections.insert(name.into(), section)
    }

    /// Borrow cell type metadata, if present.
    pub fn cell_types(&self) -> Option<&Section<CellType, CtSt>> {
        self.mesh_data.cell_types.as_ref()
    }

    /// Borrow discretization metadata, if present.
    pub fn discretization(&self) -> Option<&Discretization> {
        self.mesh_data.discretization.as_ref()
    }

    /// Borrow ownership metadata, if this DM has been distributed or numbered.
    pub fn ownership(&self) -> Option<&PointOwnership> {
        self.ownership.as_ref()
    }

    /// Borrow overlap/SF-like state, if this DM has been distributed.
    pub fn overlap(&self) -> Option<&Overlap> {
        self.overlap.as_ref()
    }

    /// Borrow distribution metadata, if this DM has been distributed.
    pub fn distribution(&self) -> Option<&MeshDMDistribution> {
        self.distribution.as_ref()
    }

    pub fn provenance_maps(&self) -> &MeshDMProvenance<crate::algs::communicator::NoComm> {
        &self.provenance_maps
    }

    /// Borrow DM setup options.
    pub fn options(&self) -> &MeshDMOptions {
        &self.options
    }

    /// Run local setup actions that do not require a partitioner/communicator.
    pub fn setup_serial(&mut self) -> Result<(), MeshSieveError> {
        self.run_refinement_passes(
            self.options.pre_refine,
            MeshDMTransferStrategy::PreserveCoordinatesAndLabels,
        )?;
        if let Some(ordering) = self.options.reorder_section {
            self.reorder_sections(ordering)?;
        }
        self.enforce_phase_invariants()?;
        self.run_requested_checks()?;
        self.run_refinement_passes(
            self.options.post_refine,
            MeshDMTransferStrategy::PreserveCoordinatesAndLabels,
        )?;
        self.enforce_phase_invariants()?;
        self.run_requested_checks()?;
        Ok(())
    }

    fn run_refinement_passes(
        &mut self,
        passes: usize,
        strategy: MeshDMTransferStrategy,
    ) -> Result<(), MeshSieveError> {
        for _ in 0..passes {
            self.run_single_refinement_pass(strategy)?;
            self.enforce_phase_invariants()?;
            self.run_requested_checks()?;
        }
        Ok(())
    }

    fn run_single_refinement_pass(
        &mut self,
        strategy: MeshDMTransferStrategy,
    ) -> Result<(), MeshSieveError> {
        let Some(cell_types) = self.mesh_data.cell_types.as_ref() else {
            return Ok(());
        };
        let refined = refine_mesh(&mut self.mesh_data.sieve, cell_types)?;
        self.mesh_data.sieve = refined.sieve;
        if matches!(
            strategy,
            MeshDMTransferStrategy::PreserveCoordinatesAndLabels
        ) {
            let _ = refined.coordinates;
        }
        if self.mesh_data.labels.is_some()
            && matches!(
                strategy,
                MeshDMTransferStrategy::PreserveLabels
                    | MeshDMTransferStrategy::PreserveCoordinatesAndLabels
            )
        {
            // labels are point-id keyed and existing points are preserved across refinement.
        }
        self.ensure_serial_ownership(0)?;
        if let Some(ownership) = &self.ownership {
            self.provenance_maps.section_map = Some(crate::algs::point_sf::create_process_sf::<
                crate::algs::communicator::NoComm,
            >(ownership, 0));
        }
        Ok(())
    }

    fn enforce_phase_invariants(&mut self) -> Result<(), MeshSieveError> {
        let _ = compute_strata(&self.mesh_data.sieve)?;
        self.mesh_data.sieve.invalidate_cache();
        if self.options.reorder_section.is_none() {
            self.reorder_sections(StratifiedOrdering::CellFirst)?;
        }
        Ok(())
    }

    fn closure_index_for_points(
        &self,
        point: PointId,
        order: &ClosureOrder,
    ) -> Result<ClosureIndex<i32>, MeshSieveError> {
        let mut atlas = Atlas::default();
        let mut closure_points: Vec<_> =
            self.mesh_data.sieve.closure_iter_sorted([point]).collect();
        closure_points.sort_unstable();
        closure_points.dedup();
        for point in closure_points {
            atlas.try_insert(point, 1)?;
        }
        let section = Section::<(), VecStorage<()>>::new(atlas);
        build_closure_index(
            &self.mesh_data.sieve,
            &section,
            point,
            0,
            order,
            &IdentitySectionSym,
        )
    }

    /// Run topology/geometry checks requested by [`MeshDMOptions`].
    pub fn run_requested_checks(&mut self) -> Result<(), MeshSieveError> {
        let check_all = self.options.check_all;
        let check_options = MeshCheckOptions {
            check_symmetry: check_all || self.options.check_symmetry,
            check_skeleton: check_all || self.options.check_skeleton,
            check_faces: check_all || self.options.check_faces,
            check_geometry: check_all || self.options.check_geometry,
            check_overlap: check_all,
            check_ownership: check_all,
            check_sections: check_all,
        };

        if check_options.check_faces {
            let cells = self.height_stratum(0)?;
            let _ = self.cell_adjacency_graph(cells, Default::default(), AdjacencyWeighting::None);
        }

        run_mesh_checks(
            &mut self.mesh_data.sieve,
            self.mesh_data.cell_types.as_ref(),
            self.mesh_data.coordinates.as_ref(),
            self.ownership.as_ref(),
            self.overlap.as_ref(),
            self.mesh_data.sections.values(),
            check_options,
        )
    }

    /// Return points in a height stratum (height 0 are cells in DMPLEX terms).
    pub fn height_stratum(&self, height: u32) -> Result<Vec<PointId>, MeshSieveError> {
        let strata = compute_strata(&self.mesh_data.sieve)?;
        Ok(strata
            .strata
            .get(height as usize)
            .cloned()
            .unwrap_or_default())
    }

    /// Return points in a depth stratum (depth 0 are vertices in DMPLEX terms).
    pub fn depth_stratum(&self, depth: u32) -> Result<Vec<PointId>, MeshSieveError> {
        let strata = compute_strata(&self.mesh_data.sieve)?;
        let mut points: Vec<_> = strata
            .depth
            .iter()
            .filter_map(|(&point, &d)| (d == depth).then_some(point))
            .collect();
        points.sort_unstable();
        Ok(points)
    }

    /// Return an FE-compatible point order for the downward transitive closure of `point`.
    pub fn closure_points(
        &self,
        point: PointId,
        order: &ClosureOrder,
    ) -> Result<Vec<PointId>, MeshSieveError> {
        Ok(self
            .closure_index_for_points(point, order)?
            .point_order()
            .collect())
    }

    /// Alias for [`Self::closure_points`] mirroring DMPLEX transitive-closure terminology.
    pub fn transitive_closure_points(
        &self,
        point: PointId,
        order: &ClosureOrder,
    ) -> Result<Vec<PointId>, MeshSieveError> {
        self.closure_points(point, order)
    }

    /// Return a deterministic upward star from `point`.
    pub fn star_points(&self, point: PointId) -> Vec<PointId> {
        let mut points: Vec<_> = self.mesh_data.sieve.star_iter_sorted([point]).collect();
        points.sort_unstable();
        points.dedup();
        points
    }

    /// Return a deterministic downward/upward transitive set from `point`.
    pub fn closure_both_points(&self, point: PointId) -> Vec<PointId> {
        let mut points: Vec<_> = self
            .mesh_data
            .sieve
            .closure_both_iter_sorted([point])
            .collect();
        points.sort_unstable();
        points.dedup();
        points
    }

    /// Extract section values over a cell closure using the same ordering as FE assembly helpers.
    pub fn get_closure_values(
        &self,
        section_name: &str,
        cell: PointId,
        order: &ClosureOrder,
    ) -> Result<ElementClosureData<V>, MeshSieveError> {
        let section = self.mesh_data.sections.get(section_name).ok_or_else(|| {
            MeshSieveError::MissingSectionName {
                name: section_name.to_string(),
            }
        })?;
        let global_map = self.global_sections.get(section_name);
        extract_oriented_element_closure(
            &self.mesh_data.sieve,
            section,
            global_map,
            cell,
            0,
            order,
            &IdentitySectionSym,
        )
    }

    /// Return points selected by a label, topology policy, and optional section-atlas filter.
    pub fn points_by_label_selection(
        &self,
        selection: &MeshDMLabelSelection,
    ) -> Result<Vec<PointId>, MeshSieveError> {
        let labels = self
            .labels()
            .ok_or_else(|| MeshSieveError::MissingSectionName {
                name: "labels".to_string(),
            })?;
        let seeds: Vec<PointId> = labels
            .points_with_label(&selection.label_name, selection.label_value)
            .collect();
        let mut points = match selection.topology {
            SubmeshSelection::FullClosure => {
                self.mesh_data.sieve.closure_iter_sorted(seeds).collect()
            }
            SubmeshSelection::ClosureDepth(depth) => {
                crate::algs::traversal::TraversalBuilder::new(&self.mesh_data.sieve)
                    .seeds(seeds)
                    .dir(crate::algs::traversal::Dir::Down)
                    .max_depth(Some(depth))
                    .run()
            }
            SubmeshSelection::TargetStratum { axis, index } => {
                let closure: Vec<PointId> =
                    self.mesh_data.sieve.closure_iter_sorted(seeds).collect();
                let strata = compute_strata(&self.mesh_data.sieve)?;
                let stratum_map = match axis {
                    crate::topology::sieve::strata::StratumAxis::Height => &strata.height,
                    crate::topology::sieve::strata::StratumAxis::Depth => &strata.depth,
                };
                let target = closure
                    .into_iter()
                    .filter(|p| stratum_map.get(p).copied() == Some(index));
                self.mesh_data.sieve.closure_iter_sorted(target).collect()
            }
        };
        if let Some(section_name) = &selection.section {
            let section = self.mesh_data.sections.get(section_name).ok_or_else(|| {
                MeshSieveError::MissingSectionName {
                    name: section_name.clone(),
                }
            })?;
            points.retain(|point| section.atlas().contains(*point));
        }
        points.sort_unstable();
        points.dedup();
        Ok(points)
    }

    /// Convenience label query for points that also have DOFs in a named section.
    pub fn points_by_label_in_section(
        &self,
        label_name: &str,
        label_value: i32,
        section_name: &str,
        topology: SubmeshSelection,
    ) -> Result<Vec<PointId>, MeshSieveError> {
        self.points_by_label_selection(
            &MeshDMLabelSelection::new(label_name, label_value)
                .topology(topology)
                .section(section_name),
        )
    }

    /// Create a constrained view of an existing section, deriving point DOFs from its atlas.
    pub fn create_constrained_view_from_labels(
        &self,
        section_name: &str,
        constraints: &[LabelConstraintSpec],
    ) -> Result<ConstrainedSection<V, St>, MeshSieveError> {
        let section = self.mesh_data.sections.get(section_name).ok_or_else(|| {
            MeshSieveError::MissingSectionName {
                name: section_name.to_string(),
            }
        })?;
        let point_dofs: Vec<_> = section
            .atlas()
            .points()
            .filter_map(|point| section.atlas().get(point).map(|(_, len)| (point, len)))
            .collect();
        self.create_constrained_section_from_labels(section_name, &point_dofs, constraints)
    }

    /// Build a compact sub-DM from labeled points, preserving parent/sub point maps and numbering metadata.
    pub fn sub_dm_by_label(
        &self,
        label_name: &str,
        label_value: i32,
        topology: SubmeshSelection,
    ) -> Result<MeshDMSubmesh<V, St, CtSt>, MeshSieveError> {
        self.sub_dm_by_label_with_sections(label_name, label_value, topology, None)
    }

    /// Build a compact sub-DM from labeled points and retain only selected named sections when requested.
    pub fn sub_dm_by_label_with_sections(
        &self,
        label_name: &str,
        label_value: i32,
        topology: SubmeshSelection,
        section_names: Option<&[&str]>,
    ) -> Result<MeshDMSubmesh<V, St, CtSt>, MeshSieveError> {
        let labels = self
            .labels()
            .ok_or_else(|| MeshSieveError::MissingSectionName {
                name: "labels".to_string(),
            })?;
        let (mut mesh_data, maps) =
            extract_by_label(&self.mesh_data, labels, label_name, label_value, topology)?;
        if let Some(names) = section_names {
            let keep: HashSet<&str> = names.iter().copied().collect();
            mesh_data
                .sections
                .retain(|name, _| keep.contains(name.as_str()));
        }
        let mut dm = MeshDM::from_mesh_data_with_options(mesh_data, self.options.clone());
        dm.ownership = remap_ownership_to_submesh(self.ownership.as_ref(), &maps)?;
        dm.global_sections = remap_global_sections_to_submesh(&self.global_sections, &maps)?;
        dm.provenance_maps = MeshDMProvenance {
            load_map: None,
            redistribute_map: None,
            section_map: Some(point_sf_for_submesh_maps(&maps)),
            storage_version: self.provenance_maps.storage_version,
            permutation_source: Some("sub_dm_by_label".to_string()),
            redistribution_map_id: self.provenance_maps.redistribution_map_id.clone(),
        };
        Ok(MeshDMSubmesh { dm, maps })
    }

    /// Build a dual graph for the provided cells.
    pub fn dual_graph(&self, cells: impl IntoIterator<Item = PointId>) -> DualGraph {
        build_dual(&self.mesh_data.sieve, cells)
    }

    /// Build a cell adjacency/preallocation graph for the provided cells.
    pub fn cell_adjacency_graph(
        &self,
        cells: impl IntoIterator<Item = PointId>,
        opts: crate::algs::adjacency_graph::CellAdjacencyOpts,
        weighting: AdjacencyWeighting,
    ) -> MeshGraph {
        cell_adjacency_graph_with_cells(&self.mesh_data.sieve, cells, opts, weighting)
    }

    /// Build a matrix preallocation graph from cell adjacency.
    pub fn matrix_preallocation_graph(
        &self,
        cells: impl IntoIterator<Item = PointId>,
        opts: crate::algs::adjacency_graph::CellAdjacencyOpts,
    ) -> PreallocationGraph {
        let graph = self.cell_adjacency_graph(cells, opts, AdjacencyWeighting::None);
        let row_nnz = graph
            .xadj
            .windows(2)
            .map(|w| w[1].saturating_sub(w[0]))
            .collect();
        PreallocationGraph {
            xadj: graph.xadj,
            adjncy: graph.adjncy,
            order: graph.order,
            row_nnz,
        }
    }

    /// Prepare this DM for solver assembly with a DMPLEX-like setup flow.
    ///
    /// The flow is intentionally deterministic: prerequisite diagnostics,
    /// section/global numbering, matrix preallocation, ownership/overlap
    /// validation, and optional ghost-section synchronization are all reported
    /// in stable point/name order. Missing required prerequisites are returned
    /// as structured diagnostics instead of panicking or partially preparing the
    /// DM.
    pub fn prepare_for_solve<C>(
        &mut self,
        comm: &C,
        options: PrepareForSolveOptions,
    ) -> Result<PrepareForSolveDiagnostics, MeshSieveError>
    where
        C: Communicator + Sync,
        V: Send + PartialEq + bytemuck::Pod + 'static,
    {
        if options.create_serial_ownership && comm.size() == 1 {
            self.ensure_serial_ownership(comm.rank())?;
        }

        let prerequisites = prepare_for_solve_prerequisites(
            &self.mesh_data.sieve,
            self.mesh_data.coordinates.as_ref(),
            self.mesh_data.cell_types.as_ref(),
            self.ownership.as_ref(),
            self.overlap.as_ref(),
            options,
        )?;
        let mut diagnostics = PrepareForSolveDiagnostics {
            ready: false,
            prerequisites,
            ..PrepareForSolveDiagnostics::default()
        };

        if diagnostics.has_missing_required_prerequisites() {
            diagnostics.steps.push(PrepareForSolveStepDiagnostic {
                name: "section_global_numbering",
                status: "skipped",
                detail: "required prerequisites are missing".to_string(),
            });
            diagnostics.steps.push(PrepareForSolveStepDiagnostic {
                name: "matrix_preallocation_graph",
                status: "skipped",
                detail: "required prerequisites are missing".to_string(),
            });
            diagnostics.steps.push(PrepareForSolveStepDiagnostic {
                name: "ownership_overlap_checks",
                status: "skipped",
                detail: "required prerequisites are missing".to_string(),
            });
            diagnostics.steps.push(PrepareForSolveStepDiagnostic {
                name: "section_synchronization",
                status: "skipped",
                detail: "required prerequisites are missing".to_string(),
            });
            return Ok(diagnostics);
        }

        self.build_global_sections(comm)?;
        diagnostics.global_sections = self.global_sections.keys().cloned().collect();
        diagnostics.steps.push(PrepareForSolveStepDiagnostic {
            name: "section_global_numbering",
            status: "completed",
            detail: format!("numbered_sections={}", diagnostics.global_sections.len()),
        });

        let cells = self.height_stratum(0)?;
        let preallocation = self.matrix_preallocation_graph(cells, Default::default());
        diagnostics.preallocation = Some(PrepareForSolvePreallocationDiagnostic {
            rows: preallocation.order.len(),
            edges: preallocation.adjncy.len(),
            order: preallocation.order,
            row_nnz: preallocation.row_nnz,
        });
        diagnostics.steps.push(PrepareForSolveStepDiagnostic {
            name: "matrix_preallocation_graph",
            status: "completed",
            detail: diagnostics
                .preallocation
                .as_ref()
                .map(|p| format!("rows={}, edges={}", p.rows, p.edges))
                .unwrap_or_else(|| "rows=0, edges=0".to_string()),
        });

        run_mesh_checks(
            &mut self.mesh_data.sieve,
            self.mesh_data.cell_types.as_ref(),
            self.mesh_data.coordinates.as_ref(),
            self.ownership.as_ref(),
            self.overlap.as_ref(),
            self.mesh_data.sections.values(),
            MeshCheckOptions {
                check_symmetry: true,
                check_skeleton: true,
                check_faces: true,
                check_geometry: true,
                check_overlap: true,
                check_ownership: true,
                check_sections: true,
            },
        )?;
        diagnostics.steps.push(PrepareForSolveStepDiagnostic {
            name: "ownership_overlap_checks",
            status: "completed",
            detail: "ownership and overlap topology are consistent".to_string(),
        });

        if options.synchronize_ghost_sections {
            let has_overlap = self.overlap.is_some();
            let has_ownership = self.ownership.is_some();
            if has_overlap && has_ownership {
                let mut synchronized = Vec::new();
                if self.mesh_data.coordinates.is_some() {
                    synchronized.push("coordinates".to_string());
                }
                synchronized.extend(self.mesh_data.sections.keys().cloned());
                self.distribute_fields(comm)?;
                diagnostics.synchronized_sections = synchronized;
                diagnostics.steps.push(PrepareForSolveStepDiagnostic {
                    name: "section_synchronization",
                    status: "completed",
                    detail: format!(
                        "synchronized_sections={}",
                        diagnostics.synchronized_sections.len()
                    ),
                });
            } else {
                diagnostics.steps.push(PrepareForSolveStepDiagnostic {
                    name: "section_synchronization",
                    status: "skipped",
                    detail: "no overlap/ownership state available for ghost synchronization"
                        .to_string(),
                });
            }
        } else {
            diagnostics.steps.push(PrepareForSolveStepDiagnostic {
                name: "section_synchronization",
                status: "skipped",
                detail: "ghost section synchronization disabled".to_string(),
            });
        }

        diagnostics.ready = true;
        Ok(diagnostics)
    }

    /// Create a zero-initialized local vector matching a named section.
    pub fn create_local_vector(&self, section_name: &str) -> Result<MeshVector<V>, MeshSieveError> {
        let section = self.mesh_data.sections.get(section_name).ok_or_else(|| {
            MeshSieveError::MissingSectionName {
                name: section_name.to_string(),
            }
        })?;
        Ok(MeshVector {
            section: Some(section_name.to_string()),
            values: vec![V::default(); section.atlas().total_len()],
        })
    }

    /// Build and store global numbering maps for all named local sections.
    pub fn build_global_sections<C>(&mut self, comm: &C) -> Result<(), MeshSieveError>
    where
        C: Communicator + Sync,
    {
        self.ensure_serial_ownership(comm.rank())?;
        let ownership = self.ownership.as_ref().expect("ownership inserted");
        let empty_overlap = Overlap::default();
        let overlap = self.overlap.as_ref().unwrap_or(&empty_overlap);
        let mut maps = BTreeMap::new();
        for (name, section) in &self.mesh_data.sections {
            maps.insert(
                name.clone(),
                LocalToGlobalMap::from_section_with_ownership(
                    section,
                    overlap,
                    ownership,
                    comm,
                    comm.rank(),
                )?,
            );
        }
        self.global_sections = maps;
        Ok(())
    }

    /// Borrow a stored global section map.
    pub fn global_section(&self, name: &str) -> Option<&LocalToGlobalMap> {
        self.global_sections.get(name)
    }

    /// Create a zero-initialized global vector matching a named global section.
    pub fn create_global_vector(
        &self,
        section_name: &str,
    ) -> Result<MeshVector<V>, MeshSieveError> {
        let map = self.global_sections.get(section_name).ok_or_else(|| {
            MeshSieveError::MissingSectionName {
                name: section_name.to_string(),
            }
        })?;
        Ok(MeshVector {
            section: Some(section_name.to_string()),
            values: global_vector_for_map(map),
        })
    }

    /// Create a constrained section for a field using label constraints.
    pub fn create_constrained_section_from_labels(
        &self,
        field_name: &str,
        point_dofs: &[(PointId, usize)],
        constraints: &[LabelConstraintSpec],
    ) -> Result<ConstrainedSection<V, St>, MeshSieveError> {
        if let Some(discretization) = self.discretization()
            && discretization.field(field_name).is_none()
        {
            return Err(MeshSieveError::MissingSectionName {
                name: field_name.to_string(),
            });
        }
        let labels = self
            .labels()
            .ok_or_else(|| MeshSieveError::MissingSectionName {
                name: "labels".to_string(),
            })?;
        constrained_section_from_label_specs(point_dofs, labels, constraints)
    }

    /// Distribute this DM through the lower-level distribution pipeline and
    /// return the local DM for the calling rank.
    pub fn distribute_with<P, C>(
        &self,
        cells: &[PointId],
        partitioner: &P,
        comm: &C,
    ) -> Result<Self, MeshSieveError>
    where
        P: CellPartitioner<MeshSieve>,
        C: Communicator + Sync,
        V: Send + PartialEq + bytemuck::Pod + 'static,
    {
        let distributed = distribute_with_overlap(
            &self.mesh_data,
            cells,
            partitioner,
            self.options.distribution_config(),
            comm,
        )?;
        let mut dm =
            Self::from_distributed(distributed, self.options.clone(), comm.rank(), comm.size());
        dm.run_refinement_passes(
            dm.options.post_refine,
            MeshDMTransferStrategy::PreserveCoordinatesAndLabels,
        )?;
        dm.enforce_phase_invariants()?;
        dm.run_requested_checks()?;
        Ok(dm)
    }

    /// Complete/synchronize all registered fields through the owned overlap/SF state.
    pub fn distribute_fields<C>(&mut self, comm: &C) -> Result<(), MeshSieveError>
    where
        C: Communicator + Sync,
        V: Send + PartialEq + bytemuck::Pod + 'static,
    {
        let Some(ownership) = &self.ownership else {
            return Ok(());
        };
        let Some(overlap) = &self.overlap else {
            return Ok(());
        };
        let sf =
            crate::algs::point_sf::PointSF::with_ownership(overlap, ownership, comm, comm.rank());
        sf.validate()?;
        self.provenance_maps.section_map = Some(crate::algs::point_sf::create_process_sf::<
            crate::algs::communicator::NoComm,
        >(ownership, comm.rank()));
        if let Some(coords) = &mut self.mesh_data.coordinates {
            sf.complete_section(coords.section_mut())?;
            if let Some(high_order) = coords.high_order_mut() {
                sf.complete_section(high_order.section_mut())?;
            }
        }
        for section in self.mesh_data.sections.values_mut() {
            sf.complete_section(section)?;
        }
        Ok(())
    }

    fn from_distributed(
        data: DistributedMeshData<V, St, CtSt>,
        options: MeshDMOptions,
        rank: usize,
        size: usize,
    ) -> Self {
        let redistribute_map = crate::algs::point_sf::create_process_sf::<
            crate::algs::communicator::NoComm,
        >(&data.ownership, rank);
        let mesh_data = MeshData {
            sieve: data.sieve,
            coordinates: data.coordinates,
            sections: data.sections,
            mixed_sections: data.mixed_sections,
            labels: data.labels,
            cell_types: data.cell_types,
            discretization: data.discretization,
        };
        Self {
            mesh_data,
            options,
            ownership: Some(data.ownership),
            overlap: Some(data.overlap),
            global_sections: BTreeMap::new(),
            distribution: Some(MeshDMDistribution {
                point_owners: data.point_owners,
                cell_parts: data.cell_parts,
                rank,
                size,
            }),
            provenance_maps: MeshDMProvenance {
                load_map: None,
                redistribute_map: Some(redistribute_map),
                section_map: None,
                storage_version: None,
                permutation_source: None,
                redistribution_map_id: None,
            },
        }
    }

    fn ensure_serial_ownership(&mut self, rank: usize) -> Result<(), MeshSieveError> {
        if self.ownership.is_some() {
            return Ok(());
        }
        let mut ownership = PointOwnership::default();
        for point in self.mesh_data.sieve.points() {
            ownership.set(point, rank, false)?;
        }
        self.ownership = Some(ownership);
        Ok(())
    }

    fn reorder_sections(&mut self, ordering: StratifiedOrdering) -> Result<(), MeshSieveError> {
        let permutation = stratified_permutation(&self.mesh_data.sieve, ordering)?;
        for section in self.mesh_data.sections.values_mut() {
            *section = reorder_section_by_points(section, &permutation)?;
        }
        if let Some(coords) = &mut self.mesh_data.coordinates {
            let topological_dimension = coords.topological_dimension();
            let embedding_dimension = coords.embedding_dimension();
            let section = reorder_section_by_points(coords.section(), &permutation)?;
            *coords =
                Coordinates::from_section(topological_dimension, embedding_dimension, section)?;
        }
        if let Some(cell_types) = &mut self.mesh_data.cell_types {
            *cell_types = reorder_section_by_points(cell_types, &permutation)?;
        }
        Ok(())
    }
}

fn reorder_section_by_points<V, St>(
    section: &Section<V, St>,
    order: &[PointId],
) -> Result<Section<V, St>, MeshSieveError>
where
    V: Clone + Default,
    St: Storage<V> + Clone,
{
    let mut atlas = Atlas::default();
    for &point in order {
        if let Some((_offset, len)) = section.atlas().get(point) {
            atlas.try_insert(point, len)?;
        }
    }
    for point in section.atlas().points() {
        if !atlas.contains(point) {
            let (_offset, len) = section
                .atlas()
                .get(point)
                .ok_or(MeshSieveError::PointNotInAtlas(point))?;
            atlas.try_insert(point, len)?;
        }
    }
    let mut reordered = Section::new(atlas);
    for point in section.atlas().points() {
        let values = section.try_restrict(point)?.to_vec();
        reordered.try_set(point, &values)?;
    }
    Ok(reordered)
}

fn remap_ownership_to_submesh(
    ownership: Option<&PointOwnership>,
    maps: &SubmeshMaps,
) -> Result<Option<PointOwnership>, MeshSieveError> {
    let Some(ownership) = ownership else {
        return Ok(None);
    };
    let mut remapped = PointOwnership::default();
    for (&parent, &sub) in &maps.parent_to_sub {
        if let Some(entry) = ownership.entry(parent) {
            remapped.set(sub, entry.owner, entry.is_ghost)?;
        }
    }
    Ok(Some(remapped))
}

fn remap_global_sections_to_submesh(
    global_sections: &BTreeMap<String, LocalToGlobalMap>,
    maps: &SubmeshMaps,
) -> Result<BTreeMap<String, LocalToGlobalMap>, MeshSieveError> {
    let new_to_old = maps
        .sub_to_parent
        .iter()
        .enumerate()
        .map(|(idx, &parent)| PointId::new((idx + 1) as u64).map(|sub| (sub, parent)))
        .collect::<Result<Vec<_>, _>>()?;
    let mut remapped = BTreeMap::new();
    for (name, map) in global_sections {
        remapped.insert(name.clone(), map.remap_points(new_to_old.iter().copied())?);
    }
    Ok(remapped)
}

fn point_sf_for_submesh_maps(
    maps: &SubmeshMaps,
) -> PointSF<'static, crate::algs::communicator::NoComm> {
    let leaves = maps
        .sub_to_parent
        .iter()
        .enumerate()
        .filter_map(|(idx, &parent)| {
            PointId::new((idx + 1) as u64).ok().map(|sub| PointSfLeaf {
                local: sub,
                remote: RemotePoint {
                    rank: 0,
                    point: parent,
                },
                owner_rank: 0,
                is_ghost: false,
            })
        })
        .collect::<Vec<_>>();
    PointSF::from_leaves(0, maps.sub_to_parent.iter().copied(), leaves)
}
