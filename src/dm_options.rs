//! DMPLEX-style configuration profiles for [`crate::dm::MeshDM`].
//!
//! This module is deliberately separate from the core topology and section data
//! structures.  It provides a serializable, PETSc-DMPLEX-familiar option layer
//! that applications can load from JSON/TOML-like configuration, environment
//! variables, or command-line arguments, then lower into mesh-sieve's native
//! runtime structs.
//!
//! # PETSc DMPLEX option mapping
//!
//! | PETSc-style option | mesh-sieve field / builder target |
//! | --- | --- |
//! | `-dm_refine_pre <n>` | [`DmplexRefinementProfile::pre_refine`] → [`crate::dm::MeshDMOptions::pre_refine`] |
//! | `-dm_refine <n>` / `-dm_refine_post <n>` | [`DmplexRefinementProfile::post_refine`] → [`crate::dm::MeshDMOptions::post_refine`] |
//! | `-dm_distribute <bool>` | [`DmplexDistributionProfile::distribute`] → [`crate::dm::MeshDMOptions::distribute`] |
//! | `-dm_distribute_overlap <n>` | [`DmplexOverlapProfile::depth`] → [`crate::algs::distribute::DistributionConfig::overlap_depth`] |
//! | `-dm_plex_localize <bool>` | [`DmplexDistributionProfile::localize_coordinates`] → [`crate::dm::MeshDMOptions::localize_coordinates`] |
//! | `-dm_plex_check_all <bool>` | [`DmplexCheckProfile::check_all`] → [`crate::diagnostics::MeshCheckOptions::all`] |
//! | `-dm_plex_check_symmetry <bool>` | [`DmplexCheckProfile::check_symmetry`] → [`crate::diagnostics::MeshCheckOptions::check_symmetry`] |
//! | `-dm_plex_check_skeleton <bool>` | [`DmplexCheckProfile::check_skeleton`] → [`crate::diagnostics::MeshCheckOptions::check_skeleton`] |
//! | `-dm_plex_check_faces <bool>` | [`DmplexCheckProfile::check_faces`] → [`crate::diagnostics::MeshCheckOptions::check_faces`] |
//! | `-dm_plex_check_geometry <bool>` | [`DmplexCheckProfile::check_geometry`] → [`crate::diagnostics::MeshCheckOptions::check_geometry`] |
//! | `-dm_plex_reorder_section <vertex_first|cell_first>` | [`DmplexConfigProfile::ordering`] → [`crate::dm::MeshDMOptions::reorder_section`] |
//! | `-dm_plex_metric_target_complexity <x>` | [`DmplexMetricProfile::target_complexity`] → metric normalization |
//! | `-dm_plex_metric_gradation <x>` | [`DmplexMetricProfile::gradation`] → metric normalization |
//! | `-dm_plex_metric_h_min <x>` | [`DmplexMetricProfile::min_magnitude`] → metric normalization |
//! | `-dm_plex_metric_h_max <x>` | [`DmplexMetricProfile::max_magnitude`] → metric normalization |
//! | `-dm_plex_metric_a_min <x>` | [`DmplexMetricProfile::min_anisotropy`] → metric normalization |
//! | `-dm_plex_metric_a_max <x>` | [`DmplexMetricProfile::max_anisotropy`] → metric normalization |
//! | `-dm_plex_metric_hausdorff_number <x>` | [`DmplexMetricProfile::hausdorff_number`] → metric normalization |
//! | `-dm_plex_metric_backend <internal|triangle|tetgen|gmsh|mmg>` | [`DmplexMetricProfile::backend`] → [`crate::adapt::MetricRemeshingBackend`] |
//! | `-dm_plex_filename <path>` | [`DmplexIoProfile::filename`] for application I/O plumbing |
//! | `-dm_plex_format <name>` | [`DmplexIoProfile::format`] for application I/O plumbing |
//! | `-dm_plex_interpolate <bool>` | [`DmplexIoProfile::interpolate`] for application I/O plumbing |
//!
//! The parser also accepts mesh-sieve-specific preparation options such as
//! `-dm_prepare_require_coordinates`, `-dm_prepare_require_cell_types`,
//! `-dm_prepare_require_ownership`, `-dm_prepare_require_overlap`,
//! `-dm_prepare_synchronize_ghost_sections`, and
//! `-dm_prepare_create_serial_ownership`.

use std::env;
use std::ffi::OsString;

use serde::{Deserialize, Serialize};

use crate::adapt::{
    ExternalRemeshingBackend, FvStabilityThresholds, MetricNormalizationControls,
    MetricRemeshingBackend, MetricThresholds,
};
use crate::algs::distribute::DistributionConfig;
use crate::algs::renumber::StratifiedOrdering;
use crate::diagnostics::{MeshCheckOptions, PrepareForSolveOptions};
use crate::dm::{
    MeshDMAdaptLabelPolicy, MeshDMMetricAdaptOptions, MeshDMOptions, MeshDMTransferStrategy,
};

/// Top-level DMPLEX-style configuration profile.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DmplexConfigProfile {
    /// Refinement controls matching `-dm_refine_pre` and `-dm_refine` workflows.
    pub refinement: DmplexRefinementProfile,
    /// Distribution and ghost-section synchronization controls.
    pub distribution: DmplexDistributionProfile,
    /// Overlap/ghost-layer controls.
    pub overlap: DmplexOverlapProfile,
    /// DMPLEX-style diagnostic switches.
    pub checks: DmplexCheckProfile,
    /// Metric adaptation and remeshing controls.
    pub metric: DmplexMetricProfile,
    /// Mesh I/O options carried for application-level loaders/writers.
    pub io: DmplexIoProfile,
    /// Optional topology-stratified section ordering during setup.
    pub ordering: Option<DmplexOrderingProfile>,
    /// Solver preparation prerequisite controls.
    pub solver: DmplexSolverPreparationProfile,
}

/// Refinement pass counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DmplexRefinementProfile {
    /// Number of pre-distribution refinement passes (`-dm_refine_pre`).
    pub pre_refine: usize,
    /// Number of regular/post-distribution refinement passes (`-dm_refine`).
    pub post_refine: usize,
}

/// Distribution controls.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DmplexDistributionProfile {
    /// Enable mesh distribution (`-dm_distribute`).
    pub distribute: bool,
    /// Ensure coordinates are localized on ghost points (`-dm_plex_localize`).
    pub localize_coordinates: bool,
    /// Synchronize section data onto ghost points.
    pub synchronize_sections: bool,
    /// Balance ownership of partition-boundary points.
    pub balance_boundary_ownership: bool,
}

impl Default for DmplexDistributionProfile {
    fn default() -> Self {
        Self {
            distribute: false,
            localize_coordinates: false,
            synchronize_sections: true,
            balance_boundary_ownership: false,
        }
    }
}

/// Overlap/ghost-layer controls.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DmplexOverlapProfile {
    /// Ghost layer depth (`-dm_distribute_overlap`).
    pub depth: usize,
}

impl Default for DmplexOverlapProfile {
    fn default() -> Self {
        Self { depth: 1 }
    }
}

/// Mesh validation switches.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DmplexCheckProfile {
    pub check_symmetry: bool,
    pub check_skeleton: bool,
    pub check_faces: bool,
    pub check_geometry: bool,
    pub check_overlap: bool,
    pub check_ownership: bool,
    pub check_sections: bool,
    pub check_all: bool,
}

/// Serializable section/topology ordering values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DmplexOrderingProfile {
    VertexFirst,
    CellFirst,
}

impl From<DmplexOrderingProfile> for StratifiedOrdering {
    fn from(value: DmplexOrderingProfile) -> Self {
        match value {
            DmplexOrderingProfile::VertexFirst => StratifiedOrdering::VertexFirst,
            DmplexOrderingProfile::CellFirst => StratifiedOrdering::CellFirst,
        }
    }
}

/// Serializable transfer policy for adaptation workflows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DmplexTransferProfile {
    #[default]
    TopologyOnly,
    PreserveLabels,
    PreserveCoordinatesAndLabels,
    PreserveAll,
}

impl From<DmplexTransferProfile> for MeshDMTransferStrategy {
    fn from(value: DmplexTransferProfile) -> Self {
        match value {
            DmplexTransferProfile::TopologyOnly => MeshDMTransferStrategy::TopologyOnly,
            DmplexTransferProfile::PreserveLabels => MeshDMTransferStrategy::PreserveLabels,
            DmplexTransferProfile::PreserveCoordinatesAndLabels => {
                MeshDMTransferStrategy::PreserveCoordinatesAndLabels
            }
            DmplexTransferProfile::PreserveAll => MeshDMTransferStrategy::PreserveAll,
        }
    }
}

/// Serializable remeshing backend selector.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DmplexMetricBackendProfile {
    #[default]
    Internal,
    Triangle,
    TetGen,
    Gmsh,
    Mmg,
}

impl From<DmplexMetricBackendProfile> for MetricRemeshingBackend {
    fn from(value: DmplexMetricBackendProfile) -> Self {
        match value {
            DmplexMetricBackendProfile::Internal => MetricRemeshingBackend::Internal,
            DmplexMetricBackendProfile::Triangle => {
                MetricRemeshingBackend::External(ExternalRemeshingBackend::Triangle)
            }
            DmplexMetricBackendProfile::TetGen => {
                MetricRemeshingBackend::External(ExternalRemeshingBackend::TetGen)
            }
            DmplexMetricBackendProfile::Gmsh => {
                MetricRemeshingBackend::External(ExternalRemeshingBackend::Gmsh)
            }
            DmplexMetricBackendProfile::Mmg => {
                MetricRemeshingBackend::External(ExternalRemeshingBackend::Mmg)
            }
        }
    }
}

/// Metric-adaptation profile mirroring common DMPLEX metric options.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DmplexMetricProfile {
    pub refine_max_edge_length: f64,
    pub coarsen_max_edge_length: f64,
    pub split_edge_length: f64,
    pub split_face_length: f64,
    pub check_geometry: bool,
    pub target_complexity: Option<f64>,
    pub gradation: Option<f64>,
    pub hausdorff_number: Option<f64>,
    pub min_anisotropy: Option<f64>,
    pub max_anisotropy: Option<f64>,
    pub min_magnitude: Option<f64>,
    pub max_magnitude: Option<f64>,
    pub backend: DmplexMetricBackendProfile,
    pub transfer: DmplexTransferProfile,
    pub fixed_boundary_labels: Vec<(String, i32)>,
    pub protected_region_labels: Vec<(String, i32)>,
    pub relax_region_labels: Vec<(String, i32)>,
}

impl Default for DmplexMetricProfile {
    fn default() -> Self {
        let thresholds = MetricThresholds::default();
        Self {
            refine_max_edge_length: thresholds.refine_max_edge_length,
            coarsen_max_edge_length: thresholds.coarsen_max_edge_length,
            split_edge_length: thresholds.split_edge_length,
            split_face_length: thresholds.split_face_length,
            check_geometry: thresholds.check_geometry,
            target_complexity: None,
            gradation: None,
            hausdorff_number: None,
            min_anisotropy: None,
            max_anisotropy: None,
            min_magnitude: None,
            max_magnitude: None,
            backend: DmplexMetricBackendProfile::Internal,
            transfer: DmplexTransferProfile::TopologyOnly,
            fixed_boundary_labels: Vec::new(),
            protected_region_labels: Vec::new(),
            relax_region_labels: Vec::new(),
        }
    }
}

/// Application-level mesh I/O profile.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DmplexIoProfile {
    pub filename: Option<String>,
    pub format: Option<String>,
    pub interpolate: Option<bool>,
    pub use_viewer: Option<String>,
}

/// Solver-preparation prerequisite profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DmplexSolverPreparationProfile {
    pub require_coordinates: bool,
    pub require_cell_types: bool,
    pub require_ownership: bool,
    pub require_overlap: bool,
    pub synchronize_ghost_sections: bool,
    pub create_serial_ownership: bool,
}

impl Default for DmplexSolverPreparationProfile {
    fn default() -> Self {
        let options = PrepareForSolveOptions::default();
        Self {
            require_coordinates: options.require_coordinates,
            require_cell_types: options.require_cell_types,
            require_ownership: options.require_ownership,
            require_overlap: options.require_overlap,
            synchronize_ghost_sections: options.synchronize_ghost_sections,
            create_serial_ownership: options.create_serial_ownership,
        }
    }
}

impl DmplexMetricProfile {
    /// Lower numeric metric thresholds into mesh-sieve's adaptation thresholds.
    pub fn metric_thresholds(&self) -> MetricThresholds {
        MetricThresholds {
            refine_max_edge_length: self.refine_max_edge_length,
            coarsen_max_edge_length: self.coarsen_max_edge_length,
            split_edge_length: self.split_edge_length,
            split_face_length: self.split_face_length,
            check_geometry: self.check_geometry,
        }
    }

    /// Lower DMPLEX-style normalization controls.
    pub fn normalization_controls(&self) -> MetricNormalizationControls {
        MetricNormalizationControls {
            target_complexity: self.target_complexity,
            gradation: self.gradation,
            hausdorff_number: self.hausdorff_number,
            min_anisotropy: self.min_anisotropy,
            max_anisotropy: self.max_anisotropy,
            min_magnitude: self.min_magnitude,
            max_magnitude: self.max_magnitude,
        }
    }

    /// Lower label policy fields into the DM-level adaptation policy.
    pub fn label_policy(&self) -> MeshDMAdaptLabelPolicy {
        MeshDMAdaptLabelPolicy {
            fixed_boundary_labels: self.fixed_boundary_labels.clone(),
            protected_region_labels: self.protected_region_labels.clone(),
            relax_region_labels: self.relax_region_labels.clone(),
        }
    }

    /// Lower this profile into DM metric adaptation options.
    pub fn metric_adapt_options(&self) -> MeshDMMetricAdaptOptions {
        MeshDMMetricAdaptOptions {
            thresholds: self.metric_thresholds(),
            labels: self.label_policy(),
            transfer: self.transfer.into(),
            normalization: self.normalization_controls(),
            backend: self.backend.into(),
        }
    }
}

impl DmplexConfigProfile {
    /// Lower setup-related fields into [`MeshDMOptions`].
    pub fn mesh_dm_options(&self) -> MeshDMOptions {
        MeshDMOptions {
            pre_refine: self.refinement.pre_refine,
            distribute: self.distribution.distribute,
            distribute_overlap: self.overlap.depth,
            post_refine: self.refinement.post_refine,
            localize_coordinates: self.distribution.localize_coordinates,
            check_symmetry: self.checks.check_symmetry,
            check_skeleton: self.checks.check_skeleton,
            check_faces: self.checks.check_faces,
            check_geometry: self.checks.check_geometry,
            check_all: self.checks.check_all,
            reorder_section: self.ordering.map(Into::into),
            balance_boundary_ownership: self.distribution.balance_boundary_ownership,
            synchronize_sections: self.distribution.synchronize_sections,
        }
    }

    /// Lower distribution fields into [`DistributionConfig`].
    pub fn distribution_config(&self) -> DistributionConfig {
        DistributionConfig {
            overlap_depth: self.overlap.depth,
            synchronize_sections: self.distribution.synchronize_sections
                || self.distribution.localize_coordinates,
            balance_boundary_ownership: self.distribution.balance_boundary_ownership,
        }
    }

    /// Lower diagnostic switches into [`MeshCheckOptions`].
    pub fn mesh_check_options(&self) -> MeshCheckOptions {
        if self.checks.check_all {
            MeshCheckOptions::all()
        } else {
            MeshCheckOptions {
                check_symmetry: self.checks.check_symmetry,
                check_skeleton: self.checks.check_skeleton,
                check_faces: self.checks.check_faces,
                check_geometry: self.checks.check_geometry,
                check_overlap: self.checks.check_overlap,
                check_ownership: self.checks.check_ownership,
                check_sections: self.checks.check_sections,
            }
        }
    }

    /// Lower solver-preparation fields into [`PrepareForSolveOptions`].
    pub fn prepare_for_solve_options(&self) -> PrepareForSolveOptions {
        PrepareForSolveOptions {
            require_coordinates: self.solver.require_coordinates,
            require_cell_types: self.solver.require_cell_types,
            require_ownership: self.solver.require_ownership,
            require_overlap: self.solver.require_overlap,
            synchronize_ghost_sections: self.solver.synchronize_ghost_sections,
            create_serial_ownership: self.solver.create_serial_ownership,
        }
    }

    /// Lower metric adaptation fields into [`MeshDMMetricAdaptOptions`].
    pub fn metric_adapt_options(&self) -> MeshDMMetricAdaptOptions {
        self.metric.metric_adapt_options()
    }

    /// Parse PETSc/DMPLEX-style command-line options from any string iterator.
    ///
    /// Unknown options are ignored so applications can pass a full PETSc-like
    /// argv without pre-filtering mesh-sieve keys.
    pub fn from_cli_args<I, S>(args: I) -> Result<Self, String>
    where
        I: IntoIterator<Item = S>,
        S: Into<OsString>,
    {
        let mut profile = Self::default();
        profile.apply_cli_args(args)?;
        Ok(profile)
    }

    /// Start with defaults and apply supported environment variables.
    ///
    /// Variables are uppercase option names without a leading dash, for example
    /// `DM_DISTRIBUTE=true`, `DM_DISTRIBUTE_OVERLAP=2`, or
    /// `DM_PLEX_METRIC_TARGET_COMPLEXITY=1000.0`.
    pub fn from_env() -> Result<Self, String> {
        let mut profile = Self::default();
        profile.apply_env()?;
        Ok(profile)
    }

    /// Apply PETSc/DMPLEX-style CLI arguments to an existing profile.
    pub fn apply_cli_args<I, S>(&mut self, args: I) -> Result<(), String>
    where
        I: IntoIterator<Item = S>,
        S: Into<OsString>,
    {
        let tokens: Vec<String> = args
            .into_iter()
            .map(|arg| arg.into().to_string_lossy().into_owned())
            .collect();
        let mut index = 0;
        while index < tokens.len() {
            let token = &tokens[index];
            if !token.starts_with('-') {
                index += 1;
                continue;
            }
            let (key, inline_value) = split_key_value(token);
            let next_is_value = tokens
                .get(index + 1)
                .is_some_and(|next| !next.starts_with('-'));
            let value = inline_value.or_else(|| next_is_value.then(|| tokens[index + 1].as_str()));
            self.apply_option(key, value)?;
            index += if inline_value.is_none() && next_is_value {
                2
            } else {
                1
            };
        }
        Ok(())
    }

    /// Apply supported environment variables to an existing profile.
    pub fn apply_env(&mut self) -> Result<(), String> {
        for (key, value) in env::vars() {
            if let Some(option) = env_key_to_option(&key) {
                self.apply_option(option.as_str(), Some(value.as_str()))?;
            }
        }
        Ok(())
    }

    /// Apply one PETSc-style option. The key may include or omit the leading `-`.
    pub fn apply_option(&mut self, key: &str, value: Option<&str>) -> Result<bool, String> {
        let key = normalize_key(key);
        match key.as_str() {
            "dm_refine_pre" => self.refinement.pre_refine = parse_value(key.as_str(), value)?,
            "dm_refine" | "dm_refine_post" => {
                self.refinement.post_refine = parse_value(key.as_str(), value)?
            }
            "dm_distribute" => self.distribution.distribute = parse_bool_flag(key.as_str(), value)?,
            "dm_distribute_overlap" => self.overlap.depth = parse_value(key.as_str(), value)?,
            "dm_plex_localize" | "dm_localize" => {
                self.distribution.localize_coordinates = parse_bool_flag(key.as_str(), value)?
            }
            "dm_distribute_synchronize_sections" | "dm_synchronize_sections" => {
                self.distribution.synchronize_sections = parse_bool_flag(key.as_str(), value)?
            }
            "dm_distribute_balance_boundary_ownership" | "dm_balance_boundary_ownership" => {
                self.distribution.balance_boundary_ownership = parse_bool_flag(key.as_str(), value)?
            }
            "dm_plex_check_all" => self.checks.check_all = parse_bool_flag(key.as_str(), value)?,
            "dm_plex_check_symmetry" => {
                self.checks.check_symmetry = parse_bool_flag(key.as_str(), value)?
            }
            "dm_plex_check_skeleton" => {
                self.checks.check_skeleton = parse_bool_flag(key.as_str(), value)?
            }
            "dm_plex_check_faces" => {
                self.checks.check_faces = parse_bool_flag(key.as_str(), value)?
            }
            "dm_plex_check_geometry" => {
                self.checks.check_geometry = parse_bool_flag(key.as_str(), value)?
            }
            "dm_plex_check_overlap" => {
                self.checks.check_overlap = parse_bool_flag(key.as_str(), value)?
            }
            "dm_plex_check_ownership" => {
                self.checks.check_ownership = parse_bool_flag(key.as_str(), value)?
            }
            "dm_plex_check_sections" => {
                self.checks.check_sections = parse_bool_flag(key.as_str(), value)?
            }
            "dm_plex_reorder_section" | "dm_reorder_section" => {
                self.ordering = Some(parse_ordering(key.as_str(), value)?)
            }
            "dm_plex_metric_refine_max_edge_length" => {
                self.metric.refine_max_edge_length = parse_value(key.as_str(), value)?
            }
            "dm_plex_metric_coarsen_max_edge_length" => {
                self.metric.coarsen_max_edge_length = parse_value(key.as_str(), value)?
            }
            "dm_plex_metric_split_edge_length" => {
                self.metric.split_edge_length = parse_value(key.as_str(), value)?
            }
            "dm_plex_metric_split_face_length" => {
                self.metric.split_face_length = parse_value(key.as_str(), value)?
            }
            "dm_plex_metric_check_geometry" => {
                self.metric.check_geometry = parse_bool_flag(key.as_str(), value)?
            }
            "dm_plex_metric_target_complexity" => {
                self.metric.target_complexity = Some(parse_value(key.as_str(), value)?)
            }
            "dm_plex_metric_gradation" => {
                self.metric.gradation = Some(parse_value(key.as_str(), value)?)
            }
            "dm_plex_metric_hausdorff_number" | "dm_plex_metric_hausdorff" => {
                self.metric.hausdorff_number = Some(parse_value(key.as_str(), value)?)
            }
            "dm_plex_metric_a_min" => {
                self.metric.min_anisotropy = Some(parse_value(key.as_str(), value)?)
            }
            "dm_plex_metric_a_max" => {
                self.metric.max_anisotropy = Some(parse_value(key.as_str(), value)?)
            }
            "dm_plex_metric_h_min" => {
                self.metric.min_magnitude = Some(parse_value(key.as_str(), value)?)
            }
            "dm_plex_metric_h_max" => {
                self.metric.max_magnitude = Some(parse_value(key.as_str(), value)?)
            }
            "dm_plex_metric_backend" => self.metric.backend = parse_backend(key.as_str(), value)?,
            "dm_adapt_transfer" => self.metric.transfer = parse_transfer(key.as_str(), value)?,
            "dm_plex_filename" | "dm_filename" => {
                self.io.filename = Some(require_value(key.as_str(), value)?.to_string())
            }
            "dm_plex_format" | "dm_format" => {
                self.io.format = Some(require_value(key.as_str(), value)?.to_string())
            }
            "dm_plex_interpolate" => {
                self.io.interpolate = Some(parse_bool_flag(key.as_str(), value)?)
            }
            "dm_viewer" | "dm_plex_viewer" => {
                self.io.use_viewer = Some(require_value(key.as_str(), value)?.to_string())
            }
            "dm_prepare_require_coordinates" => {
                self.solver.require_coordinates = parse_bool_flag(key.as_str(), value)?
            }
            "dm_prepare_require_cell_types" => {
                self.solver.require_cell_types = parse_bool_flag(key.as_str(), value)?
            }
            "dm_prepare_require_ownership" => {
                self.solver.require_ownership = parse_bool_flag(key.as_str(), value)?
            }
            "dm_prepare_require_overlap" => {
                self.solver.require_overlap = parse_bool_flag(key.as_str(), value)?
            }
            "dm_prepare_synchronize_ghost_sections" => {
                self.solver.synchronize_ghost_sections = parse_bool_flag(key.as_str(), value)?
            }
            "dm_prepare_create_serial_ownership" => {
                self.solver.create_serial_ownership = parse_bool_flag(key.as_str(), value)?
            }
            _ => return Ok(false),
        }
        Ok(true)
    }
}

/// Human-readable supported mapping table for generated docs/help text.
pub const PETSC_DMPLEX_OPTION_MAPPINGS: &[(&str, &str)] = &[
    (
        "-dm_refine_pre",
        "refinement.pre_refine / MeshDMOptions::pre_refine",
    ),
    (
        "-dm_refine",
        "refinement.post_refine / MeshDMOptions::post_refine",
    ),
    (
        "-dm_distribute",
        "distribution.distribute / MeshDMOptions::distribute",
    ),
    (
        "-dm_distribute_overlap",
        "overlap.depth / DistributionConfig::overlap_depth",
    ),
    ("-dm_plex_localize", "distribution.localize_coordinates"),
    (
        "-dm_plex_check_all",
        "checks.check_all / MeshCheckOptions::all",
    ),
    ("-dm_plex_check_symmetry", "checks.check_symmetry"),
    ("-dm_plex_check_skeleton", "checks.check_skeleton"),
    ("-dm_plex_check_faces", "checks.check_faces"),
    ("-dm_plex_check_geometry", "checks.check_geometry"),
    (
        "-dm_plex_metric_target_complexity",
        "metric.target_complexity",
    ),
    ("-dm_plex_metric_gradation", "metric.gradation"),
    ("-dm_plex_metric_h_min", "metric.min_magnitude"),
    ("-dm_plex_metric_h_max", "metric.max_magnitude"),
    ("-dm_plex_metric_a_min", "metric.min_anisotropy"),
    ("-dm_plex_metric_a_max", "metric.max_anisotropy"),
    (
        "-dm_plex_metric_backend",
        "metric.backend / MetricRemeshingBackend",
    ),
    ("-dm_plex_filename", "io.filename"),
    ("-dm_plex_format", "io.format"),
    ("-dm_plex_interpolate", "io.interpolate"),
];

/// Defaults for FV-threshold adaptation profiles that downstream apps may expose
/// next to DMPLEX metric controls.
pub fn default_fv_stability_thresholds() -> FvStabilityThresholds {
    FvStabilityThresholds::default()
}

fn split_key_value(token: &str) -> (&str, Option<&str>) {
    let token = token.trim_start_matches('-');
    if let Some((key, value)) = token.split_once('=') {
        (key, Some(value))
    } else {
        (token, None)
    }
}

fn env_key_to_option(key: &str) -> Option<String> {
    let normalized = key.to_ascii_lowercase();
    if normalized.starts_with("dm_") {
        Some(normalized)
    } else if let Some(stripped) = normalized.strip_prefix("mesh_sieve_") {
        Some(stripped.to_string())
    } else {
        None
    }
}

fn normalize_key(key: &str) -> String {
    key.trim_start_matches('-')
        .replace('-', "_")
        .to_ascii_lowercase()
}

fn parse_value<T>(key: &str, value: Option<&str>) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let value = require_value(key, value)?;
    value
        .parse::<T>()
        .map_err(|err| format!("invalid value for -{key}: {value:?}: {err}"))
}

fn require_value<'a>(key: &str, value: Option<&'a str>) -> Result<&'a str, String> {
    value.ok_or_else(|| format!("missing value for -{key}"))
}

fn parse_bool_flag(key: &str, value: Option<&str>) -> Result<bool, String> {
    match value {
        None => Ok(true),
        Some(value) => match value.to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Ok(true),
            "0" | "false" | "no" | "off" => Ok(false),
            _ => Err(format!("invalid boolean for -{key}: {value:?}")),
        },
    }
}

fn parse_ordering(key: &str, value: Option<&str>) -> Result<DmplexOrderingProfile, String> {
    match require_value(key, value)?.to_ascii_lowercase().as_str() {
        "vertex" | "vertex_first" | "depth" | "depth_first" => {
            Ok(DmplexOrderingProfile::VertexFirst)
        }
        "cell" | "cell_first" | "height" | "height_first" => Ok(DmplexOrderingProfile::CellFirst),
        other => Err(format!("invalid ordering for -{key}: {other:?}")),
    }
}

fn parse_backend(key: &str, value: Option<&str>) -> Result<DmplexMetricBackendProfile, String> {
    match require_value(key, value)?.to_ascii_lowercase().as_str() {
        "internal" | "none" => Ok(DmplexMetricBackendProfile::Internal),
        "triangle" => Ok(DmplexMetricBackendProfile::Triangle),
        "tetgen" => Ok(DmplexMetricBackendProfile::TetGen),
        "gmsh" => Ok(DmplexMetricBackendProfile::Gmsh),
        "mmg" => Ok(DmplexMetricBackendProfile::Mmg),
        other => Err(format!("invalid metric backend for -{key}: {other:?}")),
    }
}

fn parse_transfer(key: &str, value: Option<&str>) -> Result<DmplexTransferProfile, String> {
    match require_value(key, value)?.to_ascii_lowercase().as_str() {
        "topology" | "topology_only" => Ok(DmplexTransferProfile::TopologyOnly),
        "labels" | "preserve_labels" => Ok(DmplexTransferProfile::PreserveLabels),
        "coordinates" | "coords" | "preserve_coordinates_and_labels" => {
            Ok(DmplexTransferProfile::PreserveCoordinatesAndLabels)
        }
        "all" | "preserve_all" => Ok(DmplexTransferProfile::PreserveAll),
        other => Err(format!("invalid adaptation transfer for -{key}: {other:?}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_profile_maps_to_native_options() {
        let profile = DmplexConfigProfile::from_cli_args([
            "app",
            "-dm_refine_pre",
            "2",
            "-dm_refine=1",
            "-dm_distribute",
            "true",
            "-dm_distribute_overlap",
            "3",
            "-dm_plex_check_all",
            "-dm_plex_reorder_section",
            "vertex_first",
            "-dm_plex_metric_target_complexity",
            "2500.0",
            "-dm_plex_metric_backend",
            "gmsh",
            "-dm_prepare_require_overlap",
            "yes",
        ])
        .unwrap();

        let dm_options = profile.mesh_dm_options();
        assert_eq!(dm_options.pre_refine, 2);
        assert_eq!(dm_options.post_refine, 1);
        assert!(dm_options.distribute);
        assert_eq!(dm_options.distribute_overlap, 3);
        assert!(dm_options.check_all);
        assert_eq!(
            dm_options.reorder_section,
            Some(StratifiedOrdering::VertexFirst)
        );

        let distribution = profile.distribution_config();
        assert_eq!(distribution.overlap_depth, 3);
        assert!(distribution.synchronize_sections);

        let checks = profile.mesh_check_options();
        assert!(checks.check_sections);
        assert!(checks.check_ownership);

        let prepare = profile.prepare_for_solve_options();
        assert!(prepare.require_overlap);

        let adapt = profile.metric_adapt_options();
        assert_eq!(adapt.normalization.target_complexity, Some(2500.0));
        assert!(matches!(
            adapt.backend,
            MetricRemeshingBackend::External(ExternalRemeshingBackend::Gmsh)
        ));
    }

    #[test]
    fn parser_accepts_boolean_flags_and_ignores_unknown_options() {
        let profile = DmplexConfigProfile::from_cli_args([
            "-unknown_petSc_option",
            "42",
            "-dm_distribute",
            "-dm_plex_check_faces",
        ])
        .unwrap();
        assert!(profile.distribution.distribute);
        assert!(profile.checks.check_faces);
    }
}
