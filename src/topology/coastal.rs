//! Coastal mesh metadata utilities.
//!
//! This module standardizes common label names/values used by coastal and ocean
//! meshes and provides helpers to query point sets and validate consistency.

use std::collections::BTreeSet;

use crate::topology::labels::LabelSet;
use crate::topology::point::PointId;

/// Canonical label name for boundary class tags.
pub const BOUNDARY_CLASS_LABEL: &str = "boundary_class";
/// Canonical label name for boundary role tags (open boundary sub-type).
pub const BOUNDARY_ROLE_LABEL: &str = "boundary_role";
/// Canonical label name for vertical layer indices.
pub const VERTICAL_LAYER_LABEL: &str = "vertical_layer";
/// Canonical label name for optional wet/dry masks.
pub const WET_DRY_MASK_LABEL: &str = "wet_dry_mask";

/// Boundary class values for [`BOUNDARY_CLASS_LABEL`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum BoundaryClass {
    /// Free surface boundary entities.
    FreeSurface = 1,
    /// Bed boundary entities.
    Bed = 2,
    /// Open boundary entities.
    Open = 3,
}

impl BoundaryClass {
    /// Integer code for this boundary class.
    pub const fn code(self) -> i32 {
        self as i32
    }
}

/// Open-boundary role values for [`BOUNDARY_ROLE_LABEL`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum OpenBoundaryRole {
    /// Inflow boundary.
    Inflow = 11,
    /// Outflow boundary.
    Outflow = 12,
    /// Tidal forcing boundary.
    Tidal = 13,
}

impl OpenBoundaryRole {
    /// Integer code for this open-boundary role.
    pub const fn code(self) -> i32 {
        self as i32
    }
}

/// Wet/dry mask values for [`WET_DRY_MASK_LABEL`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum WetDryMask {
    /// Wet point/entity.
    Wet = 1,
    /// Dry point/entity.
    Dry = 0,
}

impl WetDryMask {
    /// Integer code for this wet/dry value.
    pub const fn code(self) -> i32 {
        self as i32
    }
}

/// Vertical coordinate system conventions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerticalCoordinateSystem {
    /// Terrain-following sigma layers.
    Sigma,
    /// Fixed z layers.
    ZLayer,
}

/// Validation options for coastal metadata checks.
#[derive(Clone, Copy, Debug)]
pub struct CoastalValidationOptions {
    /// Require free-surface, bed, and open classes to exactly partition `expected_boundary_points`.
    pub require_complete_boundary_partition: bool,
    /// Require every open-boundary point to have at least one open role.
    pub require_open_role_on_open_boundary: bool,
    /// Require vertical layer labels on every point in `expected_vertical_points`.
    pub require_complete_vertical_coverage: bool,
}

impl Default for CoastalValidationOptions {
    fn default() -> Self {
        Self {
            require_complete_boundary_partition: false,
            require_open_role_on_open_boundary: true,
            require_complete_vertical_coverage: false,
        }
    }
}

/// Validation errors emitted by coastal metadata checks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CoastalMetadataError {
    /// A point appears in multiple incompatible boundary classes.
    BoundaryClassOverlap { points: Vec<PointId> },
    /// A boundary point was expected but left unlabeled.
    MissingBoundaryClass { points: Vec<PointId> },
    /// A non-boundary point was unexpectedly labeled as a boundary class.
    UnexpectedBoundaryClass { points: Vec<PointId> },
    /// Open boundary points were missing an explicit role label.
    MissingOpenBoundaryRole { points: Vec<PointId> },
    /// A point carried an open-boundary role but was not in open boundary class.
    OpenRoleWithoutOpenClass { points: Vec<PointId> },
    /// Missing vertical layer labels on expected points.
    MissingVerticalLayer { points: Vec<PointId> },
}

/// Point-set queries for canonical coastal labels.
pub trait CoastalLabelQueries {
    /// Returns free-surface boundary points.
    fn free_surface_points(&self) -> Vec<PointId>;
    /// Returns bed boundary points.
    fn bed_points(&self) -> Vec<PointId>;
    /// Returns open-boundary points.
    fn open_boundary_points(&self) -> Vec<PointId>;
    /// Returns open-boundary inflow points.
    fn inflow_points(&self) -> Vec<PointId>;
    /// Returns open-boundary outflow points.
    fn outflow_points(&self) -> Vec<PointId>;
    /// Returns open-boundary tidal points.
    fn tidal_points(&self) -> Vec<PointId>;
    /// Returns points for the supplied vertical layer index.
    fn vertical_layer_points(&self, layer_index: i32) -> Vec<PointId>;
    /// Returns wet points from optional wet/dry mask.
    fn wet_points(&self) -> Vec<PointId>;
    /// Returns dry points from optional wet/dry mask.
    fn dry_points(&self) -> Vec<PointId>;
}

impl CoastalLabelQueries for LabelSet {
    fn free_surface_points(&self) -> Vec<PointId> {
        self.stratum_points(BOUNDARY_CLASS_LABEL, BoundaryClass::FreeSurface.code())
    }
    fn bed_points(&self) -> Vec<PointId> {
        self.stratum_points(BOUNDARY_CLASS_LABEL, BoundaryClass::Bed.code())
    }
    fn open_boundary_points(&self) -> Vec<PointId> {
        self.stratum_points(BOUNDARY_CLASS_LABEL, BoundaryClass::Open.code())
    }
    fn inflow_points(&self) -> Vec<PointId> {
        self.stratum_points(BOUNDARY_ROLE_LABEL, OpenBoundaryRole::Inflow.code())
    }
    fn outflow_points(&self) -> Vec<PointId> {
        self.stratum_points(BOUNDARY_ROLE_LABEL, OpenBoundaryRole::Outflow.code())
    }
    fn tidal_points(&self) -> Vec<PointId> {
        self.stratum_points(BOUNDARY_ROLE_LABEL, OpenBoundaryRole::Tidal.code())
    }
    fn vertical_layer_points(&self, layer_index: i32) -> Vec<PointId> {
        self.stratum_points(VERTICAL_LAYER_LABEL, layer_index)
    }
    fn wet_points(&self) -> Vec<PointId> {
        self.stratum_points(WET_DRY_MASK_LABEL, WetDryMask::Wet.code())
    }
    fn dry_points(&self) -> Vec<PointId> {
        self.stratum_points(WET_DRY_MASK_LABEL, WetDryMask::Dry.code())
    }
}

/// Validate coastal metadata consistency.
pub fn validate_coastal_metadata(
    labels: &LabelSet,
    expected_boundary_points: Option<&[PointId]>,
    expected_vertical_points: Option<&[PointId]>,
    options: CoastalValidationOptions,
) -> Result<(), CoastalMetadataError> {
    let fs: BTreeSet<_> = labels.free_surface_points().into_iter().collect();
    let bed: BTreeSet<_> = labels.bed_points().into_iter().collect();
    let open: BTreeSet<_> = labels.open_boundary_points().into_iter().collect();

    let mut overlaps = BTreeSet::new();
    overlaps.extend(fs.intersection(&bed).copied());
    overlaps.extend(fs.intersection(&open).copied());
    overlaps.extend(bed.intersection(&open).copied());
    if !overlaps.is_empty() {
        return Err(CoastalMetadataError::BoundaryClassOverlap {
            points: overlaps.into_iter().collect(),
        });
    }

    if options.require_complete_boundary_partition {
        if let Some(expected) = expected_boundary_points {
            let expected: BTreeSet<_> = expected.iter().copied().collect();
            let mut labeled = BTreeSet::new();
            labeled.extend(fs.iter().copied());
            labeled.extend(bed.iter().copied());
            labeled.extend(open.iter().copied());

            let missing: Vec<_> = expected.difference(&labeled).copied().collect();
            if !missing.is_empty() {
                return Err(CoastalMetadataError::MissingBoundaryClass { points: missing });
            }

            let extra: Vec<_> = labeled.difference(&expected).copied().collect();
            if !extra.is_empty() {
                return Err(CoastalMetadataError::UnexpectedBoundaryClass { points: extra });
            }
        }
    }

    let inflow: BTreeSet<_> = labels.inflow_points().into_iter().collect();
    let outflow: BTreeSet<_> = labels.outflow_points().into_iter().collect();
    let tidal: BTreeSet<_> = labels.tidal_points().into_iter().collect();

    let mut open_roles = BTreeSet::new();
    open_roles.extend(inflow.iter().copied());
    open_roles.extend(outflow.iter().copied());
    open_roles.extend(tidal.iter().copied());

    if options.require_open_role_on_open_boundary {
        let missing_roles: Vec<_> = open.difference(&open_roles).copied().collect();
        if !missing_roles.is_empty() {
            return Err(CoastalMetadataError::MissingOpenBoundaryRole {
                points: missing_roles,
            });
        }
    }

    let non_open_role_points: Vec<_> = open_roles.difference(&open).copied().collect();
    if !non_open_role_points.is_empty() {
        return Err(CoastalMetadataError::OpenRoleWithoutOpenClass {
            points: non_open_role_points,
        });
    }

    if options.require_complete_vertical_coverage {
        if let Some(expected) = expected_vertical_points {
            let expected: BTreeSet<_> = expected.iter().copied().collect();
            let labeled: BTreeSet<_> = labels
                .iter()
                .filter_map(|(name, point, _)| (name == VERTICAL_LAYER_LABEL).then_some(point))
                .collect();
            let missing: Vec<_> = expected.difference(&labeled).copied().collect();
            if !missing.is_empty() {
                return Err(CoastalMetadataError::MissingVerticalLayer { points: missing });
            }
        }
    }

    Ok(())
}
