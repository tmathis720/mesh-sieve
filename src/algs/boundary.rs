//! Boundary classification utilities.
//!
//! These helpers classify points as boundary or interior based on the number of
//! incident cells found in the upward (support) closure.

use std::collections::HashSet;

use crate::mesh_error::MeshSieveError;
use crate::topology::labels::LabelSet;
use crate::topology::point::PointId;
use crate::topology::sieve::strata::compute_strata;
use crate::topology::sieve::Sieve;

/// Default label name for boundary points.
pub const DEFAULT_BOUNDARY_LABEL: &str = "boundary";
/// Default label name for interior points.
pub const DEFAULT_INTERIOR_LABEL: &str = "interior";

/// Label values for boundary classification.
#[derive(Clone, Copy, Debug)]
pub struct BoundaryLabelValues {
    pub boundary: i32,
    pub interior: i32,
}

impl Default for BoundaryLabelValues {
    fn default() -> Self {
        Self {
            boundary: 1,
            interior: 0,
        }
    }
}

/// Classified boundary/interior point sets.
#[derive(Clone, Debug, Default)]
pub struct BoundaryClassification {
    pub boundary: Vec<PointId>,
    pub interior: Vec<PointId>,
}

/// Classify points by the number of incident cells in their upward closure.
///
/// Points with **zero or one** incident cell are labeled as boundary; points with
/// two or more incident cells are labeled as interior.
pub fn classify_boundary_points<S, I>(
    sieve: &S,
    points: I,
) -> Result<BoundaryClassification, MeshSieveError>
where
    S: Sieve<Point = PointId>,
    I: IntoIterator<Item = PointId>,
{
    let strata = compute_strata(sieve)?;
    let mut boundary = Vec::new();
    let mut interior = Vec::new();

    for p in points {
        let mut incident_cells: HashSet<PointId> = HashSet::new();
        for q in sieve.star_iter([p]) {
            if strata.height.get(&q).copied() == Some(0) {
                incident_cells.insert(q);
                if incident_cells.len() > 1 {
                    break;
                }
            }
        }

        if incident_cells.len() <= 1 {
            boundary.push(p);
        } else {
            interior.push(p);
        }
    }

    boundary.sort_unstable();
    interior.sort_unstable();

    Ok(BoundaryClassification { boundary, interior })
}

/// Classify points and attach boundary/interior labels.
pub fn label_boundary_points<S, I>(
    sieve: &S,
    points: I,
    labels: &mut LabelSet,
) -> Result<BoundaryClassification, MeshSieveError>
where
    S: Sieve<Point = PointId>,
    I: IntoIterator<Item = PointId>,
{
    label_boundary_points_with(
        sieve,
        points,
        labels,
        DEFAULT_BOUNDARY_LABEL,
        DEFAULT_INTERIOR_LABEL,
        BoundaryLabelValues::default(),
    )
}

/// Classify points and attach custom boundary/interior labels.
pub fn label_boundary_points_with<S, I>(
    sieve: &S,
    points: I,
    labels: &mut LabelSet,
    boundary_label: &str,
    interior_label: &str,
    values: BoundaryLabelValues,
) -> Result<BoundaryClassification, MeshSieveError>
where
    S: Sieve<Point = PointId>,
    I: IntoIterator<Item = PointId>,
{
    let classification = classify_boundary_points(sieve, points)?;
    for &p in &classification.boundary {
        labels.set_label(p, boundary_label, values.boundary);
    }
    for &p in &classification.interior {
        labels.set_label(p, interior_label, values.interior);
    }
    Ok(classification)
}
