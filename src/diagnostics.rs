//! Centralized mesh diagnostics and DMPLEX-style validation options.

use serde::Serialize;
use std::fmt::Write as _;

use crate::data::coordinates::Coordinates;
use crate::data::section::Section;
use crate::data::storage::Storage;
use crate::geometry::fvm::build_fvm_face_metrics;
use crate::mesh_error::MeshSieveError;
use crate::overlap::overlap::Overlap;
use crate::topology::cell_type::CellType;
use crate::topology::ownership::PointOwnership;
use crate::topology::point::PointId;
use crate::topology::sieve::strata::compute_strata;
use crate::topology::sieve::{MeshSieve, Sieve};
use crate::topology::validation::{
    TopologyValidationOptions, validate_oriented_sieve_topology,
    validate_overlap_ownership_topology,
};

#[derive(Clone, Debug, Default, Serialize)]
pub struct MetricSummary {
    pub max: f64,
    pub p95: f64,
    pub mean: f64,
    pub count: usize,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct FvmQualityDiagnostics {
    pub non_orthogonality_deg: MetricSummary,
    pub skewness: MetricSummary,
    pub cell_char_length: MetricSummary,
}

#[derive(Clone, Debug, Serialize)]
pub struct FvmCellDiagnostic {
    pub cell: PointId,
    pub max_non_orthogonality_deg: f64,
    pub max_skewness: f64,
    pub char_length: f64,
    pub boundary_face_count: usize,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct FvReadinessThresholds {
    pub max_non_orthogonality_deg: f64,
    pub max_skewness: f64,
    pub min_cell_char_length: f64,
}

impl Default for FvReadinessThresholds {
    fn default() -> Self {
        Self {
            max_non_orthogonality_deg: 75.0,
            max_skewness: 4.0,
            min_cell_char_length: 1.0e-8,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct FvReadinessOptions {
    pub strict: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct FvReadinessViolation {
    pub metric: &'static str,
    pub point: PointId,
    pub value: f64,
    pub threshold: f64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct FvReadinessSummary {
    pub failed: usize,
    pub total: usize,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct FvReadinessReport {
    pub passed: bool,
    pub thresholds: FvReadinessThresholds,
    pub non_orthogonality: FvReadinessSummary,
    pub skewness: FvReadinessSummary,
    pub cell_char_length: FvReadinessSummary,
    pub violations: Vec<FvReadinessViolation>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MeshCheckOptions {
    pub check_symmetry: bool,
    pub check_skeleton: bool,
    pub check_faces: bool,
    pub check_geometry: bool,
    pub check_overlap: bool,
    pub check_ownership: bool,
    pub check_sections: bool,
}

impl MeshCheckOptions {
    pub fn all() -> Self {
        Self {
            check_symmetry: true,
            check_skeleton: true,
            check_faces: true,
            check_geometry: true,
            check_overlap: true,
            check_ownership: true,
            check_sections: true,
        }
    }
}

pub fn run_mesh_checks<'a, V, St, CtSt>(
    sieve: &mut MeshSieve,
    cell_types: Option<&Section<CellType, CtSt>>,
    coords: Option<&Coordinates<V, St>>,
    ownership: Option<&PointOwnership>,
    overlap: Option<&Overlap>,
    sections: impl Iterator<Item = &'a Section<V, St>>,
    options: MeshCheckOptions,
) -> Result<(), MeshSieveError>
where
    V: Clone + Default + 'a,
    St: Storage<V> + Clone + 'a,
    CtSt: Storage<CellType>,
{
    if options.check_skeleton {
        compute_strata(sieve)?;
    }

    if options.check_symmetry {
        if let Some(cell_types) = cell_types {
            validate_oriented_sieve_topology(sieve, cell_types, TopologyValidationOptions::all())?;
        }
    }

    if options.check_geometry {
        if let Some(coords) = coords {
            for (point, (_offset, len)) in coords.section().atlas().iter_entries() {
                if len != coords.embedding_dimension() {
                    return Err(MeshSieveError::SliceLengthMismatch {
                        point,
                        expected: coords.embedding_dimension(),
                        found: len,
                    });
                }
            }
        }
    }

    if options.check_overlap || options.check_ownership {
        if let Some(ownership) = ownership {
            validate_overlap_ownership_topology(sieve, ownership, overlap, 0)?;
        }
    }

    if options.check_sections {
        for section in sections {
            for (point, (_offset, len)) in section.atlas().iter_entries() {
                if len == 0 {
                    return Err(MeshSieveError::SliceLengthMismatch {
                        point,
                        expected: 1,
                        found: 0,
                    });
                }
            }
        }
    }
    Ok(())
}

pub fn run_mesh_checks_f64_with_report<'a, St, CtSt>(
    sieve: &mut MeshSieve,
    cell_types: Option<&Section<CellType, CtSt>>,
    coords: Option<&Coordinates<f64, St>>,
    ownership: Option<&PointOwnership>,
    overlap: Option<&Overlap>,
    sections: impl Iterator<Item = &'a Section<f64, St>>,
    options: MeshCheckOptions,
) -> Result<Option<FvmQualityDiagnostics>, MeshSieveError>
where
    St: Storage<f64> + Clone + 'a,
    CtSt: Storage<CellType>,
{
    run_mesh_checks(
        sieve, cell_types, coords, ownership, overlap, sections, options,
    )?;
    if let (Some(ct), Some(cd)) = (cell_types, coords) {
        Ok(Some(fvm_quality_diagnostics(sieve, ct, cd)?))
    } else {
        Ok(None)
    }
}

pub fn fvm_quality_diagnostics<CT, CS>(
    sieve: &MeshSieve,
    cell_types: &Section<CellType, CT>,
    coordinates: &Coordinates<f64, CS>,
) -> Result<FvmQualityDiagnostics, MeshSieveError>
where
    CT: Storage<CellType>,
    CS: Storage<f64>,
{
    let face_metrics = build_fvm_face_metrics(sieve, cell_types, coordinates)?;
    let non_ortho: Vec<f64> = face_metrics
        .iter()
        .filter_map(|m| m.non_orthogonality_deg)
        .collect();
    let skewness: Vec<f64> = face_metrics
        .iter()
        .map(|m| {
            let denom = m
                .owner_to_neighbor
                .map(norm3)
                .unwrap_or_else(|| norm3(m.owner_to_face));
            if denom > 1e-12 {
                norm3(m.skewness_vector) / denom
            } else {
                0.0
            }
        })
        .collect();
    let mut cell_area_sum: std::collections::BTreeMap<PointId, f64> =
        std::collections::BTreeMap::new();
    for fm in &face_metrics {
        *cell_area_sum.entry(fm.owner).or_insert(0.0) += fm.area_magnitude;
        if let Some(n) = fm.neighbor {
            *cell_area_sum.entry(n).or_insert(0.0) += fm.area_magnitude;
        }
    }
    let mut char_lengths = Vec::with_capacity(cell_area_sum.len());
    for (cell, area_sum) in cell_area_sum {
        if let Some(cell_type) = cell_types
            .try_restrict(cell)
            .ok()
            .and_then(|v| v.first().copied())
        {
            let mut verts: Vec<_> = sieve
                .closure_iter([cell])
                .filter(|p| sieve.cone_points(*p).next().is_none())
                .collect();
            verts.sort_unstable();
            verts.dedup();
            let quality = crate::geometry::quality::cell_quality_from_section(
                cell_type,
                &verts,
                coordinates,
            )?;
            if area_sum > 1e-12 {
                char_lengths.push(quality.jacobian_sign.abs() / area_sum);
            }
        }
    }
    Ok(FvmQualityDiagnostics {
        non_orthogonality_deg: summarize(&non_ortho),
        skewness: summarize(&skewness),
        cell_char_length: summarize(&char_lengths),
    })
}

pub fn fvm_cell_diagnostics<CT, CS>(
    sieve: &MeshSieve,
    cell_types: &Section<CellType, CT>,
    coordinates: &Coordinates<f64, CS>,
) -> Result<Vec<FvmCellDiagnostic>, MeshSieveError>
where
    CT: Storage<CellType>,
    CS: Storage<f64>,
{
    let face_metrics = build_fvm_face_metrics(sieve, cell_types, coordinates)?;
    let mut by_cell: std::collections::BTreeMap<PointId, FvmCellDiagnostic> =
        std::collections::BTreeMap::new();
    let mut cell_area_sum: std::collections::BTreeMap<PointId, f64> =
        std::collections::BTreeMap::new();
    for fm in &face_metrics {
        let denom = fm
            .owner_to_neighbor
            .map(norm3)
            .unwrap_or_else(|| norm3(fm.owner_to_face));
        let skewness = if denom > 1e-12 {
            norm3(fm.skewness_vector) / denom
        } else {
            0.0
        };
        let owner = by_cell.entry(fm.owner).or_insert(FvmCellDiagnostic {
            cell: fm.owner,
            max_non_orthogonality_deg: 0.0,
            max_skewness: 0.0,
            char_length: 0.0,
            boundary_face_count: 0,
        });
        owner.max_non_orthogonality_deg = owner
            .max_non_orthogonality_deg
            .max(fm.non_orthogonality_deg.unwrap_or(0.0));
        owner.max_skewness = owner.max_skewness.max(skewness);
        if fm.neighbor.is_none() {
            owner.boundary_face_count += 1;
        }
        *cell_area_sum.entry(fm.owner).or_insert(0.0) += fm.area_magnitude;
        if let Some(n) = fm.neighbor {
            let neigh = by_cell.entry(n).or_insert(FvmCellDiagnostic {
                cell: n,
                max_non_orthogonality_deg: 0.0,
                max_skewness: 0.0,
                char_length: 0.0,
                boundary_face_count: 0,
            });
            neigh.max_non_orthogonality_deg = neigh
                .max_non_orthogonality_deg
                .max(fm.non_orthogonality_deg.unwrap_or(0.0));
            neigh.max_skewness = neigh.max_skewness.max(skewness);
            *cell_area_sum.entry(n).or_insert(0.0) += fm.area_magnitude;
        }
    }
    for (cell, diag) in &mut by_cell {
        if let Some(cell_type) = cell_types.try_restrict(*cell).ok().and_then(|v| v.first().copied()) {
            let mut verts: Vec<_> = sieve
                .closure_iter([*cell])
                .filter(|p| sieve.cone_points(*p).next().is_none())
                .collect();
            verts.sort_unstable();
            verts.dedup();
            let quality = crate::geometry::quality::cell_quality_from_section(cell_type, &verts, coordinates)?;
            let area_sum = *cell_area_sum.get(cell).unwrap_or(&0.0);
            if area_sum > 1e-12 {
                diag.char_length = quality.jacobian_sign.abs() / area_sum;
            }
        }
    }
    Ok(by_cell.into_values().collect())
}

pub fn fvm_quality_diagnostics_json<CT, CS>(
    sieve: &MeshSieve,
    cell_types: &Section<CellType, CT>,
    coordinates: &Coordinates<f64, CS>,
) -> Result<String, MeshSieveError>
where
    CT: Storage<CellType>,
    CS: Storage<f64>,
{
    let report = fvm_quality_diagnostics(sieve, cell_types, coordinates)?;
    serde_json::to_string(&report)
        .map_err(|e| MeshSieveError::InvalidGeometry(format!("serialize diagnostics: {e}")))
}

pub fn fv_readiness_report<CT, CS>(
    sieve: &MeshSieve,
    cell_types: &Section<CellType, CT>,
    coordinates: &Coordinates<f64, CS>,
    thresholds: FvReadinessThresholds,
    options: FvReadinessOptions,
) -> Result<FvReadinessReport, MeshSieveError>
where
    CT: Storage<CellType>,
    CS: Storage<f64>,
{
    let face_metrics = build_fvm_face_metrics(sieve, cell_types, coordinates)?;
    let mut report = FvReadinessReport {
        passed: true,
        thresholds,
        non_orthogonality: FvReadinessSummary {
            total: face_metrics.len(),
            ..FvReadinessSummary::default()
        },
        skewness: FvReadinessSummary {
            total: face_metrics.len(),
            ..FvReadinessSummary::default()
        },
        ..FvReadinessReport::default()
    };
    for fm in &face_metrics {
        let non_ortho = fm.non_orthogonality_deg.unwrap_or(0.0);
        if non_ortho > thresholds.max_non_orthogonality_deg {
            report.non_orthogonality.failed += 1;
            report.violations.push(FvReadinessViolation {
                metric: "non_orthogonality_deg",
                point: fm.face,
                value: non_ortho,
                threshold: thresholds.max_non_orthogonality_deg,
            });
        }
        let denom = fm
            .owner_to_neighbor
            .map(norm3)
            .unwrap_or_else(|| norm3(fm.owner_to_face));
        let skewness = if denom > 1e-12 {
            norm3(fm.skewness_vector) / denom
        } else {
            0.0
        };
        if skewness > thresholds.max_skewness {
            report.skewness.failed += 1;
            report.violations.push(FvReadinessViolation {
                metric: "skewness",
                point: fm.face,
                value: skewness,
                threshold: thresholds.max_skewness,
            });
        }
    }
    let mut cell_area_sum: std::collections::BTreeMap<PointId, f64> =
        std::collections::BTreeMap::new();
    for fm in &face_metrics {
        *cell_area_sum.entry(fm.owner).or_insert(0.0) += fm.area_magnitude;
        if let Some(n) = fm.neighbor {
            *cell_area_sum.entry(n).or_insert(0.0) += fm.area_magnitude;
        }
    }
    report.cell_char_length.total = cell_area_sum.len();
    for (cell, area_sum) in cell_area_sum {
        if let Some(cell_type) = cell_types
            .try_restrict(cell)
            .ok()
            .and_then(|v| v.first().copied())
        {
            let mut verts: Vec<_> = sieve
                .closure_iter([cell])
                .filter(|p| sieve.cone_points(*p).next().is_none())
                .collect();
            verts.sort_unstable();
            verts.dedup();
            let quality = crate::geometry::quality::cell_quality_from_section(
                cell_type,
                &verts,
                coordinates,
            )?;
            let char_len = if area_sum > 1e-12 {
                quality.jacobian_sign.abs() / area_sum
            } else {
                0.0
            };
            if char_len < thresholds.min_cell_char_length {
                report.cell_char_length.failed += 1;
                report.violations.push(FvReadinessViolation {
                    metric: "cell_char_length",
                    point: cell,
                    value: char_len,
                    threshold: thresholds.min_cell_char_length,
                });
            }
        }
    }
    report.passed = report.violations.is_empty();
    if options.strict && !report.passed {
        return Err(MeshSieveError::InvalidGeometry(format!(
            "FV readiness failed with {} violation(s)",
            report.violations.len()
        )));
    }
    Ok(report)
}

pub fn fv_readiness_report_json(report: &FvReadinessReport) -> Result<String, MeshSieveError> {
    serde_json::to_string(report)
        .map_err(|e| MeshSieveError::InvalidGeometry(format!("serialize readiness report: {e}")))
}

pub fn fv_readiness_report_text(report: &FvReadinessReport) -> String {
    let mut out = String::new();
    let _ = writeln!(&mut out, "fv_readiness_passed={}", report.passed);
    let _ = writeln!(
        &mut out,
        "thresholds: max_non_orthogonality_deg={}, max_skewness={}, min_cell_char_length={}",
        report.thresholds.max_non_orthogonality_deg,
        report.thresholds.max_skewness,
        report.thresholds.min_cell_char_length
    );
    let _ = writeln!(
        &mut out,
        "non_orthogonality_failed={}/{}",
        report.non_orthogonality.failed, report.non_orthogonality.total
    );
    let _ = writeln!(
        &mut out,
        "skewness_failed={}/{}",
        report.skewness.failed, report.skewness.total
    );
    let _ = writeln!(
        &mut out,
        "cell_char_length_failed={}/{}",
        report.cell_char_length.failed, report.cell_char_length.total
    );
    for v in &report.violations {
        let _ = writeln!(
            &mut out,
            "violation metric={} point={} value={} threshold={}",
            v.metric,
            v.point.get(),
            v.value,
            v.threshold
        );
    }
    out
}

fn summarize(values: &[f64]) -> MetricSummary {
    if values.is_empty() {
        return MetricSummary::default();
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let p95_idx = ((sorted.len() - 1) as f64 * 0.95).round() as usize;
    MetricSummary {
        max: *sorted.last().unwrap_or(&0.0),
        p95: sorted[p95_idx.min(sorted.len() - 1)],
        mean: sorted.iter().sum::<f64>() / (sorted.len() as f64),
        count: sorted.len(),
    }
}

fn norm3(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

pub fn mesh_text_summary<S: Sieve<Point = PointId>>(sieve: &S) -> Result<String, MeshSieveError> {
    let strata = compute_strata(sieve)?;
    Ok(format!(
        "points={}, topological_dimension={}, strata={:?}",
        strata.chart_points.len(),
        strata.diameter,
        strata.strata
    ))
}

pub fn mesh_dot_graph<S: Sieve<Point = PointId>>(sieve: &S) -> String {
    let mut out = String::from("digraph MeshSieve {\n");
    for src in sieve.points() {
        for dst in sieve.cone_points(src) {
            let _ = writeln!(&mut out, "  \"{:?}\" -> \"{:?}\";", src, dst);
        }
    }
    out.push_str("}\n");
    out
}

pub fn mesh_json_debug_dump<S: Sieve<Point = PointId>>(sieve: &S) -> String {
    let mut edges = Vec::new();
    for src in sieve.points() {
        for dst in sieve.cone_points(src) {
            edges.push(format!("[{},{}]", src.get(), dst.get()));
        }
    }
    format!("{{\"edges\":[{}]}}", edges.join(","))
}

pub fn mesh_tikz_viewer<S: Sieve<Point = PointId>>(sieve: &S, max_points: usize) -> Option<String> {
    let points: Vec<_> = sieve.points().collect();
    if points.len() > max_points {
        return None;
    }
    let mut out = String::from("\\begin{tikzpicture}[scale=1.0]\n");
    for (i, p) in points.iter().enumerate() {
        let _ = writeln!(
            &mut out,
            "\\node (p{}) at ({},0) {{{}}};",
            p.get(),
            i,
            p.get()
        );
    }
    for src in sieve.points() {
        for dst in sieve.cone_points(src) {
            let _ = writeln!(&mut out, "\\draw[->] (p{}) -- (p{});", src.get(), dst.get());
        }
    }
    out.push_str("\\end{tikzpicture}\n");
    Some(out)
}

pub fn fem_diagnostics_report(
    element_closure: &[PointId],
    basis_values: &[f64],
    quadrature_weights: &[f64],
    local_matrix: &[f64],
    local_vector: &[f64],
) -> String {
    format!(
        "FEM diagnostics: closure={:?}, basis={:?}, quadrature={:?}, Ke={:?}, Fe={:?}",
        element_closure, basis_values, quadrature_weights, local_matrix, local_vector
    )
}

pub fn fvm_flux_diagnostics(flux_updates: &[(PointId, f64)]) -> String {
    let mut out = String::from("FVM flux updates:");
    for (p, f) in flux_updates {
        let _ = write!(&mut out, " ({:?}: {f})", p);
    }
    out
}
