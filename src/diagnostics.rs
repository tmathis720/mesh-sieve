//! Centralized mesh diagnostics and DMPLEX-style validation options.

use std::fmt::Write as _;

use crate::data::coordinates::Coordinates;
use crate::data::section::Section;
use crate::data::storage::Storage;
use crate::mesh_error::MeshSieveError;
use crate::overlap::overlap::Overlap;
use crate::topology::cell_type::CellType;
use crate::topology::ownership::PointOwnership;
use crate::topology::point::PointId;
use crate::topology::sieve::strata::compute_strata;
use crate::topology::sieve::{MeshSieve, Sieve};
use crate::topology::validation::{
    TopologyValidationOptions, validate_oriented_sieve_topology, validate_overlap_ownership_topology,
};

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

pub fn mesh_text_summary<S: Sieve<Point = PointId>>(sieve: &S) -> Result<String, MeshSieveError> {
    let strata = compute_strata(sieve)?;
    Ok(format!(
        "points={}, topological_dimension={}, strata={:?}",
        strata.chart_points.len(), strata.diameter, strata.strata
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
        let _ = writeln!(&mut out, "\\node (p{}) at ({},0) {{{}}};", p.get(), i, p.get());
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
