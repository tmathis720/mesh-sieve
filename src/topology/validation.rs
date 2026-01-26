//! Topology validation helpers.

use crate::data::section::Section;
use crate::data::storage::Storage;
use crate::mesh_error::MeshSieveError;
use crate::topology::cell_type::CellType;
use crate::topology::point::PointId;
use crate::topology::sieve::Sieve;
use std::collections::HashSet;

/// Optional validation toggles for sieve topology checks.
#[derive(Debug, Clone, Copy)]
pub struct TopologyValidationOptions {
    /// Ensure the cone size matches the expected size for each cell type.
    pub check_cone_sizes: bool,
    /// Ensure each `(src, dst)` arrow appears at most once.
    pub check_duplicate_arrows: bool,
    /// Ensure closure-derived vertex counts match the expected size for each cell type.
    pub check_closure_consistency: bool,
}

impl TopologyValidationOptions {
    /// Enable all topology validation checks.
    pub fn all() -> Self {
        Self {
            check_cone_sizes: true,
            check_duplicate_arrows: true,
            check_closure_consistency: true,
        }
    }
}

/// Validate sieve topology against the provided cell type section.
pub fn validate_sieve_topology<S, CtSt>(
    sieve: &S,
    cell_types: &Section<CellType, CtSt>,
    options: TopologyValidationOptions,
) -> Result<(), MeshSieveError>
where
    S: Sieve<Point = PointId>,
    CtSt: Storage<CellType>,
{
    if options.check_duplicate_arrows {
        for src in sieve.points() {
            let mut seen = HashSet::new();
            for dst in sieve.cone_points(src) {
                if !seen.insert(dst) {
                    return Err(MeshSieveError::DuplicateArrow { src, dst });
                }
            }
        }
    }

    for (cell, cell_slice) in cell_types.iter() {
        if cell_slice.len() != 1 {
            return Err(MeshSieveError::SliceLengthMismatch {
                point: cell,
                expected: 1,
                found: cell_slice.len(),
            });
        }
        let cell_type = cell_slice[0];

        if options.check_cone_sizes {
            if let Some(expected) = expected_cone_size(cell_type) {
                let found = sieve.cone_points(cell).count();
                if found != expected {
                    return Err(MeshSieveError::ConeSizeMismatch {
                        cell,
                        cell_type,
                        expected,
                        found,
                    });
                }
            }
        }

        if options.check_closure_consistency {
            if let Some(expected) = expected_cone_size(cell_type) {
                let found = closure_vertex_count(sieve, cell);
                if found != expected {
                    return Err(MeshSieveError::ClosureVertexCountMismatch {
                        cell,
                        cell_type,
                        expected,
                        found,
                    });
                }
            }
        }
    }

    Ok(())
}

fn expected_cone_size(cell_type: CellType) -> Option<usize> {
    match cell_type {
        CellType::Vertex => Some(1),
        CellType::Segment => Some(2),
        CellType::Triangle => Some(3),
        CellType::Quadrilateral => Some(4),
        CellType::Tetrahedron => Some(4),
        CellType::Hexahedron => Some(8),
        CellType::Prism => Some(6),
        CellType::Pyramid => Some(5),
        CellType::Polygon(sides) => Some(usize::from(sides.max(1))),
        CellType::Simplex(dim) => Some(dim.saturating_add(1) as usize),
        CellType::Polyhedron => None,
    }
}

fn closure_vertex_count<S: Sieve<Point = PointId>>(sieve: &S, cell: PointId) -> usize {
    let mut vertices = HashSet::new();
    for point in sieve.closure_iter([cell]) {
        if sieve.cone(point).next().is_none() {
            vertices.insert(point);
        }
    }
    vertices.len()
}
