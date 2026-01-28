//! Topology validation helpers.

use crate::data::section::Section;
use crate::data::storage::Storage;
use crate::mesh_error::MeshSieveError;
use crate::overlap::overlap::{Overlap, OvlId};
use crate::topology::cell_type::CellType;
use crate::topology::ownership::PointOwnership;
use crate::topology::point::PointId;
use crate::topology::sieve::Sieve;
use crate::topology::sieve::strata::compute_strata;
use std::collections::{BTreeSet, HashMap, HashSet};

/// Optional validation toggles for sieve topology checks.
#[derive(Debug, Clone, Copy)]
pub struct TopologyValidationOptions {
    /// Ensure the cone size matches the expected size for each cell type.
    pub check_cone_sizes: bool,
    /// Ensure each `(src, dst)` arrow appears at most once.
    pub check_duplicate_arrows: bool,
    /// Ensure closure-derived vertex counts match the expected size for each cell type.
    pub check_closure_consistency: bool,
    /// How to handle non-manifold edges/faces found by incident cell counting.
    pub non_manifold: NonManifoldHandling,
}

impl TopologyValidationOptions {
    /// Enable all topology validation checks.
    pub fn all() -> Self {
        Self {
            check_cone_sizes: true,
            check_duplicate_arrows: true,
            check_closure_consistency: true,
            non_manifold: NonManifoldHandling::Error,
        }
    }
}

/// Behavior for non-manifold detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NonManifoldHandling {
    /// Skip non-manifold detection.
    Ignore,
    /// Log a warning on non-manifold entities.
    Warn,
    /// Return an error on non-manifold entities.
    Error,
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

    validate_non_manifold(sieve, options.non_manifold)?;

    Ok(())
}

/// Validate ownership and overlap consistency with the local topology.
pub fn validate_overlap_ownership_topology<S>(
    sieve: &S,
    ownership: &PointOwnership,
    overlap: Option<&Overlap>,
    my_rank: usize,
) -> Result<(), MeshSieveError>
where
    S: Sieve<Point = PointId>,
{
    let sieve_points: HashSet<PointId> = sieve.points().collect();

    for point in ownership.local_points() {
        if !sieve_points.contains(&point) {
            return Err(MeshSieveError::OwnershipPointMissingTopology { point });
        }
    }

    for point in &sieve_points {
        if ownership.entry(*point).is_none() {
            return Err(MeshSieveError::TopologyPointMissingOwnership { point: *point });
        }
    }

    for point in ownership.local_points() {
        let entry = ownership
            .entry(point)
            .expect("ownership entry should exist");
        if entry.is_ghost == (entry.owner == my_rank) {
            return Err(MeshSieveError::OwnershipGhostMismatch {
                point,
                owner: entry.owner,
                my_rank,
            });
        }
    }

    if let Some(overlap) = overlap {
        let mut link_map: HashMap<PointId, BTreeSet<usize>> = HashMap::new();
        for src in overlap.base_points() {
            if let OvlId::Local(point) = src {
                if !sieve_points.contains(&point) {
                    return Err(MeshSieveError::OverlapPointMissingTopology { point });
                }
                if ownership.entry(point).is_none() {
                    return Err(MeshSieveError::OverlapPointMissingOwnership { point });
                }
                for (dst, rem) in overlap.cone(src) {
                    if let OvlId::Part(rank) = dst {
                        debug_assert_eq!(rem.rank, rank);
                        link_map.entry(point).or_default().insert(rank);
                    }
                }
            }
        }

        for point in ownership.ghost_points() {
            let owner = ownership
                .owner(point)
                .expect("ghost points should have owner");
            let links = link_map.get(&point);
            if !links.is_some_and(|set| set.contains(&owner)) {
                return Err(MeshSieveError::GhostPointMissingOverlapLink { point, owner });
            }
        }
    }

    Ok(())
}

/// Detect non-manifold edges/faces by counting incident cells in the star.
fn validate_non_manifold<S>(sieve: &S, handling: NonManifoldHandling) -> Result<(), MeshSieveError>
where
    S: Sieve<Point = PointId>,
{
    if handling == NonManifoldHandling::Ignore {
        return Ok(());
    }

    let cache = compute_strata(sieve)?;
    let top_dim = cache.diameter;
    if top_dim < 2 {
        return Ok(());
    }

    for &point in &cache.chart_points {
        let height = match cache.height.get(&point) {
            Some(height) => *height,
            None => continue,
        };
        let dim = top_dim.saturating_sub(height);
        let check_entity =
            (top_dim >= 2 && dim == top_dim - 1) || (top_dim >= 3 && dim == top_dim - 2);
        if !check_entity {
            continue;
        }

        let mut incident_cells = HashSet::new();
        for candidate in sieve.star_iter([point]) {
            if let Some(candidate_height) = cache.height.get(&candidate) {
                let candidate_dim = top_dim.saturating_sub(*candidate_height);
                if candidate_dim == top_dim {
                    incident_cells.insert(candidate);
                }
            }
        }

        let count = incident_cells.len();
        if count > 2 {
            match handling {
                NonManifoldHandling::Warn => {
                    log::warn!(
                        "Non-manifold entity detected: point={point:?} dim={dim} incident_cells={count}"
                    );
                }
                NonManifoldHandling::Error => {
                    return Err(MeshSieveError::NonManifoldIncidentCells {
                        point,
                        dimension: dim,
                        incident_cells: count,
                    });
                }
                NonManifoldHandling::Ignore => {}
            }
        }
    }

    Ok(())
}

#[cfg(any(
    debug_assertions,
    feature = "strict-invariants",
    feature = "check-invariants"
))]
/// Debug-only ownership/overlap validation (enabled in strict builds).
pub fn debug_validate_overlap_ownership_topology<S>(
    sieve: &S,
    ownership: &PointOwnership,
    overlap: Option<&Overlap>,
    my_rank: usize,
) -> Result<(), MeshSieveError>
where
    S: Sieve<Point = PointId>,
{
    validate_overlap_ownership_topology(sieve, ownership, overlap, my_rank)
}

#[cfg(not(any(
    debug_assertions,
    feature = "strict-invariants",
    feature = "check-invariants"
)))]
/// No-op ownership/overlap validation for release builds.
pub fn debug_validate_overlap_ownership_topology<S>(
    _sieve: &S,
    _ownership: &PointOwnership,
    _overlap: Option<&Overlap>,
    _my_rank: usize,
) -> Result<(), MeshSieveError>
where
    S: Sieve<Point = PointId>,
{
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
