//! Reference refinement templates and mesh refinement helpers.
//!
//! # Expected invariants
//! - The coarse topology is a **2D cell complex** where each cell points directly to
//!   its vertices (no intermediate edges/faces).
//! - `cell_types` contains **exactly one** [`CellType`] entry for every cell to refine.
//! - Triangle cells have exactly **3** vertices in their cone; quad cells have **4**.
//! - Vertex ordering in each cone is **consistent** (e.g., counter-clockwise) so the
//!   reference templates preserve orientation.
//! - Shared edges must appear with the same vertex pair so mid-edge points can be reused.
//!
//! The refinement map returned by [`refine_mesh`] is suitable for
//! [`crate::data::refine::SievedArray::try_refine_with_sifter`], enabling transfer of
//! cell-wise data from the coarse mesh to the refined mesh.

use crate::data::section::Section;
use crate::data::storage::Storage;
use crate::mesh_error::MeshSieveError;
use crate::topology::arrow::Polarity;
use crate::topology::cell_type::CellType;
use crate::topology::point::PointId;
use crate::topology::sieve::{InMemorySieve, Sieve};
use std::collections::HashMap;

/// Mapping from coarse cells to refined cells, with orientation hints.
pub type RefinementMap = Vec<(PointId, Vec<(PointId, Polarity)>)>;

/// Output of [`refine_mesh`], including the refined topology and transfer map.
#[derive(Clone, Debug)]
pub struct RefinedMesh {
    /// Refined topology (cells → vertices).
    pub sieve: InMemorySieve<PointId, ()>,
    /// Mapping from coarse cell to refined cells (all `Polarity::Forward`).
    pub cell_refinement: RefinementMap,
}

/// Reference 1→4 subdivision of a triangle given vertices and mid-edge points.
pub fn triangle_subdivision(vertices: [PointId; 3], midpoints: [PointId; 3]) -> [[PointId; 3]; 4] {
    let [v0, v1, v2] = vertices;
    let [m01, m12, m20] = midpoints;
    [
        [v0, m01, m20],
        [v1, m12, m01],
        [v2, m20, m12],
        [m01, m12, m20],
    ]
}

/// Reference 1→4 subdivision of a quad given vertices, mid-edge points, and a center point.
pub fn quadrilateral_subdivision(
    vertices: [PointId; 4],
    midpoints: [PointId; 4],
    center: PointId,
) -> [[PointId; 4]; 4] {
    let [v0, v1, v2, v3] = vertices;
    let [m01, m12, m23, m30] = midpoints;
    [
        [v0, m01, center, m30],
        [m01, v1, m12, center],
        [center, m12, v2, m23],
        [m30, center, m23, v3],
    ]
}

/// Refine a 2D mesh using reference triangle/quad subdivision templates.
///
/// Returns a refined [`InMemorySieve`] plus a refinement map for carrying
/// cell-wise data via [`crate::data::refine::SievedArray::try_refine_with_sifter`].
///
/// # Errors
/// - [`MeshSieveError::UnsupportedRefinementCellType`] for unsupported cell types.
/// - [`MeshSieveError::RefinementTopologyMismatch`] if cone sizes do not match templates.
/// - [`MeshSieveError::SliceLengthMismatch`] if the cell-type section is not length-1.
pub fn refine_mesh<S>(
    coarse: &mut impl Sieve<Point = PointId>,
    cell_types: &Section<CellType, S>,
) -> Result<RefinedMesh, MeshSieveError>
where
    S: Storage<CellType>,
{
    let mut max_id = 0u64;
    for p in coarse.chart_points()? {
        max_id = max_id.max(p.get());
    }
    let mut next_id = max_id
        .checked_add(1)
        .ok_or(MeshSieveError::InvalidPointId)?;

    let mut refined = InMemorySieve::<PointId, ()>::default();
    let mut refinement_map: RefinementMap = Vec::new();
    let mut edge_midpoints: HashMap<(PointId, PointId), PointId> = HashMap::new();

    for (cell, cell_slice) in cell_types.iter() {
        if cell_slice.len() != 1 {
            return Err(MeshSieveError::SliceLengthMismatch {
                point: cell,
                expected: 1,
                found: cell_slice.len(),
            });
        }
        let cell_type = cell_slice[0];
        let cone: Vec<PointId> = coarse.cone(cell).collect();
        let mut fine_cells = Vec::new();

        match cell_type {
            CellType::Triangle => {
                if cone.len() != 3 {
                    return Err(MeshSieveError::RefinementTopologyMismatch {
                        cell,
                        template: "triangle",
                        expected: 3,
                        found: cone.len(),
                    });
                }
                let vertices = [cone[0], cone[1], cone[2]];
                let midpoints = [
                    midpoint(&mut edge_midpoints, &mut next_id, cone[0], cone[1])?,
                    midpoint(&mut edge_midpoints, &mut next_id, cone[1], cone[2])?,
                    midpoint(&mut edge_midpoints, &mut next_id, cone[2], cone[0])?,
                ];
                for verts in triangle_subdivision(vertices, midpoints) {
                    let fine_cell = alloc_point(&mut next_id)?;
                    for v in verts {
                        refined.add_arrow(fine_cell, v, ());
                    }
                    fine_cells.push((fine_cell, Polarity::Forward));
                }
            }
            CellType::Quadrilateral => {
                if cone.len() != 4 {
                    return Err(MeshSieveError::RefinementTopologyMismatch {
                        cell,
                        template: "quadrilateral",
                        expected: 4,
                        found: cone.len(),
                    });
                }
                let vertices = [cone[0], cone[1], cone[2], cone[3]];
                let midpoints = [
                    midpoint(&mut edge_midpoints, &mut next_id, cone[0], cone[1])?,
                    midpoint(&mut edge_midpoints, &mut next_id, cone[1], cone[2])?,
                    midpoint(&mut edge_midpoints, &mut next_id, cone[2], cone[3])?,
                    midpoint(&mut edge_midpoints, &mut next_id, cone[3], cone[0])?,
                ];
                let center = alloc_point(&mut next_id)?;
                for verts in quadrilateral_subdivision(vertices, midpoints, center) {
                    let fine_cell = alloc_point(&mut next_id)?;
                    for v in verts {
                        refined.add_arrow(fine_cell, v, ());
                    }
                    fine_cells.push((fine_cell, Polarity::Forward));
                }
            }
            _ => return Err(MeshSieveError::UnsupportedRefinementCellType { cell, cell_type }),
        }

        refinement_map.push((cell, fine_cells));
    }

    Ok(RefinedMesh {
        sieve: refined,
        cell_refinement: refinement_map,
    })
}

fn alloc_point(next_id: &mut u64) -> Result<PointId, MeshSieveError> {
    let id = PointId::new(*next_id)?;
    *next_id = next_id
        .checked_add(1)
        .ok_or(MeshSieveError::InvalidPointId)?;
    Ok(id)
}

fn midpoint(
    edge_midpoints: &mut HashMap<(PointId, PointId), PointId>,
    next_id: &mut u64,
    a: PointId,
    b: PointId,
) -> Result<PointId, MeshSieveError> {
    let key = if a < b { (a, b) } else { (b, a) };
    if let Some(p) = edge_midpoints.get(&key) {
        return Ok(*p);
    }
    let p = alloc_point(next_id)?;
    edge_midpoints.insert(key, p);
    Ok(p)
}
