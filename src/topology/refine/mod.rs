//! Reference refinement templates and mesh refinement helpers.
//!
//! # Expected invariants
//! - The coarse topology is a cell complex where each cell points directly to its vertices
//!   or to intermediate edges/faces that can be reduced to vertices.
//! - `cell_types` contains **exactly one** [`CellType`] entry for every cell to refine.
//! - Each supported cell type has the expected number of vertices in its closure.
//! - Vertex ordering in each cone is **consistent** (e.g., counter-clockwise) so the
//!   reference templates preserve orientation. When intermediate edges are present,
//!   we try to recover this ordering from the edge cycle; otherwise vertices are
//!   canonicalized by PointId order.
//! - Shared edges/faces must appear with the same vertex sets so mid-edge and face-center
//!   points can be reused.
//!
//! The refinement map returned by [`refine_mesh`] is suitable for
//! [`crate::data::refine::SievedArray::try_refine_with_sifter`], enabling transfer of
//! cell-wise data from the coarse mesh to the refined mesh.

use crate::algs::interpolate::{interpolate_edges_faces, InterpolationResult};
use crate::data::atlas::Atlas;
use crate::data::coordinates::Coordinates;
use crate::data::section::Section;
use crate::data::storage::{Storage, VecStorage};
use crate::geometry::quality::validate_cell_geometry;
use crate::mesh_error::MeshSieveError;
use crate::topology::arrow::Polarity;
use crate::topology::cell_type::CellType;
use crate::topology::ownership::PointOwnership;
use crate::topology::point::PointId;
use crate::topology::sieve::{InMemorySieve, Sieve};
use std::collections::{HashMap, HashSet};

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

/// Output of [`refine_mesh_with_ownership`], including ownership metadata updates.
#[derive(Clone, Debug)]
pub struct RefinedMeshWithOwnership {
    /// Refined topology (cells → vertices).
    pub sieve: InMemorySieve<PointId, ()>,
    /// Mapping from coarse cell to refined cells (all `Polarity::Forward`).
    pub cell_refinement: RefinementMap,
    /// Updated ownership metadata including new points.
    pub ownership: PointOwnership,
}

/// Output of [`refine_mesh_full_topology`], including a full topology and types.
#[derive(Clone, Debug)]
pub struct RefinedMeshWithTopology {
    /// Refined topology (cells → faces → edges → vertices).
    pub sieve: InMemorySieve<PointId, ()>,
    /// Mapping from coarse cell to refined cells (all `Polarity::Forward`).
    pub cell_refinement: RefinementMap,
    /// Cell types for the refined topology (cells, edges, faces, vertices).
    pub cell_types: Section<CellType, VecStorage<CellType>>,
}

/// Optional settings for refinement.
#[derive(Clone, Copy, Debug, Default)]
pub struct RefineOptions {
    /// When enabled, validate cell geometry and reject inverted or degenerate elements.
    pub check_geometry: bool,
}

/// Pre-interpolate a cell→vertex topology to a full cell→face→edge→vertex mesh.
pub fn pre_interpolate_topology<S, CtSt>(
    sieve: &mut S,
    cell_types: &mut Section<CellType, CtSt>,
) -> Result<InterpolationResult, MeshSieveError>
where
    S: crate::topology::sieve::MutableSieve<Point = PointId>,
    S::Payload: Default,
    CtSt: Storage<CellType> + Clone,
{
    interpolate_edges_faces(sieve, cell_types)
}

/// Collapse a full topology to cell→vertex connectivity for refinement.
pub fn collapse_to_cell_vertices<CtSt>(
    sieve: &mut impl Sieve<Point = PointId>,
    cell_types: &Section<CellType, CtSt>,
) -> Result<InMemorySieve<PointId, ()>, MeshSieveError>
where
    CtSt: Storage<CellType>,
{
    let cells = collect_top_cells(cell_types)?;
    let mut collapsed = InMemorySieve::<PointId, ()>::default();

    for (cell, cell_type) in cells {
        let expected = expected_vertex_count(cell_type)
            .ok_or(MeshSieveError::UnsupportedRefinementCellType { cell, cell_type })?;
        let vertices = cell_vertices(sieve, cell, expected)?;
        for v in vertices {
            collapsed.add_arrow(cell, v, ());
        }
    }

    Ok(collapsed)
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

/// Reference 1→8 subdivision of a tetrahedron given vertices and mid-edge points.
pub fn tetrahedron_subdivision(
    vertices: [PointId; 4],
    midpoints: [PointId; 6],
) -> [[PointId; 4]; 8] {
    let [v0, v1, v2, v3] = vertices;
    let [m01, m12, m20, m03, m13, m23] = midpoints;
    [
        [v0, m01, m20, m03],
        [v1, m12, m01, m13],
        [v2, m20, m12, m23],
        [v3, m03, m13, m23],
        [m01, m12, m13, m23],
        [m01, m13, m03, m23],
        [m01, m03, m20, m23],
        [m01, m20, m12, m23],
    ]
}

/// Reference 1→8 subdivision of a hex using edge midpoints, face centers, and cell center.
pub fn hexahedron_subdivision(
    vertices: [PointId; 8],
    edge_midpoints: [PointId; 12],
    face_centers: [PointId; 6],
    center: PointId,
) -> [[PointId; 8]; 8] {
    let [v0, v1, v2, v3, v4, v5, v6, v7] = vertices;
    let [m01, m12, m23, m30, m45, m56, m67, m74, m04, m15, m26, m37] = edge_midpoints;
    let [f0123, f4567, f0154, f1265, f2376, f3047] = face_centers;

    let mut grid: HashMap<(u8, u8, u8), PointId> = HashMap::new();
    let mut insert = |x: u8, y: u8, z: u8, p: PointId| {
        grid.insert((x, y, z), p);
    };

    insert(0, 0, 0, v0);
    insert(2, 0, 0, v1);
    insert(2, 2, 0, v2);
    insert(0, 2, 0, v3);
    insert(0, 0, 2, v4);
    insert(2, 0, 2, v5);
    insert(2, 2, 2, v6);
    insert(0, 2, 2, v7);

    insert(1, 0, 0, m01);
    insert(2, 1, 0, m12);
    insert(1, 2, 0, m23);
    insert(0, 1, 0, m30);
    insert(1, 0, 2, m45);
    insert(2, 1, 2, m56);
    insert(1, 2, 2, m67);
    insert(0, 1, 2, m74);
    insert(0, 0, 1, m04);
    insert(2, 0, 1, m15);
    insert(2, 2, 1, m26);
    insert(0, 2, 1, m37);

    insert(1, 1, 0, f0123);
    insert(1, 1, 2, f4567);
    insert(1, 0, 1, f0154);
    insert(2, 1, 1, f1265);
    insert(1, 2, 1, f2376);
    insert(0, 1, 1, f3047);
    insert(1, 1, 1, center);

    let mut sub_hexes = Vec::with_capacity(8);
    for (ix, iy, iz) in [
        (0u8, 0u8, 0u8),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 1),
        (0, 1, 1),
    ] {
        let (x0, x1) = if ix == 0 { (0, 1) } else { (1, 2) };
        let (y0, y1) = if iy == 0 { (0, 1) } else { (1, 2) };
        let (z0, z1) = if iz == 0 { (0, 1) } else { (1, 2) };
        let verts = [
            grid[&(x0, y0, z0)],
            grid[&(x1, y0, z0)],
            grid[&(x1, y1, z0)],
            grid[&(x0, y1, z0)],
            grid[&(x0, y0, z1)],
            grid[&(x1, y0, z1)],
            grid[&(x1, y1, z1)],
            grid[&(x0, y1, z1)],
        ];
        sub_hexes.push(verts);
    }
    sub_hexes
        .try_into()
        .expect("hexahedron subdivision should yield 8 sub-hexes")
}

/// Reference 1→4 subdivision of a prism via triangle subdivisions on bottom/top.
pub fn prism_subdivision(
    vertices: [PointId; 6],
    bottom_midpoints: [PointId; 3],
    top_midpoints: [PointId; 3],
) -> [[PointId; 6]; 4] {
    let [v0, v1, v2, v3, v4, v5] = vertices;
    let bottom = triangle_subdivision([v0, v1, v2], bottom_midpoints);
    let top = triangle_subdivision([v3, v4, v5], top_midpoints);
    let mut prisms = Vec::with_capacity(4);
    for i in 0..4 {
        prisms.push([
            bottom[i][0],
            bottom[i][1],
            bottom[i][2],
            top[i][0],
            top[i][1],
            top[i][2],
        ]);
    }
    prisms
        .try_into()
        .expect("prism subdivision should yield 4 sub-prisms")
}

/// Reference 1→4 subdivision of a pyramid via quad subdivision of the base.
pub fn pyramid_subdivision(
    vertices: [PointId; 5],
    base_midpoints: [PointId; 4],
    base_center: PointId,
) -> [[PointId; 5]; 4] {
    let [v0, v1, v2, v3, apex] = vertices;
    let base = quadrilateral_subdivision([v0, v1, v2, v3], base_midpoints, base_center);
    let mut pyramids = Vec::with_capacity(4);
    for quad in base {
        pyramids.push([quad[0], quad[1], quad[2], quad[3], apex]);
    }
    pyramids
        .try_into()
        .expect("pyramid subdivision should yield 4 sub-pyramids")
}

/// Refine a 2D/3D mesh using reference subdivision templates.
///
/// Returns a refined [`InMemorySieve`] plus a refinement map for carrying
/// cell-wise data via [`crate::data::refine::SievedArray::try_refine_with_sifter`].
///
/// # Errors
/// - [`MeshSieveError::UnsupportedRefinementCellType`] for unsupported cell types.
/// - [`MeshSieveError::RefinementTopologyMismatch`] if vertex counts do not match templates.
/// - [`MeshSieveError::SliceLengthMismatch`] if the cell-type section is not length-1.
pub fn refine_mesh<S>(
    coarse: &mut impl Sieve<Point = PointId>,
    cell_types: &Section<CellType, S>,
) -> Result<RefinedMesh, MeshSieveError>
where
    S: Storage<CellType>,
{
    refine_mesh_with_options::<S, crate::data::storage::VecStorage<f64>>(
        coarse,
        cell_types,
        None,
        RefineOptions::default(),
    )
}

/// Refine a 2D/3D mesh using reference subdivision templates with optional checks.
///
/// Returns a refined [`InMemorySieve`] plus a refinement map for carrying
/// cell-wise data via [`crate::data::refine::SievedArray::try_refine_with_sifter`].
///
/// # Errors
/// - [`MeshSieveError::InvalidGeometry`] when geometry checks are requested and fail.
/// - [`MeshSieveError::UnsupportedRefinementCellType`] for unsupported cell types.
/// - [`MeshSieveError::RefinementTopologyMismatch`] if vertex counts do not match templates.
/// - [`MeshSieveError::SliceLengthMismatch`] if the cell-type section is not length-1.
pub fn refine_mesh_with_options<S, Cs>(
    coarse: &mut impl Sieve<Point = PointId>,
    cell_types: &Section<CellType, S>,
    coordinates: Option<&Coordinates<f64, Cs>>,
    options: RefineOptions,
) -> Result<RefinedMesh, MeshSieveError>
where
    S: Storage<CellType>,
    Cs: Storage<f64>,
{
    if options.check_geometry {
        let coords = coordinates.ok_or_else(|| {
            MeshSieveError::InvalidGeometry("geometry checks requested without coordinates".into())
        })?;
        for (cell, cell_slice) in cell_types.iter() {
            if cell_slice.len() != 1 {
                return Err(MeshSieveError::SliceLengthMismatch {
                    point: cell,
                    expected: 1,
                    found: cell_slice.len(),
                });
            }
            let cell_type = cell_slice[0];
            let expected = expected_vertex_count(cell_type)
                .ok_or(MeshSieveError::UnsupportedRefinementCellType { cell, cell_type })?;
            let vertices = cell_vertices(coarse, cell, expected)?;
            if let Err(err) = validate_cell_geometry(cell_type, &vertices, coords) {
                return Err(MeshSieveError::InvalidGeometry(format!(
                    "cell {cell:?}: {err}"
                )));
            }
        }
    }

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
    let mut face_centers: HashMap<Vec<PointId>, PointId> = HashMap::new();

    for (cell, cell_slice) in cell_types.iter() {
        if cell_slice.len() != 1 {
            return Err(MeshSieveError::SliceLengthMismatch {
                point: cell,
                expected: 1,
                found: cell_slice.len(),
            });
        }
        let cell_type = cell_slice[0];
        let expected = expected_vertex_count(cell_type)
            .ok_or(MeshSieveError::UnsupportedRefinementCellType { cell, cell_type })?;
        let vertices = cell_vertices(coarse, cell, expected)?;
        let mut fine_cells = Vec::new();

        match cell_type {
            CellType::Triangle => {
                let vertices = [vertices[0], vertices[1], vertices[2]];
                let midpoints = [
                    midpoint(&mut edge_midpoints, &mut next_id, vertices[0], vertices[1])?,
                    midpoint(&mut edge_midpoints, &mut next_id, vertices[1], vertices[2])?,
                    midpoint(&mut edge_midpoints, &mut next_id, vertices[2], vertices[0])?,
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
                let vertices = [vertices[0], vertices[1], vertices[2], vertices[3]];
                let midpoints = [
                    midpoint(&mut edge_midpoints, &mut next_id, vertices[0], vertices[1])?,
                    midpoint(&mut edge_midpoints, &mut next_id, vertices[1], vertices[2])?,
                    midpoint(&mut edge_midpoints, &mut next_id, vertices[2], vertices[3])?,
                    midpoint(&mut edge_midpoints, &mut next_id, vertices[3], vertices[0])?,
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
            CellType::Tetrahedron => {
                let vertices = [vertices[0], vertices[1], vertices[2], vertices[3]];
                let midpoints = [
                    midpoint(&mut edge_midpoints, &mut next_id, vertices[0], vertices[1])?,
                    midpoint(&mut edge_midpoints, &mut next_id, vertices[1], vertices[2])?,
                    midpoint(&mut edge_midpoints, &mut next_id, vertices[2], vertices[0])?,
                    midpoint(&mut edge_midpoints, &mut next_id, vertices[0], vertices[3])?,
                    midpoint(&mut edge_midpoints, &mut next_id, vertices[1], vertices[3])?,
                    midpoint(&mut edge_midpoints, &mut next_id, vertices[2], vertices[3])?,
                ];
                for verts in tetrahedron_subdivision(vertices, midpoints) {
                    let fine_cell = alloc_point(&mut next_id)?;
                    for v in verts {
                        refined.add_arrow(fine_cell, v, ());
                    }
                    fine_cells.push((fine_cell, Polarity::Forward));
                }
            }
            CellType::Hexahedron => {
                let vertices = [
                    vertices[0],
                    vertices[1],
                    vertices[2],
                    vertices[3],
                    vertices[4],
                    vertices[5],
                    vertices[6],
                    vertices[7],
                ];
                let edge_midpoints = [
                    midpoint(&mut edge_midpoints, &mut next_id, vertices[0], vertices[1])?,
                    midpoint(&mut edge_midpoints, &mut next_id, vertices[1], vertices[2])?,
                    midpoint(&mut edge_midpoints, &mut next_id, vertices[2], vertices[3])?,
                    midpoint(&mut edge_midpoints, &mut next_id, vertices[3], vertices[0])?,
                    midpoint(&mut edge_midpoints, &mut next_id, vertices[4], vertices[5])?,
                    midpoint(&mut edge_midpoints, &mut next_id, vertices[5], vertices[6])?,
                    midpoint(&mut edge_midpoints, &mut next_id, vertices[6], vertices[7])?,
                    midpoint(&mut edge_midpoints, &mut next_id, vertices[7], vertices[4])?,
                    midpoint(&mut edge_midpoints, &mut next_id, vertices[0], vertices[4])?,
                    midpoint(&mut edge_midpoints, &mut next_id, vertices[1], vertices[5])?,
                    midpoint(&mut edge_midpoints, &mut next_id, vertices[2], vertices[6])?,
                    midpoint(&mut edge_midpoints, &mut next_id, vertices[3], vertices[7])?,
                ];
                let face_centers = [
                    face_center(
                        &mut face_centers,
                        &mut next_id,
                        &[vertices[0], vertices[1], vertices[2], vertices[3]],
                    )?,
                    face_center(
                        &mut face_centers,
                        &mut next_id,
                        &[vertices[4], vertices[5], vertices[6], vertices[7]],
                    )?,
                    face_center(
                        &mut face_centers,
                        &mut next_id,
                        &[vertices[0], vertices[1], vertices[5], vertices[4]],
                    )?,
                    face_center(
                        &mut face_centers,
                        &mut next_id,
                        &[vertices[1], vertices[2], vertices[6], vertices[5]],
                    )?,
                    face_center(
                        &mut face_centers,
                        &mut next_id,
                        &[vertices[2], vertices[3], vertices[7], vertices[6]],
                    )?,
                    face_center(
                        &mut face_centers,
                        &mut next_id,
                        &[vertices[3], vertices[0], vertices[4], vertices[7]],
                    )?,
                ];
                let center = alloc_point(&mut next_id)?;
                for verts in hexahedron_subdivision(vertices, edge_midpoints, face_centers, center)
                {
                    let fine_cell = alloc_point(&mut next_id)?;
                    for v in verts {
                        refined.add_arrow(fine_cell, v, ());
                    }
                    fine_cells.push((fine_cell, Polarity::Forward));
                }
            }
            CellType::Prism => {
                let vertices = [
                    vertices[0],
                    vertices[1],
                    vertices[2],
                    vertices[3],
                    vertices[4],
                    vertices[5],
                ];
                let bottom_midpoints = [
                    midpoint(&mut edge_midpoints, &mut next_id, vertices[0], vertices[1])?,
                    midpoint(&mut edge_midpoints, &mut next_id, vertices[1], vertices[2])?,
                    midpoint(&mut edge_midpoints, &mut next_id, vertices[2], vertices[0])?,
                ];
                let top_midpoints = [
                    midpoint(&mut edge_midpoints, &mut next_id, vertices[3], vertices[4])?,
                    midpoint(&mut edge_midpoints, &mut next_id, vertices[4], vertices[5])?,
                    midpoint(&mut edge_midpoints, &mut next_id, vertices[5], vertices[3])?,
                ];
                for verts in prism_subdivision(vertices, bottom_midpoints, top_midpoints) {
                    let fine_cell = alloc_point(&mut next_id)?;
                    for v in verts {
                        refined.add_arrow(fine_cell, v, ());
                    }
                    fine_cells.push((fine_cell, Polarity::Forward));
                }
            }
            CellType::Pyramid => {
                let vertices = [
                    vertices[0],
                    vertices[1],
                    vertices[2],
                    vertices[3],
                    vertices[4],
                ];
                let base_midpoints = [
                    midpoint(&mut edge_midpoints, &mut next_id, vertices[0], vertices[1])?,
                    midpoint(&mut edge_midpoints, &mut next_id, vertices[1], vertices[2])?,
                    midpoint(&mut edge_midpoints, &mut next_id, vertices[2], vertices[3])?,
                    midpoint(&mut edge_midpoints, &mut next_id, vertices[3], vertices[0])?,
                ];
                let base_center = face_center(
                    &mut face_centers,
                    &mut next_id,
                    &[vertices[0], vertices[1], vertices[2], vertices[3]],
                )?;
                for verts in pyramid_subdivision(vertices, base_midpoints, base_center) {
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

/// Refine a mesh and return a full cell→face→edge→vertex topology.
///
/// This path is suitable for mixed-element meshes that already carry edges and faces.
/// The refined mesh includes interpolated edges/faces that are kept consistent across
/// shared boundaries via canonical vertex keys.
pub fn refine_mesh_full_topology<S>(
    coarse: &mut impl Sieve<Point = PointId>,
    cell_types: &Section<CellType, S>,
) -> Result<RefinedMeshWithTopology, MeshSieveError>
where
    S: Storage<CellType>,
{
    refine_mesh_full_topology_with_options::<S, crate::data::storage::VecStorage<f64>>(
        coarse,
        cell_types,
        None,
        RefineOptions::default(),
    )
}

/// Refine a mesh and return a full cell→face→edge→vertex topology with options.
pub fn refine_mesh_full_topology_with_options<S, Cs>(
    coarse: &mut impl Sieve<Point = PointId>,
    cell_types: &Section<CellType, S>,
    coordinates: Option<&Coordinates<f64, Cs>>,
    options: RefineOptions,
) -> Result<RefinedMeshWithTopology, MeshSieveError>
where
    S: Storage<CellType>,
    Cs: Storage<f64>,
{
    let cell_only = cell_only_section(cell_types)?;
    let mut refined = refine_mesh_with_options(coarse, &cell_only, coordinates, options)?;
    let mut refined_cell_types = build_refined_cell_types(&refined, &cell_only)?;
    interpolate_edges_faces(&mut refined.sieve, &mut refined_cell_types)?;

    Ok(RefinedMeshWithTopology {
        sieve: refined.sieve,
        cell_refinement: refined.cell_refinement,
        cell_types: refined_cell_types,
    })
}

/// Refine a mesh and update point ownership metadata consistently.
///
/// Ownership for new points is inherited from the owning rank of the coarse
/// cell that spawned them. When a new point is shared by multiple refined cells,
/// the smallest owner rank encountered is retained to guarantee deterministic
/// ownership assignment.
pub fn refine_mesh_with_ownership<S>(
    coarse: &mut impl Sieve<Point = PointId>,
    cell_types: &Section<CellType, S>,
    ownership: &PointOwnership,
    my_rank: usize,
) -> Result<RefinedMeshWithOwnership, MeshSieveError>
where
    S: Storage<CellType>,
{
    let refined = refine_mesh_with_options::<S, crate::data::storage::VecStorage<f64>>(
        coarse,
        cell_types,
        None,
        RefineOptions::default(),
    )?;
    let mut refined_ownership = ownership.clone();

    for (cell, fine_cells) in &refined.cell_refinement {
        let owner = refined_ownership.owner_or_err(*cell)?;
        for (fine_cell, _) in fine_cells {
            refined_ownership.set_owner_min(*fine_cell, owner, my_rank)?;
            for p in refined.sieve.cone_points(*fine_cell) {
                if refined_ownership.entry(p).is_none() {
                    refined_ownership.set_owner_min(p, owner, my_rank)?;
                }
            }
        }
    }

    Ok(RefinedMeshWithOwnership {
        sieve: refined.sieve,
        cell_refinement: refined.cell_refinement,
        ownership: refined_ownership,
    })
}

/// Refine a mesh and update point ownership metadata with optional geometry checks.
pub fn refine_mesh_with_ownership_and_options<S, Cs>(
    coarse: &mut impl Sieve<Point = PointId>,
    cell_types: &Section<CellType, S>,
    ownership: &PointOwnership,
    my_rank: usize,
    coordinates: Option<&Coordinates<f64, Cs>>,
    options: RefineOptions,
) -> Result<RefinedMeshWithOwnership, MeshSieveError>
where
    S: Storage<CellType>,
    Cs: Storage<f64>,
{
    let refined = refine_mesh_with_options(coarse, cell_types, coordinates, options)?;
    let mut refined_ownership = ownership.clone();

    for (cell, fine_cells) in &refined.cell_refinement {
        let owner = refined_ownership.owner_or_err(*cell)?;
        for (fine_cell, _) in fine_cells {
            refined_ownership.set_owner_min(*fine_cell, owner, my_rank)?;
            for p in refined.sieve.cone_points(*fine_cell) {
                if refined_ownership.entry(p).is_none() {
                    refined_ownership.set_owner_min(p, owner, my_rank)?;
                }
            }
        }
    }

    Ok(RefinedMeshWithOwnership {
        sieve: refined.sieve,
        cell_refinement: refined.cell_refinement,
        ownership: refined_ownership,
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

fn face_center(
    face_centers: &mut HashMap<Vec<PointId>, PointId>,
    next_id: &mut u64,
    vertices: &[PointId],
) -> Result<PointId, MeshSieveError> {
    let mut key = vertices.to_vec();
    key.sort();
    if let Some(p) = face_centers.get(&key) {
        return Ok(*p);
    }
    let p = alloc_point(next_id)?;
    face_centers.insert(key, p);
    Ok(p)
}

fn expected_vertex_count(cell_type: CellType) -> Option<usize> {
    match cell_type {
        CellType::Triangle => Some(3),
        CellType::Quadrilateral => Some(4),
        CellType::Tetrahedron => Some(4),
        CellType::Hexahedron => Some(8),
        CellType::Prism => Some(6),
        CellType::Pyramid => Some(5),
        _ => None,
    }
}

fn collect_top_cells<CtSt>(
    cell_types: &Section<CellType, CtSt>,
) -> Result<Vec<(PointId, CellType)>, MeshSieveError>
where
    CtSt: Storage<CellType>,
{
    let mut cells = Vec::new();
    let mut max_dim = None;

    for (point, cell_slice) in cell_types.iter() {
        if cell_slice.len() != 1 {
            return Err(MeshSieveError::SliceLengthMismatch {
                point,
                expected: 1,
                found: cell_slice.len(),
            });
        }
        let cell_type = cell_slice[0];
        let dim = cell_type.dimension();
        max_dim = Some(max_dim.map_or(dim, |current| current.max(dim)));
        cells.push((point, cell_type));
    }

    let max_dim = max_dim.unwrap_or(0);
    cells.retain(|(_, cell_type)| cell_type.dimension() == max_dim);
    Ok(cells)
}

fn cell_only_section<CtSt>(
    cell_types: &Section<CellType, CtSt>,
) -> Result<Section<CellType, VecStorage<CellType>>, MeshSieveError>
where
    CtSt: Storage<CellType>,
{
    let cells = collect_top_cells(cell_types)?;
    let mut atlas = Atlas::default();
    for (cell, _) in &cells {
        atlas
            .try_insert(*cell, 1)
            .map_err(|e| MeshSieveError::AtlasInsertionFailed(*cell, Box::new(e)))?;
    }
    let mut filtered = Section::<CellType, VecStorage<CellType>>::new(atlas);
    for (cell, cell_type) in cells {
        filtered.try_set(cell, &[cell_type])?;
    }
    Ok(filtered)
}

fn build_refined_cell_types<CtSt>(
    refined: &RefinedMesh,
    coarse_cells: &Section<CellType, CtSt>,
) -> Result<Section<CellType, VecStorage<CellType>>, MeshSieveError>
where
    CtSt: Storage<CellType>,
{
    let mut cell_map = HashMap::new();
    for (cell, cell_slice) in coarse_cells.iter() {
        if cell_slice.len() != 1 {
            return Err(MeshSieveError::SliceLengthMismatch {
                point: cell,
                expected: 1,
                found: cell_slice.len(),
            });
        }
        cell_map.insert(cell, cell_slice[0]);
    }

    let mut atlas = Atlas::default();
    let mut points = HashSet::new();
    for p in refined.sieve.points() {
        if points.insert(p) {
            atlas
                .try_insert(p, 1)
                .map_err(|e| MeshSieveError::AtlasInsertionFailed(p, Box::new(e)))?;
        }
    }
    let mut cell_types = Section::<CellType, VecStorage<CellType>>::new(atlas);

    for p in refined.sieve.points() {
        if refined.sieve.cone_points(p).next().is_none() {
            cell_types.try_set(p, &[CellType::Vertex])?;
        }
    }

    for (cell, fine_cells) in &refined.cell_refinement {
        let cell_type = *cell_map
            .get(cell)
            .ok_or(MeshSieveError::PointNotInAtlas(*cell))?;
        for (fine_cell, _) in fine_cells {
            cell_types.try_set(*fine_cell, &[cell_type])?;
        }
    }

    Ok(cell_types)
}

fn cell_vertices(
    coarse: &mut impl Sieve<Point = PointId>,
    cell: PointId,
    expected: usize,
) -> Result<Vec<PointId>, MeshSieveError> {
    let cone: Vec<PointId> = coarse.cone_points(cell).collect();
    if cone.len() == expected && cone.iter().all(|p| coarse.cone_points(*p).next().is_none()) {
        return Ok(cone);
    }

    if (expected == 3 || expected == 4)
        && !cone.is_empty()
        && cone.iter().all(|p| coarse.cone_points(*p).count() == 2)
    {
        if let Some(ordered) = ordered_vertices_from_edges(coarse, &cone) {
            if ordered.len() == expected {
                return Ok(ordered);
            }
        }
    }

    let mut vertices = Vec::new();
    for p in coarse.closure(std::iter::once(cell)) {
        if coarse.cone_points(p).next().is_none() {
            vertices.push(p);
        }
    }
    vertices.sort();
    vertices.dedup();
    if vertices.len() != expected {
        return Err(MeshSieveError::RefinementTopologyMismatch {
            cell,
            template: "cell",
            expected,
            found: vertices.len(),
        });
    }
    Ok(vertices)
}

fn ordered_vertices_from_edges(
    coarse: &mut impl Sieve<Point = PointId>,
    edges: &[PointId],
) -> Option<Vec<PointId>> {
    let mut edge_vertices = Vec::with_capacity(edges.len());
    for edge in edges {
        let verts: Vec<PointId> = coarse.cone_points(*edge).collect();
        if verts.len() != 2 {
            return None;
        }
        edge_vertices.push([verts[0], verts[1]]);
    }
    let mut ordered = Vec::with_capacity(edges.len());
    let first = edge_vertices.first()?.to_owned();
    ordered.push(first[0]);
    ordered.push(first[1]);
    for verts in edge_vertices.iter().skip(1) {
        let last = *ordered.last()?;
        if verts[0] == last {
            ordered.push(verts[1]);
        } else if verts[1] == last {
            ordered.push(verts[0]);
        } else {
            return None;
        }
    }
    if ordered.len() > 1 && ordered.first()? != ordered.last()? {
        return None;
    }
    ordered.pop();
    Some(ordered)
}
