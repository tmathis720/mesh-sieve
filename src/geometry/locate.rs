//! Point location, periodic coordinate localization, and projection helpers.
//!
//! The APIs in this module are deliberately geometry-only: they build spatial
//! acceleration data from a sieve, a cell-type section, and coordinate section,
//! then use the existing reference/physical mapping routines to test candidate
//! cells and interpolate linear nodal fields.

use crate::data::atlas::Atlas;
use crate::data::coordinates::Coordinates;
use crate::data::section::Section;
use crate::data::storage::{Storage, VecStorage};
use crate::geometry::metrics::{physical_to_reference, reference_to_physical};
use crate::mesh_error::MeshSieveError;
use crate::topology::cell_type::CellType;
use crate::topology::point::PointId;
use crate::topology::sieve::Sieve;
use std::collections::{BTreeSet, HashMap};

const DEFAULT_TOLERANCE: f64 = 1.0e-10;

/// Axis-aligned bounding box in padded three-dimensional coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoundingBox {
    /// Minimum coordinate in each embedding direction.
    pub min: [f64; 3],
    /// Maximum coordinate in each embedding direction.
    pub max: [f64; 3],
}

impl BoundingBox {
    /// Create an empty box that can be grown with [`include`](Self::include).
    pub fn empty() -> Self {
        Self {
            min: [f64::INFINITY; 3],
            max: [f64::NEG_INFINITY; 3],
        }
    }

    /// Build a bounding box for a list of vertices.
    pub fn from_points(points: &[[f64; 3]]) -> Result<Self, MeshSieveError> {
        if points.is_empty() {
            return Err(MeshSieveError::InvalidGeometry(
                "cannot build a bounding box for zero points".to_string(),
            ));
        }
        let mut bbox = Self::empty();
        for point in points {
            bbox.include(*point);
        }
        Ok(bbox)
    }

    /// Grow the box so it contains `point`.
    pub fn include(&mut self, point: [f64; 3]) {
        for d in 0..3 {
            self.min[d] = self.min[d].min(point[d]);
            self.max[d] = self.max[d].max(point[d]);
        }
    }

    /// Return a copy inflated by `amount` in every direction.
    pub fn inflated(mut self, amount: f64) -> Self {
        for d in 0..3 {
            self.min[d] -= amount;
            self.max[d] += amount;
        }
        self
    }

    /// Test whether the box contains a point, allowing `tol` slack.
    pub fn contains(&self, point: [f64; 3], tol: f64) -> bool {
        (0..3).all(|d| point[d] >= self.min[d] - tol && point[d] <= self.max[d] + tol)
    }

    fn union(a: &Self, b: &Self) -> Self {
        let mut out = *a;
        for d in 0..3 {
            out.min[d] = out.min[d].min(b.min[d]);
            out.max[d] = out.max[d].max(b.max[d]);
        }
        out
    }

    fn center(&self, axis: usize) -> f64 {
        0.5 * (self.min[axis] + self.max[axis])
    }
}

/// A cell record stored in a point-location index.
#[derive(Clone, Debug)]
pub struct LocatedCell {
    /// Mesh point representing the cell.
    pub cell: PointId,
    /// Supported cell type.
    pub cell_type: CellType,
    /// Cell vertices in the order used by the coordinate mapping.
    pub vertices: Vec<PointId>,
    /// Physical coordinates for the vertices, padded to three components.
    pub vertex_coordinates: Vec<[f64; 3]>,
    /// Cell bounding box.
    pub bbox: BoundingBox,
}

/// Successful point-location result.
#[derive(Clone, Debug, PartialEq)]
pub struct PointLocation {
    /// Cell containing the physical point.
    pub cell: PointId,
    /// Supported cell type.
    pub cell_type: CellType,
    /// Reference coordinates in the cell's reference element.
    pub reference_coordinates: Vec<f64>,
}

/// Periodic directions and extents used for coordinate localization.
#[derive(Clone, Debug)]
pub struct PeriodicDomain {
    min: [f64; 3],
    max: [f64; 3],
    periodic: [bool; 3],
    dimension: usize,
}

impl PeriodicDomain {
    /// Create a periodic domain from embedding-space bounds and active flags.
    pub fn new(min: &[f64], max: &[f64], periodic: &[bool]) -> Result<Self, MeshSieveError> {
        if min.len() != max.len() || min.len() != periodic.len() || min.is_empty() || min.len() > 3
        {
            return Err(MeshSieveError::InvalidGeometry(
                "periodic domain arrays must have matching length 1..=3".to_string(),
            ));
        }
        let mut mn = [0.0; 3];
        let mut mx = [0.0; 3];
        let mut per = [false; 3];
        for d in 0..min.len() {
            if periodic[d] && max[d] <= min[d] {
                return Err(MeshSieveError::InvalidGeometry(format!(
                    "periodic extent in direction {d} must be positive"
                )));
            }
            mn[d] = min[d];
            mx[d] = max[d];
            per[d] = periodic[d];
        }
        Ok(Self {
            min: mn,
            max: mx,
            periodic: per,
            dimension: min.len(),
        })
    }

    /// Coordinate dimension covered by this domain.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    fn length(&self, axis: usize) -> f64 {
        self.max[axis] - self.min[axis]
    }

    /// Wrap a point into the domain's fundamental periodic interval.
    pub fn localize_point(&self, point: &[f64]) -> Result<Vec<f64>, MeshSieveError> {
        if point.len() != self.dimension {
            return Err(MeshSieveError::InvalidGeometry(format!(
                "point dimension mismatch: expected {}, got {}",
                self.dimension,
                point.len()
            )));
        }
        let mut out = point.to_vec();
        for d in 0..self.dimension {
            if self.periodic[d] {
                let len = self.length(d);
                out[d] = self.min[d] + (out[d] - self.min[d]).rem_euclid(len);
                if (out[d] - self.max[d]).abs() <= DEFAULT_TOLERANCE {
                    out[d] = self.min[d];
                }
            }
        }
        Ok(out)
    }

    fn localize_vertex_near(&self, vertex: [f64; 3], target: [f64; 3]) -> [f64; 3] {
        let mut out = vertex;
        for d in 0..self.dimension {
            if self.periodic[d] {
                let len = self.length(d);
                let shift = ((target[d] - out[d]) / len).round();
                out[d] += shift * len;
            }
        }
        out
    }
}

/// Cell-local coordinate view, optionally shifted across periodic boundaries.
#[derive(Clone, Debug)]
pub struct LocalizedCellCoordinates {
    /// Mesh point representing the cell.
    pub cell: PointId,
    /// Cell vertices in interpolation order.
    pub vertices: Vec<PointId>,
    /// Localized physical coordinates, padded to three components.
    pub coordinates: Vec<[f64; 3]>,
}

#[derive(Clone, Debug)]
enum SpatialBackend {
    Linear,
    Grid(GridHash),
    Bvh(BvhNode),
}

/// Spatial point-location index over supported mesh cells.
#[derive(Clone, Debug)]
pub struct PointLocator {
    cells: Vec<LocatedCell>,
    domain_bbox: BoundingBox,
    backend: SpatialBackend,
    tolerance: f64,
}

impl PointLocator {
    /// Build a deterministic linear locator.
    pub fn linear<Sv, St, CtSt>(
        sieve: &Sv,
        cell_types: &Section<CellType, CtSt>,
        coordinates: &Coordinates<f64, St>,
    ) -> Result<Self, MeshSieveError>
    where
        Sv: Sieve<Point = PointId>,
        St: Storage<f64>,
        CtSt: Storage<CellType>,
    {
        Self::build(sieve, cell_types, coordinates, SpatialBackendKind::Linear)
    }

    /// Build a uniform grid-hash locator.
    pub fn grid_hash<Sv, St, CtSt>(
        sieve: &Sv,
        cell_types: &Section<CellType, CtSt>,
        coordinates: &Coordinates<f64, St>,
    ) -> Result<Self, MeshSieveError>
    where
        Sv: Sieve<Point = PointId>,
        St: Storage<f64>,
        CtSt: Storage<CellType>,
    {
        Self::build(sieve, cell_types, coordinates, SpatialBackendKind::GridHash)
    }

    /// Build a BVH-style locator.
    pub fn bvh<Sv, St, CtSt>(
        sieve: &Sv,
        cell_types: &Section<CellType, CtSt>,
        coordinates: &Coordinates<f64, St>,
    ) -> Result<Self, MeshSieveError>
    where
        Sv: Sieve<Point = PointId>,
        St: Storage<f64>,
        CtSt: Storage<CellType>,
    {
        Self::build(sieve, cell_types, coordinates, SpatialBackendKind::Bvh)
    }

    fn build<Sv, St, CtSt>(
        sieve: &Sv,
        cell_types: &Section<CellType, CtSt>,
        coordinates: &Coordinates<f64, St>,
        kind: SpatialBackendKind,
    ) -> Result<Self, MeshSieveError>
    where
        Sv: Sieve<Point = PointId>,
        St: Storage<f64>,
        CtSt: Storage<CellType>,
    {
        let mut cells = collect_cells(sieve, cell_types, coordinates)?;
        cells.sort_by_key(|cell| cell.cell);
        let domain_bbox = cells
            .iter()
            .map(|cell| cell.bbox)
            .reduce(|a, b| BoundingBox::union(&a, &b))
            .ok_or_else(|| {
                MeshSieveError::InvalidGeometry("mesh has no locatable cells".to_string())
            })?;
        let backend = match kind {
            SpatialBackendKind::Linear => SpatialBackend::Linear,
            SpatialBackendKind::GridHash => {
                SpatialBackend::Grid(GridHash::new(&cells, domain_bbox))
            }
            SpatialBackendKind::Bvh => SpatialBackend::Bvh(BvhNode::build(
                &cells,
                &(0..cells.len()).collect::<Vec<_>>(),
            )),
        };
        Ok(Self {
            cells,
            domain_bbox,
            backend,
            tolerance: DEFAULT_TOLERANCE,
        })
    }

    /// Set geometric tolerance used by bounding-box and reference-cell tests.
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance.max(0.0);
        self
    }

    /// Number of indexed cells.
    pub fn len(&self) -> usize {
        self.cells.len()
    }

    /// Whether the index contains no cells.
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    /// Indexed cells in deterministic point-id order.
    pub fn cells(&self) -> &[LocatedCell] {
        &self.cells
    }

    /// Return an indexed cell by point id.
    pub fn cell(&self, cell: PointId) -> Option<&LocatedCell> {
        self.cells.iter().find(|entry| entry.cell == cell)
    }

    /// Locate `point` in a mesh cell and return reference coordinates.
    pub fn locate_point(&self, point: &[f64]) -> Result<Option<PointLocation>, MeshSieveError> {
        let query = pad_point(point)?;
        if !self.domain_bbox.contains(query, self.tolerance) {
            return Ok(None);
        }
        for idx in self.candidate_indices(query) {
            let cell = &self.cells[idx];
            if !cell.bbox.contains(query, self.tolerance) {
                continue;
            }
            let reference =
                physical_to_reference(cell.cell_type, &cell.vertex_coordinates, &query)?;
            if reference_coordinates_in_cell(cell.cell_type, &reference, self.tolerance) {
                return Ok(Some(PointLocation {
                    cell: cell.cell,
                    cell_type: cell.cell_type,
                    reference_coordinates: reference,
                }));
            }
        }
        Ok(None)
    }

    /// Locate a point after first wrapping it into a periodic fundamental domain.
    pub fn locate_periodic_point(
        &self,
        point: &[f64],
        domain: &PeriodicDomain,
    ) -> Result<Option<PointLocation>, MeshSieveError> {
        let localized = domain.localize_point(point)?;
        self.locate_point(&localized)
    }

    /// Return a localized coordinate view for `cell` near `point`.
    pub fn localized_cell_coordinates(
        &self,
        cell: PointId,
        point: &[f64],
        domain: Option<&PeriodicDomain>,
    ) -> Result<LocalizedCellCoordinates, MeshSieveError> {
        let target = pad_point(point)?;
        let located = self
            .cells
            .iter()
            .find(|entry| entry.cell == cell)
            .ok_or_else(|| {
                MeshSieveError::InvalidGeometry(format!("cell {cell:?} is not in the locator"))
            })?;
        let coordinates = match domain {
            Some(domain) => located
                .vertex_coordinates
                .iter()
                .map(|coord| domain.localize_vertex_near(*coord, target))
                .collect(),
            None => located.vertex_coordinates.clone(),
        };
        Ok(LocalizedCellCoordinates {
            cell,
            vertices: located.vertices.clone(),
            coordinates,
        })
    }

    /// Interpolate a linear nodal section at a physical point.
    pub fn interpolate_section_at_point<St>(
        &self,
        section: &Section<f64, St>,
        point: &[f64],
    ) -> Result<Option<Vec<f64>>, MeshSieveError>
    where
        St: Storage<f64>,
    {
        let Some(location) = self.locate_point(point)? else {
            return Ok(None);
        };
        self.interpolate_section_at_location(section, &location)
            .map(Some)
    }

    /// Interpolate a linear nodal section at an already located point.
    pub fn interpolate_section_at_location<St>(
        &self,
        section: &Section<f64, St>,
        location: &PointLocation,
    ) -> Result<Vec<f64>, MeshSieveError>
    where
        St: Storage<f64>,
    {
        let cell = self
            .cells
            .iter()
            .find(|entry| entry.cell == location.cell)
            .ok_or_else(|| {
                MeshSieveError::InvalidGeometry(format!(
                    "cell {:?} is not in the locator",
                    location.cell
                ))
            })?;
        interpolate_on_cell(cell, section, &location.reference_coordinates)
    }

    fn candidate_indices(&self, point: [f64; 3]) -> Vec<usize> {
        let mut candidates = match &self.backend {
            SpatialBackend::Linear => (0..self.cells.len()).collect(),
            SpatialBackend::Grid(grid) => grid.query(point),
            SpatialBackend::Bvh(root) => {
                let mut out = Vec::new();
                root.query_point(point, self.tolerance, &mut out);
                out
            }
        };
        candidates.sort_by_key(|idx| self.cells[*idx].cell);
        candidates.dedup();
        candidates
    }
}

enum SpatialBackendKind {
    Linear,
    GridHash,
    Bvh,
}

#[derive(Clone, Debug)]
struct GridHash {
    origin: [f64; 3],
    cell_width: [f64; 3],
    dims: [usize; 3],
    buckets: HashMap<(usize, usize, usize), Vec<usize>>,
}

impl GridHash {
    fn new(cells: &[LocatedCell], domain_bbox: BoundingBox) -> Self {
        let target = (cells.len() as f64).cbrt().ceil().max(1.0) as usize;
        let mut dims = [target; 3];
        let mut cell_width = [1.0; 3];
        for d in 0..3 {
            let span = domain_bbox.max[d] - domain_bbox.min[d];
            if span <= DEFAULT_TOLERANCE {
                dims[d] = 1;
                cell_width[d] = 1.0;
            } else {
                cell_width[d] = span / dims[d] as f64;
            }
        }
        let mut grid = Self {
            origin: domain_bbox.min,
            cell_width,
            dims,
            buckets: HashMap::new(),
        };
        for (idx, cell) in cells.iter().enumerate() {
            let lo = grid.key_for(cell.bbox.min);
            let hi = grid.key_for(cell.bbox.max);
            for i in lo.0..=hi.0 {
                for j in lo.1..=hi.1 {
                    for k in lo.2..=hi.2 {
                        grid.buckets.entry((i, j, k)).or_default().push(idx);
                    }
                }
            }
        }
        grid
    }

    fn key_for(&self, point: [f64; 3]) -> (usize, usize, usize) {
        let mut key = [0usize; 3];
        for d in 0..3 {
            if self.dims[d] == 1 {
                key[d] = 0;
            } else {
                let raw = ((point[d] - self.origin[d]) / self.cell_width[d]).floor() as isize;
                key[d] = raw.clamp(0, self.dims[d] as isize - 1) as usize;
            }
        }
        (key[0], key[1], key[2])
    }

    fn query(&self, point: [f64; 3]) -> Vec<usize> {
        self.buckets
            .get(&self.key_for(point))
            .cloned()
            .unwrap_or_default()
    }
}

#[derive(Clone, Debug)]
struct BvhNode {
    bbox: BoundingBox,
    children: Option<(Box<BvhNode>, Box<BvhNode>)>,
    cells: Vec<usize>,
}

impl BvhNode {
    fn build(cells: &[LocatedCell], indices: &[usize]) -> Self {
        let bbox = indices
            .iter()
            .map(|idx| cells[*idx].bbox)
            .reduce(|a, b| BoundingBox::union(&a, &b))
            .unwrap_or_else(BoundingBox::empty);
        if indices.len() <= 8 {
            return Self {
                bbox,
                children: None,
                cells: indices.to_vec(),
            };
        }
        let mut axis = 0;
        let mut best = bbox.max[0] - bbox.min[0];
        for d in 1..3 {
            let span = bbox.max[d] - bbox.min[d];
            if span > best {
                best = span;
                axis = d;
            }
        }
        let mut sorted = indices.to_vec();
        sorted.sort_by(|a, b| {
            cells[*a]
                .bbox
                .center(axis)
                .partial_cmp(&cells[*b].bbox.center(axis))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| cells[*a].cell.cmp(&cells[*b].cell))
        });
        let mid = sorted.len() / 2;
        Self {
            bbox,
            children: Some((
                Box::new(Self::build(cells, &sorted[..mid])),
                Box::new(Self::build(cells, &sorted[mid..])),
            )),
            cells: Vec::new(),
        }
    }

    fn query_point(&self, point: [f64; 3], tol: f64, out: &mut Vec<usize>) {
        if !self.bbox.contains(point, tol) {
            return;
        }
        if let Some((left, right)) = &self.children {
            left.query_point(point, tol, out);
            right.query_point(point, tol, out);
        } else {
            out.extend(self.cells.iter().copied());
        }
    }
}

/// Locate a physical point with a transient grid-hash index.
pub fn locate_point<Sv, St, CtSt>(
    sieve: &Sv,
    cell_types: &Section<CellType, CtSt>,
    coordinates: &Coordinates<f64, St>,
    point: &[f64],
) -> Result<Option<PointLocation>, MeshSieveError>
where
    Sv: Sieve<Point = PointId>,
    St: Storage<f64>,
    CtSt: Storage<CellType>,
{
    PointLocator::grid_hash::<Sv, St, CtSt>(sieve, cell_types, coordinates)?.locate_point(point)
}

/// Project/interpolate a linear nodal source section onto target mesh vertices.
pub fn project_section_to_vertices<
    SvSrc,
    SvDst,
    SrcCoordSt,
    DstCoordSt,
    SrcFieldSt,
    SrcCtSt,
    DstCtSt,
>(
    source_sieve: &SvSrc,
    source_cell_types: &Section<CellType, SrcCtSt>,
    source_coordinates: &Coordinates<f64, SrcCoordSt>,
    source_section: &Section<f64, SrcFieldSt>,
    target_sieve: &SvDst,
    target_cell_types: &Section<CellType, DstCtSt>,
    target_coordinates: &Coordinates<f64, DstCoordSt>,
) -> Result<Section<f64, VecStorage<f64>>, MeshSieveError>
where
    SvSrc: Sieve<Point = PointId>,
    SvDst: Sieve<Point = PointId>,
    SrcCoordSt: Storage<f64>,
    DstCoordSt: Storage<f64>,
    SrcFieldSt: Storage<f64>,
    SrcCtSt: Storage<CellType>,
    DstCtSt: Storage<CellType>,
{
    let locator = PointLocator::grid_hash::<SvSrc, SrcCoordSt, SrcCtSt>(
        source_sieve,
        source_cell_types,
        source_coordinates,
    )?;
    let dof = source_section
        .iter()
        .next()
        .map(|(_, values)| values.len())
        .ok_or_else(|| MeshSieveError::InvalidGeometry("source section is empty".to_string()))?;
    let target_vertices = mesh_vertices(target_sieve, target_cell_types)?;
    let mut atlas = Atlas::default();
    for vertex in &target_vertices {
        atlas.try_insert(*vertex, dof)?;
    }
    let mut out = Section::<f64, VecStorage<f64>>::new(atlas);
    for vertex in target_vertices {
        let point = target_coordinates.try_restrict(vertex)?;
        let values = locator
            .interpolate_section_at_point(source_section, point)?
            .ok_or_else(|| {
                MeshSieveError::InvalidGeometry(format!(
                    "target point {vertex:?} lies outside the source mesh"
                ))
            })?;
        out.try_set(vertex, &values)?;
    }
    Ok(out)
}

fn collect_cells<Sv, St, CtSt>(
    sieve: &Sv,
    cell_types: &Section<CellType, CtSt>,
    coordinates: &Coordinates<f64, St>,
) -> Result<Vec<LocatedCell>, MeshSieveError>
where
    Sv: Sieve<Point = PointId>,
    St: Storage<f64>,
    CtSt: Storage<CellType>,
{
    let mut cells = Vec::new();
    for (point, cell_type_slice) in cell_types.iter() {
        let cell_type = *cell_type_slice.first().ok_or_else(|| {
            MeshSieveError::InvalidGeometry(format!("missing cell type for {point:?}"))
        })?;
        if expected_vertex_count(cell_type).is_none() || cell_type == CellType::Vertex {
            continue;
        }
        let mut vertices = ordered_cell_vertices(sieve, point, cell_type)?;
        let mut vertex_coordinates = gather_vertices(coordinates, &vertices)?;
        canonicalize_vertex_order(cell_type, &mut vertices, &mut vertex_coordinates);
        let bbox = BoundingBox::from_points(&vertex_coordinates)?.inflated(DEFAULT_TOLERANCE);
        cells.push(LocatedCell {
            cell: point,
            cell_type,
            vertices,
            vertex_coordinates,
            bbox,
        });
    }
    Ok(cells)
}

fn mesh_vertices<Sv, CtSt>(
    sieve: &Sv,
    cell_types: &Section<CellType, CtSt>,
) -> Result<Vec<PointId>, MeshSieveError>
where
    Sv: Sieve<Point = PointId>,
    CtSt: Storage<CellType>,
{
    let mut vertices = BTreeSet::new();
    for (point, cell_type_slice) in cell_types.iter() {
        if cell_type_slice.first() == Some(&CellType::Vertex)
            || sieve.cone_points(point).next().is_none()
        {
            vertices.insert(point);
        }
    }
    Ok(vertices.into_iter().collect())
}

fn ordered_cell_vertices<Sv>(
    sieve: &Sv,
    cell: PointId,
    cell_type: CellType,
) -> Result<Vec<PointId>, MeshSieveError>
where
    Sv: Sieve<Point = PointId>,
{
    let expected = expected_vertex_count(cell_type).ok_or_else(|| {
        MeshSieveError::InvalidGeometry(format!("unsupported cell type: {cell_type:?}"))
    })?;
    let cone: Vec<_> = sieve.cone_points(cell).collect();
    if cone.len() == expected
        && cone
            .iter()
            .all(|point| sieve.cone_points(*point).next().is_none())
    {
        return Ok(cone);
    }
    let mut vertices = Vec::new();
    for point in sieve.closure(std::iter::once(cell)) {
        if sieve.cone_points(point).next().is_none() {
            vertices.push(point);
        }
    }
    vertices.sort();
    vertices.dedup();
    if vertices.len() != expected {
        return Err(MeshSieveError::InvalidGeometry(format!(
            "expected {expected} vertices for cell {cell:?}, found {}",
            vertices.len()
        )));
    }
    Ok(vertices)
}

fn canonicalize_vertex_order(
    cell_type: CellType,
    vertices: &mut Vec<PointId>,
    coordinates: &mut Vec<[f64; 3]>,
) {
    match cell_type {
        CellType::Triangle | CellType::Simplex(2) | CellType::Quadrilateral => {
            order_planar_polygon(vertices, coordinates);
        }
        CellType::Hexahedron if vertices.len() == 8 => {
            order_hexahedron(vertices, coordinates);
        }
        _ => {}
    }
}

fn order_planar_polygon(vertices: &mut Vec<PointId>, coordinates: &mut Vec<[f64; 3]>) {
    if vertices.len() < 3 {
        return;
    }
    let centroid = coordinates.iter().fold([0.0; 3], |mut acc, coord| {
        acc[0] += coord[0];
        acc[1] += coord[1];
        acc[2] += coord[2];
        acc
    });
    let inv_n = 1.0 / coordinates.len() as f64;
    let center = [
        centroid[0] * inv_n,
        centroid[1] * inv_n,
        centroid[2] * inv_n,
    ];
    let mut pairs: Vec<_> = vertices
        .iter()
        .copied()
        .zip(coordinates.iter().copied())
        .collect();
    pairs.sort_by(|(_, a), (_, b)| {
        (a[1] - center[1])
            .atan2(a[0] - center[0])
            .partial_cmp(&(b[1] - center[1]).atan2(b[0] - center[0]))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    rotate_smallest_coordinate_first(&mut pairs);
    *vertices = pairs.iter().map(|(vertex, _)| *vertex).collect();
    *coordinates = pairs.iter().map(|(_, coord)| *coord).collect();
}

fn order_hexahedron(vertices: &mut Vec<PointId>, coordinates: &mut Vec<[f64; 3]>) {
    let mut pairs: Vec<_> = vertices
        .iter()
        .copied()
        .zip(coordinates.iter().copied())
        .collect();
    pairs.sort_by(|(_, a), (_, b)| {
        a[2].partial_cmp(&b[2])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a[1].partial_cmp(&b[1]).unwrap_or(std::cmp::Ordering::Equal))
            .then_with(|| a[0].partial_cmp(&b[0]).unwrap_or(std::cmp::Ordering::Equal))
    });
    let mut bottom = pairs[..4].to_vec();
    let mut top = pairs[4..].to_vec();
    order_planar_pairs(&mut bottom);
    order_planar_pairs(&mut top);
    pairs.clear();
    pairs.extend(bottom);
    pairs.extend(top);
    *vertices = pairs.iter().map(|(vertex, _)| *vertex).collect();
    *coordinates = pairs.iter().map(|(_, coord)| *coord).collect();
}

fn order_planar_pairs(pairs: &mut Vec<(PointId, [f64; 3])>) {
    let centroid = pairs.iter().fold([0.0; 3], |mut acc, (_, coord)| {
        acc[0] += coord[0];
        acc[1] += coord[1];
        acc[2] += coord[2];
        acc
    });
    let inv_n = 1.0 / pairs.len() as f64;
    let center = [
        centroid[0] * inv_n,
        centroid[1] * inv_n,
        centroid[2] * inv_n,
    ];
    pairs.sort_by(|(_, a), (_, b)| {
        (a[1] - center[1])
            .atan2(a[0] - center[0])
            .partial_cmp(&(b[1] - center[1]).atan2(b[0] - center[0]))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    rotate_smallest_coordinate_first(pairs);
}

fn rotate_smallest_coordinate_first(pairs: &mut Vec<(PointId, [f64; 3])>) {
    if pairs.is_empty() {
        return;
    }
    let start = pairs
        .iter()
        .enumerate()
        .min_by(|(_, (_, a)), (_, (_, b))| compare_coords(*a, *b))
        .map(|(idx, _)| idx)
        .unwrap_or(0);
    pairs.rotate_left(start);
}

fn compare_coords(a: [f64; 3], b: [f64; 3]) -> std::cmp::Ordering {
    for d in 0..3 {
        let ord = a[d].partial_cmp(&b[d]).unwrap_or(std::cmp::Ordering::Equal);
        if ord != std::cmp::Ordering::Equal {
            return ord;
        }
    }
    std::cmp::Ordering::Equal
}

fn gather_vertices<St>(
    coordinates: &Coordinates<f64, St>,
    vertices: &[PointId],
) -> Result<Vec<[f64; 3]>, MeshSieveError>
where
    St: Storage<f64>,
{
    vertices
        .iter()
        .map(|vertex| pad_point(coordinates.try_restrict(*vertex)?))
        .collect()
}

fn pad_point(point: &[f64]) -> Result<[f64; 3], MeshSieveError> {
    if point.is_empty() || point.len() > 3 {
        return Err(MeshSieveError::InvalidGeometry(format!(
            "point dimension must be 1..=3, got {}",
            point.len()
        )));
    }
    let mut out = [0.0; 3];
    out[..point.len()].copy_from_slice(point);
    Ok(out)
}

fn expected_vertex_count(cell_type: CellType) -> Option<usize> {
    match cell_type {
        CellType::Vertex => Some(1),
        CellType::Segment | CellType::Simplex(1) => Some(2),
        CellType::Triangle | CellType::Simplex(2) => Some(3),
        CellType::Quadrilateral => Some(4),
        CellType::Tetrahedron | CellType::Simplex(3) => Some(4),
        CellType::Hexahedron => Some(8),
        CellType::Prism => Some(6),
        CellType::Pyramid => Some(5),
        _ => None,
    }
}

fn reference_coordinates_in_cell(cell_type: CellType, reference: &[f64], tol: f64) -> bool {
    match cell_type {
        CellType::Vertex => reference.is_empty(),
        CellType::Segment | CellType::Simplex(1) => {
            reference.len() == 1 && reference[0] >= -tol && reference[0] <= 1.0 + tol
        }
        CellType::Triangle | CellType::Simplex(2) => {
            reference.len() == 2
                && reference[0] >= -tol
                && reference[1] >= -tol
                && reference[0] + reference[1] <= 1.0 + tol
        }
        CellType::Quadrilateral => {
            reference.len() == 2 && reference.iter().all(|r| *r >= -tol && *r <= 1.0 + tol)
        }
        CellType::Tetrahedron | CellType::Simplex(3) => {
            reference.len() == 3
                && reference.iter().all(|r| *r >= -tol)
                && reference.iter().sum::<f64>() <= 1.0 + tol
        }
        CellType::Hexahedron | CellType::Prism | CellType::Pyramid => {
            if reference.len() != 3 || !reference.iter().all(|r| *r >= -tol && *r <= 1.0 + tol) {
                return false;
            }
            if matches!(cell_type, CellType::Prism) {
                reference[0] + reference[1] <= 1.0 + tol
            } else {
                true
            }
        }
        _ => false,
    }
}

fn reference_weights(cell_type: CellType, reference: &[f64]) -> Result<Vec<f64>, MeshSieveError> {
    match cell_type {
        CellType::Vertex => Ok(vec![1.0]),
        CellType::Segment | CellType::Simplex(1) => Ok(vec![1.0 - reference[0], reference[0]]),
        CellType::Triangle | CellType::Simplex(2) => Ok(vec![
            1.0 - reference[0] - reference[1],
            reference[0],
            reference[1],
        ]),
        CellType::Quadrilateral => {
            let r = reference[0];
            let s = reference[1];
            Ok(vec![
                (1.0 - r) * (1.0 - s),
                r * (1.0 - s),
                r * s,
                (1.0 - r) * s,
            ])
        }
        CellType::Tetrahedron | CellType::Simplex(3) => Ok(vec![
            1.0 - reference[0] - reference[1] - reference[2],
            reference[0],
            reference[1],
            reference[2],
        ]),
        CellType::Hexahedron => {
            let r = reference[0];
            let s = reference[1];
            let t = reference[2];
            let (rm, sm, tm) = (1.0 - r, 1.0 - s, 1.0 - t);
            Ok(vec![
                rm * sm * tm,
                r * sm * tm,
                r * s * tm,
                rm * s * tm,
                rm * sm * t,
                r * sm * t,
                r * s * t,
                rm * s * t,
            ])
        }
        CellType::Prism => {
            let r = reference[0];
            let s = reference[1];
            let t = reference[2];
            let rm = 1.0 - r - s;
            let tm = 1.0 - t;
            Ok(vec![rm * tm, r * tm, s * tm, rm * t, r * t, s * t])
        }
        CellType::Pyramid => {
            let r = reference[0];
            let s = reference[1];
            let t = reference[2];
            let tm = 1.0 - t;
            Ok(vec![
                tm * (1.0 - r) * (1.0 - s),
                tm * r * (1.0 - s),
                tm * r * s,
                tm * (1.0 - r) * s,
                t,
            ])
        }
        _ => Err(MeshSieveError::InvalidGeometry(format!(
            "unsupported cell type: {cell_type:?}"
        ))),
    }
}

fn interpolate_on_cell<St>(
    cell: &LocatedCell,
    section: &Section<f64, St>,
    reference: &[f64],
) -> Result<Vec<f64>, MeshSieveError>
where
    St: Storage<f64>,
{
    let weights = reference_weights(cell.cell_type, reference)?;
    if weights.len() != cell.vertices.len() {
        return Err(MeshSieveError::InvalidGeometry(format!(
            "basis/vertex mismatch for cell {:?}: {} weights, {} vertices",
            cell.cell,
            weights.len(),
            cell.vertices.len()
        )));
    }
    let first = section.try_restrict(cell.vertices[0])?;
    let dof = first.len();
    let mut out = vec![0.0; dof];
    for (weight, vertex) in weights.iter().zip(cell.vertices.iter()) {
        let values = section.try_restrict(*vertex)?;
        if values.len() != dof {
            return Err(MeshSieveError::InvalidGeometry(format!(
                "section dof mismatch at {vertex:?}: expected {dof}, got {}",
                values.len()
            )));
        }
        for (dst, value) in out.iter_mut().zip(values.iter()) {
            *dst += weight * value;
        }
    }
    Ok(out)
}

/// Map reference coordinates to physical coordinates using a localized cell view.
pub fn localized_reference_to_physical(
    cell_type: CellType,
    localized: &LocalizedCellCoordinates,
    reference: &[f64],
) -> Result<[f64; 3], MeshSieveError> {
    reference_to_physical(cell_type, &localized.coordinates, reference)
}
