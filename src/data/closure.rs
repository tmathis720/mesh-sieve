//! Orientation-aware closure DOF extraction in the style of PETSc DMPlex.
//!
//! A [`ClosureIndexCache`] stores reusable closure indices keyed by
//! `(cell, section_version, topology_version)`.  The indices flatten all DOFs
//! in a cell closure once, so repeated element assembly can call
//! [`get_closure`], [`set_closure`], or [`add_closure`] without re-traversing the
//! topology DAG.

use crate::data::section::Section;
use crate::data::storage::Storage;
use crate::mesh_error::MeshSieveError;
use crate::topology::point::PointId;
use crate::topology::sieve::{Orientation, OrientedSieve, Sieve};
use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::Hash;
use std::ops::AddAssign;

/// Monotonic version supplied by the caller for topology changes.
pub type TopologyVersion = u64;

/// Stable cache key for closure indices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ClosureIndexKey {
    /// Seed cell for the transitive closure.
    pub cell: PointId,
    /// [`Atlas`](crate::data::atlas::Atlas) version backing the section.
    pub section_version: u64,
    /// Caller-maintained topology version.
    pub topology_version: TopologyVersion,
}

/// Ordering policy for points in a cell closure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ClosureOrder {
    /// Breadth-first DMPlex-style order: cell, cone, cone-of-cone, ... .
    BreadthFirstDmpLex,
    /// Deterministic lexicographic tensor order.  `dims` describes the tensor
    /// grid shape when known; this implementation uses stable point ids as the
    /// lexicographic coordinate surrogate for topological closures.
    LexicographicTensor {
        /// Tensor grid dimensions used by callers to describe the lexicographic shape.
        dims: Vec<usize>,
    },
    /// User-provided point order.  Points not listed are appended in
    /// breadth-first order.
    Custom(Vec<PointId>),
}

impl Default for ClosureOrder {
    fn default() -> Self {
        Self::BreadthFirstDmpLex
    }
}

/// One point-sized span inside a flattened closure vector.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClosurePointIndex<O> {
    /// Topology point contributing these DOFs.
    pub point: PointId,
    /// Accumulated orientation from the closure seed to `point`.
    pub orientation: O,
    /// Offset in the flattened closure vector.
    pub offset: usize,
    /// Number of DOFs contributed by this point.
    pub len: usize,
    /// Maps closure-local DOF slots to section-local indices for this point.
    pub permutation: Vec<usize>,
}

/// Reusable flattened index for a closure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClosureIndex<O> {
    /// Cache key used to build this index.
    pub key: ClosureIndexKey,
    /// Point spans in flattened closure order.
    pub points: Vec<ClosurePointIndex<O>>,
    /// Total number of flattened DOFs.
    pub len: usize,
}

impl<O> ClosureIndex<O> {
    /// Closure points in flattened order.
    pub fn point_order(&self) -> impl Iterator<Item = PointId> + '_ {
        self.points.iter().map(|entry| entry.point)
    }
}

/// PetscSectionSym-like per-point orientation symmetries.
///
/// The returned permutation maps each closure-local DOF slot to the source
/// index within that point's section slice. Returning `None` means identity.
pub trait SectionSym<O> {
    /// Return the orientation-dependent DOF permutation for `point`.
    fn permutation(&self, point: PointId, orientation: O, dof_count: usize) -> Option<Vec<usize>>;
}

/// Identity section symmetry.
#[derive(Clone, Copy, Debug, Default)]
pub struct IdentitySectionSym;

impl<O: Copy> SectionSym<O> for IdentitySectionSym {
    #[inline]
    fn permutation(
        &self,
        _point: PointId,
        _orientation: O,
        _dof_count: usize,
    ) -> Option<Vec<usize>> {
        None
    }
}

/// Table-backed section symmetry keyed by `(point, orientation)`.
#[derive(Clone, Debug, Default)]
pub struct TableSectionSym<O> {
    by_point_orientation: HashMap<(PointId, O), Vec<usize>>,
    by_orientation: HashMap<O, Vec<usize>>,
}

impl<O> TableSectionSym<O>
where
    O: Copy + Eq + Hash,
{
    /// Create an empty table.
    pub fn new() -> Self {
        Self {
            by_point_orientation: HashMap::new(),
            by_orientation: HashMap::new(),
        }
    }

    /// Set a permutation used for a specific point and orientation.
    pub fn insert_point(&mut self, point: PointId, orientation: O, permutation: Vec<usize>) {
        self.by_point_orientation
            .insert((point, orientation), permutation);
    }

    /// Set a default permutation for an orientation, independent of point.
    pub fn insert_orientation(&mut self, orientation: O, permutation: Vec<usize>) {
        self.by_orientation.insert(orientation, permutation);
    }
}

impl<O> SectionSym<O> for TableSectionSym<O>
where
    O: Copy + Eq + Hash,
{
    fn permutation(&self, point: PointId, orientation: O, _dof_count: usize) -> Option<Vec<usize>> {
        self.by_point_orientation
            .get(&(point, orientation))
            .or_else(|| self.by_orientation.get(&orientation))
            .map(|perm| perm.clone())
    }
}

/// User-provided symmetry closure.
pub struct FnSectionSym<F>(pub F);

impl<O, F> SectionSym<O> for FnSectionSym<F>
where
    F: Fn(PointId, O, usize) -> Option<Vec<usize>>,
{
    fn permutation(&self, point: PointId, orientation: O, dof_count: usize) -> Option<Vec<usize>> {
        (self.0)(point, orientation, dof_count)
    }
}

/// Cache for repeated closure-index construction.
#[derive(Clone, Debug, Default)]
pub struct ClosureIndexCache<O> {
    entries: HashMap<ClosureIndexKey, ClosureIndex<O>>,
}

impl<O> ClosureIndexCache<O>
where
    O: Orientation + Eq + Hash,
{
    /// Create an empty cache.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Drop all cached indices.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Return a cached index or build and insert it.
    pub fn get_or_build<T, V, Sct, Sym>(
        &mut self,
        topology: &T,
        section: &Section<V, Sct>,
        cell: PointId,
        topology_version: TopologyVersion,
        order: &ClosureOrder,
        sym: &Sym,
    ) -> Result<&ClosureIndex<O>, MeshSieveError>
    where
        T: OrientedSieve<Point = PointId, Orient = O>,
        Sct: Storage<V>,
        Sym: SectionSym<O>,
    {
        let key = ClosureIndexKey {
            cell,
            section_version: section.atlas().version(),
            topology_version,
        };
        if !self.entries.contains_key(&key) {
            let index = build_closure_index(topology, section, cell, topology_version, order, sym)?;
            self.entries.insert(key, index);
        }
        Ok(self.entries.get(&key).expect("inserted or present"))
    }
}

impl ClosureIndexCache<()> {
    /// Return a cached non-oriented index or build and insert it.
    pub fn get_or_build_unoriented<T, V, Sct, Sym>(
        &mut self,
        topology: &T,
        section: &Section<V, Sct>,
        cell: PointId,
        topology_version: TopologyVersion,
        order: &ClosureOrder,
        sym: &Sym,
    ) -> Result<&ClosureIndex<()>, MeshSieveError>
    where
        T: Sieve<Point = PointId>,
        Sct: Storage<V>,
        Sym: SectionSym<()>,
    {
        let key = ClosureIndexKey {
            cell,
            section_version: section.atlas().version(),
            topology_version,
        };
        if !self.entries.contains_key(&key) {
            let index = build_closure_index_unoriented(
                topology,
                section,
                cell,
                topology_version,
                order,
                sym,
            )?;
            self.entries.insert(key, index);
        }
        Ok(self.entries.get(&key).expect("inserted or present"))
    }
}

/// Build an orientation-aware closure index without caching.
pub fn build_closure_index<T, V, Sct, O, Sym>(
    topology: &T,
    section: &Section<V, Sct>,
    cell: PointId,
    topology_version: TopologyVersion,
    order: &ClosureOrder,
    sym: &Sym,
) -> Result<ClosureIndex<O>, MeshSieveError>
where
    T: OrientedSieve<Point = PointId, Orient = O>,
    Sct: Storage<V>,
    O: Orientation + Eq + Hash,
    Sym: SectionSym<O>,
{
    let key = ClosureIndexKey {
        cell,
        section_version: section.atlas().version(),
        topology_version,
    };
    let oriented_points = ordered_oriented_closure(topology, cell, order);
    let mut offset = 0usize;
    let mut points = Vec::with_capacity(oriented_points.len());
    for (point, orientation) in oriented_points {
        let len = match section.atlas().get(point) {
            Some((_, len)) => len,
            None => continue,
        };
        let permutation =
            normalized_permutation(point, len, sym.permutation(point, orientation, len))?;
        points.push(ClosurePointIndex {
            point,
            orientation,
            offset,
            len,
            permutation,
        });
        offset += len;
    }
    Ok(ClosureIndex {
        key,
        points,
        len: offset,
    })
}

/// Build a non-oriented closure index for a plain [`Sieve`].
pub fn build_closure_index_unoriented<T, V, Sct, Sym>(
    topology: &T,
    section: &Section<V, Sct>,
    cell: PointId,
    topology_version: TopologyVersion,
    order: &ClosureOrder,
    sym: &Sym,
) -> Result<ClosureIndex<()>, MeshSieveError>
where
    T: Sieve<Point = PointId>,
    Sct: Storage<V>,
    Sym: SectionSym<()>,
{
    let key = ClosureIndexKey {
        cell,
        section_version: section.atlas().version(),
        topology_version,
    };
    let points_only = ordered_closure(topology, cell, order);
    let mut offset = 0usize;
    let mut points = Vec::with_capacity(points_only.len());
    for point in points_only {
        let len = match section.atlas().get(point) {
            Some((_, len)) => len,
            None => continue,
        };
        let permutation = normalized_permutation(point, len, sym.permutation(point, (), len))?;
        points.push(ClosurePointIndex {
            point,
            orientation: (),
            offset,
            len,
            permutation,
        });
        offset += len;
    }
    Ok(ClosureIndex {
        key,
        points,
        len: offset,
    })
}

/// Extract closure DOFs using a precomputed index.
pub fn get_closure<V, Sct, O>(
    section: &Section<V, Sct>,
    index: &ClosureIndex<O>,
) -> Result<Vec<V>, MeshSieveError>
where
    V: Clone,
    Sct: Storage<V>,
{
    validate_section_version(section, index)?;
    let mut out = Vec::with_capacity(index.len);
    for entry in &index.points {
        let slice = section.try_restrict(entry.point)?;
        for &src in &entry.permutation {
            out.push(slice[src].clone());
        }
    }
    Ok(out)
}

/// Set closure DOFs using a precomputed index.
pub fn set_closure<V, Sct, O>(
    section: &mut Section<V, Sct>,
    index: &ClosureIndex<O>,
    values: &[V],
) -> Result<(), MeshSieveError>
where
    V: Clone,
    Sct: Storage<V>,
{
    validate_section_version(section, index)?;
    validate_closure_len(index, values.len())?;
    for entry in &index.points {
        let slice = section.try_restrict_mut(entry.point)?;
        for (local_slot, &section_slot) in entry.permutation.iter().enumerate() {
            slice[section_slot] = values[entry.offset + local_slot].clone();
        }
    }
    Ok(())
}

/// Add closure DOFs into a section using a precomputed index.
pub fn add_closure<V, Sct, O>(
    section: &mut Section<V, Sct>,
    index: &ClosureIndex<O>,
    values: &[V],
) -> Result<(), MeshSieveError>
where
    V: Clone + AddAssign<V>,
    Sct: Storage<V>,
{
    validate_section_version(section, index)?;
    validate_closure_len(index, values.len())?;
    for entry in &index.points {
        let slice = section.try_restrict_mut(entry.point)?;
        for (local_slot, &section_slot) in entry.permutation.iter().enumerate() {
            slice[section_slot] += values[entry.offset + local_slot].clone();
        }
    }
    Ok(())
}

/// Build an index and extract closure DOFs in one call.
pub fn get_closure_oriented<T, V, Sct, O, Sym>(
    topology: &T,
    section: &Section<V, Sct>,
    cell: PointId,
    topology_version: TopologyVersion,
    order: &ClosureOrder,
    sym: &Sym,
) -> Result<Vec<V>, MeshSieveError>
where
    T: OrientedSieve<Point = PointId, Orient = O>,
    V: Clone,
    Sct: Storage<V>,
    O: Orientation + Eq + Hash,
    Sym: SectionSym<O>,
{
    let index = build_closure_index(topology, section, cell, topology_version, order, sym)?;
    get_closure(section, &index)
}

fn validate_section_version<V, Sct, O>(
    section: &Section<V, Sct>,
    index: &ClosureIndex<O>,
) -> Result<(), MeshSieveError>
where
    Sct: Storage<V>,
{
    let current = section.atlas().version();
    if current != index.key.section_version {
        return Err(MeshSieveError::AtlasPlanStale {
            expected: index.key.section_version,
            found: current,
        });
    }
    Ok(())
}

fn validate_closure_len<O>(index: &ClosureIndex<O>, found: usize) -> Result<(), MeshSieveError> {
    if index.len != found {
        return Err(MeshSieveError::ScatterLengthMismatch {
            expected: index.len,
            found,
        });
    }
    Ok(())
}

fn normalized_permutation(
    point: PointId,
    len: usize,
    permutation: Option<Vec<usize>>,
) -> Result<Vec<usize>, MeshSieveError> {
    let perm = permutation.unwrap_or_else(|| (0..len).collect());
    if perm.len() != len {
        return Err(MeshSieveError::SliceLengthMismatch {
            point,
            expected: len,
            found: perm.len(),
        });
    }
    let mut seen = vec![false; len];
    for &idx in &perm {
        if idx >= len || seen[idx] {
            return Err(MeshSieveError::InvalidPermutation(format!(
                "invalid closure permutation for {point:?}: {perm:?}"
            )));
        }
        seen[idx] = true;
    }
    Ok(perm)
}

fn ordered_closure<T>(topology: &T, cell: PointId, order: &ClosureOrder) -> Vec<PointId>
where
    T: Sieve<Point = PointId>,
{
    let bfs = bfs_closure(topology, cell);
    apply_order(bfs, order)
}

fn ordered_oriented_closure<T, O>(
    topology: &T,
    cell: PointId,
    order: &ClosureOrder,
) -> Vec<(PointId, O)>
where
    T: OrientedSieve<Point = PointId, Orient = O>,
    O: Orientation + Eq + Hash,
{
    let bfs = bfs_oriented_closure(topology, cell);
    let ordered_points = apply_order(bfs.iter().map(|(point, _)| *point).collect(), order);
    let orientations: HashMap<PointId, O> = bfs.into_iter().collect();
    ordered_points
        .into_iter()
        .filter_map(|point| orientations.get(&point).copied().map(|o| (point, o)))
        .collect()
}

fn apply_order(mut bfs: Vec<PointId>, order: &ClosureOrder) -> Vec<PointId> {
    match order {
        ClosureOrder::BreadthFirstDmpLex => bfs,
        ClosureOrder::LexicographicTensor { .. } => {
            if let Some((&cell, rest)) = bfs.split_first() {
                let mut out = vec![cell];
                let mut rest = rest.to_vec();
                rest.sort_unstable();
                out.extend(rest);
                out
            } else {
                bfs
            }
        }
        ClosureOrder::Custom(custom) => {
            let bfs_set: HashSet<_> = bfs.iter().copied().collect();
            let mut used = HashSet::new();
            let mut out = Vec::with_capacity(bfs.len());
            for &point in custom {
                if bfs_set.contains(&point) && used.insert(point) {
                    out.push(point);
                }
            }
            for point in bfs.drain(..) {
                if used.insert(point) {
                    out.push(point);
                }
            }
            out
        }
    }
}

fn bfs_closure<T>(topology: &T, cell: PointId) -> Vec<PointId>
where
    T: Sieve<Point = PointId>,
{
    let mut seen = HashSet::new();
    let mut queue = VecDeque::from([cell]);
    let mut out = Vec::new();
    while let Some(point) = queue.pop_front() {
        if !seen.insert(point) {
            continue;
        }
        out.push(point);
        for child in topology.cone_points(point) {
            if !seen.contains(&child) {
                queue.push_back(child);
            }
        }
    }
    out
}

fn bfs_oriented_closure<T, O>(topology: &T, cell: PointId) -> Vec<(PointId, O)>
where
    T: OrientedSieve<Point = PointId, Orient = O>,
    O: Orientation + Eq + Hash,
{
    let mut seen = HashSet::new();
    let mut queue = VecDeque::from([(cell, O::default())]);
    let mut out = Vec::new();
    while let Some((point, orientation)) = queue.pop_front() {
        if !seen.insert(point) {
            continue;
        }
        out.push((point, orientation));
        for (child, edge_orientation) in topology.cone_o(point) {
            if !seen.contains(&child) {
                queue.push_back((child, O::compose(orientation, edge_orientation)));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::atlas::Atlas;
    use crate::data::storage::VecStorage;
    use crate::topology::orientation::Sign;
    use crate::topology::sieve::{InMemoryOrientedSieve, InMemorySieve};

    fn p(id: u64) -> PointId {
        PointId::new(id).unwrap()
    }

    #[test]
    fn get_set_add_unoriented_closure() {
        let mut topo = InMemorySieve::<PointId, ()>::default();
        topo.add_arrow(p(1), p(2), ());
        topo.add_arrow(p(1), p(3), ());

        let mut atlas = Atlas::default();
        atlas.try_insert(p(1), 1).unwrap();
        atlas.try_insert(p(2), 2).unwrap();
        atlas.try_insert(p(3), 1).unwrap();
        let mut section = Section::<i32, VecStorage<i32>>::new(atlas);
        section.try_scatter_in_order(&[10, 20, 21, 30]).unwrap();

        let index = build_closure_index_unoriented(
            &topo,
            &section,
            p(1),
            0,
            &ClosureOrder::BreadthFirstDmpLex,
            &IdentitySectionSym,
        )
        .unwrap();
        assert_eq!(get_closure(&section, &index).unwrap(), vec![10, 20, 21, 30]);

        set_closure(&mut section, &index, &[1, 2, 3, 4]).unwrap();
        add_closure(&mut section, &index, &[10, 20, 30, 40]).unwrap();
        assert_eq!(get_closure(&section, &index).unwrap(), vec![11, 22, 33, 44]);

        let mut cache = ClosureIndexCache::new();
        let cached = cache
            .get_or_build_unoriented(
                &topo,
                &section,
                p(1),
                0,
                &ClosureOrder::BreadthFirstDmpLex,
                &IdentitySectionSym,
            )
            .unwrap();
        assert_eq!(cached.key.cell, p(1));
    }

    #[test]
    fn oriented_symmetry_permutates_point_dofs() {
        let mut topo = InMemoryOrientedSieve::<PointId, (), Sign>::default();
        topo.add_arrow_o(p(1), p(2), (), Sign(true));

        let mut atlas = Atlas::default();
        atlas.try_insert(p(2), 3).unwrap();
        let mut section = Section::<i32, VecStorage<i32>>::new(atlas);
        section.try_scatter_in_order(&[1, 2, 3]).unwrap();

        let mut sym = TableSectionSym::new();
        sym.insert_orientation(Sign(true), vec![2, 1, 0]);
        let index = build_closure_index(
            &topo,
            &section,
            p(1),
            7,
            &ClosureOrder::BreadthFirstDmpLex,
            &sym,
        )
        .unwrap();
        assert_eq!(index.key.topology_version, 7);
        assert_eq!(get_closure(&section, &index).unwrap(), vec![3, 2, 1]);
    }

    #[test]
    fn custom_order_appends_unlisted_points() {
        let mut topo = InMemorySieve::<PointId, ()>::default();
        topo.add_arrow(p(1), p(2), ());
        topo.add_arrow(p(1), p(3), ());
        let ordered = ordered_closure(&topo, p(1), &ClosureOrder::Custom(vec![p(3)]));
        assert_eq!(ordered, vec![p(3), p(1), p(2)]);
    }
}
