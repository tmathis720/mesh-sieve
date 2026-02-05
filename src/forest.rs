//! Quad/oct-tree AMR forest representation with Sieve mappings.

use crate::topology::point::PointId;
use crate::topology::sieve::{InMemorySieve, Sieve};
use std::collections::{HashMap, HashSet};

/// A cell in a quadtree/octree forest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TreeCell<const D: usize> {
    /// Refinement level (0 is root).
    pub level: u8,
    /// Integer coordinates at the given level.
    pub coords: [u32; D],
}

impl<const D: usize> TreeCell<D> {
    /// Returns the parent cell, or `None` for the root.
    pub fn parent(&self) -> Option<Self> {
        if self.level == 0 {
            None
        } else {
            let mut coords = self.coords;
            for coord in &mut coords {
                *coord /= 2;
            }
            Some(Self {
                level: self.level - 1,
                coords,
            })
        }
    }

    /// Returns the `2^D` children of this cell.
    pub fn children(&self) -> Vec<Self> {
        let count = 1usize << D;
        let mut children = Vec::with_capacity(count);
        for idx in 0..count {
            let mut coords = [0u32; D];
            for axis in 0..D {
                let bit = (idx >> axis) & 1;
                coords[axis] = self.coords[axis] * 2 + bit as u32;
            }
            children.push(Self {
                level: self.level + 1,
                coords,
            });
        }
        children
    }
}

/// A vertex on the conforming mesh view (coordinates on the max-level grid).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ForestVertex<const D: usize> {
    pub coords: [u32; D],
}

/// A conforming mesh view extracted from the forest.
#[derive(Debug, Clone)]
pub struct ForestMeshView<const D: usize> {
    /// Sieve topology with cell → vertex arrows.
    pub sieve: InMemorySieve<PointId, ()>,
    /// Mapping from forest leaf cells to sieve cell points.
    pub cell_points: HashMap<TreeCell<D>, PointId>,
    /// Mapping from vertex coordinates to sieve vertex points.
    pub vertex_points: HashMap<ForestVertex<D>, PointId>,
    /// Maximum refinement level represented by the view.
    pub max_level: u8,
}

/// Forest representation for quadtrees (`D = 2`) or octrees (`D = 3`).
#[derive(Debug, Clone)]
pub struct Forest<const D: usize> {
    leaves: HashSet<TreeCell<D>>,
}

/// A quadtree forest (`D = 2`).
pub type QuadForest = Forest<2>;
/// An octree forest (`D = 3`).
pub type OctForest = Forest<3>;

impl<const D: usize> Forest<D> {
    /// Create a new forest with a single root cell.
    pub fn new() -> Self {
        let mut leaves = HashSet::new();
        leaves.insert(TreeCell {
            level: 0,
            coords: [0; D],
        });
        Self { leaves }
    }

    /// Return an iterator over leaf cells.
    pub fn leaves(&self) -> impl Iterator<Item = &TreeCell<D>> {
        self.leaves.iter()
    }

    /// Return the number of leaf cells.
    pub fn leaf_count(&self) -> usize {
        self.leaves.len()
    }

    /// Refine all leaf cells whose indicator exceeds the threshold.
    pub fn refine_by_indicator<F>(&mut self, indicator: F, threshold: f64) -> usize
    where
        F: Fn(&TreeCell<D>) -> f64,
    {
        let to_refine: Vec<_> = self
            .leaves
            .iter()
            .copied()
            .filter(|cell| indicator(cell) > threshold)
            .collect();
        self.refine_cells(&to_refine)
    }

    /// Coarsen all leaf siblings whose indicators are below the threshold.
    pub fn coarsen_by_indicator<F>(&mut self, indicator: F, threshold: f64) -> usize
    where
        F: Fn(&TreeCell<D>) -> f64,
    {
        let mut parent_to_children: HashMap<TreeCell<D>, Vec<TreeCell<D>>> = HashMap::new();
        for leaf in &self.leaves {
            if let Some(parent) = leaf.parent() {
                parent_to_children.entry(parent).or_default().push(*leaf);
            }
        }

        let mut to_coarsen = Vec::new();
        let sibling_count = 1 << D;
        for (parent, children) in parent_to_children {
            if children.len() == sibling_count
                && children.iter().all(|child| indicator(child) < threshold)
            {
                to_coarsen.push((parent, children));
            }
        }

        let mut coarsened = 0;
        for (parent, children) in to_coarsen {
            let mut removed = 0;
            for child in children {
                if self.leaves.remove(&child) {
                    removed += 1;
                }
            }
            if removed == sibling_count {
                self.leaves.insert(parent);
                coarsened += 1;
            }
        }
        coarsened
    }

    /// Build a conforming mesh view by refining until neighboring cells match in level.
    pub fn conforming_view(&self) -> ForestMeshView<D> {
        let mut balanced = self.clone();
        balanced.balance();
        balanced.build_view()
    }

    fn refine_cells(&mut self, cells: &[TreeCell<D>]) -> usize {
        let mut refined = 0;
        for cell in cells {
            if self.leaves.remove(cell) {
                for child in cell.children() {
                    self.leaves.insert(child);
                }
                refined += 1;
            }
        }
        refined
    }

    fn max_level(&self) -> u8 {
        self.leaves.iter().map(|cell| cell.level).max().unwrap_or(0)
    }

    fn balance(&mut self) {
        loop {
            let leaves: Vec<_> = self.leaves.iter().copied().collect();
            let max_level = leaves.iter().map(|cell| cell.level).max().unwrap_or(0);
            let mut to_refine = HashSet::new();
            for (i, cell) in leaves.iter().enumerate() {
                for other in leaves.iter().skip(i + 1) {
                    if are_face_neighbors(cell, other, max_level) {
                        if cell.level < other.level {
                            to_refine.insert(*cell);
                        } else if other.level < cell.level {
                            to_refine.insert(*other);
                        }
                    }
                }
            }

            if to_refine.is_empty() {
                break;
            }

            let cells: Vec<_> = to_refine.into_iter().collect();
            self.refine_cells(&cells);
        }
    }

    fn build_view(&self) -> ForestMeshView<D> {
        let max_level = self.max_level();
        let leaves: Vec<_> = self.leaves.iter().copied().collect();
        let mut sieve = InMemorySieve::<PointId, ()>::default();
        let mut cell_points = HashMap::new();
        let mut vertex_points = HashMap::new();
        let mut next_id = 1u64;

        for cell in &leaves {
            let point = PointId::new(next_id).expect("cell point id");
            next_id += 1;
            cell_points.insert(*cell, point);
        }

        for cell in &leaves {
            let cell_point = cell_points[cell];
            for vertex in cell_vertices(cell, max_level) {
                let vertex_point = vertex_points.entry(vertex).or_insert_with(|| {
                    let point = PointId::new(next_id).expect("vertex point id");
                    next_id += 1;
                    point
                });
                sieve.add_arrow(cell_point, *vertex_point, ());
            }
        }

        ForestMeshView {
            sieve,
            cell_points,
            vertex_points,
            max_level,
        }
    }
}

fn cell_bounds<const D: usize>(cell: &TreeCell<D>, max_level: u8) -> [(u32, u32); D] {
    let scale = 1u32 << (max_level - cell.level);
    let mut bounds = [(0u32, 0u32); D];
    for axis in 0..D {
        let start = cell.coords[axis] * scale;
        bounds[axis] = (start, start + scale);
    }
    bounds
}

fn are_face_neighbors<const D: usize>(a: &TreeCell<D>, b: &TreeCell<D>, max_level: u8) -> bool {
    let a_bounds = cell_bounds(a, max_level);
    let b_bounds = cell_bounds(b, max_level);
    let mut touching_axis = None;
    for axis in 0..D {
        let (a0, a1) = a_bounds[axis];
        let (b0, b1) = b_bounds[axis];
        if a1 == b0 || b1 == a0 {
            if touching_axis.is_some() {
                return false;
            }
            touching_axis = Some(axis);
        } else if a0 >= b1 || b0 >= a1 {
            return false;
        }
    }
    if let Some(axis) = touching_axis {
        for other_axis in 0..D {
            if other_axis == axis {
                continue;
            }
            let (a0, a1) = a_bounds[other_axis];
            let (b0, b1) = b_bounds[other_axis];
            if a0 >= b1 || b0 >= a1 {
                return false;
            }
        }
        true
    } else {
        false
    }
}

fn cell_vertices<const D: usize>(cell: &TreeCell<D>, max_level: u8) -> Vec<ForestVertex<D>> {
    let bounds = cell_bounds(cell, max_level);
    let mut vertices = Vec::with_capacity(1 << D);
    for idx in 0..(1 << D) {
        let mut coords = [0u32; D];
        for axis in 0..D {
            let bit = (idx >> axis) & 1;
            coords[axis] = if bit == 0 {
                bounds[axis].0
            } else {
                bounds[axis].1
            };
        }
        vertices.push(ForestVertex { coords });
    }
    vertices
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forest_refine_and_coarsen_by_indicator() {
        let mut forest = QuadForest::new();
        assert_eq!(forest.leaf_count(), 1);

        let refined = forest.refine_by_indicator(
            |cell| {
                if cell.level == 0 { 1.0 } else { 0.0 }
            },
            0.5,
        );
        assert_eq!(refined, 1);
        assert_eq!(forest.leaf_count(), 4);

        forest.refine_by_indicator(
            |cell| {
                if cell.level == 1 && cell.coords == [0, 0] {
                    1.0
                } else {
                    0.0
                }
            },
            0.5,
        );
        assert_eq!(forest.leaf_count(), 7);

        let coarsened = forest.coarsen_by_indicator(|_| 0.0, 0.1);
        assert!(coarsened > 0);
        assert_eq!(forest.leaf_count(), 4);

        let coarsened_again = forest.coarsen_by_indicator(|_| 0.0, 0.1);
        assert!(coarsened_again > 0);
        assert_eq!(forest.leaf_count(), 1);
    }

    #[test]
    fn forest_conforming_view_has_consistent_topology() {
        let mut forest = QuadForest::new();
        forest.refine_by_indicator(|cell| if cell.level == 0 { 1.0 } else { 0.0 }, 0.0);
        forest.refine_by_indicator(
            |cell| {
                if cell.level == 1 && cell.coords == [0, 0] {
                    1.0
                } else {
                    0.0
                }
            },
            0.5,
        );

        let view = forest.conforming_view();
        assert_eq!(view.max_level, 2);
        assert_eq!(view.cell_points.len(), 16);

        for cell_point in view.cell_points.values() {
            let cone: Vec<_> = view.sieve.cone_points(*cell_point).collect();
            assert_eq!(cone.len(), 4);
        }
    }
}
