//! Parallel graph trait abstraction for partitioning.
//!
//! This module defines the [`PartitionableGraph`] trait, which provides a parallel, read-only,
//! thread-safe interface for graph structures used in partitioning algorithms. All methods must
//! be safe for concurrent use and must not mutate the graph.

use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use std::hash::Hash;

/// Trait for graphs that can be partitioned in parallel.
///
/// All methods are read-only, thread-safe, and require no interior mutability.
/// Implementors must guarantee that all returned iterators are safe for concurrent use and do not mutate the graph.
///
/// This trait is only available when the `partitioning` feature is enabled.
#[cfg(feature = "mpi-support")]
pub trait PartitionableGraph: Sync {
    /// Vertex identifier (must be totally ordered to define `u < v`).
    type VertexId: Copy + Send + Sync + Eq + Ord + Hash;
    /// Parallel iterator over all vertices (indexable for `.len()`).
    type VertexParIter<'a>: IndexedParallelIterator<Item = Self::VertexId> + 'a
    where
        Self: 'a;
    /// Parallel iterator over neighbors of a vertex.
    type NeighParIter<'a>: ParallelIterator<Item = Self::VertexId> + 'a
    where
        Self: 'a;
    /// **Serial** iterator over neighbors of a vertex (no allocation).
    type NeighIter<'a>: Iterator<Item = Self::VertexId> + 'a
    where
        Self: 'a;
    /// Parallel iterator over undirected edges `(u, v)` with `u < v`.
    /// Implementors may override for maximum performance.
    type EdgeParIter<'a>: ParallelIterator<Item = (Self::VertexId, Self::VertexId)> + 'a
    where
        Self: 'a;

    /// All vertices.
    fn vertices(&self) -> Self::VertexParIter<'_>;

    /// Parallel neighbors of `v` (for algorithms that need parallel expansion).
    fn neighbors(&self, v: Self::VertexId) -> Self::NeighParIter<'_>;

    /// Serial neighbors of `v` (used to build zero-alloc `edges()` by default).
    fn neighbors_seq(&self, v: Self::VertexId) -> Self::NeighIter<'_>;

    /// Degree of a vertex.
    fn degree(&self, v: Self::VertexId) -> usize;

    /// **Default**: parallel undirected edge stream built without per-vertex `Vec`s.
    ///
    /// Contract:
    /// - Yields each undirected edge exactly once with `u < v`.
    /// - Thread-safe and read-only.
    fn edges(&self) -> Self::EdgeParIter<'_>
    where
        Self::VertexId: 'static, Self: Sized,
    {
        self.vertices().flat_map_iter(move |u| {
            self.neighbors_seq(u)
                .filter(move |&v| u < v)
                .map(move |v| (u, v))
        })
    }

    /// Deterministic O(1) helper: count of vertices.
    fn vertex_count(&self) -> usize {
        self.vertices().len()
    }

    /// Deterministic O(E) helper: count of undirected edges.
    fn edge_count(&self) -> usize
    where
        Self::VertexId: 'static, Self: Sized,
    {
        self.edges().count()
    }
}

/// Debug-time verification that `edges()` upholds its contract.
#[cfg(any(debug_assertions, feature = "check-graph-edges"))]
pub fn assert_edges_well_formed<G: PartitionableGraph>(g: &G) where <G as partitioning::graph_traits::PartitionableGraph>::VertexId: std::fmt::Debug {
    use std::collections::HashSet;
    let mut seen = HashSet::new();
    g.edges().for_each(|(u, v)| {
        debug_assert!(u < v, "edges() must yield u < v");
        let ok = seen.insert((u, v));
        debug_assert!(ok, "duplicate edge ({u:?}, {v:?}) from edges()");
    });

    #[cfg(feature = "expensive-checks")]
    {
        use std::collections::HashMap;
        let mut deg = HashMap::<G::VertexId, usize>::new();
        g.edges().for_each(|(u, v)| {
            *deg.entry(u).or_default() += 1;
            *deg.entry(v).or_default() += 1;
        });
        g.vertices().for_each(|u| {
            let e = *deg.get(&u).unwrap_or(&0);
            let d = g.degree(u);
            debug_assert_eq!(e, d, "degree mismatch for {u:?}: edges() says {e}, degree() says {d}");
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rayon::iter::IntoParallelIterator;
    use std::collections::HashMap;

    /// Simple in-memory undirected graph for testing.
    struct TestGraph {
        adj: HashMap<usize, Vec<usize>>,
    }

    impl PartitionableGraph for TestGraph {
        type VertexId = usize;
        type VertexParIter<'a> = rayon::vec::IntoIter<usize>;
        type NeighParIter<'a> = rayon::vec::IntoIter<usize>;
        type NeighIter<'a> = std::iter::Copied<std::slice::Iter<'a, usize>>;
        type EdgeParIter<'a> = rayon::vec::IntoIter<(usize, usize)>;

        fn vertices(&self) -> Self::VertexParIter<'_> {
            let vs: Vec<_> = self.adj.keys().copied().collect();
            vs.into_par_iter()
        }
        fn neighbors(&self, v: Self::VertexId) -> Self::NeighParIter<'_> {
            let ns = self.adj.get(&v).cloned().unwrap_or_default();
            ns.into_par_iter()
        }
        fn neighbors_seq(&self, v: Self::VertexId) -> Self::NeighIter<'_> {
            self.adj
                .get(&v)
                .map(|ns| ns.iter().copied())
                .unwrap_or_else(|| [].iter().copied())
        }
        fn degree(&self, v: Self::VertexId) -> usize {
            self.adj.get(&v).map_or(0, |n| n.len())
        }
        fn edges(&self) -> Self::EdgeParIter<'_> {
            let mut es = Vec::new();
            for (&u, ns) in &self.adj {
                for &v in ns {
                    if u < v {
                        es.push((u, v));
                    }
                }
            }
            es.into_par_iter()
        }
    }

    #[test]
    fn test_vertices_and_neighbors_cycle() {
        // 5-node cycle: 0-1-2-3-4-0
        let mut adj = HashMap::new();
        adj.insert(0, vec![1, 4]);
        adj.insert(1, vec![0, 2]);
        adj.insert(2, vec![1, 3]);
        adj.insert(3, vec![2, 4]);
        adj.insert(4, vec![3, 0]);
        let g = TestGraph { adj };
        let mut vs: Vec<_> = g.vertices().collect();
        vs.sort();
        assert_eq!(vs, vec![0, 1, 2, 3, 4]);
        let ns: Vec<_> = g.neighbors(0).collect();
        assert!(ns.contains(&1) && ns.contains(&4));
    }

    #[test]
    fn test_edges_path() {
        // 4-node path: 0-1-2-3
        let mut adj = HashMap::new();
        adj.insert(0, vec![1]);
        adj.insert(1, vec![0, 2]);
        adj.insert(2, vec![1, 3]);
        adj.insert(3, vec![2]);
        let g = TestGraph { adj };
        let mut edges: Vec<_> = g.edges().collect();
        edges.sort();
        edges.dedup();
        assert_eq!(edges, vec![(0, 1), (1, 2), (2, 3)]);
    }
}
