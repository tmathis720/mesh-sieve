// Graph trait abstraction for partitioning
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use std::hash::Hash;

/// Trait for graphs that can be partitioned in parallel.
//
/// All methods are read-only, thread-safe, and require no interior mutability.
/// Implementors must guarantee that all returned iterators are safe for concurrent use and do not mutate the graph.
///
/// This trait is only available when the `partitioning` feature is enabled.
#[cfg_attr(docsrs, doc(cfg(feature = "partitioning")))]
pub trait PartitionableGraph: Sync {
    /// Vertex identifier type (must be copyable, hashable, and thread-safe).
    type VertexId: Copy + Hash + Eq + Send + Sync;
    /// Parallel iterator over all vertices.
    type VertexParIter<'a>: IndexedParallelIterator<Item = Self::VertexId> + 'a
    where Self: 'a;
    /// Parallel iterator over neighbors.
    type NeighParIter<'a>: ParallelIterator<Item = Self::VertexId> + 'a
    where Self: 'a;

    /// Returns a parallel, indexable iterator over all vertices.
    fn vertices(&self) -> Self::VertexParIter<'_>;

    /// Returns a parallel iterator over neighbours of `v`.
    fn neighbors(&self, v: Self::VertexId) -> Self::NeighParIter<'_>;

    /// Degree of a vertex (number of neighbors).
    fn degree(&self, v: Self::VertexId) -> usize;

    /// Returns a parallel iterator over all undirected edges (u, v) with u < v.
    fn edges(&self) -> impl ParallelIterator<Item = (Self::VertexId, Self::VertexId)> + '_
    where
        Self::VertexId: PartialOrd,
    {
        self.vertices().flat_map_iter(move |u| {
            self.neighbors(u)
                .filter(move |&v| u < v)
                .map(move |v| (u, v))
                .collect::<Vec<_>>()
                .into_iter()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rayon::iter::{IntoParallelRefIterator, IntoParallelIterator};
    use std::collections::HashMap;

    /// Simple in-memory undirected graph for testing.
    struct TestGraph {
        adj: HashMap<usize, Vec<usize>>,
    }

    impl PartitionableGraph for TestGraph {
        type VertexId = usize;
        type VertexParIter<'a> = rayon::vec::IntoIter<usize>;
        type NeighParIter<'a> = rayon::vec::IntoIter<usize>;
        fn vertices(&self) -> Self::VertexParIter<'_> {
            let vs: Vec<_> = self.adj.keys().copied().collect();
            vs.into_par_iter()
        }
        fn neighbors(&self, v: Self::VertexId) -> Self::NeighParIter<'_> {
            let ns = self.adj.get(&v).cloned().unwrap_or_default();
            ns.into_par_iter()
        }
        fn degree(&self, v: Self::VertexId) -> usize {
            self.adj.get(&v).map_or(0, |n| n.len())
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
