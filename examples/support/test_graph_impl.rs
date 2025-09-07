use mesh_sieve::partitioning::graph_traits::PartitionableGraph;
use rayon::prelude::*;

/// Simple adjacency-list graph for examples/tests.
#[derive(Clone)]
pub struct AdjListGraph {
    pub nbrs: Vec<Vec<usize>>,
}

impl AdjListGraph {
    pub fn from_undirected(n: usize, edges: &[(usize, usize)]) -> Self {
        let mut nbrs = vec![Vec::new(); n];
        for &(u, v) in edges {
            nbrs[u].push(v);
            nbrs[v].push(u);
        }
        for ns in &mut nbrs {
            ns.sort_unstable();
            ns.dedup();
        }
        Self { nbrs }
    }
}

impl PartitionableGraph for AdjListGraph {
    type VertexId = usize;
    type VertexParIter<'a> = rayon::vec::IntoIter<usize>;
    type NeighParIter<'a> = rayon::vec::IntoIter<usize>;
    type NeighIter<'a> = std::iter::Copied<std::slice::Iter<'a, usize>>;

    fn vertices(&self) -> Self::VertexParIter<'_> {
        (0..self.nbrs.len()).collect::<Vec<_>>().into_par_iter()
    }

    fn neighbors(&self, v: usize) -> Self::NeighParIter<'_> {
        self.nbrs[v].clone().into_par_iter()
    }

    fn neighbors_seq(&self, v: usize) -> Self::NeighIter<'_> {
        self.nbrs[v].iter().copied()
    }

    fn degree(&self, v: usize) -> usize {
        self.nbrs[v].len()
    }
}
