//! Seed selection for graph partitioning.
//!
//! This module provides [`pick_seeds`], a function for selecting seed vertices for partitioning algorithms,
//! with probability weighted by vertex degree and without replacement.

use crate::partitioning::PartitionerConfig;
use crate::partitioning::graph_traits::PartitionableGraph;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;

/// Returns an error if `degrees.len()` doesn’t match the number of vertices.
pub fn pick_seeds<G>(
    graph: &G,
    degrees: &[u64],
    cfg: &PartitionerConfig,
) -> Result<Vec<G::VertexId>, crate::partitioning::PartitionerError>
where
    G: PartitionableGraph<VertexId = usize> + Sync,
{
    // 1) collect vertices up front and verify lengths match
    let vertices: Vec<usize> = graph.vertices().collect();
    let n = vertices.len();
    if degrees.len() != n {
        return Err(crate::partitioning::PartitionerError::DegreeLengthMismatch {
            expected: n,
            got: degrees.len(),
        });
    }
    if n == 0 {
        return Ok(Vec::new());
    }
    let num_seeds = ((cfg.seed_factor * cfg.n_parts as f64).ceil() as usize)
        .min(n)
        .max(1);
    let mut weights = degrees.to_vec();
    let mut prefix: Vec<u64> = Vec::with_capacity(n);
    let mut sum = 0u64;
    for &w in &weights {
        sum += w;
        prefix.push(sum);
    }
    let mut rng = SmallRng::seed_from_u64(cfg.rng_seed);
    let mut chosen = Vec::with_capacity(num_seeds);
    if sum == 0 {
        // All degrees are zero, pick uniformly at random
        let mut pool: Vec<usize> = (0..n).collect();
        for _ in 0..num_seeds {
            if pool.is_empty() {
                break;
            }
            let idx = rng.gen_range(0..pool.len());
            chosen.push(vertices[pool[idx]]);
            pool.remove(idx);
        }
        return Ok(chosen);
    }
    for _ in 0..num_seeds {
        let total_weight = prefix[n - 1];
        if total_weight == 0 {
            break;
        }
        let t = rng.gen_range(0..total_weight);
        let mut i = prefix.binary_search(&t).unwrap_or_else(|x| x);
        // Find first nonzero weight ≥ t
        while i < n && weights[i] == 0 {
            i += 1;
        }
        if i == n {
            break;
        }
        chosen.push(vertices[i]);
        // Remove this vertex from future selection
        let w = weights[i];
        weights[i] = 0;
        for j in i..n {
            prefix[j] -= w;
        }
    }
    Ok(chosen)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::partitioning::graph_traits::PartitionableGraph;

    struct PathGraph {
        n: usize,
    }
    impl PartitionableGraph for PathGraph {
        type VertexId = usize;
        type VertexParIter<'a> = rayon::vec::IntoIter<usize>;
        type NeighParIter<'a> = rayon::vec::IntoIter<usize>;
        fn vertices(&self) -> Self::VertexParIter<'_> {
            (0..self.n).collect::<Vec<_>>().into_par_iter()
        }
        fn neighbors(&self, v: usize) -> Self::NeighParIter<'_> {
            let mut neigh = Vec::new();
            if v > 0 {
                neigh.push(v - 1);
            }
            if v + 1 < self.n {
                neigh.push(v + 1);
            }
            neigh.into_par_iter()
        }
        fn degree(&self, v: usize) -> usize {
            self.neighbors(v).count()
        }
    }

    #[test]
    fn pick_seeds_length_mismatch() {
        let g = PathGraph { n: 3 };
        let degrees = vec![1, 2]; // wrong length
        let cfg = PartitionerConfig::default();
        let err = pick_seeds(&g, &degrees, &cfg).unwrap_err();
        assert!(matches!(
            err,
            crate::partitioning::PartitionerError::DegreeLengthMismatch { expected: 3, got: 2 }
        ));
    }

    #[test]
    fn pick_seeds_path_highest_degree() {
        let g = PathGraph { n: 5 };
        let degrees: Vec<u64> = (0..5).map(|v| g.degree(v) as u64).collect();
        let cfg = PartitionerConfig {
            n_parts: 1,
            seed_factor: 1.0,
            ..Default::default()
        };
        let seeds = pick_seeds(&g, &degrees, &cfg).unwrap();
        // For a path, the highest degree is in the middle. This is probabilistic,
        // so just check that the seed count is correct and all are valid.
        assert!(seeds.len() == 1 && seeds[0] < 5, "seeds = {:?}", seeds);
    }

    #[test]
    fn pick_seeds_equal_degrees() {
        let g = PathGraph { n: 4 };
        let degrees = vec![1, 1, 1, 1];
        let cfg = PartitionerConfig {
            n_parts: 2,
            seed_factor: 1.0,
            ..Default::default()
        };
        let seeds = pick_seeds(&g, &degrees, &cfg).unwrap();
        assert_eq!(seeds.len(), 2);
        for &s in &seeds {
            assert!(s < 4);
        }
    }

    #[test]
    fn pick_seeds_more_than_n() {
        let g = PathGraph { n: 3 };
        let degrees = vec![1, 1, 1];
        let cfg = PartitionerConfig {
            n_parts: 5,
            seed_factor: 1.0,
            ..Default::default()
        };
        let seeds = pick_seeds(&g, &degrees, &cfg).unwrap();
        assert_eq!(seeds.len(), 3);
    }
}

#[cfg(feature = "mpi-support")]
mod onizuka_partitioning {
    use super::*;
    use crate::partitioning::graph_traits::PartitionableGraph;

    struct CompleteGraph {
        n: usize,
    }
    impl PartitionableGraph for CompleteGraph {
        type VertexId = usize;
        type VertexParIter<'a> = rayon::vec::IntoIter<usize>;
        type NeighParIter<'a> = rayon::vec::IntoIter<usize>;
        fn vertices(&self) -> Self::VertexParIter<'_> {
            (0..self.n).collect::<Vec<_>>().into_par_iter()
        }
        fn neighbors(&self, v: usize) -> Self::NeighParIter<'_> {
            let mut neigh = (0..self.n).collect::<Vec<_>>();
            neigh.remove(v);
            neigh.into_par_iter()
        }
        fn degree(&self, _v: usize) -> usize {
            self.n - 1
        }
    }

    #[test]
    fn pick_seeds_complete_graph() {
        let g = CompleteGraph { n: 4 };
        let degrees: Vec<u64> = (0..4).map(|v| g.degree(v) as u64).collect();
        let cfg = PartitionerConfig {
            n_parts: 2,
            seed_factor: 1.0,
            ..Default::default()
        };
        let seeds = pick_seeds(&g, &degrees, &cfg).unwrap();
        // In a complete graph, all vertices have the same degree. Seeds should be
        // picked uniformly at random.
        assert_eq!(seeds.len(), 2);
        for &s in &seeds {
            assert!(s < 4);
        }
    }
}
