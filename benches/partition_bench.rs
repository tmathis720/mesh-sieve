#![cfg(feature = "mpi-support")]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::{SeedableRng, Rng};
use rand::rngs::SmallRng;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use sieve_rs::partitioning::{partition, PartitionerConfig};
use sieve_rs::partitioning::graph_traits::PartitionableGraph;

// 1) Synthetic Erdos-Renyi graph
struct RandomGraph {
    n: usize,
    edges: Vec<(usize,usize)>,
}
impl RandomGraph {
    fn with_params(n: usize, p: f64, seed: u64) -> Self {
        let mut edges = Vec::new();
        let mut rng = SmallRng::seed_from_u64(seed);
        for u in 0..n {
            for v in (u+1)..n {
                if rng.r#gen::<f64>() < p {
                    edges.push((u,v));
                }
            }
        }
        RandomGraph { n, edges }
    }
}

impl PartitionableGraph for RandomGraph {
    type VertexId = usize;
    type VertexParIter<'a> = rayon::vec::IntoIter<usize> where Self: 'a;
    type NeighParIter<'a>   = rayon::vec::IntoIter<usize> where Self: 'a;

    fn vertices(&self) -> Self::VertexParIter<'_> {
        (0..self.n).collect::<Vec<_>>().into_par_iter()
    }
    fn neighbors(&self, u: usize) -> Self::NeighParIter<'_> {
        let nbrs: Vec<_> = self.edges.iter()
            .filter_map(|&(a,b)| if a==u { Some(b) } else if b==u { Some(a) } else { None })
            .collect();
        nbrs.into_par_iter()
    }
    fn degree(&self, u: usize) -> usize {
        self.neighbors(u).count()
    }
}

fn bench_partition(c: &mut Criterion) {
    let mut group = c.benchmark_group("mpi-support");

    // Test a couple of graph sizes and densities
    for &(n, p) in &[(1_000, 0.01), (5_000, 0.005), (10_000, 0.001)] {
        let graph = RandomGraph::with_params(n, p, 42);
        let cfg = PartitionerConfig {
            n_parts: 4,
            alpha:   0.75,
            seed_factor: 4.0,
            rng_seed:    42,
            max_iters:   20,
            epsilon:     0.05,
            enable_phase1: true,
            enable_phase2: true,
            enable_phase3: true,
        };

        group.bench_with_input(
            BenchmarkId::new(format!("n{}_p{}", n, p), ""),
            &(graph, cfg),
            |b, (g, cfg)| {
                b.iter(|| {
                    // we ignore the result; just measure timing
                    let _ = partition(g, cfg).unwrap();
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_partition);
criterion_main!(benches);
