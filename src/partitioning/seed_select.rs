//! Seed selection for graph partitioning.
//!
//! Provides [`pick_seeds`], a deterministic, weighted sampling routine that
//! selects vertices without replacement.  Two strategies are used depending on
//! the regime:
//!   * A Fenwick tree (binary indexed tree) for `k << n`.
//!   * The A-ExpJ method of Efraimidis–Spirakis for larger `k`.
//!
//! The RNG is seeded from [`PartitionerConfig::rng_seed`] ensuring stable
//! results across runs.

use crate::partitioning::{graph_traits::PartitionableGraph, PartitionerConfig, PartitionerError};
use rand::{rngs::SmallRng, seq::SliceRandom, Rng, RngCore, SeedableRng};
use rayon::iter::ParallelIterator;

/// Returns an error if `degrees.len()` doesn’t match the number of vertices.
pub fn pick_seeds<G>(
    graph: &G,
    degrees: &[u64],
    cfg: &PartitionerConfig,
) -> Result<Vec<G::VertexId>, PartitionerError>
where
    G: PartitionableGraph<VertexId = usize> + Sync,
{
    let vertices: Vec<usize> = graph.vertices().collect();
    let n = vertices.len();
    if degrees.len() != n {
        return Err(PartitionerError::DegreeLengthMismatch {
            expected: n,
            got: degrees.len(),
        });
    }
    if n == 0 {
        return Ok(Vec::new());
    }

    let mut rng = SmallRng::seed_from_u64(cfg.rng_seed);
    let k = ((cfg.seed_factor * cfg.n_parts as f64).ceil() as usize).clamp(0, n);
    if k == 0 {
        return Ok(Vec::new());
    }

    let total_w: u128 = degrees.iter().map(|&w| w as u128).sum();
    if total_w == 0 {
        return Ok(pick_uniform_without_replacement(&vertices, k, &mut rng));
    }

    let small_k = (k as f64) <= 0.12 * (n as f64);
    let chosen = if small_k {
        pick_seeds_bit(&vertices, degrees, k, &mut rng)
    } else {
        pick_seeds_aexpj(&vertices, degrees, k, &mut rng)
    };

    Ok(chosen)
}

#[derive(Debug)]
struct Fenwick {
    tree: Vec<u128>,
}

impl Fenwick {
    fn from_weights(ws: &[u64]) -> (Self, u128) {
        let n = ws.len();
        let mut t = vec![0u128; n + 1];
        for (i, &w) in ws.iter().enumerate() {
            let mut idx = (i + 1) as usize;
            let add = w as u128;
            while idx <= n {
                t[idx] += add;
                idx += idx & (!idx + 1); // idx += idx & -idx
            }
        }
        let total = ws.iter().map(|&w| w as u128).sum();
        (Self { tree: t }, total)
    }

    fn add(&mut self, i0: usize, delta: i128) {
        let n = self.tree.len() - 1;
        let mut idx = i0 + 1;
        while idx <= n {
            let cur = self.tree[idx] as i128;
            self.tree[idx] = (cur + delta) as u128;
            idx += idx & (!idx + 1);
        }
    }

    fn find_by_prefix(&self, mut r: u128) -> usize {
        let n = self.tree.len() - 1;
        let mut step = 1usize;
        while step << 1 <= n {
            step <<= 1;
        }
        let mut idx = 0usize;
        let mut k = step;
        while k != 0 {
            let next = idx + k;
            if next <= n && self.tree[next] <= r {
                r -= self.tree[next];
                idx = next;
            }
            k >>= 1;
        }
        idx
    }
}

fn pick_seeds_bit(vertices: &[usize], weights: &[u64], k: usize, rng: &mut SmallRng) -> Vec<usize> {
    let n = vertices.len();
    let mut ws = weights.to_vec();
    let (mut bit, mut total) = Fenwick::from_weights(&ws);
    let mut out = Vec::with_capacity(k.min(n));

    for _ in 0..k.min(n) {
        if total == 0 {
            break;
        }
        let r = (rng.next_u64() as u128) % total;
        let idx = bit.find_by_prefix(r);
        out.push(vertices[idx]);

        let w = ws[idx] as u128;
        ws[idx] = 0;
        total -= w;
        bit.add(idx, -(w as i128));
    }
    out
}

fn pick_seeds_aexpj(
    vertices: &[usize],
    weights: &[u64],
    k: usize,
    rng: &mut SmallRng,
) -> Vec<usize> {
    use std::cmp::Ordering;

    let n = vertices.len();
    let k = k.min(n);
    if k == 0 {
        return Vec::new();
    }

    let mut keys: Vec<(f64, usize)> = Vec::with_capacity(n);
    for (i, &w) in weights.iter().enumerate() {
        if w == 0 {
            continue;
        }
        let mut u: f64 = rand::Rng::r#gen(rng);
        if u <= f64::MIN_POSITIVE {
            u = f64::MIN_POSITIVE;
        }
        if u >= 1.0 {
            u = 0.999_999_999_999_999_9;
        }
        let key = -u.ln() / (w as f64);
        keys.push((key, i));
    }
    if keys.is_empty() {
        return pick_uniform_without_replacement(vertices, k, rng);
    }

    let kth = k - 1;
    keys.select_nth_unstable_by(kth, |a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

    let mut chosen = keys[..=kth].to_vec();
    chosen.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    chosen.truncate(k);
    chosen.into_iter().map(|(_, i)| vertices[i]).collect()
}

fn pick_uniform_without_replacement(
    vertices: &[usize],
    k: usize,
    rng: &mut SmallRng,
) -> Vec<usize> {
    let n = vertices.len();
    let mut idxs: Vec<usize> = (0..n).collect();
    idxs.shuffle(rng);
    idxs.truncate(k.min(n));
    idxs.into_iter().map(|i| vertices[i]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::partitioning::graph_traits::PartitionableGraph;
    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    struct PathGraph {
        n: usize,
    }
    impl PartitionableGraph for PathGraph {
        type VertexId = usize;
        type VertexParIter<'a> = rayon::vec::IntoIter<usize>;
        type NeighParIter<'a> = rayon::vec::IntoIter<usize>;
        type NeighIter<'a> = std::vec::IntoIter<usize>;

        fn vertices(&self) -> Self::VertexParIter<'_> {
            (0..self.n).collect::<Vec<_>>().into_par_iter()
        }
        fn neighbors(&self, v: usize) -> Self::NeighParIter<'_> {
            self.neighbors_seq(v).collect::<Vec<_>>().into_par_iter()
        }
        fn neighbors_seq(&self, v: usize) -> Self::NeighIter<'_> {
            let mut neigh = Vec::new();
            if v > 0 {
                neigh.push(v - 1);
            }
            if v + 1 < self.n {
                neigh.push(v + 1);
            }
            neigh.into_iter()
        }
        fn degree(&self, v: usize) -> usize {
            self.neighbors_seq(v).count()
        }
        fn edges(&self) -> rayon::vec::IntoIter<(usize, usize)> {
            (0..self.n.saturating_sub(1))
                .map(|i| (i, i + 1))
                .collect::<Vec<_>>()
                .into_par_iter()
        }
    }

    #[test]
    fn pick_seeds_length_mismatch() {
        let g = PathGraph { n: 3 };
        let degrees = vec![1, 2];
        let cfg = PartitionerConfig::default();
        let err = pick_seeds(&g, &degrees, &cfg).unwrap_err();
        assert!(matches!(
            err,
            PartitionerError::DegreeLengthMismatch {
                expected: 3,
                got: 2
            }
        ));
    }

    #[test]
    fn determinism() {
        let g = PathGraph { n: 10 };
        let degrees = vec![1u64; 10];
        let cfg = PartitionerConfig {
            n_parts: 2,
            seed_factor: 1.0,
            rng_seed: 7,
            ..Default::default()
        };
        let s1 = pick_seeds(&g, &degrees, &cfg).unwrap();
        let s2 = pick_seeds(&g, &degrees, &cfg).unwrap();
        assert_eq!(s1, s2);
    }

    #[test]
    fn zero_degrees_uniform() {
        let g = PathGraph { n: 5 };
        let degrees = vec![0u64; 5];
        let cfg = PartitionerConfig {
            n_parts: 2,
            seed_factor: 1.0,
            rng_seed: 3,
            ..Default::default()
        };
        let seeds = pick_seeds(&g, &degrees, &cfg).unwrap();
        assert_eq!(seeds.len(), 2);
        let mut s = seeds.clone();
        s.sort();
        s.dedup();
        assert_eq!(s.len(), seeds.len());
    }

    #[test]
    fn k_greater_than_n() {
        let g = PathGraph { n: 3 };
        let degrees = vec![1u64; 3];
        let cfg = PartitionerConfig {
            n_parts: 5,
            seed_factor: 1.0,
            ..Default::default()
        };
        let seeds = pick_seeds(&g, &degrees, &cfg).unwrap();
        assert_eq!(seeds.len(), 3);
    }
}
