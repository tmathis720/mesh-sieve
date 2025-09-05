//! Bin-packing and cluster assignment utilities for partitioning.
//!
//! This module provides algorithms for assigning clusters to parts using
//! first-fit decreasing (FFD) bin-packing and adjacency-guided merging,
//! supporting balanced and locality-aware partitioning.

use rayon::prelude::*;
use std::cmp::{Ordering, Reverse};
use std::collections::{HashMap, HashSet};

/// Represents a “cluster” as an item to be packed into one of `k` parts.
///
/// - `cid`: Cluster ID.
/// - `load`: The load or weight of the cluster.
/// - `adj`: Adjacency list of (neighbor cluster ID, edge weight) pairs.
#[derive(Debug, Clone)]
pub struct Item {
    pub cid: usize,
    pub load: u64,
    pub adj: Vec<(usize, u64)>,
}

/// Assigns clusters to `k` parts using an adjacency-aware first-fit decreasing strategy.
///
/// Greedy placement never violates the balance threshold; if no admissible adjacency choice
/// exists, the cluster is placed into the lightest feasible part. If no such part exists,
/// `PartitionerError::Unbalanced` is returned.
pub fn partition_clusters(
    items: &[Item],
    k: usize,
    epsilon: f64,
) -> Result<Vec<usize>, crate::partitioning::PartitionerError> {
    use crate::partitioning::PartitionerError;

    assert!(k > 0, "k must be ≥ 1");
    let n = items.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.par_sort_unstable_by_key(|&i| Reverse(items[i].load));

    let cid_to_index: HashMap<usize, usize> = items
        .iter()
        .enumerate()
        .map(|(i, it)| (it.cid, i))
        .collect();

    let total_load: u64 = items.iter().map(|it| it.load).sum();
    let threshold: u64 = ((1.0 + epsilon) * (total_load as f64 / k as f64)).ceil() as u64;

    let mut part_loads = vec![0u64; k];
    let mut assign = vec![usize::MAX; n];

    let choose_best = |cands: &[(usize, u64)], part_loads: &[u64]| -> Option<usize> {
        cands
            .iter()
            .copied()
            .max_by(|&(pa, wa), &(pb, wb)| match wa.cmp(&wb) {
                Ordering::Equal => match part_loads[pa].cmp(&part_loads[pb]) {
                    Ordering::Equal => pa.cmp(&pb).reverse(),
                    other => other.reverse(),
                },
                other => other,
            })
            .map(|(p, _)| p)
    };

    for &idx in &order {
        let item = &items[idx];

        let mut into = vec![0u64; k];
        for &(nbr_cid, w) in &item.adj {
            if let Some(&nbr_idx) = cid_to_index.get(&nbr_cid) {
                let p = assign[nbr_idx];
                if p != usize::MAX {
                    into[p] = into[p].saturating_add(w);
                }
            }
        }

        let feasible: Vec<(usize, u64)> = (0..k)
            .filter(|&p| part_loads[p].saturating_add(item.load) <= threshold)
            .map(|p| (p, into[p]))
            .collect();

        let chosen = if !feasible.is_empty() {
            choose_best(&feasible, &part_loads)
        } else {
            None
        };

        let part = match chosen {
            Some(p) => p,
            None => {
                let max_load = *part_loads.iter().max().unwrap_or(&0);
                let min_load = *part_loads.iter().min().unwrap_or(&0);
                let ratio = if min_load == 0 {
                    f64::INFINITY
                } else {
                    max_load as f64 / min_load as f64
                };
                return Err(PartitionerError::Unbalanced {
                    max_load,
                    min_load,
                    ratio,
                    tolerance: 1.0 + epsilon + 1e-6,
                });
            }
        };

        assign[idx] = part;
        part_loads[part] = part_loads[part].saturating_add(item.load);
    }

    Ok(assign)
}

/// Phase 2 merge: adjacency-guided cluster assignment.
///
/// Greedy merges respect the balance threshold. When no admissible positive merge remains,
/// remaining clusters are assigned to the lightest feasible part. If no feasible part exists,
/// the function returns `PartitionerError::Unbalanced`.
pub fn merge_clusters_into_parts(
    items: &[Item],
    k: usize,
    epsilon: f64,
) -> Result<Vec<usize>, crate::partitioning::PartitionerError> {
    use crate::partitioning::PartitionerError;

    assert!(k > 0 && !items.is_empty());

    let n = items.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by_key(|&i| Reverse(items[i].load));

    let seed_count = k.min(n);
    let seed_idxs = &order[..seed_count];

    let total_load: u64 = items.iter().map(|it| it.load).sum();
    let threshold: u64 = ((1.0 + epsilon) * (total_load as f64 / k as f64)).ceil() as u64;

    let mut part_loads: Vec<u64> = seed_idxs.iter().map(|&i| items[i].load).collect();
    part_loads.resize(k, 0);
    let mut part_members: Vec<Vec<usize>> = seed_idxs.iter().map(|&i| vec![i]).collect();
    part_members.resize(k, Vec::new());

    let mut assign = vec![usize::MAX; n];
    for (p, &i) in seed_idxs.iter().enumerate() {
        if items[i].load > threshold {
            return Err(PartitionerError::Unbalanced {
                max_load: items[i].load,
                min_load: 0,
                ratio: f64::INFINITY,
                tolerance: 1.0 + epsilon + 1e-6,
            });
        }
        assign[i] = p;
    }

    let cid_to_index: HashMap<usize, usize> = items
        .iter()
        .enumerate()
        .map(|(i, it)| (it.cid, i))
        .collect();

    let mut unassigned: HashSet<usize> = (0..n).collect();
    for &i in seed_idxs {
        unassigned.remove(&i);
    }

    loop {
        let mut best: Option<(usize, usize, u64)> = None;

        for p in 0..k {
            for &m in &part_members[p] {
                for &(nbr_cid, w) in &items[m].adj {
                    if w == 0 {
                        continue;
                    }
                    if let Some(&nbr_idx) = cid_to_index.get(&nbr_cid) {
                        if unassigned.contains(&nbr_idx)
                            && part_loads[p].saturating_add(items[nbr_idx].load) <= threshold
                        {
                            match best {
                                None => best = Some((p, nbr_idx, w)),
                                Some((_, _, bw)) if w > bw => best = Some((p, nbr_idx, w)),
                                _ => {}
                            }
                        }
                    }
                }
            }
        }

        match best {
            Some((p, c, _w)) => {
                assign[c] = p;
                part_members[p].push(c);
                part_loads[p] = part_loads[p].saturating_add(items[c].load);
                unassigned.remove(&c);
            }
            None => break,
        }
    }

    for &i in unassigned.iter() {
        let mut best_p: Option<usize> = None;
        let mut best_load = u64::MAX;
        for p in 0..k {
            let new_load = part_loads[p].saturating_add(items[i].load);
            if new_load <= threshold && new_load < best_load {
                best_load = new_load;
                best_p = Some(p);
            }
        }
        let p = match best_p {
            Some(p) => p,
            None => {
                let max_load = *part_loads.iter().max().unwrap_or(&0);
                let min_load = *part_loads.iter().min().unwrap_or(&0);
                let ratio = if min_load == 0 {
                    f64::INFINITY
                } else {
                    max_load as f64 / min_load as f64
                };
                return Err(PartitionerError::Unbalanced {
                    max_load,
                    min_load,
                    ratio,
                    tolerance: 1.0 + epsilon + 1e-6,
                });
            }
        };
        assign[i] = p;
        part_members[p].push(i);
        part_loads[p] = part_loads[p].saturating_add(items[i].load);
    }

    let min_load = *part_loads.iter().min().unwrap();
    let max_load = *part_loads.iter().max().unwrap();
    let ratio = max_load as f64 / (min_load as f64 + std::f64::EPSILON);
    let tol = 1.0 + epsilon + 1e-6;
    if ratio > tol {
        return Err(PartitionerError::Unbalanced {
            max_load,
            min_load,
            ratio,
            tolerance: tol,
        });
    }

    Ok(assign)
}

#[cfg(feature = "binpack-retry")]
pub fn partition_clusters_with_retry(
    items: &[Item],
    k: usize,
    epsilon: f64,
    max_retries: usize,
    eps_cap: f64,
    growth: f64,
) -> Result<Vec<usize>, crate::partitioning::PartitionerError> {
    use crate::partitioning::PartitionerError;

    let mut eps = epsilon;
    for attempt in 0..=max_retries {
        match partition_clusters(items, k, eps) {
            Ok(a) => return Ok(a),
            Err(PartitionerError::Unbalanced { .. }) if attempt < max_retries => {
                let new_eps = (eps * growth).min(eps_cap);
                if (new_eps - eps).abs() < f64::EPSILON {
                    break;
                }
                #[cfg(feature = "log")]
                log::warn!("binpack retry: epsilon {:.4} -> {:.4}", eps, new_eps);
                eps = new_eps;
            }
            Err(e) => return Err(e),
        }
    }
    partition_clusters(items, k, eps)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threshold_admissibility() {
        let items = vec![
            Item {
                cid: 0,
                load: 60,
                adj: vec![(1, 1)],
            },
            Item {
                cid: 1,
                load: 40,
                adj: vec![(0, 1)],
            },
            Item {
                cid: 2,
                load: 40,
                adj: vec![],
            },
        ];

        let parts = partition_clusters(&items, 2, 0.2).unwrap();

        let mut loads = [0u64; 2];
        for (i, &p) in parts.iter().enumerate() {
            loads[p] += items[i].load;
        }
        let total_load: u64 = items.iter().map(|it| it.load).sum();
        let threshold = ((1.0 + 0.2) * (total_load as f64 / 2.0)).ceil() as u64;
        assert!(loads.iter().all(|&l| l <= threshold));
    }

    #[test]
    fn merge_fallback_to_lightest_feasible() {
        let items = vec![
            Item {
                cid: 0,
                load: 10,
                adj: vec![],
            },
            Item {
                cid: 1,
                load: 20,
                adj: vec![],
            },
            Item {
                cid: 2,
                load: 30,
                adj: vec![],
            },
        ];

        let parts = merge_clusters_into_parts(&items, 2, 0.05).unwrap();
        assert_eq!(parts[0], parts[1]);
        assert_eq!(parts[2], 0);
    }

    #[test]
    fn adjacency_chain_prefers_merge() {
        let items = vec![
            Item {
                cid: 0,
                load: 10,
                adj: vec![(1, 10)],
            },
            Item {
                cid: 1,
                load: 10,
                adj: vec![(0, 10), (2, 10)],
            },
            Item {
                cid: 2,
                load: 10,
                adj: vec![(1, 10)],
            },
            Item {
                cid: 3,
                load: 1,
                adj: vec![],
            },
        ];

        let parts = partition_clusters(&items, 2, 100.0).unwrap();
        assert_eq!(parts[0], parts[1]);
        assert_eq!(parts[1], parts[2]);
        assert_ne!(parts[3], parts[0]);
    }

    #[test]
    fn heavy_seed_unbalanced() {
        let items = vec![
            Item {
                cid: 0,
                load: 100,
                adj: vec![],
            },
            Item {
                cid: 1,
                load: 1,
                adj: vec![],
            },
        ];
        assert!(matches!(
            partition_clusters(&items, 2, 0.0),
            Err(crate::partitioning::PartitionerError::Unbalanced { .. })
        ));
        assert!(matches!(
            merge_clusters_into_parts(&items, 2, 0.0),
            Err(crate::partitioning::PartitionerError::Unbalanced { .. })
        ));
    }

    #[cfg(feature = "binpack-retry")]
    #[test]
    fn retry_succeeds_with_larger_epsilon() {
        let items = vec![
            Item {
                cid: 0,
                load: 55,
                adj: vec![],
            },
            Item {
                cid: 1,
                load: 54,
                adj: vec![],
            },
        ];
        let parts = partition_clusters_with_retry(&items, 2, 0.01, 2, 0.05, 2.0).unwrap();
        assert_eq!(parts.len(), 2);
    }
}
