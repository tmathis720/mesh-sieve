//! Bin-packing and cluster assignment utilities for partitioning.
//!
//! This module provides algorithms for assigning clusters to parts using
//! first-fit decreasing (FFD) bin-packing and adjacency-guided merging,
//! supporting balanced and locality-aware partitioning.

use rayon::prelude::ParallelSliceMut;
use std::cmp::Reverse;
use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};

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

/// Assigns each cluster to one of `k` parts using first-fit decreasing (FFD) bin-packing.
///
/// # Arguments
/// - `items`: Slice of clusters to assign.
/// - `k`: Number of parts.
/// - `epsilon`: Allowed balance tolerance (e.g., 0.05 for 5% imbalance).
///
/// # Returns
/// A vector mapping each cluster to a part, or an error if balance cannot be achieved.
///
/// # Errors
/// Returns `PartitionerError::Unbalanced` if the resulting assignment exceeds the allowed tolerance.
pub fn partition_clusters(
    items: &[Item],
    k: usize,
    epsilon: f64,
) -> Result<Vec<usize>, crate::partitioning::PartitionerError> {
    assert!(k > 0, "Number of parts (k) must be ≥ 1");
    let n = items.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    // 1. Build a vector of indices [0, 1, 2, ..., n-1].
    let mut order: Vec<usize> = (0..n).collect();

    // 2. Sort `order` by descending `items[i].load`.
    //    We're using a parallel sort here so that large cluster lists
    //    finish quickly.  For small `n`, this falls back to a serial sort.
    order.par_sort_unstable_by_key(|&i| Reverse(items[i].load));

    // 3. Create `k` buckets, each with an atomic load counter initialized to 0.
    let buckets: Vec<AtomicU64> = (0..k).map(|_| AtomicU64::new(0)).collect();

    // 4. Prepare the output vector: cluster_id → part_id.
    let mut cluster_to_part = vec![0; n];

    // 5. Compute balance threshold
    let total_load: u64 = items.iter().map(|it| it.load).sum();
    let threshold = ((1.0 + epsilon) * (total_load as f64 / k as f64)).ceil() as u64;

    // 6. Greedily assign each cluster (in `order`) to a bucket
    for &idx in &order {
        // (a) Try to place in a part containing a neighbor (adjacency-aware)
        let mut chosen_bucket = None;
        for &(nbr_cid, _) in &items[idx].adj {
            if nbr_cid < cluster_to_part.len() {
                let part = cluster_to_part[nbr_cid];
                let load_b = buckets[part].load(Ordering::Relaxed);
                if load_b + items[idx].load <= threshold {
                    chosen_bucket = Some(part);
                    break;
                }
            }
        }
        // (b) Fallback: pick the bucket with minimal load
        if chosen_bucket.is_none() {
            let (b0, _) = (0..k)
                .map(|b| (b, buckets[b].load(Ordering::Relaxed)))
                .min_by_key(|&(_, w)| w)
                .unwrap();
            chosen_bucket = Some(b0);
        }
        let bucket = chosen_bucket.unwrap();
        cluster_to_part[idx] = bucket;
        buckets[bucket].fetch_add(items[idx].load, Ordering::Relaxed);
    }

    // 7. Check balance
    let loads: Vec<u64> = buckets.iter().map(|b| b.load(Ordering::Relaxed)).collect();
    let min_load = *loads.iter().min().unwrap();
    let max_load = *loads.iter().max().unwrap();
    let ratio = max_load as f64 / (min_load as f64 + std::f64::EPSILON);
    let tol = 1.0 + epsilon + 1e-6;
    if ratio > tol {
        return Err(crate::partitioning::PartitionerError::Unbalanced {
            max_load,
            min_load,
            ratio,
            tolerance: tol,
        });
    }
    Ok(cluster_to_part)
}

/// Phase 2 merge: adjacency-guided cluster assignment.
///
/// Assigns clusters to parts by merging based on adjacency weights, then fills remaining clusters
/// to balance load.
///
/// # Arguments
/// - `items`: Slice of clusters to assign.
/// - `k`: Number of parts.
/// - `epsilon`: Allowed balance tolerance.
///
/// # Returns
/// A vector mapping each cluster to a part, or an error if no positive merge is possible or balance fails.
///
/// # Errors
/// Returns `PartitionerError::NoPositiveMerge` or `PartitionerError::Unbalanced`.
pub fn merge_clusters_into_parts(
    items: &[Item],
    k: usize,
    epsilon: f64,
) -> Result<Vec<usize>, crate::partitioning::PartitionerError> {
    assert!(k > 0 && !items.is_empty());
    let n = items.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by_key(|&i| Reverse(items[i].load));
    let seed_idxs = &order[..k];
    let mut seed_loads: Vec<u64> = seed_idxs.iter().map(|&i| items[i].load).collect();
    let mut seed_members: Vec<Vec<usize>> = seed_idxs.iter().map(|&i| vec![i]).collect();
    let mut unassigned: HashSet<usize> = (0..n).collect();
    for &i in seed_idxs { unassigned.remove(&i); }
    let mut did_positive_merge = false;
    loop {
        let mut best: Option<(usize, usize, u64)> = None;
        for (s, members) in seed_members.iter().enumerate() {
            for &cid in members {
                for &(nbr, w) in &items[cid].adj {
                    if unassigned.contains(&nbr) {
                        if best.as_ref().map_or(true, |&(_,_,bw)| w > bw) {
                            best = Some((s, nbr, w));
                        }
                    }
                }
            }
        }
        let (seed_idx, pick, _) = match best { Some(t) => t, None => break };
        did_positive_merge = true;
        seed_members[seed_idx].push(pick);
        seed_loads[seed_idx] += items[pick].load;
        unassigned.remove(&pick);
    }
    if !did_positive_merge && unassigned.len() > 0 {
        return Err(crate::partitioning::PartitionerError::NoPositiveMerge);
    }
    for &cid in unassigned.iter() {
        let (s, _) = seed_loads.iter().enumerate().min_by_key(|&(_, &l)| l).unwrap();
        seed_members[s].push(cid);
        seed_loads[s] += items[cid].load;
    }
    let mut result = vec![0usize; n];
    for (part, members) in seed_members.into_iter().enumerate() {
        for cid in members {
            result[cid] = part;
        }
    }
    let min_load = *seed_loads.iter().min().unwrap();
    let max_load = *seed_loads.iter().max().unwrap();
    let ratio = max_load as f64 / (min_load as f64 + std::f64::EPSILON);
    let tol = 1.0 + epsilon + 1e-6;
    if ratio > tol {
        return Err(crate::partitioning::PartitionerError::Unbalanced {
            max_load,
            min_load,
            ratio,
            tolerance: tol,
        });
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_binpack_test() {
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
                load: 5,
                adj: vec![],
            },
            Item {
                cid: 3,
                load: 15,
                adj: vec![],
            },
        ];

        let parts = partition_clusters(&items, 2, 0.05).unwrap();

        let mut load0 = 0u64;
        let mut load1 = 0u64;
        for (idx, &p) in parts.iter().enumerate() {
            if p == 0 {
                load0 += items[idx].load;
            } else if p == 1 {
                load1 += items[idx].load;
            } else {
                panic!("Invalid part id {:?}", p);
            }
        }

        let max_load = std::cmp::max(load0, load1);
        let min_load = std::cmp::min(load0, load1);
        assert!(
            (max_load as f64) / (min_load as f64 + 1e-9) <= 1.05,
            "Buckets are unbalanced: max={}, min={}",
            max_load,
            min_load
        );
    }

    #[test]
    fn many_buckets_few_items() {
        let items = vec![
            Item {
                cid: 0,
                load: 7,
                adj: vec![],
            },
            Item {
                cid: 1,
                load: 3,
                adj: vec![],
            },
        ];

        let parts = partition_clusters(&items, 2, 2.0).unwrap(); // Allow up to 3x imbalance
        assert!(parts[0] != parts[1]);
        assert!(parts[0] < 2 && parts[1] < 2);
    }

    #[test]
    fn unbalanced_error() {
        let items = vec![
            Item { cid: 0, load: 100, adj: vec![] },
            Item { cid: 1, load: 1, adj: vec![] },
        ];
        let res = partition_clusters(&items, 2, 0.0);
        assert!(matches!(res, Err(crate::partitioning::PartitionerError::Unbalanced { .. })));
    }

    #[test]
    fn no_positive_merge_error() {
        let items = vec![
            Item { cid: 0, load: 10, adj: vec![] },
            Item { cid: 1, load: 20, adj: vec![] },
            Item { cid: 2, load: 30, adj: vec![] },
        ];
        let res = merge_clusters_into_parts(&items, 2, 0.05);
        assert!(matches!(res, Err(crate::partitioning::PartitionerError::NoPositiveMerge)));
    }
}
