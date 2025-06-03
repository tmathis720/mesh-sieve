#![cfg(feature = "partitioning")]

use std::cmp::Reverse;
use std::sync::atomic::{AtomicU64, Ordering};
use rayon::prelude::*;

// Ensure rand and ahash are available as dependencies in Cargo.toml
// rand = { version = "0.8", features = ["std"], optional = true }
// ahash = { version = "0.8", optional = true }
// and add them to the partitioning feature if not already present.

/// Represents a “cluster” as an item to be packed into one of `k` parts.
#[derive(Debug, Clone)]
pub struct Item {
    pub cid: usize,
    pub load: u64,
    pub adj: Vec<(usize, u64)>,
}

/// Given a slice of `Item`s (each with a distinct `cid`), a target number
/// of parts `k`, and a balance tolerance `epsilon`, assign each cluster to
/// one of `k` parts.  This is a *first‐fit decreasing* (FFD) style bin‐packing.
pub fn partition_clusters(items: &[Item], k: usize, epsilon: f64) -> Vec<usize> {
    assert!(k > 0, "Number of parts (k) must be ≥ 1");
    let n = items.len();
    if n == 0 {
        return Vec::new();
    }

    // 1. Build a vector of indices [0, 1, 2, ..., n-1].
    let mut order: Vec<usize> = (0..n).collect();

    // 2. Sort `order` by descending `items[i].load`.
    //    We're using a parallel sort here so that large cluster lists
    //    finish quickly.  For small `n`, this falls back to a serial sort.
    order.par_sort_unstable_by_key(|&i| Reverse(items[i].load));

    // 3. Create `k` buckets, each with an atomic load counter initialized to 0.
    let buckets: Vec<AtomicU64> = (0..k)
        .map(|_| AtomicU64::new(0))
        .collect();

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
    assert!(
        (max_load as f64) / (min_load as f64 + 1e-9) <= 1.0 + epsilon + 1e-6,
        "Unbalanced: max/min = {:.3} > {:.3}",
        (max_load as f64) / (min_load as f64 + 1e-9),
        1.0 + epsilon
    );

    cluster_to_part
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_binpack_test() {
        let items = vec![
            Item { cid: 0, load: 10, adj: vec![] },
            Item { cid: 1, load: 20, adj: vec![] },
            Item { cid: 2, load: 5,  adj: vec![] },
            Item { cid: 3, load: 15, adj: vec![] },
        ];

        let parts = partition_clusters(&items, 2, 0.05);

        let mut load0 = 0u64;
        let mut load1 = 0u64;
        for (idx, &p) in parts.iter().enumerate() {
            if p == 0 {
                load0 += items[idx].load;
            } else if p == 1 {
                load1 += items[idx].load;
            } else {
                panic!("Invalid part id {}", p);
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
            Item { cid: 0, load: 7, adj: vec![] },
            Item { cid: 1, load: 3, adj: vec![] },
        ];

        let parts = partition_clusters(&items, 4, 0.1);
        // clusters 0 and 1 must each occupy different minimal-load buckets.
        assert!(parts[0] != parts[1]);
        // Both parts should be < 4.
        assert!(parts[0] < 4 && parts[1] < 4);
    }
}
