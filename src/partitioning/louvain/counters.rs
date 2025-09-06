use crate::partitioning::graph_traits::PartitionableGraph;
use hashbrown::HashMap;
use rayon::prelude::*;

/// Brute-force cluster counters for modularity verification.
#[derive(Debug, Clone)]
pub struct ClusterCounters {
    pub m_edges: u64,
    pub vol: Vec<u64>,
    pub e_ij: HashMap<(u32, u32), u64>,
    pub e_cc: Vec<u64>,
}

/// Compute cluster counters for a graph and clustering.
pub fn compute_counters<G>(g: &G, clusters: &[u32]) -> ClusterCounters
where
    G: PartitionableGraph<VertexId = usize>,
{
    let m_edges = g.edges().count() as u64;
    let n_clusters = clusters.iter().max().map(|c| *c as usize + 1).unwrap_or(0);
    let mut vol = vec![0u64; n_clusters];
    let mut e_cc = vec![0u64; n_clusters];
    let mut e_ij: HashMap<(u32, u32), u64> = HashMap::new();

    for (v, &cid) in clusters.iter().enumerate() {
        vol[cid as usize] += g.degree(v) as u64;
    }

    for (u, v) in g.edges().collect::<Vec<_>>() {
        let cu = clusters[u as usize];
        let cv = clusters[v as usize];
        if cu == cv {
            e_cc[cu as usize] += 1;
        } else {
            let key = (cu.min(cv), cu.max(cv));
            *e_ij.entry(key).or_insert(0) += 1;
        }
    }

    ClusterCounters {
        m_edges,
        vol,
        e_ij,
        e_cc,
    }
}
