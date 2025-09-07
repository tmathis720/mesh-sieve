mod util;
use util::*;
use mesh_sieve::algs::rcm::{distributed_rcm, distributed_rcm_with, RcmAdjacency};
use mesh_sieve::algs::communicator::NoComm;
use mesh_sieve::topology::sieve::Sieve;

fn path_sieve(n: usize) -> mesh_sieve::topology::sieve::InMemorySieve<mesh_sieve::topology::point::PointId, ()> {
    let mut s = mesh_sieve::topology::sieve::InMemorySieve::<_, ()>::default();
    for i in 0..n - 1 {
        s.add_arrow(pid((i + 1) as u64), pid((i + 2) as u64), ());
        s.add_arrow(pid((i + 2) as u64), pid((i + 1) as u64), ());
    }
    s
}

#[test]
fn rcm_path_bandwidth_is_one_and_is_permutation() {
    let s = path_sieve(6);
    let comm = NoComm;
    let order_pts = distributed_rcm_with(&s, &comm, RcmAdjacency::Undirected);
    let mut pts: Vec<_> = s.base_points().collect();
    pts.sort_unstable_by_key(|p| p.get());
    let n = pts.len();
    let idx_of = |p: mesh_sieve::topology::point::PointId| -> usize { (p.get() - 1) as usize };
    let perm: Vec<usize> = order_pts.iter().map(|&p| idx_of(p)).collect();
    assert_eq!(perm.len(), n);
    let want: Vec<usize> = (0..n).collect();
    assert_permutation(&perm, &want);
    let edges: Vec<(usize, usize)> = (0..n - 1).map(|i| (i, i + 1)).collect();
    assert_eq!(bandwidth(&perm, &edges), 1);
}

#[test]
fn rcm_undirected_equals_default() {
    let s = path_sieve(5);
    let comm = NoComm;
    let undirected = distributed_rcm_with(&s, &comm, RcmAdjacency::Undirected);
    let default = distributed_rcm(&s, &comm);
    assert_eq!(undirected, default);
}
