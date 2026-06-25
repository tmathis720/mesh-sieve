#![allow(dead_code)]
use mesh_sieve::{
    algs::communicator::RayonComm,
    topology::point::PointId,
    topology::sieve::{InMemorySieve, Sieve},
};

pub fn pid(u: u64) -> PointId {
    PointId::new(u).unwrap()
}

/// Build a sieve from arrows (u -> v) with unit payload ().
pub fn sieve_from(arrows: &[(u64, u64)]) -> InMemorySieve<PointId, ()> {
    let mut s = InMemorySieve::<PointId, ()>::default();
    for &(u, v) in arrows {
        s.add_arrow(pid(u), pid(v), ());
    }
    s
}

/// Two-rank Rayon comms (ranks 0 and 1).
pub fn rayons() -> (RayonComm, RayonComm) {
    (RayonComm::new(0, 2), RayonComm::new(1, 2))
}

/// Return an MPI world-size hint only when a recognized launcher populated
/// its process environment. This lets ordinary `cargo test` skip MPI runtime
/// initialization, which some MPI implementations do not support in singleton
/// mode and may abort rather than return an error.
#[cfg(feature = "mpi-support")]
pub fn mpi_launcher_world_size() -> Option<usize> {
    [
        "OMPI_COMM_WORLD_SIZE",
        "PMI_SIZE",
        "PMIX_UNIV_SIZE",
        "MV2_COMM_WORLD_SIZE",
    ]
    .into_iter()
    .find_map(|name| {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|size| *size > 0)
    })
}

/// Assert vec is a permutation of another vec (order-agnostic).
pub fn assert_permutation<T: Ord + Copy + std::fmt::Debug>(got: &[T], want: &[T]) {
    let mut a = got.to_vec();
    a.sort_unstable();
    let mut b = want.to_vec();
    b.sort_unstable();
    assert_eq!(a, b, "not a permutation\n got={:?}\nwant={:?}", got, want);
}

/// Bandwidth of an ordering π over an undirected simple graph E (u,v).
pub fn bandwidth(order: &[usize], edges: &[(usize, usize)]) -> usize {
    let mut pos = vec![0usize; order.len()];
    for (i, &v) in order.iter().enumerate() {
        pos[v] = i;
    }
    edges
        .iter()
        .map(|&(u, v)| pos[u].abs_diff(pos[v]))
        .max()
        .unwrap_or(0)
}
