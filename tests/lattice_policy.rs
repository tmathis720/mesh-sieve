mod util;
use mesh_sieve::algs::lattice::{AdjacencyOpts, adjacent_with};
use util::*;

#[test]
fn face_only_vs_vertex_expansion_differs() {
    let arrows = &[
        (1, 10),
        (1, 11),
        (10, 100),
        (11, 101),
        (2, 12),
        (2, 13),
        (12, 100),
        (13, 102),
        (2, 100),
    ];
    let sieve = sieve_from(arrows);

    let a1 = adjacent_with(
        &sieve,
        pid(1),
        AdjacencyOpts {
            max_down_depth: Some(1),
            same_stratum_only: true,
        },
    );
    assert!(a1.iter().all(|&p| p != pid(2)));

    let a2 = adjacent_with(
        &sieve,
        pid(1),
        AdjacencyOpts {
            max_down_depth: Some(2),
            same_stratum_only: true,
        },
    );
    assert!(a2.contains(&pid(2)));
}
