mod util;
use mesh_sieve::algs::dual_graph::build_dual_with_order;
use util::*;

#[test]
fn non_manifold_three_cells_share_one_face_is_clique() {
    let arrows = &[(1, 10), (1, 11), (2, 10), (2, 12), (3, 10), (3, 13)];
    let sieve = sieve_from(arrows);
    let (dg, order) = build_dual_with_order(&sieve, [pid(1), pid(2), pid(3)]);
    let mut edges = Vec::new();
    for i in 0..order.len() {
        let start = dg.xadj[i];
        let end = dg.xadj[i + 1];
        for &j in &dg.adjncy[start..end] {
            if i < j {
                edges.push((i, j));
            }
        }
    }
    edges.sort_unstable();
    let want = vec![(0, 1), (0, 2), (1, 2)];
    assert_eq!(
        edges, want,
        "dual graph should be K3 for a face shared by three cells"
    );
}
