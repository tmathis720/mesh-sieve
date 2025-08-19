use mesh_sieve::algs::traversal_ref::{closure_ref, depth_map_ref, star_ref};
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::sieve::in_memory::InMemorySieve;
use mesh_sieve::topology::sieve::{Sieve, SieveRef};

#[test]
fn closure_star_depth_map_ref_work() {
    let mut s = InMemorySieve::<PointId, String>::new();
    let p1 = PointId::new(1).unwrap();
    let p2 = PointId::new(2).unwrap();
    let p3 = PointId::new(3).unwrap();
    let p4 = PointId::new(4).unwrap();
    s.add_arrow(p1, p2, "e12".into());
    s.add_arrow(p2, p3, "e23".into());
    s.add_arrow(p2, p4, "e24".into());

    // Clone-free variants return same points as legacy versions:
    let mut c = closure_ref(&s, [p1]);
    c.sort_unstable();
    assert_eq!(c, vec![p1, p2, p3, p4]);

    let mut st = star_ref(&s, [p4]);
    st.sort_unstable();
    assert_eq!(st, vec![p1, p2, p4]);

    let dm = depth_map_ref(&s, p1);
    assert_eq!(dm, vec![(p1, 0), (p2, 1), (p3, 2), (p4, 2)]);
}

// Demonstrate direct borrowing access to payloads (no clones):
#[test]
fn cone_support_ref_borrow_payloads() {
    let mut s = InMemorySieve::<PointId, Vec<u8>>::new();
    let p10 = PointId::new(10).unwrap();
    let p20 = PointId::new(20).unwrap();
    let p30 = PointId::new(30).unwrap();
    s.add_arrow(p10, p20, vec![1, 2, 3]);
    s.add_arrow(p10, p30, vec![4, 5]);

    // Borrowing cone gives &Vec<u8>
    let mut sizes: Vec<_> = s.cone_ref(p10).map(|(_q, pay)| pay.len()).collect();
    sizes.sort_unstable();
    assert_eq!(sizes, vec![2, 3]);

    // Borrowing support likewise
    let sum: usize = s
        .support_ref(p20)
        .map(|(_p, pay)| pay.iter().copied().sum::<u8>() as usize)
        .sum();
    assert_eq!(sum, 1 + 2 + 3);
}
