use mesh_sieve::topology::sieve::Sieve;
use mesh_sieve::topology::sieve::in_memory::InMemorySieve;

#[test]
fn reserve_cone_and_support_do_not_change_topology() {
    let mut s = InMemorySieve::<u32, ()>::new();
    s.add_arrow(1, 2, ());
    s.add_arrow(1, 3, ());
    s.add_arrow(4, 2, ());

    // Snapshot current cones/supports
    let mut c1: Vec<_> = s.cone(1).map(|(d, _)| d).collect();
    let mut s2: Vec<_> = s.support(2).map(|(u, _)| u).collect();
    c1.sort_unstable();
    s2.sort_unstable();

    // Reserve additional capacity
    s.reserve_cone(1, 16);
    s.reserve_support(2, 32);

    // Topology must be unchanged
    let mut c1_after: Vec<_> = s.cone(1).map(|(d, _)| d).collect();
    let mut s2_after: Vec<_> = s.support(2).map(|(u, _)| u).collect();
    c1_after.sort_unstable();
    s2_after.sort_unstable();
    assert_eq!(c1, c1_after);
    assert_eq!(s2, s2_after);

    // Cache is untouched by reserve
    let d1 = s.diameter().unwrap();
    let ptr_before = s.strata.get().unwrap() as *const _;
    let d2 = s.diameter().unwrap();
    let ptr_after = s.strata.get().unwrap() as *const _;
    assert_eq!(d1, d2);
    assert_eq!(ptr_before, ptr_after);
}

#[test]
fn set_cone_is_degree_local_and_keeps_mirrors_consistent() {
    let mut s = InMemorySieve::<u32, ()>::new();
    // Initial: 1 -> {2,3}, 4 -> {2}
    s.add_arrow(1, 2, ());
    s.add_arrow(1, 3, ());
    s.add_arrow(4, 2, ());
    // Change only cone(1): now 1 -> {3,5}
    s.set_cone(1, vec![(3, ()), (5, ())]);

    // Check cone(1)
    let mut c1: Vec<_> = s.cone(1).map(|(d, _)| d).collect();
    c1.sort_unstable();
    assert_eq!(c1, vec![3, 5]);

    // Check mirrors in support(*)
    let mut s2: Vec<_> = s.support(2).map(|(u, _)| u).collect();
    s2.sort_unstable();
    assert_eq!(s2, vec![4]); // 1->2 was removed, 4->2 remains

    let mut s3: Vec<_> = s.support(3).map(|(u, _)| u).collect();
    s3.sort_unstable();
    assert_eq!(s3, vec![1]); // still owned by 1

    let mut s5: Vec<_> = s.support(5).map(|(u, _)| u).collect();
    s5.sort_unstable();
    assert_eq!(s5, vec![1]); // new mirror added
}

#[test]
fn set_support_is_degree_local_and_keeps_mirrors_consistent() {
    let mut s = InMemorySieve::<u32, ()>::new();
    // Initial: {1,4} -> 2, and 1 -> 3
    s.add_arrow(1, 2, ());
    s.add_arrow(4, 2, ());
    s.add_arrow(1, 3, ());
    // Change only support(2): now {4,7} -> 2
    s.set_support(2, vec![(4, ()), (7, ())]);

    // Check support(2)
    let mut s2: Vec<_> = s.support(2).map(|(u, _)| u).collect();
    s2.sort_unstable();
    assert_eq!(s2, vec![4, 7]);

    // Mirrors in cone(*)
    let mut c1: Vec<_> = s.cone(1).map(|(d, _)| d).collect();
    c1.sort_unstable();
    assert_eq!(c1, vec![3]); // 1->2 removed, 1->3 remains

    let mut c4: Vec<_> = s.cone(4).map(|(d, _)| d).collect();
    c4.sort_unstable();
    assert_eq!(c4, vec![2]); // 4->2 still present

    let mut c7: Vec<_> = s.cone(7).map(|(d, _)| d).collect();
    c7.sort_unstable();
    assert_eq!(c7, vec![2]); // new mirror added
}

#[test]
fn add_cone_is_incremental_without_global_rebuild() {
    let mut s = InMemorySieve::<u32, ()>::new();
    s.add_arrow(1, 2, ());
    s.add_arrow(3, 4, ());
    // add_cone should affect only p=1 and its mirrors
    s.add_cone(1, vec![(5, ()), (6, ())]);

    let mut c1: Vec<_> = s.cone(1).map(|(d, _)| d).collect();
    c1.sort_unstable();
    assert_eq!(c1, vec![2, 5, 6]);

    // Mirrors exist:
    let sup5: Vec<_> = s.support(5).map(|(u, _)| u).collect();
    let sup6: Vec<_> = s.support(6).map(|(u, _)| u).collect();
    assert_eq!(sup5, vec![1]);
    assert_eq!(sup6, vec![1]);

    // Unrelated adjacency remains unchanged
    let c3: Vec<_> = s.cone(3).map(|(d, _)| d).collect();
    assert_eq!(c3, vec![4]);
}
