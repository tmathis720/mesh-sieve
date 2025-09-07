use mesh_sieve::algs::traversal::{Dir, TraversalBuilder};
use mesh_sieve::topology::sieve::Sieve;
use mesh_sieve::topology::sieve::in_memory::InMemorySieve;

#[test]
fn dfs_down_max_depth() {
    let mut s = InMemorySieve::<u32, ()>::new();
    s.add_arrow(0, 1, ());
    s.add_arrow(1, 2, ());
    s.add_arrow(2, 3, ());
    let v = TraversalBuilder::new(&s)
        .dir(Dir::Down)
        .dfs()
        .max_depth(Some(2))
        .seeds([0])
        .run();
    assert_eq!(v, vec![0, 1, 2]); // 3 is beyond depth 2
}

#[test]
fn bfs_up_early_stop() {
    let mut s = InMemorySieve::<u32, ()>::new();
    s.add_arrow(10, 1, ());
    s.add_arrow(11, 1, ());
    s.add_arrow(12, 10, ());
    let stop_at = |p: u32| p == 11;
    let v = TraversalBuilder::new(&s)
        .dir(Dir::Up)
        .bfs()
        .early_stop(&stop_at)
        .seeds([1])
        .run();
    // Contains {1,10,11} but may stop before visiting 12 depending on order; 12 is not required.
    assert!(v.contains(&1) && v.contains(&10) && v.contains(&11));
}
