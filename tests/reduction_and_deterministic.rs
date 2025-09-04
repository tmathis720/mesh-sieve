use mesh_sieve::topology::sieve::{InMemorySieve, Sieve, InMemorySieveDeterministic};
use mesh_sieve::prelude::SieveQueryExt;
use mesh_sieve::algs::reduction::transitive_reduction_dag;

#[test]
fn transitive_reduction_removes_shortcuts() {
    let (a,b,c) = (1u32,2u32,3u32);
    let mut s = InMemorySieve::<u32, ()>::default();
    s.add_arrow(a,b,()); s.add_arrow(b,c,()); s.add_arrow(a,c,());
    let removed = transitive_reduction_dag(&mut s).unwrap();
    assert_eq!(removed,1);
    assert!(s.has_arrow(a,b) && s.has_arrow(b,c));
    assert!(!s.has_arrow(a,c));
}

#[test]
fn deterministic_backend_sorted_neighbors() {
    let mut s = InMemorySieveDeterministic::<u32, ()>::default();
    s.add_arrow(1,3,()); s.add_arrow(1,2,()); s.add_arrow(1,5,()); s.add_arrow(1,4,());
    let neigh: Vec<_> = s.cone(1).map(|(d,_ )| d).collect();
    assert_eq!(neigh, vec![2,3,4,5]);
    assert_eq!(SieveQueryExt::out_degree(&s,1),4);
}
