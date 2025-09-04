use mesh_sieve::topology::sieve::{InMemorySieve, Sieve};
use mesh_sieve::prelude::SieveQueryExt;
use mesh_sieve::topology::sieve::build_ext::SieveBuildExt;

#[test]
fn edge_queries_work() {
    let mut s = InMemorySieve::<u32, ()>::default();
    s.add_arrow(1,2,());
    s.add_arrow(1,3,());
    s.add_arrow(4,1,());
    assert!(s.has_arrow(1,2));
    assert!(!s.has_arrow(2,1));
    assert_eq!(SieveQueryExt::out_degree(&s,1),2); // explicit UFCS to avoid import issues
    assert_eq!(SieveQueryExt::in_degree(&s,1),1);
}

#[test]
fn bulk_add_last_wins_and_dedup() {
    let mut s = InMemorySieve::<u32, i32>::default();
    s.add_arrows_from([(1,2,7),(1,2,9),(1,3,5)]);
    let v: Vec<_> = s.cone(1).collect();
    assert!(v.contains(&(2,9)) && v.contains(&(3,5)));

    let mut s2 = InMemorySieve::<u32, i32>::default();
    s2.add_arrows_dedup_from([(1,2,1),(1,2,2),(1,3,3)]);
    let v: Vec<_> = s2.cone(1).collect();
    assert!(v.contains(&(2,2)) && v.contains(&(3,3)) && v.len()==2);
}
