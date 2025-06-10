// Lattice operation tests for Sieve
use crate::topology::sieve::{InMemorySieve, Sieve};
use crate::topology::point::PointId as P;

fn p(x: u64) -> P {
    P::new(x)
}

/// Build the 1D chain 1→2→3→4
fn chain() -> InMemorySieve<P, ()> {
    let mut s = InMemorySieve::default();
    for (u, v) in &[(1, 2), (2, 3), (3, 4)] {
        s.add_arrow(p(*u), p(*v), ());
    }
    s
}

#[test]
fn meet_chain() {
    let s = chain();
    // closure(2) = {2,3,4}, closure(3) = {3,4}; so meet = [] (since closure({2,3}) = {2,3,4})
    let mut m: Vec<_> = s.meet(p(2), p(3)).collect();
    m.sort();
    m.dedup();
    assert_eq!(m, vec![]);
}

#[test]
fn join_chain() {
    let s = chain();
    // star(2) = {2,1}, star(3) = {3,2,1}, union = {1,2,3}
    let mut j: Vec<_> = s.join(p(2), p(3)).collect();
    j.sort();
    j.dedup();
    assert_eq!(j, vec![p(1), p(2), p(3)]);
}
