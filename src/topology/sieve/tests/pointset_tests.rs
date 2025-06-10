// Pointset and base/cap/points tests for Sieve
use crate::topology::sieve::InMemorySieve;
use crate::topology::point::PointId as P;

fn s() -> InMemorySieve<P, ()> {
    let mut s = InMemorySieve::default();
    s.add_arrow(P::new(1), P::new(2), ());
    s.add_arrow(P::new(3), P::new(2), ());
    s
}

#[test]
fn points_union_base_and_cap() {
    let s = s();
    let mut pts: Vec<_> = s.points().collect();
    pts.sort_unstable();
    assert_eq!(pts, vec![P::new(1), P::new(2), P::new(3)]);
}

#[test]
fn base_vs_cap_points() {
    let s = s();
    let mut base: Vec<_> = s.base_points().collect();
    base.sort_unstable();
    assert_eq!(base, vec![P::new(1), P::new(3)]);

    let mut cap: Vec<_> = s.cap_points().collect();
    cap.sort_unstable();
    assert_eq!(cap, vec![P::new(2)]);
}
