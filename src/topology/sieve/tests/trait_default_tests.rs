// Tests for Sieve trait default implementations and genericity
use crate::topology::sieve::Sieve;
use std::collections::HashSet;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct Tiny(u8);

struct FakeSieve {
    arrows: Vec<(Tiny, Tiny)>,
    points: HashSet<Tiny>,
}

impl FakeSieve {
    fn new(arrows: &[(u8, u8)]) -> Self {
        let mut points = HashSet::new();
        for &(a, b) in arrows {
            points.insert(Tiny(a));
            points.insert(Tiny(b));
        }
        Self {
            arrows: arrows.iter().map(|&(a, b)| (Tiny(a), Tiny(b))).collect(),
            points,
        }
    }
}

impl Sieve for FakeSieve {
    type Point = Tiny;
    type Payload = ();
    fn cone<'a>(&'a mut self, p: Self::Point) -> Box<dyn Iterator<Item = (Self::Point, &'a Self::Payload)> + 'a> {
        Box::new(self.arrows.iter().filter_map(move |(src, dst)| {
            if *src == p { Some((*dst, &())) } else { None }
        }))
    }
    fn support<'a>(&'a mut self, p: Self::Point) -> Box<dyn Iterator<Item = (Self::Point, &'a Self::Payload)> + 'a> {
        Box::new(self.arrows.iter().filter_map(move |(src, dst)| {
            if *dst == p { Some((*src, &())) } else { None }
        }))
    }
    // No override for points/base_points/cap_points: use default
}

#[test]
fn default_points_methods_work() {
    let mut s = FakeSieve::new(&[(1,2), (2,3)]);
    let mut pts: Vec<_> = s.points().collect();
    pts.sort_by_key(|x| x.0);
    assert_eq!(pts, vec![Tiny(1), Tiny(2), Tiny(3)]);
    let mut base: Vec<_> = s.base_points().collect();
    base.sort_by_key(|x| x.0);
    assert_eq!(base, vec![Tiny(1)]);
    let mut cap: Vec<_> = s.cap_points().collect();
    cap.sort_by_key(|x| x.0);
    assert_eq!(cap, vec![Tiny(3)]);
}

#[test]
fn default_closure_and_star_work() {
    let mut s = FakeSieve::new(&[(1,2), (2,3)]);
    let mut closure: Vec<_> = Sieve::closure(&mut s, Tiny(1)).collect();
    closure.sort_by_key(|x| x.0);
    assert_eq!(closure, vec![Tiny(1), Tiny(2), Tiny(3)]);
    let mut star: Vec<_> = Sieve::star(&mut s, Tiny(3)).collect();
    star.sort_by_key(|x| x.0);
    assert_eq!(star, vec![Tiny(3), Tiny(2), Tiny(1)]);
}

#[test]
fn default_meet_and_join_work() {
    let mut s = FakeSieve::new(&[(1,2), (2,3), (1,4)]);
    // meet(2,4) should be empty (no separator)
    let sep: Vec<_> = s.meet(Tiny(2), Tiny(4)).collect();
    assert!(sep.is_empty());
    // join(2,4) should be star(2) ∪ star(4)
    let mut join: Vec<_> = s.join(Tiny(2), Tiny(4)).collect();
    join.sort_by_key(|x| x.0);
    assert_eq!(join, vec![Tiny(2), Tiny(1), Tiny(4)]);
}

#[test]
fn default_height_depth_diameter() {
    let mut s = FakeSieve::new(&[(1,2), (2,3), (1,4)]);
    assert_eq!(s.height(Tiny(1)), 2);
    assert_eq!(s.depth(Tiny(3)), 2);
    assert_eq!(s.diameter(), 2);
}

#[test]
fn genericity_on_non_pointid() {
    let mut s = FakeSieve::new(&[(10,11), (11,12)]);
    let pts: HashSet<_> = s.points().collect();
    assert!(pts.contains(&Tiny(10)) && pts.contains(&Tiny(12)));
}
