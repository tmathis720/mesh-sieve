use mesh_sieve::algs::communicator::NoComm;
use mesh_sieve::algs::distribute::distribute_mesh;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::sieve::{InMemorySieve, Sieve};
use mesh_sieve::topology::InvalidateCache;

#[derive(Default)]
struct NonBaseOriginSieve {
    inner: InMemorySieve<PointId, ()>,
}

impl InvalidateCache for NonBaseOriginSieve {
    fn invalidate_cache(&mut self) {
        self.inner.invalidate_cache();
    }
}

impl Sieve for NonBaseOriginSieve {
    type Point = PointId;
    type Payload = ();
    type ConeIter<'a> = <InMemorySieve<PointId, ()> as Sieve>::ConeIter<'a> where Self: 'a;
    type SupportIter<'a> = <InMemorySieve<PointId, ()> as Sieve>::SupportIter<'a> where Self: 'a;

    fn cone<'a>(&'a self, p: Self::Point) -> Self::ConeIter<'a> {
        self.inner.cone(p)
    }

    fn support<'a>(&'a self, p: Self::Point) -> Self::SupportIter<'a> {
        self.inner.support(p)
    }

    fn add_arrow(&mut self, src: Self::Point, dst: Self::Point, payload: Self::Payload) {
        self.inner.add_arrow(src, dst, payload);
    }

    fn remove_arrow(&mut self, src: Self::Point, dst: Self::Point) -> Option<Self::Payload> {
        self.inner.remove_arrow(src, dst)
    }

    fn base_points<'a>(&'a self) -> Box<dyn Iterator<Item = Self::Point> + 'a> {
        Box::new(std::iter::empty())
    }

    fn cap_points<'a>(&'a self) -> Box<dyn Iterator<Item = Self::Point> + 'a> {
        self.inner.cap_points()
    }

    fn points<'a>(&'a self) -> Box<dyn Iterator<Item = Self::Point> + 'a> {
        self.inner.points()
    }
}

#[test]
fn structural_overlap_only() {
    let mut g = InMemorySieve::<PointId, ()>::default();
    let p = |x| PointId::new(x).unwrap();
    g.add_arrow(p(1), p(2), ());
    g.add_arrow(p(2), p(3), ());
    let parts = vec![0, 1, 1];

    let comm = NoComm;
    let (local0, ov0) = distribute_mesh(&g, &parts, &comm).unwrap();
    assert_eq!(local0.points().count(), 0);

    let ranks: Vec<_> = ov0.neighbor_ranks().collect();
    assert_eq!(ranks, vec![1]);
    let links: Vec<_> = ov0.links_to(1).collect();
    assert!(
        links
            .iter()
            .any(|(p, rp)| *p == PointId::new(3).unwrap() && rp.is_none())
    );
}

#[test]
fn invariants_hold() {
    let mut g = InMemorySieve::<PointId, ()>::default();
    let p = |x| PointId::new(x).unwrap();
    g.add_arrow(p(1), p(2), ());
    g.add_arrow(p(2), p(3), ());
    let parts = vec![0, 1, 1];
    let comm = NoComm;
    let (_local, ov) = distribute_mesh(&g, &parts, &comm).unwrap();

    #[cfg(any(
        debug_assertions,
        feature = "strict-invariants",
        feature = "check-invariants"
    ))]
    ov.validate_invariants().unwrap();
}

#[test]
fn preserves_non_base_origin_edges() {
    let mut g = NonBaseOriginSieve::default();
    let p = |x| PointId::new(x).unwrap();
    g.add_arrow(p(1), p(2), ());
    g.add_arrow(p(2), p(3), ());
    let parts = vec![0, 0, 0];
    let comm = NoComm;

    let (local, _ov) = distribute_mesh(&g, &parts, &comm).unwrap();

    assert!(local.has_arrow(p(1), p(2)));
    assert!(local.has_arrow(p(2), p(3)));
}
