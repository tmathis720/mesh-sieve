use mesh_sieve::mesh_error::MeshSieveError;
use mesh_sieve::topology::cache::InvalidateCache;
use mesh_sieve::topology::sieve::Sieve;
use mesh_sieve::topology::sieve::in_memory::InMemorySieve;
use mesh_sieve::topology::sieve::strata::compute_strata;
use std::cell::RefCell;
use std::collections::HashMap;

// Wrapper that delegates everything except `points()` (returns empty).
#[derive(Default)]
struct NoPoints<S: Sieve>(S);

impl<S: Sieve> InvalidateCache for NoPoints<S> {
    fn invalidate_cache(&mut self) {
        self.0.invalidate_cache()
    }
}

impl<S: Sieve> Sieve for NoPoints<S> {
    type Point = S::Point;
    type Payload = S::Payload;
    type ConeIter<'a>
        = S::ConeIter<'a>
    where
        S: 'a;
    type SupportIter<'a>
        = S::SupportIter<'a>
    where
        S: 'a;

    fn cone<'a>(&'a self, p: Self::Point) -> Self::ConeIter<'a> {
        self.0.cone(p)
    }
    fn support<'a>(&'a self, p: Self::Point) -> Self::SupportIter<'a> {
        self.0.support(p)
    }
    fn add_arrow(&mut self, src: Self::Point, dst: Self::Point, payload: Self::Payload) {
        self.0.add_arrow(src, dst, payload)
    }
    fn remove_arrow(&mut self, src: Self::Point, dst: Self::Point) -> Option<Self::Payload> {
        self.0.remove_arrow(src, dst)
    }
    fn base_points<'a>(&'a self) -> Box<dyn Iterator<Item = Self::Point> + 'a> {
        self.0.base_points()
    }
    fn cap_points<'a>(&'a self) -> Box<dyn Iterator<Item = Self::Point> + 'a> {
        self.0.cap_points()
    }
    fn points<'a>(&'a self) -> Box<dyn Iterator<Item = Self::Point> + 'a> {
        Box::new(std::iter::empty())
    }
}

#[test]
fn strata_ignores_points_method() {
    let mut inner = InMemorySieve::<u32, ()>::new();
    inner.add_arrow(1, 2, ());
    inner.add_arrow(2, 3, ());
    let wrapper = NoPoints(inner);
    let strata = compute_strata(&wrapper).unwrap();
    assert_eq!(strata.height.get(&3), Some(&2));
}

// Sieve with manually specified role sets (possibly inconsistent)
struct BrokenRoles {
    inner: InMemorySieve<u32, ()>,
    base: Vec<u32>,
    cap: Vec<u32>,
}

impl Default for BrokenRoles {
    fn default() -> Self {
        Self {
            inner: InMemorySieve::new(),
            base: Vec::new(),
            cap: Vec::new(),
        }
    }
}

impl InvalidateCache for BrokenRoles {
    fn invalidate_cache(&mut self) {
        self.inner.invalidate_cache()
    }
}

impl Sieve for BrokenRoles {
    type Point = u32;
    type Payload = ();
    type ConeIter<'a>
        = <InMemorySieve<u32, ()> as Sieve>::ConeIter<'a>
    where
        Self: 'a;
    type SupportIter<'a>
        = <InMemorySieve<u32, ()> as Sieve>::SupportIter<'a>
    where
        Self: 'a;

    fn cone<'a>(&'a self, p: u32) -> Self::ConeIter<'a> {
        self.inner.cone(p)
    }
    fn support<'a>(&'a self, p: u32) -> Self::SupportIter<'a> {
        self.inner.support(p)
    }
    fn add_arrow(&mut self, src: u32, dst: u32, payload: ()) {
        self.inner.add_arrow(src, dst, payload)
    }
    fn remove_arrow(&mut self, src: u32, dst: u32) -> Option<()> {
        self.inner.remove_arrow(src, dst)
    }
    fn base_points<'a>(&'a self) -> Box<dyn Iterator<Item = u32> + 'a> {
        Box::new(self.base.clone().into_iter())
    }
    fn cap_points<'a>(&'a self) -> Box<dyn Iterator<Item = u32> + 'a> {
        Box::new(self.cap.clone().into_iter())
    }
}

#[test]
fn aggregates_missing_vertices() {
    let mut s = BrokenRoles::default();
    s.inner.add_arrow(1, 2, ());
    s.inner.add_arrow(3, 4, ());
    s.inner.add_arrow(5, 6, ());
    s.base = vec![1, 3]; // omit src 5
    s.cap = vec![2, 6]; // omit dst 4
    let err = compute_strata(&s).unwrap_err();
    assert!(matches!(err, MeshSieveError::MissingPointInCone(_)));
    let msg = err.to_string();
    assert!(msg.contains("2 missing total"));
}

// Wrapper that panics if `cone` is called more than once per vertex
struct PanicOnSecondCone {
    inner: InMemorySieve<u32, ()>,
    base: Vec<u32>,
    cap: Vec<u32>,
    counts: RefCell<HashMap<u32, u32>>,
}

impl Default for PanicOnSecondCone {
    fn default() -> Self {
        Self {
            inner: InMemorySieve::new(),
            base: Vec::new(),
            cap: Vec::new(),
            counts: RefCell::new(HashMap::new()),
        }
    }
}

impl InvalidateCache for PanicOnSecondCone {
    fn invalidate_cache(&mut self) {
        self.inner.invalidate_cache()
    }
}

impl Sieve for PanicOnSecondCone {
    type Point = u32;
    type Payload = ();
    type ConeIter<'a>
        = <InMemorySieve<u32, ()> as Sieve>::ConeIter<'a>
    where
        Self: 'a;
    type SupportIter<'a>
        = <InMemorySieve<u32, ()> as Sieve>::SupportIter<'a>
    where
        Self: 'a;

    fn cone<'a>(&'a self, p: u32) -> Self::ConeIter<'a> {
        let mut c = self.counts.borrow_mut();
        let e = c.entry(p).or_insert(0);
        *e += 1;
        assert!(*e <= 1, "Kahn pass should not start after validation error");
        self.inner.cone(p)
    }
    fn support<'a>(&'a self, p: u32) -> Self::SupportIter<'a> {
        self.inner.support(p)
    }
    fn add_arrow(&mut self, src: u32, dst: u32, payload: ()) {
        self.inner.add_arrow(src, dst, payload)
    }
    fn remove_arrow(&mut self, src: u32, dst: u32) -> Option<()> {
        self.inner.remove_arrow(src, dst)
    }
    fn base_points<'a>(&'a self) -> Box<dyn Iterator<Item = u32> + 'a> {
        Box::new(self.base.clone().into_iter())
    }
    fn cap_points<'a>(&'a self) -> Box<dyn Iterator<Item = u32> + 'a> {
        Box::new(self.cap.clone().into_iter())
    }
}

#[test]
fn fails_before_kahn() {
    let mut s = PanicOnSecondCone::default();
    s.inner.add_arrow(1, 3, ());
    s.base = vec![1];
    s.cap = vec![2]; // dst 3 missing
    let err = compute_strata(&s).unwrap_err();
    assert!(matches!(err, MeshSieveError::MissingPointInCone(_)));
}
