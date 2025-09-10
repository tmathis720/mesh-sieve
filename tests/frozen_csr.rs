use mesh_sieve::mesh_error::MeshSieveError;
use mesh_sieve::topology::sieve::{InMemorySieve, Sieve, try_freeze_csr};

#[test]
fn try_freeze_ok() {
    let mut s = InMemorySieve::<u32, ()>::default();
    s.add_arrow(1, 2, ());
    s.add_arrow(2, 3, ());
    let csr = try_freeze_csr(s).expect("should freeze");
    let v: Vec<_> = csr.cone(1).map(|(q, _)| q).collect();
    assert_eq!(v, vec![2]);
}

#[test]
fn unknown_point_is_non_panicking_and_empty() {
    let mut s = InMemorySieve::<u32, ()>::default();
    s.add_arrow(1, 2, ());
    let csr = try_freeze_csr(s).unwrap();

    let v: Vec<_> = csr.cone(99).collect();
    assert!(v.is_empty());

    let err = csr.cone_checked(99).err().unwrap();
    assert!(matches!(err, MeshSieveError::UnknownPoint(_)));
}

#[test]
fn try_freeze_err_on_missing_point_in_cone() {
    use mesh_sieve::topology::cache::InvalidateCache;
    use mesh_sieve::topology::sieve::sieve_trait::Sieve as SieveTrait;

    struct Bad;
    impl SieveTrait for Bad {
        type Point = u32;
        type Payload = ();
        type ConeIter<'a> = std::iter::Once<(u32, ())>;
        type SupportIter<'a> = std::iter::Empty<(u32, ())>;
        fn cone<'a>(&'a self, _p: u32) -> Self::ConeIter<'a> {
            std::iter::once((42, ()))
        }
        fn support<'a>(&'a self, _p: u32) -> Self::SupportIter<'a> {
            std::iter::empty()
        }
        fn add_arrow(&mut self, _: u32, _: u32, _: ()) {
            unimplemented!()
        }
        fn remove_arrow(&mut self, _: u32, _: u32) -> Option<()> {
            unimplemented!()
        }
        fn base_points<'a>(&'a self) -> Box<dyn Iterator<Item = u32> + 'a> {
            Box::new([1u32].into_iter())
        }
        fn cap_points<'a>(&'a self) -> Box<dyn Iterator<Item = u32> + 'a> {
            Box::new(std::iter::empty())
        }
        fn points_chart_order(&mut self) -> Result<Vec<u32>, MeshSieveError> {
            Ok(vec![1])
        }
    }
    impl InvalidateCache for Bad {
        fn invalidate_cache(&mut self) {}
    }
    impl Default for Bad {
        fn default() -> Self {
            Bad
        }
    }

    let err = try_freeze_csr::<_, u32, ()>(Bad).unwrap_err();
    assert!(matches!(err, MeshSieveError::MissingPointInCone(_)));
}
