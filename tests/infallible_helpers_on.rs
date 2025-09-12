#[cfg(feature = "map-adapter")]
#[test]
fn infallible_helpers_exist() {
    use mesh_sieve::data::{
        atlas::Atlas, refine::restrict_closure_vec, section::Section, storage::VecStorage,
    };
    use mesh_sieve::topology::{point::PointId, sieve::in_memory::InMemorySieve};
    let s = InMemorySieve::<PointId, ()>::default();
    let sec: Section<u8, VecStorage<u8>> = Section::new(Atlas::default());
    let _ = restrict_closure_vec(&s, &sec, std::iter::empty::<PointId>());
}
