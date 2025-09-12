use mesh_sieve::data::{
    atlas::Atlas, refine::try_restrict_closure_vec, section::Section, storage::VecStorage,
};
use mesh_sieve::topology::{point::PointId, sieve::in_memory::InMemorySieve};

#[test]
fn fallible_helpers_exist() {
    let s = InMemorySieve::<PointId, ()>::default();
    let sec: Section<u8, VecStorage<u8>> = Section::new(Atlas::default());
    let _ = try_restrict_closure_vec(&s, &sec, std::iter::empty::<PointId>());
}
