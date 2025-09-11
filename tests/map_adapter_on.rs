#[cfg(feature = "map-adapter")]
#[test]
fn map_trait_is_exposed() {
    use mesh_sieve::data::section::{Map, Section};
    let atlas = mesh_sieve::data::atlas::Atlas::default();
    let sec: Section<u32> = Section::new(atlas);
    // compile-time check that Map is in scope
    fn takes_map<M: Map<u32>>(_m: &M) {}
    takes_map(&sec);
}
