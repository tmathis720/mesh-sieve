use mesh_sieve::data::section::Section;

// No Map here; just ensure fallible API is available.
#[test]
fn fallible_core_is_present() {
    let atlas = mesh_sieve::data::atlas::Atlas::default();
    let _sec: Section<u32> = Section::new(atlas);
}
