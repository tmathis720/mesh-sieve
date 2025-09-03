use mesh_sieve::mesh_error::MeshSieveError;
use mesh_sieve::topology::sieve::{InMemoryOrientedSieve, InMemorySieve, Sieve};

#[test]
fn strata_single_cache_path() -> Result<(), Box<dyn std::error::Error>> {
    let mut s = InMemorySieve::<u32, ()>::default();
    s.add_arrow(1, 2, ());
    s.add_arrow(2, 3, ());
    // First call populates cache
    assert_eq!(s.diameter()?, 2);
    // Mutate -> cache invalidated
    s.add_arrow(3, 4, ());
    assert_eq!(s.diameter()?, 3);
    Ok(())
}

#[test]
fn strata_consistency_oriented() -> Result<(), Box<dyn std::error::Error>> {
    let mut s = InMemoryOrientedSieve::<u32, (), i32>::default();
    s.add_arrow(1, 2, ());
    s.add_arrow(2, 3, ());
    assert_eq!(s.diameter()?, 2);
    s.add_arrow(3, 4, ());
    assert_eq!(s.diameter()?, 3);
    Ok(())
}

#[test]
fn strata_cycle_error() {
    let mut s = InMemorySieve::<u32, ()>::default();
    s.add_arrow(1, 2, ());
    s.add_arrow(2, 1, ());
    assert!(matches!(s.diameter(), Err(MeshSieveError::CycleDetected)));
}
