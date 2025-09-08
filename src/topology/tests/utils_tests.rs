use crate::mesh_error::MeshSieveError;
use crate::topology::sieve::{InMemorySieve, Sieve};
use crate::topology::utils::check_dag;

#[test]
fn check_dag_accepts_simple_dag() -> Result<(), Box<dyn std::error::Error>> {
    let mut s = InMemorySieve::<u32, ()>::default();
    // 1 → 2 → 3, 1 → 4
    s.add_arrow(1, 2, ());
    s.add_arrow(2, 3, ());
    s.add_arrow(1, 4, ());
    assert!(check_dag(&s).is_ok());
    Ok(())
}

#[test]
fn check_dag_detects_cycle() {
    let mut s = InMemorySieve::<u32, ()>::default();
    s.add_arrow(1, 2, ());
    s.add_arrow(2, 1, ());
    let err = check_dag(&s).unwrap_err();
    assert!(matches!(err, MeshSieveError::CycleDetected));
}

#[test]
fn check_dag_handles_isolated_points() -> Result<(), Box<dyn std::error::Error>> {
    let mut s = InMemorySieve::<u32, ()>::default();
    s.add_point(10); // isolated
    s.add_arrow(1, 2, ());
    assert!(check_dag(&s).is_ok());
    Ok(())
}

#[test]
fn check_dag_oriented_backend() -> Result<(), Box<dyn std::error::Error>> {
    use crate::topology::sieve::in_memory_oriented::InMemoryOrientedSieve;
    let mut s = InMemoryOrientedSieve::<u32, (), i32>::default();
    s.add_arrow(1, 2, ());
    s.add_arrow(2, 3, ());
    assert!(check_dag(&s).is_ok());
    s.add_arrow(3, 1, ());
    assert!(matches!(check_dag(&s), Err(MeshSieveError::CycleDetected)));
    Ok(())
}

#[test]
fn check_dag_ref_accepts_simple_dag() -> Result<(), Box<dyn std::error::Error>> {
    use crate::topology::utils::check_dag_ref;
    let mut s = InMemorySieve::<u32, ()>::default();
    s.add_arrow(1, 2, ());
    s.add_arrow(2, 3, ());
    assert!(check_dag_ref(&s).is_ok());
    Ok(())
}
