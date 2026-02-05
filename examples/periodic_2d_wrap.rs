// cargo run --example periodic_2d_wrap
use mesh_sieve::mesh_error::MeshSieveError;
use mesh_sieve::topology::periodic::{PeriodicMap, collapse_points, quotient_sieve};
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::sieve::{InMemorySieve, Sieve};

fn main() -> Result<(), MeshSieveError> {
    let mut sieve = InMemorySieve::<PointId, ()>::default();

    let v = |i| PointId::new(i).unwrap();
    let c = |i| PointId::new(100_u64 + i).unwrap();

    // 2x1 quad mesh (two cells) with vertices laid out on x = 0, 1, 2.
    // Cell c0 uses vertices v0, v1, v4, v3.
    // Cell c1 uses vertices v1, v2, v5, v4.
    let (v0, v1, v2, v3, v4, v5) = (v(1), v(2), v(3), v(4), v(5), v(6));
    let (c0, c1) = (c(0), c(1));

    for &vertex in &[v0, v1, v4, v3] {
        Sieve::add_arrow(&mut sieve, c0, vertex, ());
    }
    for &vertex in &[v1, v2, v5, v4] {
        Sieve::add_arrow(&mut sieve, c1, vertex, ());
    }

    // Wrap the x-boundary (x = 2) to x = 0.
    let mut periodic = PeriodicMap::new();
    periodic.insert_pair(v0, v2)?;
    periodic.insert_pair(v3, v5)?;

    let mut equivalence = periodic.equivalence();
    let collapsed = collapse_points([v0, v1, v2, v3, v4, v5], &mut equivalence);
    println!("Collapsed vertex map: {collapsed:?}");

    let mut equivalence = periodic.equivalence();
    let quotient = quotient_sieve(&sieve, &mut equivalence);
    let quotient_points: Vec<_> = quotient.points().collect();
    println!("Quotient sieve points: {quotient_points:?}");

    Ok(())
}
