use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::bundle::Bundle;
use mesh_sieve::data::section::Section;
use mesh_sieve::overlap::delta::CopyDelta;
use mesh_sieve::topology::arrow::Polarity;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::sieve::Sieve;
use mesh_sieve::topology::stack::{InMemoryStack, Stack};

#[test]
fn refine_disjoint_slices_no_allocations() -> Result<(), Box<dyn std::error::Error>> {
    let b = PointId::new(1)?;
    let c = PointId::new(2)?;
    let mut atlas = Atlas::default();
    atlas.try_insert(b, 3)?;
    atlas.try_insert(c, 3)?;
    let mut section = Section::<i32>::new(atlas);
    section.try_set(b, &[10, 20, 30])?;

    let mut stack = InMemoryStack::<PointId, PointId, Polarity>::default();
    stack.base_mut().unwrap().add_arrow(b, b, ());
    stack.cap_mut().unwrap().add_arrow(c, c, ());
    stack.add_arrow(b, c, Polarity::Forward)?;

    let mut bundle = Bundle {
        stack,
        section,
        delta: CopyDelta,
    };
    bundle.refine([b])?;

    let cap_vals = bundle.section.try_restrict(c)?;
    assert_eq!(cap_vals, &[10, 20, 30]);
    Ok(())
}

#[test]
fn refine_overlapping_slices_safe_reverse() -> Result<(), Box<dyn std::error::Error>> {
    let p = PointId::new(42)?;
    let mut atlas = Atlas::default();
    atlas.try_insert(p, 4)?;
    let mut section = Section::<i32>::new(atlas);
    section.try_set(p, &[1, 2, 3, 4])?;

    let mut stack = InMemoryStack::<PointId, PointId, Polarity>::default();
    stack.base_mut().unwrap().add_arrow(p, p, ());
    stack.cap_mut().unwrap().add_arrow(p, p, ());
    stack.add_arrow(p, p, Polarity::Reverse)?;

    let mut bundle = Bundle {
        stack,
        section,
        delta: CopyDelta,
    };
    bundle.refine([p])?;

    let vals = bundle.section.try_restrict(p)?;
    assert_eq!(vals, &[4, 3, 2, 1]);
    Ok(())
}
