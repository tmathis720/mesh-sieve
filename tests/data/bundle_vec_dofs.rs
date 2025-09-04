use mesh_sieve::data::{atlas::Atlas, bundle::Bundle, section::Section};
use mesh_sieve::overlap::delta::CopyDelta;
use mesh_sieve::topology::arrow::Orientation;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::stack::{InMemoryStack, Stack};

#[test]
fn bundle_refine_with_forward_and_reverse() -> Result<(), Box<dyn std::error::Error>> {
    let b = PointId::new(10)?;
    let c1 = PointId::new(11)?;
    let c2 = PointId::new(12)?;
    let mut atlas = Atlas::default();
    atlas.try_insert(b, 3)?;
    atlas.try_insert(c1, 3)?;
    atlas.try_insert(c2, 3)?;

    let mut section = Section::<i32>::new(atlas.clone());
    section.try_set(b, &[1, 2, 3])?;

    let mut stack = InMemoryStack::<PointId, PointId, Orientation>::default();
    stack.add_arrow(b, c1, Orientation::Forward)?;
    stack.add_arrow(b, c2, Orientation::Reverse)?;

    let mut bundle = Bundle { stack, section, delta: CopyDelta };
    bundle.refine([b])?;

    assert_eq!(bundle.section.try_restrict(c1)?, &[1, 2, 3]);
    assert_eq!(bundle.section.try_restrict(c2)?, &[3, 2, 1]);
    Ok(())
}

#[test]
fn bundle_assemble_elementwise_average() -> Result<(), Box<dyn std::error::Error>> {
    let b = PointId::new(20)?;
    let c1 = PointId::new(21)?;
    let c2 = PointId::new(22)?;
    let mut atlas = Atlas::default();
    atlas.try_insert(b, 3)?;
    atlas.try_insert(c1, 3)?;
    atlas.try_insert(c2, 3)?;
    let mut section = Section::<f64>::new(atlas.clone());

    let mut stack = InMemoryStack::<PointId, PointId, Orientation>::default();
    stack.add_arrow(b, c1, Orientation::Forward)?;
    stack.add_arrow(b, c2, Orientation::Forward)?;
    let mut bundle = Bundle { stack, section, delta: CopyDelta };

    bundle.section.try_set(c1, &[2.0, 4.0, 6.0])?;
    bundle.section.try_set(c2, &[6.0, 8.0, 10.0])?;

    bundle.assemble([b])?;

    assert_eq!(bundle.section.try_restrict(b)?, &[4.0, 6.0, 8.0]);
    Ok(())
}
