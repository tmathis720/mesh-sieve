use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::bundle::{Bundle, SliceReducer};
use mesh_sieve::data::section::Section;
use mesh_sieve::overlap::delta::CopyDelta;
use mesh_sieve::topology::arrow::Orientation;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::stack::InMemoryStack;
use mesh_sieve::mesh_error::MeshSieveError;

#[derive(Copy, Clone, Default)]
struct SumReducer;

impl<V> SliceReducer<V> for SumReducer
where
    V: Clone + Default + core::ops::AddAssign,
{
    fn make_zero(&self, len: usize) -> Vec<V> {
        vec![V::default(); len]
    }

    fn accumulate(
        &self,
        acc: &mut [V],
        src: &[V],
    ) -> Result<(), MeshSieveError> {
        if acc.len() != src.len() {
            return Err(MeshSieveError::SliceLengthMismatch {
                point: unsafe { PointId::new_unchecked(1) },
                expected: acc.len(),
                found: src.len(),
            });
        }
        for (dst, s) in acc.iter_mut().zip(src.iter()) {
            *dst += s.clone();
        }
        Ok(())
    }
}

#[test]
fn bundle_assemble_with_sum() -> Result<(), Box<dyn std::error::Error>> {
    let b = PointId::new(30)?;
    let c1 = PointId::new(31)?;
    let c2 = PointId::new(32)?;
    let mut atlas = Atlas::default();
    atlas.try_insert(b, 3)?;
    atlas.try_insert(c1, 3)?;
    atlas.try_insert(c2, 3)?;
    let mut section = Section::<i32>::new(atlas.clone());

    let mut stack = InMemoryStack::<PointId, PointId, Orientation>::default();
    stack.add_arrow(b, c1, Orientation::Forward)?;
    stack.add_arrow(b, c2, Orientation::Forward)?;
    let mut bundle = Bundle { stack, section, delta: CopyDelta };

    bundle.section.try_set(c1, &[2, 4, 6])?;
    bundle.section.try_set(c2, &[6, 8, 10])?;

    bundle.assemble_with([b], &SumReducer::default())?;

    assert_eq!(bundle.section.try_restrict(b)?, &[8, 12, 16]);
    Ok(())
}
