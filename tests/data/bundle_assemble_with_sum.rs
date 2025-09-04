use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::bundle::{Bundle, Reducer};
use mesh_sieve::data::section::Section;
use mesh_sieve::overlap::delta::CopyDelta;
use mesh_sieve::topology::arrow::Orientation;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::stack::InMemoryStack;
use mesh_sieve::mesh_error::MeshSieveError;

#[derive(Copy, Clone, Default)]
struct SumReducer;

impl<V> Reducer<V> for SumReducer
where
    V: Clone + Default + core::ops::AddAssign,
{
    fn reduce_into(&self, base: &mut [V], caps: &[&[V]]) -> Result<(), MeshSieveError> {
        if caps.is_empty() {
            return Ok(());
        }
        let k = base.len();
        for c in caps {
            if c.len() != k {
                return Err(MeshSieveError::SliceLengthMismatch {
                    point: unsafe { PointId::new_unchecked(1) },
                    expected: k,
                    found: c.len(),
                });
            }
        }
        for v in base.iter_mut() {
            *v = V::default();
        }
        for c in caps {
            for (dst, src) in base.iter_mut().zip(c.iter()) {
                *dst += src.clone();
            }
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
