//! Mesh transformation helpers for coordinate updates.

use crate::data::section::Section;
use crate::data::storage::Storage;
use crate::io::MeshData;
use crate::mesh_error::MeshSieveError;
use crate::topology::cell_type::CellType;
use crate::topology::point::PointId;
use crate::topology::sieve::Sieve;

/// Coordinate update strategies for mesh transforms.
pub enum CoordinateTransform<'a, St>
where
    St: Storage<f64>,
{
    /// Update coordinates using a user-supplied function.
    ///
    /// The function receives the point ID and a mutable slice of the
    /// coordinate tuple for that point.
    Function(&'a mut dyn FnMut(PointId, &mut [f64]) -> Result<(), MeshSieveError>),
    /// Update coordinates by adding a displacement section.
    ///
    /// Displacement slices must match the coordinate dimension for each point.
    Displacement(&'a Section<f64, St>),
}

/// Hook set for updating derived data after coordinate transforms.
pub struct TransformHooks<'a, M, St, CtSt>
where
    St: Storage<f64>,
    CtSt: Storage<CellType>,
{
    /// Invoked after coordinates are updated.
    pub after_update:
        Option<&'a mut dyn FnMut(&MeshData<M, f64, St, CtSt>) -> Result<(), MeshSieveError>>,
}

/// Apply a coordinate transformation to a mesh, leaving topology unchanged.
pub fn transform_mesh<M, St, CtSt>(
    mesh: &mut MeshData<M, f64, St, CtSt>,
    transform: CoordinateTransform<'_, St>,
    mut hooks: TransformHooks<'_, M, St, CtSt>,
) -> Result<(), MeshSieveError>
where
    M: Sieve<Point = PointId>,
    St: Storage<f64>,
    CtSt: Storage<CellType>,
{
    let coords = mesh
        .coordinates
        .as_mut()
        .ok_or_else(|| MeshSieveError::InvalidGeometry("mesh is missing coordinates".into()))?;
    let dim = coords.dimension();
    let points: Vec<PointId> = coords.section().atlas().points().collect();

    match transform {
        CoordinateTransform::Function(update) => {
            for point in points {
                let slice = coords.try_restrict_mut(point)?;
                if slice.len() != dim {
                    return Err(MeshSieveError::SliceLengthMismatch {
                        point,
                        expected: dim,
                        found: slice.len(),
                    });
                }
                update(point, slice)?;
            }
        }
        CoordinateTransform::Displacement(displacement) => {
            for point in points {
                let disp = displacement.try_restrict(point)?;
                if disp.len() != dim {
                    return Err(MeshSieveError::SliceLengthMismatch {
                        point,
                        expected: dim,
                        found: disp.len(),
                    });
                }
                let slice = coords.try_restrict_mut(point)?;
                if slice.len() != dim {
                    return Err(MeshSieveError::SliceLengthMismatch {
                        point,
                        expected: dim,
                        found: slice.len(),
                    });
                }
                for (coord, delta) in slice.iter_mut().zip(disp.iter()) {
                    *coord += *delta;
                }
            }
        }
    }

    if let Some(after_update) = hooks.after_update.as_mut() {
        after_update(mesh)?;
    }

    Ok(())
}
