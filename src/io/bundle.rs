//! Mesh bundle container for multiple mesh topologies.

use std::collections::{BTreeMap, HashSet};

use crate::data::storage::Storage;
use crate::mesh_error::MeshSieveError;
use crate::topology::cell_type::CellType;
use crate::topology::labels::LabelSet;
use crate::topology::point::PointId;
use crate::topology::sieve::Sieve;

use super::MeshData;

/// Collection of mesh topologies with separate section data.
#[derive(Debug)]
pub struct MeshBundle<S, V, St, CtSt>
where
    St: Storage<V>,
    CtSt: Storage<CellType>,
{
    /// Individual mesh containers.
    pub meshes: Vec<MeshData<S, V, St, CtSt>>,
}

impl<S, V, St, CtSt> MeshBundle<S, V, St, CtSt>
where
    St: Storage<V>,
    CtSt: Storage<CellType>,
{
    /// Construct a bundle from a list of meshes.
    pub fn new(meshes: Vec<MeshData<S, V, St, CtSt>>) -> Self {
        Self { meshes }
    }

    /// Add a mesh to the bundle.
    pub fn push(&mut self, mesh: MeshData<S, V, St, CtSt>) {
        self.meshes.push(mesh);
    }

    /// Borrow all meshes immutably.
    pub fn meshes(&self) -> &[MeshData<S, V, St, CtSt>] {
        &self.meshes
    }

    /// Borrow all meshes mutably.
    pub fn meshes_mut(&mut self) -> &mut [MeshData<S, V, St, CtSt>] {
        &mut self.meshes
    }
}

impl<S, V, St, CtSt> MeshBundle<S, V, St, CtSt>
where
    S: Sieve<Point = PointId>,
    St: Storage<V>,
    CtSt: Storage<CellType>,
{
    /// Synchronize label entries for shared points across all meshes.
    ///
    /// For every label entry appearing in any mesh, this updates every other
    /// mesh that contains the same point to carry that label value.
    ///
    /// Returns the number of label entries applied across meshes.
    pub fn sync_labels(&mut self) -> usize {
        let mut combined: Vec<(String, PointId, i32)> = Vec::new();
        for mesh in &self.meshes {
            if let Some(labels) = &mesh.labels {
                for (name, point, value) in labels.iter() {
                    combined.push((name.to_string(), point, value));
                }
            }
        }

        combined.sort_by(|a, b| {
            a.0.cmp(&b.0)
                .then_with(|| a.1.cmp(&b.1))
                .then_with(|| a.2.cmp(&b.2))
        });

        let mut applied = 0usize;
        for mesh in &mut self.meshes {
            if combined.is_empty() {
                break;
            }
            let points: HashSet<PointId> = mesh.sieve.points().collect();
            if points.is_empty() {
                continue;
            }
            let labels = mesh.labels.get_or_insert_with(LabelSet::new);
            for (name, point, value) in &combined {
                if points.contains(point) {
                    labels.set_label(*point, name, *value);
                    applied += 1;
                }
            }
        }

        applied
    }
}

impl<S, V, St, CtSt> MeshBundle<S, V, St, CtSt>
where
    S: Sieve<Point = PointId>,
    V: Clone + Default + PartialEq,
    St: Storage<V> + Clone,
    CtSt: Storage<CellType>,
{
    /// Synchronize coordinate values for shared points across all meshes.
    ///
    /// Returns an error if coordinate dimensions differ or if conflicting
    /// values are detected for the same point.
    pub fn sync_coordinates(&mut self) -> Result<(), MeshSieveError> {
        let mut dimension: Option<usize> = None;
        let mut values: BTreeMap<PointId, Vec<V>> = BTreeMap::new();

        for mesh in &self.meshes {
            let Some(coords) = &mesh.coordinates else {
                continue;
            };
            if let Some(existing_dim) = dimension {
                if coords.dimension() != existing_dim {
                    return Err(MeshSieveError::InvalidGeometry(format!(
                        "coordinate dimension mismatch: expected {existing_dim}, found {}",
                        coords.dimension()
                    )));
                }
            } else {
                dimension = Some(coords.dimension());
            }

            for point in coords.section().atlas().points() {
                let slice = coords.try_restrict(point)?;
                if let Some(existing) = values.get(&point) {
                    if existing.as_slice() != slice {
                        return Err(MeshSieveError::InvalidGeometry(format!(
                            "conflicting coordinate values for point {point:?}"
                        )));
                    }
                } else {
                    values.insert(point, slice.to_vec());
                }
            }
        }

        if values.is_empty() {
            return Ok(());
        }

        for mesh in &mut self.meshes {
            let Some(coords) = &mut mesh.coordinates else {
                continue;
            };
            for point in coords.section().atlas().points() {
                if let Some(val) = values.get(&point) {
                    coords.section_mut().try_set(point, val)?;
                }
            }
        }

        Ok(())
    }
}
