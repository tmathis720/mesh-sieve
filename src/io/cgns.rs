//! CGNS mesh reader.

use crate::data::storage::VecStorage;
use crate::io::{MeshData, SieveSectionReader};
use crate::mesh_error::MeshSieveError;
use crate::topology::cell_type::CellType;
use crate::topology::point::PointId;
use crate::topology::sieve::InMemorySieve;
use std::io::Read;

/// Placeholder CGNS reader.
#[derive(Debug, Default, Clone)]
pub struct CgnsReader;

impl SieveSectionReader for CgnsReader {
    type Sieve = InMemorySieve<PointId, ()>;
    type Value = f64;
    type Storage = VecStorage<f64>;
    type CellStorage = VecStorage<CellType>;

    fn read<R: Read>(
        &self,
        _reader: R,
    ) -> Result<MeshData<Self::Sieve, Self::Value, Self::Storage, Self::CellStorage>, MeshSieveError>
    {
        Err(MeshSieveError::MeshIoParse(
            "CGNS reader not yet implemented".into(),
        ))
    }
}
