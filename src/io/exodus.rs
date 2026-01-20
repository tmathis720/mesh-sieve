//! Exodus mesh reader.

use crate::data::storage::VecStorage;
use crate::io::{MeshData, SieveSectionReader};
use crate::mesh_error::MeshSieveError;
use crate::topology::cell_type::CellType;
use crate::topology::point::PointId;
use crate::topology::sieve::InMemorySieve;
use std::io::Read;

/// Placeholder Exodus reader.
#[derive(Debug, Default, Clone)]
pub struct ExodusReader;

impl SieveSectionReader for ExodusReader {
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
            "Exodus reader not yet implemented".into(),
        ))
    }
}
