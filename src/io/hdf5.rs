//! HDF5-based mesh reader.

use crate::data::storage::VecStorage;
use crate::io::{MeshData, SieveSectionReader};
use crate::mesh_error::MeshSieveError;
use crate::topology::cell_type::CellType;
use crate::topology::point::PointId;
use crate::topology::sieve::InMemorySieve;
use std::io::Read;

/// Placeholder HDF5 reader.
#[derive(Debug, Default, Clone)]
pub struct Hdf5Reader;

impl SieveSectionReader for Hdf5Reader {
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
            "HDF5 reader not yet implemented".into(),
        ))
    }
}
