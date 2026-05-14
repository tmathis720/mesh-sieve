//! CGNS mesh reader.
//!
//! CGNS is a broad HDF5/ADF standard with several topology encodings.  The
//! `CgnsReader` entry point is feature-gated so downstream crates do not
//! accidentally rely on a placeholder implementation. Enable the `cgns` feature
//! to opt into the experimental HDF5-backed reader as it grows.

use crate::data::storage::VecStorage;
use crate::io::{MeshData, SieveSectionReader};
use crate::mesh_error::MeshSieveError;
use crate::topology::cell_type::CellType;
use crate::topology::sieve::MeshSieve;
use std::io::Read;

/// CGNS reader entry point.
#[derive(Debug, Default, Clone)]
pub struct CgnsReader;

#[cfg(not(feature = "cgns"))]
impl SieveSectionReader for CgnsReader {
    type Sieve = MeshSieve;
    type Value = f64;
    type Storage = VecStorage<f64>;
    type CellStorage = VecStorage<CellType>;

    fn read<R: Read>(
        &self,
        _reader: R,
    ) -> Result<MeshData<Self::Sieve, Self::Value, Self::Storage, Self::CellStorage>, MeshSieveError>
    {
        Err(MeshSieveError::MeshIoParse(
            "CGNS support is not compiled in; rebuild mesh-sieve with `--features cgns`".into(),
        ))
    }
}

#[cfg(feature = "cgns")]
impl SieveSectionReader for CgnsReader {
    type Sieve = MeshSieve;
    type Value = f64;
    type Storage = VecStorage<f64>;
    type CellStorage = VecStorage<CellType>;

    fn read<R: Read>(
        &self,
        _reader: R,
    ) -> Result<MeshData<Self::Sieve, Self::Value, Self::Storage, Self::CellStorage>, MeshSieveError>
    {
        Err(MeshSieveError::MeshIoParse(
            "experimental CGNS feature is enabled, but full CGNS/HDF5 topology import is not available in this build"
                .into(),
        ))
    }
}
