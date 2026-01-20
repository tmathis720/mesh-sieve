//! Mesh I/O helpers for sieve topology and sections.
//!
//! This module provides trait-based readers and writers for loading and
//! saving `Sieve` topologies together with associated `Section` data.

pub mod cgns;
pub mod exodus;
pub mod gmsh;
pub mod hdf5;

use crate::data::coordinates::Coordinates;
use crate::data::mixed_section::MixedSectionStore;
use crate::data::section::Section;
use crate::data::storage::Storage;
use crate::mesh_error::MeshSieveError;
use crate::topology::cell_type::CellType;
use crate::topology::labels::LabelSet;
use crate::topology::sieve::Sieve;
use std::collections::BTreeMap;
use std::io::{Read, Write};

/// Combined sieve and section data returned by I/O readers.
#[derive(Debug)]
pub struct MeshData<S, V, St, CtSt>
where
    St: Storage<V>,
    CtSt: Storage<CellType>,
{
    /// The mesh topology as a sieve.
    pub sieve: S,
    /// Optional coordinate section wrapper.
    pub coordinates: Option<Coordinates<V, St>>,
    /// Named sections keyed by user-provided identifiers.
    pub sections: BTreeMap<String, Section<V, St>>,
    /// Tagged sections with mixed scalar types.
    pub mixed_sections: MixedSectionStore,
    /// Optional integer labels associated with mesh points.
    pub labels: Option<LabelSet>,
    /// Optional cell type section over mesh points.
    pub cell_types: Option<Section<CellType, CtSt>>,
}

impl<S, V, St, CtSt> MeshData<S, V, St, CtSt>
where
    St: Storage<V>,
    CtSt: Storage<CellType>,
{
    /// Create an empty container with a sieve and no sections.
    pub fn new(sieve: S) -> Self {
        Self {
            sieve,
            coordinates: None,
            sections: BTreeMap::new(),
            mixed_sections: MixedSectionStore::default(),
            labels: None,
            cell_types: None,
        }
    }
}

/// Trait for mesh readers that produce sieve + section data.
pub trait SieveSectionReader {
    /// Sieve implementation returned by the reader.
    type Sieve: Sieve;
    /// Scalar value stored in sections.
    type Value;
    /// Storage backend for section data.
    type Storage: Storage<Self::Value>;
    /// Storage backend for cell type data.
    type CellStorage: Storage<CellType>;

    /// Parse mesh data from a reader.
    fn read<R: Read>(
        &self,
        reader: R,
    ) -> Result<MeshData<Self::Sieve, Self::Value, Self::Storage, Self::CellStorage>, MeshSieveError>;
}

/// Trait for mesh writers that serialize sieve + section data.
pub trait SieveSectionWriter {
    /// Sieve implementation expected by the writer.
    type Sieve: Sieve;
    /// Scalar value stored in sections.
    type Value;
    /// Storage backend for section data.
    type Storage: Storage<Self::Value>;
    /// Storage backend for cell type data.
    type CellStorage: Storage<CellType>;

    /// Write mesh data to a writer.
    fn write<W: Write>(
        &self,
        writer: W,
        mesh: &MeshData<Self::Sieve, Self::Value, Self::Storage, Self::CellStorage>,
    ) -> Result<(), MeshSieveError>;
}
