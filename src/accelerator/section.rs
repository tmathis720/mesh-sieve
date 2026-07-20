//! Explicit section transfers.

use crate::data::section::Section;
use crate::data::storage::Storage;

use super::plan::{checked_u32, upload};
use super::{AcceleratorBackend, AcceleratorError, DeviceBuffer, DeviceValue};

/// A section layout and its values resident in backend-owned memory.
pub struct DeviceSection<T: DeviceValue, B: AcceleratorBackend> {
    /// Source atlas structural version.
    pub atlas_version: u64,
    /// Source mesh topology version supplied by the caller.
    pub topology_version: u64,
    /// Per-point offsets in atlas iteration order.
    pub offsets: B::Buffer<u32>,
    /// Per-point lengths in atlas iteration order.
    pub lengths: B::Buffer<u32>,
    /// Flat field values.
    pub values: B::Buffer<T>,
    /// Number of atlas points.
    pub point_count: usize,
    /// Number of scalar values.
    pub value_count: usize,
}

impl<T: DeviceValue, B: AcceleratorBackend> DeviceSection<T, B> {
    /// Upload atlas metadata and values from a host-accessible section.
    pub fn upload_from<S: Storage<T>>(
        backend: &B,
        section: &Section<T, S>,
        topology_version: u64,
    ) -> Result<Self, AcceleratorError> {
        checked_u32(section.atlas().len(), "atlas point count")?;
        checked_u32(section.atlas().total_len(), "section value count")?;
        let mut offsets = Vec::with_capacity(section.atlas().len());
        let mut lengths = Vec::with_capacity(section.atlas().len());
        for (offset, len) in section.atlas().iter_spans() {
            offsets.push(checked_u32(offset, "atlas offset")?);
            lengths.push(checked_u32(len, "atlas slice length")?);
        }
        Ok(Self {
            atlas_version: section.atlas().version(),
            topology_version,
            offsets: upload(backend, &offsets)?,
            lengths: upload(backend, &lengths)?,
            values: upload(backend, section.as_flat_slice())?,
            point_count: section.atlas().len(),
            value_count: section.atlas().total_len(),
        })
    }

    /// Refresh values without rebuilding layout metadata.
    pub fn refresh_values_from<S: Storage<T>>(
        &mut self,
        backend: &B,
        section: &Section<T, S>,
        topology_version: u64,
    ) -> Result<(), AcceleratorError> {
        self.validate(section.atlas().version(), topology_version)?;
        if section.as_flat_slice().len() != self.values.len() {
            return Err(AcceleratorError::LengthMismatch {
                expected: self.values.len(),
                found: section.as_flat_slice().len(),
            });
        }
        backend
            .upload_into(section.as_flat_slice(), &mut self.values)
            .map_err(|e| AcceleratorError::DeviceTransferFailed(e.to_string()))
    }

    /// Download values into a section with the same atlas layout.
    pub fn download_into<S: Storage<T> + Clone>(
        &self,
        backend: &B,
        section: &mut Section<T, S>,
        topology_version: u64,
    ) -> Result<(), AcceleratorError>
    where
        T: Clone + Default,
    {
        self.validate(section.atlas().version(), topology_version)?;
        let mut host = vec![T::zeroed(); self.value_count];
        backend
            .download(&self.values, &mut host)
            .map_err(|e| AcceleratorError::DeviceTransferFailed(e.to_string()))?;
        section
            .try_scatter_in_order(&host)
            .map_err(|e| AcceleratorError::InvalidPlan(e.to_string()))
    }

    /// Validate both structural epochs used to construct this section.
    pub fn validate(
        &self,
        atlas_version: u64,
        topology_version: u64,
    ) -> Result<(), AcceleratorError> {
        if self.atlas_version != atlas_version {
            return Err(AcceleratorError::StaleAtlasPlan {
                expected: self.atlas_version,
                found: atlas_version,
            });
        }
        if self.topology_version != topology_version {
            return Err(AcceleratorError::StaleTopologyPlan {
                expected: self.topology_version,
                found: topology_version,
            });
        }
        Ok(())
    }
}
