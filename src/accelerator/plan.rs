//! Frozen topology plans.

use std::collections::HashMap;

use crate::topology::bounds::PayloadLike;
use crate::topology::point::PointId;
use crate::topology::sieve::FrozenSieveCsr;

use super::{AcceleratorBackend, AcceleratorError, DeviceValue};

/// Versions/epochs captured when an execution plan is compiled.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PlanEpochs {
    /// Version of the mutable topology from which the frozen CSR was made.
    pub topology: u64,
    /// Version of an associated field atlas, when applicable.
    pub atlas: u64,
    /// Epoch incremented by the caller when coordinates/geometry change.
    pub geometry: u64,
}

impl PlanEpochs {
    /// Reject use after topology, atlas, or geometry changes.
    pub fn validate(self, current: Self) -> Result<(), AcceleratorError> {
        if self.topology != current.topology {
            return Err(AcceleratorError::StaleTopologyPlan {
                expected: self.topology,
                found: current.topology,
            });
        }
        if self.atlas != current.atlas {
            return Err(AcceleratorError::StaleAtlasPlan {
                expected: self.atlas,
                found: current.atlas,
            });
        }
        if self.geometry != current.geometry {
            return Err(AcceleratorError::StaleGeometryPlan {
                expected: self.geometry,
                found: current.geometry,
            });
        }
        Ok(())
    }
}

/// Device-resident dense CSR topology.
pub struct DeviceTopology<B: AcceleratorBackend> {
    /// Version of the source topology.
    pub topology_version: u64,
    /// Dense-index to stable point identifier.
    pub point_ids: B::Buffer<u64>,
    /// Cone CSR row offsets.
    pub cone_offsets: B::Buffer<u32>,
    /// Cone dense point indices.
    pub cone_points: B::Buffer<u32>,
    /// Support CSR row offsets.
    pub support_offsets: B::Buffer<u32>,
    /// Support dense point indices.
    pub support_points: B::Buffer<u32>,
    /// Number of points in the dense chart.
    pub point_count: usize,
    /// Number of directed cone incidences.
    pub incidence_count: usize,
}

/// A device topology plus the host-only stable-ID lookup needed to compile
/// operation-specific plans.
pub struct DeviceMeshPlan<B: AcceleratorBackend> {
    /// Device CSR arrays.
    pub topology: DeviceTopology<B>,
    /// Stable point ID to dense device index. This map is never uploaded.
    pub index_of: HashMap<PointId, u32>,
}

impl<B: AcceleratorBackend> DeviceMeshPlan<B> {
    /// Compile an immutable CSR topology into device arrays.
    pub fn compile<T: PayloadLike>(
        backend: &B,
        frozen: &FrozenSieveCsr<PointId, T>,
        topology_version: u64,
    ) -> Result<Self, AcceleratorError> {
        checked_u32(frozen.point_of.len(), "point count")?;
        checked_u32(frozen.out_dsts.len(), "cone incidence count")?;
        checked_u32(frozen.in_srcs.len(), "support incidence count")?;
        let point_ids: Vec<u64> = frozen.point_of.iter().map(PointId::get).collect();
        let upload = |result: Result<B::Buffer<u32>, B::Error>| {
            result.map_err(|e| AcceleratorError::DeviceTransferFailed(e.to_string()))
        };
        let point_ids = backend
            .upload(&point_ids)
            .map_err(|e| AcceleratorError::DeviceTransferFailed(e.to_string()))?;
        let cone_offsets = upload(backend.upload(&frozen.out_offsets))?;
        let cone_points = upload(backend.upload(&frozen.out_dsts))?;
        let support_offsets = upload(backend.upload(&frozen.in_offsets))?;
        let support_points = upload(backend.upload(&frozen.in_srcs))?;
        Ok(Self {
            topology: DeviceTopology {
                topology_version,
                point_ids,
                cone_offsets,
                cone_points,
                support_offsets,
                support_points,
                point_count: frozen.point_of.len(),
                incidence_count: frozen.out_dsts.len(),
            },
            index_of: frozen.index_of.clone(),
        })
    }

    /// Ensure this plan still corresponds to the current topology version.
    pub fn validate_topology(&self, current: u64) -> Result<(), AcceleratorError> {
        if self.topology.topology_version == current {
            Ok(())
        } else {
            Err(AcceleratorError::StaleTopologyPlan {
                expected: self.topology.topology_version,
                found: current,
            })
        }
    }
}

pub(crate) fn checked_u32(value: usize, what: &'static str) -> Result<u32, AcceleratorError> {
    u32::try_from(value).map_err(|_| AcceleratorError::IndexOverflow { what, value })
}

pub(crate) fn upload<T: DeviceValue, B: AcceleratorBackend>(
    backend: &B,
    values: &[T],
) -> Result<B::Buffer<T>, AcceleratorError> {
    backend
        .upload(values)
        .map_err(|e| AcceleratorError::DeviceTransferFailed(e.to_string()))
}
