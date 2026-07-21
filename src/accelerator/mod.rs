//! Explicit, plan-based accelerator execution.
//!
//! Mutable mesh and section objects remain host-owned.  This module compiles
//! them into dense plans whose buffers may live on a CPU, CUDA device, or a
//! future accelerator.  Transfers are always explicit; device buffers never
//! pretend to be host slices.

mod backend;
mod csr;
mod error;
mod fvm;
mod plan;
mod reduction;
mod section;

#[cfg(feature = "cuda")]
pub mod cuda;

pub use backend::{
    AcceleratorBackend, ComputeBackend, CpuBackend, CpuBuffer, DeviceBuffer, DeviceValue,
};
pub use csr::DeviceCsrMatrix;
pub use error::AcceleratorError;
pub use fvm::{
    DeviceBoundaryFace, DeviceFvmBoundaryConditions, DeviceFvmOperator, DeviceFvmPlan,
    DeviceFvmState, DeviceInternalFace, DeviceLeastSquaresPlan, DevicePhysicsParams, FvmScalar,
    ScalarFluxScheme,
};
pub use plan::{DeviceMeshPlan, DeviceTopology, PlanEpochs};
pub use reduction::DeviceReduction;
pub use section::DeviceSection;
