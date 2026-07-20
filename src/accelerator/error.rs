//! Accelerator-specific errors.

use thiserror::Error;

/// Failures produced while compiling or executing an accelerator plan.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum AcceleratorError {
    /// The requested backend was not compiled in or could not initialize.
    #[error("accelerator backend unavailable: {0}")]
    BackendUnavailable(String),
    /// No device exists at the requested ordinal.
    #[error("CUDA device {ordinal} was not found")]
    DeviceNotFound { ordinal: usize },
    /// Device allocation failed.
    #[error("device allocation of {bytes} bytes failed: {reason}")]
    AllocationFailed { bytes: usize, reason: String },
    /// Runtime kernel compilation failed.
    #[error("CUDA kernel compilation failed: {0}")]
    KernelCompilationFailed(String),
    /// A compiled module could not be loaded.
    #[error("CUDA module load failed: {0}")]
    ModuleLoadFailed(String),
    /// A named kernel was absent from a module.
    #[error("CUDA kernel `{0}` was not found")]
    KernelNotFound(String),
    /// A kernel launch failed.
    #[error("CUDA kernel launch failed: {0}")]
    KernelLaunchFailed(String),
    /// A host/device transfer failed.
    #[error("device transfer failed: {0}")]
    DeviceTransferFailed(String),
    /// A topology-dependent plan is stale.
    #[error("stale topology plan: expected version {expected}, found {found}")]
    StaleTopologyPlan { expected: u64, found: u64 },
    /// A section-dependent plan is stale.
    #[error("stale atlas plan: expected version {expected}, found {found}")]
    StaleAtlasPlan { expected: u64, found: u64 },
    /// Geometry changed after plan compilation.
    #[error("stale geometry plan: expected epoch {expected}, found {found}")]
    StaleGeometryPlan { expected: u64, found: u64 },
    /// Host input has a different length than its device buffer.
    #[error("buffer length mismatch: expected {expected}, found {found}")]
    LengthMismatch { expected: usize, found: usize },
    /// A count or offset cannot be represented by the device ABI.
    #[error("{what} value {value} exceeds the device ABI limit")]
    IndexOverflow { what: &'static str, value: usize },
    /// The value type cannot be used by the selected device.
    #[error("unsupported device value type: {0}")]
    UnsupportedDeviceType(&'static str),
    /// Packed FVM data is internally inconsistent.
    #[error("invalid accelerator plan: {0}")]
    InvalidPlan(String),
}
