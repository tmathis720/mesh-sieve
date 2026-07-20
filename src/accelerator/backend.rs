//! Backend-neutral buffers and transfers.

use bytemuck::{Pod, Zeroable};

use super::AcceleratorError;

/// Values that can be copied byte-for-byte between host and accelerator.
#[cfg(not(feature = "cuda"))]
pub trait DeviceValue: Pod + Zeroable + Send + Sync + 'static {}

#[cfg(not(feature = "cuda"))]
impl<T> DeviceValue for T where T: Pod + Zeroable + Send + Sync + 'static {}

/// Values that can be copied byte-for-byte between host and CUDA memory.
#[cfg(feature = "cuda")]
pub trait DeviceValue: Pod + Zeroable + Send + Sync + 'static + cudarc::driver::DeviceRepr {}

#[cfg(feature = "cuda")]
impl<T> DeviceValue for T where
    T: Pod + Zeroable + Send + Sync + 'static + cudarc::driver::DeviceRepr
{
}

/// Minimal metadata exposed by an opaque device allocation.
pub trait DeviceBuffer<T: DeviceValue>: Send + Sync {
    /// Number of elements in the allocation.
    fn len(&self) -> usize;

    /// Whether this allocation is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Explicit memory and synchronization boundary implemented by accelerators.
pub trait AcceleratorBackend: Send + Sync {
    /// Backend-owned typed allocation.
    type Buffer<T: DeviceValue>: DeviceBuffer<T>;
    /// Backend event/fence type.
    type Event: Send + Sync;
    /// Native backend error.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Copy host values into a new device allocation.
    fn upload<T: DeviceValue>(&self, values: &[T]) -> Result<Self::Buffer<T>, Self::Error>;
    /// Allocate a zero-initialized device buffer.
    fn allocate<T: DeviceValue>(&self, len: usize) -> Result<Self::Buffer<T>, Self::Error>;
    /// Replace all values in an existing allocation.
    fn upload_into<T: DeviceValue>(
        &self,
        values: &[T],
        buffer: &mut Self::Buffer<T>,
    ) -> Result<(), Self::Error>;
    /// Copy a complete device allocation into a host slice.
    fn download<T: DeviceValue>(
        &self,
        buffer: &Self::Buffer<T>,
        values: &mut [T],
    ) -> Result<(), Self::Error>;
    /// Wait for all work submitted through this backend.
    fn synchronize(&self) -> Result<(), Self::Error>;
}

/// Host buffer used by the reference backend.
#[derive(Clone, Debug, PartialEq)]
pub struct CpuBuffer<T>(pub(crate) Vec<T>);

impl<T: DeviceValue> DeviceBuffer<T> for CpuBuffer<T> {
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl<T> CpuBuffer<T> {
    /// Borrow values for CPU reference execution and tests.
    pub fn as_slice(&self) -> &[T] {
        &self.0
    }

    /// Mutably borrow values for CPU reference execution and tests.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.0
    }
}

/// Reference backend that exercises exactly the packed accelerator layouts.
#[derive(Clone, Copy, Debug, Default)]
pub struct CpuBackend;

impl AcceleratorBackend for CpuBackend {
    type Buffer<T: DeviceValue> = CpuBuffer<T>;
    type Event = ();
    type Error = AcceleratorError;

    fn upload<T: DeviceValue>(&self, values: &[T]) -> Result<Self::Buffer<T>, Self::Error> {
        Ok(CpuBuffer(values.to_vec()))
    }

    fn allocate<T: DeviceValue>(&self, len: usize) -> Result<Self::Buffer<T>, Self::Error> {
        Ok(CpuBuffer(vec![T::zeroed(); len]))
    }

    fn upload_into<T: DeviceValue>(
        &self,
        values: &[T],
        buffer: &mut Self::Buffer<T>,
    ) -> Result<(), Self::Error> {
        if values.len() != buffer.0.len() {
            return Err(AcceleratorError::LengthMismatch {
                expected: buffer.0.len(),
                found: values.len(),
            });
        }
        buffer.0.copy_from_slice(values);
        Ok(())
    }

    fn download<T: DeviceValue>(
        &self,
        buffer: &Self::Buffer<T>,
        values: &mut [T],
    ) -> Result<(), Self::Error> {
        if values.len() != buffer.0.len() {
            return Err(AcceleratorError::LengthMismatch {
                expected: buffer.0.len(),
                found: values.len(),
            });
        }
        values.copy_from_slice(&buffer.0);
        Ok(())
    }

    fn synchronize(&self) -> Result<(), Self::Error> {
        Ok(())
    }
}

/// User-facing execution backend preference.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ComputeBackend {
    /// Always execute on the host reference backend.
    Cpu,
    /// Use the portable WGPU path where supported by the operation.
    Wgpu,
    /// Use the native CUDA path on this device ordinal.
    Cuda { device_ordinal: usize },
    /// Select the best backend that initializes successfully.
    Auto,
}

impl ComputeBackend {
    /// Resolve `Auto` and report both the selected backend and why.
    ///
    /// CUDA fallback is attempted only here, during initialization. Execution
    /// errors are intentionally never converted into a silent CPU fallback.
    pub fn resolve(&self) -> Result<(Self, String), AcceleratorError> {
        match self {
            Self::Cpu => Ok((Self::Cpu, "CPU backend explicitly requested".into())),
            Self::Wgpu => {
                #[cfg(feature = "wgpu")]
                return Ok((Self::Wgpu, "WGPU backend explicitly requested".into()));
                #[cfg(not(feature = "wgpu"))]
                Err(AcceleratorError::BackendUnavailable(
                    "mesh-sieve was built without the `wgpu` feature".into(),
                ))
            }
            Self::Cuda { device_ordinal } => {
                #[cfg(feature = "cuda")]
                {
                    super::cuda::CudaBackend::new(super::cuda::CudaOptions {
                        device_ordinal: *device_ordinal,
                        ..Default::default()
                    })?;
                    Ok((
                        self.clone(),
                        format!("CUDA device {device_ordinal} initialized"),
                    ))
                }
                #[cfg(not(feature = "cuda"))]
                {
                    let _ = device_ordinal;
                    Err(AcceleratorError::BackendUnavailable(
                        "mesh-sieve was built without the `cuda` feature".into(),
                    ))
                }
            }
            Self::Auto => {
                #[cfg(feature = "cuda")]
                {
                    return match super::cuda::CudaBackend::new(super::cuda::CudaOptions::default())
                    {
                        Ok(_) => Ok((
                            Self::Cuda { device_ordinal: 0 },
                            "CUDA device 0 and NVRTC initialized successfully".into(),
                        )),
                        Err(error) => Ok((
                            Self::Cpu,
                            format!("CUDA initialization failed ({error}); using CPU"),
                        )),
                    };
                }
                #[cfg(not(feature = "cuda"))]
                Ok((
                    Self::Cpu,
                    "CUDA support is not compiled in; using CPU".into(),
                ))
            }
        }
    }
}
