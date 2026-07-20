//! CUDA context, streams, and explicit transfers.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream};

use crate::accelerator::{AcceleratorBackend, AcceleratorError, DeviceBuffer, DeviceValue};

use super::buffer::CudaBuffer;
use super::module_cache::CudaModuleCache;

/// CUDA initialization settings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CudaOptions {
    /// Zero-based CUDA device ordinal.
    pub device_ordinal: usize,
    /// Number of streams retained by the backend (minimum one).
    pub stream_count: usize,
    /// Keep profiling-oriented metadata available to callers.
    pub enable_profiling: bool,
}

impl Default for CudaOptions {
    fn default() -> Self {
        Self {
            device_ordinal: 0,
            stream_count: 1,
            enable_profiling: false,
        }
    }
}

/// Persistent CUDA context, streams, and compiled-module cache.
pub struct CudaBackend {
    pub(super) streams: Vec<Arc<CudaStream>>,
    pub(super) modules: CudaModuleCache,
    options: CudaOptions,
}

impl std::fmt::Debug for CudaBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaBackend")
            .field("options", &self.options)
            .field("stream_count", &self.streams.len())
            .finish_non_exhaustive()
    }
}

impl CudaBackend {
    /// Initialize one persistent context and the requested streams.
    pub fn new(options: CudaOptions) -> Result<Self, AcceleratorError> {
        let context = std::panic::catch_unwind(|| CudaContext::new(options.device_ordinal))
            .map_err(|payload| AcceleratorError::BackendUnavailable(panic_message(payload)))?
            .map_err(|e| {
                AcceleratorError::BackendUnavailable(format!(
                    "failed to initialize CUDA device {}: {e}",
                    options.device_ordinal
                ))
            })?;
        // Dynamic cudarc loading reports missing shared libraries by panic.
        // Probe NVRTC here so `Auto` fallback remains an initialization-time
        // decision and kernel execution never silently changes backend.
        std::panic::catch_unwind(|| {
            cudarc::nvrtc::compile_ptx("extern \"C\" __global__ void mesh_sieve_probe() {}")
        })
        .map_err(|payload| AcceleratorError::BackendUnavailable(panic_message(payload)))?
        .map_err(|e| {
            AcceleratorError::BackendUnavailable(format!("failed to initialize NVRTC: {e}"))
        })?;
        let mut streams = Vec::with_capacity(options.stream_count.max(1));
        streams.push(context.default_stream());
        for _ in 1..options.stream_count.max(1) {
            streams.push(context.new_stream().map_err(|e| {
                AcceleratorError::BackendUnavailable(format!("failed to create CUDA stream: {e}"))
            })?);
        }
        Ok(Self {
            modules: CudaModuleCache::new(context.clone()),
            streams,
            options,
        })
    }

    /// Effective initialization options.
    pub fn options(&self) -> &CudaOptions {
        &self.options
    }

    /// Compile/load a CUDA C source once and return a named function.
    pub(crate) fn function(
        &self,
        source: &str,
        source_name: &str,
        function_name: &str,
    ) -> Result<cudarc::driver::CudaFunction, AcceleratorError> {
        self.modules.function(source, source_name, function_name)
    }

    /// Compile CUDA C with NVRTC, cache the resulting module by content, and
    /// return a named entry point. The function keeps its module alive.
    pub fn compile_function(
        &self,
        source: &str,
        source_name: &str,
        function_name: &str,
    ) -> Result<cudarc::driver::CudaFunction, AcceleratorError> {
        self.modules.function(source, source_name, function_name)
    }

    /// Load caller-supplied PTX, cache its module by content, and return a
    /// named entry point for model-specific execution.
    pub fn load_ptx_function(
        &self,
        ptx: &str,
        module_name: &str,
        function_name: &str,
    ) -> Result<cudarc::driver::CudaFunction, AcceleratorError> {
        self.modules.ptx_function(ptx, module_name, function_name)
    }

    pub(super) fn stream(&self) -> &Arc<CudaStream> {
        &self.streams[0]
    }
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else {
        "CUDA dynamic library initialization panicked".to_string()
    }
}

impl AcceleratorBackend for CudaBackend {
    type Buffer<T: DeviceValue> = CudaBuffer<T>;
    type Event = ();
    type Error = AcceleratorError;

    fn upload<T: DeviceValue>(&self, values: &[T]) -> Result<Self::Buffer<T>, Self::Error> {
        let inner = self
            .stream()
            .clone_htod(values)
            .map_err(|e| AcceleratorError::DeviceTransferFailed(e.to_string()))?;
        Ok(CudaBuffer { inner })
    }

    fn allocate<T: DeviceValue>(&self, len: usize) -> Result<Self::Buffer<T>, Self::Error> {
        // `DeviceValue: Zeroable` is stronger than cudarc's marker for our
        // purposes, while still allowing downstream POD parameter structs.
        let zeros = vec![T::zeroed(); len];
        let inner =
            self.stream()
                .clone_htod(&zeros)
                .map_err(|e| AcceleratorError::AllocationFailed {
                    bytes: len.saturating_mul(std::mem::size_of::<T>()),
                    reason: e.to_string(),
                })?;
        Ok(CudaBuffer { inner })
    }

    fn upload_into<T: DeviceValue>(
        &self,
        values: &[T],
        buffer: &mut Self::Buffer<T>,
    ) -> Result<(), Self::Error> {
        if values.len() != buffer.len() {
            return Err(AcceleratorError::LengthMismatch {
                expected: buffer.len(),
                found: values.len(),
            });
        }
        self.stream()
            .memcpy_htod(values, &mut buffer.inner)
            .map_err(|e| AcceleratorError::DeviceTransferFailed(e.to_string()))
    }

    fn download<T: DeviceValue>(
        &self,
        buffer: &Self::Buffer<T>,
        values: &mut [T],
    ) -> Result<(), Self::Error> {
        if values.len() != buffer.len() {
            return Err(AcceleratorError::LengthMismatch {
                expected: buffer.len(),
                found: values.len(),
            });
        }
        self.stream()
            .memcpy_dtoh(&buffer.inner, values)
            .map_err(|e| AcceleratorError::DeviceTransferFailed(e.to_string()))
    }

    fn synchronize(&self) -> Result<(), Self::Error> {
        for stream in &self.streams {
            stream
                .synchronize()
                .map_err(|e| AcceleratorError::KernelLaunchFailed(e.to_string()))?;
        }
        Ok(())
    }
}
