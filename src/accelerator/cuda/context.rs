//! CUDA context, streams, and explicit transfers.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

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
    /// Create timing-capable events for device-side elapsed-time measurement.
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
    id: u64,
    options: CudaOptions,
}

/// A recorded point in a CUDA stream.
///
/// Events are backend-owned synchronization objects. Timing is available when
/// the backend was created with [`CudaOptions::enable_profiling`].
#[derive(Debug)]
pub struct CudaEvent {
    pub(super) inner: cudarc::driver::CudaEvent,
    backend_id: u64,
    timing_enabled: bool,
}

static NEXT_BACKEND_ID: AtomicU64 = AtomicU64::new(1);

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
        ensure_dynamic_library(&["cuda", "nvcuda"], "CUDA driver")?;
        ensure_dynamic_library(&["nvrtc"], "NVRTC")?;
        let context = std::panic::catch_unwind(|| CudaContext::new(options.device_ordinal))
            .map_err(|payload| AcceleratorError::BackendUnavailable(panic_message(payload)))?
            .map_err(|e| {
                AcceleratorError::BackendUnavailable(format!(
                    "failed to initialize CUDA 13.0.3 ABI device {}: {e}",
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
            AcceleratorError::BackendUnavailable(format!(
                "failed to initialize CUDA 13.0.3-compatible NVRTC: {e}"
            ))
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
            id: NEXT_BACKEND_ID.fetch_add(1, Ordering::Relaxed),
            options,
        })
    }

    /// Effective initialization options.
    pub fn options(&self) -> &CudaOptions {
        &self.options
    }

    /// Number of streams retained by this backend.
    pub fn stream_count(&self) -> usize {
        self.streams.len()
    }

    /// Upload values through an indexed stream.
    pub fn upload_on<T: DeviceValue>(
        &self,
        stream_index: usize,
        values: &[T],
    ) -> Result<CudaBuffer<T>, AcceleratorError> {
        let inner = self
            .stream_at(stream_index)?
            .clone_htod(values)
            .map_err(|e| AcceleratorError::DeviceTransferFailed(e.to_string()))?;
        Ok(CudaBuffer {
            inner,
            backend_id: self.id,
        })
    }

    /// Allocate a zero-initialized buffer through an indexed stream.
    pub fn allocate_on<T: DeviceValue>(
        &self,
        stream_index: usize,
        len: usize,
    ) -> Result<CudaBuffer<T>, AcceleratorError> {
        let inner = self
            .stream_at(stream_index)?
            .alloc_zeros::<T>(len)
            .map_err(|e| AcceleratorError::AllocationFailed {
                bytes: len.saturating_mul(std::mem::size_of::<T>()),
                reason: e.to_string(),
            })?;
        Ok(CudaBuffer {
            inner,
            backend_id: self.id,
        })
    }

    /// Replace a complete device allocation through an indexed stream.
    pub fn upload_into_on<T: DeviceValue>(
        &self,
        stream_index: usize,
        values: &[T],
        buffer: &mut CudaBuffer<T>,
    ) -> Result<(), AcceleratorError> {
        self.validate_buffer(buffer)?;
        if values.len() != buffer.len() {
            return Err(AcceleratorError::LengthMismatch {
                expected: buffer.len(),
                found: values.len(),
            });
        }
        self.stream_at(stream_index)?
            .memcpy_htod(values, &mut buffer.inner)
            .map_err(|e| AcceleratorError::DeviceTransferFailed(e.to_string()))
    }

    /// Download a complete device allocation through an indexed stream.
    pub fn download_on<T: DeviceValue>(
        &self,
        stream_index: usize,
        buffer: &CudaBuffer<T>,
        values: &mut [T],
    ) -> Result<(), AcceleratorError> {
        self.validate_buffer(buffer)?;
        if values.len() != buffer.len() {
            return Err(AcceleratorError::LengthMismatch {
                expected: buffer.len(),
                found: values.len(),
            });
        }
        self.stream_at(stream_index)?
            .memcpy_dtoh(&buffer.inner, values)
            .map_err(|e| AcceleratorError::DeviceTransferFailed(e.to_string()))
    }

    /// Wait for all work submitted to one indexed stream.
    pub fn synchronize_stream(&self, stream_index: usize) -> Result<(), AcceleratorError> {
        self.stream_at(stream_index)?
            .synchronize()
            .map_err(|e| AcceleratorError::KernelLaunchFailed(e.to_string()))
    }

    /// Record all work currently queued on `stream_index`.
    pub fn record_event(&self, stream_index: usize) -> Result<CudaEvent, AcceleratorError> {
        let stream = self.stream_at(stream_index)?;
        let flags = if self.options.enable_profiling {
            Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT)
        } else {
            None
        };
        let inner = stream
            .record_event(flags)
            .map_err(|e| AcceleratorError::EventFailed(e.to_string()))?;
        Ok(CudaEvent {
            inner,
            backend_id: self.id,
            timing_enabled: self.options.enable_profiling,
        })
    }

    /// Make `stream_index` wait for a previously recorded event.
    pub fn wait_event(
        &self,
        stream_index: usize,
        event: &CudaEvent,
    ) -> Result<(), AcceleratorError> {
        self.validate_event(event)?;
        self.stream_at(stream_index)?
            .wait(&event.inner)
            .map_err(|e| AcceleratorError::EventFailed(e.to_string()))
    }

    /// Block until an event has completed.
    pub fn synchronize_event(&self, event: &CudaEvent) -> Result<(), AcceleratorError> {
        self.validate_event(event)?;
        event
            .inner
            .synchronize()
            .map_err(|e| AcceleratorError::EventFailed(e.to_string()))
    }

    /// Return elapsed device time between two timing-enabled events.
    pub fn elapsed_ms(&self, start: &CudaEvent, end: &CudaEvent) -> Result<f32, AcceleratorError> {
        self.validate_event(start)?;
        self.validate_event(end)?;
        if !start.timing_enabled || !end.timing_enabled {
            return Err(AcceleratorError::EventFailed(
                "event timing requires CudaOptions::enable_profiling".into(),
            ));
        }
        start
            .inner
            .elapsed_ms(&end.inner)
            .map_err(|e| AcceleratorError::EventFailed(e.to_string()))
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

    pub(super) fn stream_at(&self, index: usize) -> Result<&Arc<CudaStream>, AcceleratorError> {
        self.streams
            .get(index)
            .ok_or(AcceleratorError::InvalidStreamIndex {
                index,
                stream_count: self.streams.len(),
            })
    }

    pub(super) fn validate_backend_id(&self, expected: u64) -> Result<(), AcceleratorError> {
        if expected == self.id {
            Ok(())
        } else {
            Err(AcceleratorError::BackendMismatch {
                expected,
                found: self.id,
            })
        }
    }

    pub(super) fn validate_buffer<T: DeviceValue>(
        &self,
        buffer: &CudaBuffer<T>,
    ) -> Result<(), AcceleratorError> {
        self.validate_backend_id(buffer.backend_id)
    }

    fn validate_event(&self, event: &CudaEvent) -> Result<(), AcceleratorError> {
        self.validate_backend_id(event.backend_id)
    }
}

fn ensure_dynamic_library(names: &[&str], description: &str) -> Result<(), AcceleratorError> {
    let candidates: Vec<_> = names
        .iter()
        .flat_map(|name| cudarc::get_lib_name_candidates(name))
        .collect();
    for candidate in &candidates {
        // SAFETY: this is an availability probe; no symbols are accessed and
        // the handle is dropped before cudarc opens its process-global handle.
        if unsafe { libloading::Library::new(candidate) }.is_ok() {
            return Ok(());
        }
    }
    Err(AcceleratorError::BackendUnavailable(format!(
        "{description} shared library is unavailable (tried {})",
        candidates.join(", ")
    )))
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

    fn identity(&self) -> u64 {
        self.id
    }

    fn upload<T: DeviceValue>(&self, values: &[T]) -> Result<Self::Buffer<T>, Self::Error> {
        self.upload_on(0, values)
    }

    fn allocate<T: DeviceValue>(&self, len: usize) -> Result<Self::Buffer<T>, Self::Error> {
        self.allocate_on(0, len)
    }

    fn upload_into<T: DeviceValue>(
        &self,
        values: &[T],
        buffer: &mut Self::Buffer<T>,
    ) -> Result<(), Self::Error> {
        self.upload_into_on(0, values, buffer)
    }

    fn download<T: DeviceValue>(
        &self,
        buffer: &Self::Buffer<T>,
        values: &mut [T],
    ) -> Result<(), Self::Error> {
        self.download_on(0, buffer, values)
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
