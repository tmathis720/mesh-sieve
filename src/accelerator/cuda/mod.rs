//! Native NVIDIA CUDA backend powered by `cudarc`.

mod buffer;
mod context;
mod executor;
mod module_cache;

pub use buffer::CudaBuffer;
pub use context::{CudaBackend, CudaOptions};
pub use executor::CudaFvmScalar;
