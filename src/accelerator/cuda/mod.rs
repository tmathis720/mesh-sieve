//! Native NVIDIA CUDA backend powered by `cudarc`.

mod buffer;
mod context;
#[cfg(feature = "cuda-cusparse")]
mod cusparse;
mod executor;
mod module_cache;

pub use buffer::CudaBuffer;
pub use context::{CudaBackend, CudaEvent, CudaOptions};
#[cfg(feature = "cuda-cusparse")]
pub use cusparse::CudaSparseScalar;
pub use executor::CudaFvmScalar;
