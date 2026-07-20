//! CUDA allocation wrapper.

use cudarc::driver::CudaSlice;

use crate::accelerator::{DeviceBuffer, DeviceValue};

/// Opaque device allocation. It intentionally exposes no host slice methods.
#[derive(Debug)]
pub struct CudaBuffer<T: DeviceValue> {
    pub(super) inner: CudaSlice<T>,
}

impl<T: DeviceValue> DeviceBuffer<T> for CudaBuffer<T> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}
