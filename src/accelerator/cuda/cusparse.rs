//! Minimal owned cuSPARSE CSR SpMV bridge.

use std::ffi::c_void;
use std::mem::MaybeUninit;

use cudarc::cusparse::{result, sys};
use cudarc::driver::{DevicePtr, DevicePtrMut};

use crate::accelerator::{
    AcceleratorBackend, AcceleratorError, DeviceBuffer, DeviceCsrMatrix, FvmScalar,
};

use super::{CudaBackend, CudaBuffer};

/// Scalar types accepted by the cuSPARSE CSR bridge.
pub trait CudaSparseScalar: FvmScalar {
    const DATA_TYPE: sys::cudaDataType;
}

impl CudaSparseScalar for f32 {
    const DATA_TYPE: sys::cudaDataType = sys::cudaDataType::CUDA_R_32F;
}

impl CudaSparseScalar for f64 {
    const DATA_TYPE: sys::cudaDataType = sys::cudaDataType::CUDA_R_64F;
}

struct Handle(sys::cusparseHandle_t);
impl Drop for Handle {
    fn drop(&mut self) {
        // SAFETY: this wrapper uniquely owns the handle.
        let _ = unsafe { result::destroy(self.0) };
    }
}

struct SparseMatrix(sys::cusparseSpMatDescr_t);
impl Drop for SparseMatrix {
    fn drop(&mut self) {
        // SAFETY: this wrapper uniquely owns the descriptor.
        let _ = unsafe { sys::cusparseDestroySpMat(self.0).result() };
    }
}

struct DenseVector(sys::cusparseDnVecDescr_t);
impl Drop for DenseVector {
    fn drop(&mut self) {
        // SAFETY: this wrapper uniquely owns the descriptor.
        let _ = unsafe { sys::cusparseDestroyDnVec(self.0).result() };
    }
}

impl<T: CudaSparseScalar> DeviceCsrMatrix<T, CudaBackend> {
    /// Compute y = alpha * A * x + beta * y with cuSPARSE on stream 0.
    pub fn spmv(
        &mut self,
        backend: &CudaBackend,
        alpha: T,
        x: &CudaBuffer<T>,
        beta: T,
        y: &mut CudaBuffer<T>,
    ) -> Result<(), AcceleratorError> {
        self.spmv_on(backend, 0, alpha, x, beta, y)
    }

    /// Compute y = alpha * A * x + beta * y with cuSPARSE on an indexed stream.
    pub fn spmv_on(
        &mut self,
        backend: &CudaBackend,
        stream_index: usize,
        alpha: T,
        x: &CudaBuffer<T>,
        beta: T,
        y: &mut CudaBuffer<T>,
    ) -> Result<(), AcceleratorError> {
        backend.validate_buffer(&self.row_offsets)?;
        backend.validate_buffer(&self.column_indices)?;
        backend.validate_buffer(&self.values)?;
        backend.validate_buffer(x)?;
        backend.validate_buffer(y)?;
        if let Some(workspace) = &self.workspace {
            backend.validate_buffer(workspace)?;
        }
        if x.len() != self.column_count {
            return Err(AcceleratorError::LengthMismatch {
                expected: self.column_count,
                found: x.len(),
            });
        }
        if y.len() != self.row_count {
            return Err(AcceleratorError::LengthMismatch {
                expected: self.row_count,
                found: y.len(),
            });
        }
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.spmv_inner(backend, stream_index, alpha, x, beta, y)
        }))
        .map_err(|payload| {
            let message = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
                .unwrap_or_else(|| "cuSPARSE dynamic loading panicked".into());
            AcceleratorError::SparseLibraryFailed(message)
        })?
    }

    fn spmv_inner(
        &mut self,
        backend: &CudaBackend,
        stream_index: usize,
        alpha: T,
        x: &CudaBuffer<T>,
        beta: T,
        y: &mut CudaBuffer<T>,
    ) -> Result<(), AcceleratorError> {
        let stream = backend.stream_at(stream_index)?;
        let handle = Handle(
            result::create()
                .map_err(|e| AcceleratorError::SparseLibraryFailed(format!("{e:?}")))?,
        );
        // SAFETY: the cuSPARSE handle and CUDA stream are live and share a context/device.
        unsafe { sys::cusparseSetStream(handle.0, stream.cu_stream().cast()) }
            .result()
            .map_err(sparse_error)?;

        let (row_ptr, _row_guard) = self.row_offsets.inner.device_ptr(stream);
        let (column_ptr, _column_guard) = self.column_indices.inner.device_ptr(stream);
        let (value_ptr, _value_guard) = self.values.inner.device_ptr(stream);
        let (x_ptr, _x_guard) = x.inner.device_ptr(stream);
        let (y_ptr, _y_guard) = y.inner.device_ptr_mut(stream);

        let mut matrix = MaybeUninit::uninit();
        // SAFETY: pointers refer to validated, live device allocations for the descriptor lifetime.
        unsafe {
            sys::cusparseCreateCsr(
                matrix.as_mut_ptr(),
                self.row_count as i64,
                self.column_count as i64,
                self.values.len() as i64,
                device_void(row_ptr),
                device_void(column_ptr),
                device_void(value_ptr),
                sys::cusparseIndexType_t::CUSPARSE_INDEX_32I,
                sys::cusparseIndexType_t::CUSPARSE_INDEX_32I,
                sys::cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO,
                T::DATA_TYPE,
            )
        }
        .result()
        .map_err(sparse_error)?;
        // SAFETY: creation above initialized the descriptor.
        let matrix = SparseMatrix(unsafe { matrix.assume_init() });

        let mut x_descriptor = MaybeUninit::uninit();
        // SAFETY: x_ptr references column_count live values.
        unsafe {
            sys::cusparseCreateDnVec(
                x_descriptor.as_mut_ptr(),
                self.column_count as i64,
                device_void(x_ptr),
                T::DATA_TYPE,
            )
        }
        .result()
        .map_err(sparse_error)?;
        // SAFETY: creation above initialized the descriptor.
        let x_descriptor = DenseVector(unsafe { x_descriptor.assume_init() });

        let mut y_descriptor = MaybeUninit::uninit();
        // SAFETY: y_ptr references row_count live writable values.
        unsafe {
            sys::cusparseCreateDnVec(
                y_descriptor.as_mut_ptr(),
                self.row_count as i64,
                device_void(y_ptr),
                T::DATA_TYPE,
            )
        }
        .result()
        .map_err(sparse_error)?;
        // SAFETY: creation above initialized the descriptor.
        let y_descriptor = DenseVector(unsafe { y_descriptor.assume_init() });

        let alpha_ptr = (&alpha as *const T).cast::<c_void>();
        let beta_ptr = (&beta as *const T).cast::<c_void>();
        let algorithm = sys::cusparseSpMVAlg_t::CUSPARSE_SPMV_ALG_DEFAULT;
        let operation = sys::cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE;
        let mut workspace_bytes = 0usize;
        // SAFETY: every descriptor and scalar pointer remains live through this call.
        unsafe {
            sys::cusparseSpMV_bufferSize(
                handle.0,
                operation,
                alpha_ptr,
                matrix.0,
                x_descriptor.0,
                beta_ptr,
                y_descriptor.0,
                T::DATA_TYPE,
                algorithm,
                &mut workspace_bytes,
            )
        }
        .result()
        .map_err(sparse_error)?;

        let needs_workspace = self
            .workspace
            .as_ref()
            .map_or(true, |workspace| workspace.len() < workspace_bytes);
        if needs_workspace {
            self.workspace = Some(backend.allocate(workspace_bytes.max(1)).map_err(|e| {
                AcceleratorError::AllocationFailed {
                    bytes: workspace_bytes.max(1),
                    reason: e.to_string(),
                }
            })?);
        }
        let (workspace_ptr, _workspace_guard) = self
            .workspace
            .as_mut()
            .map(|workspace| workspace.inner.device_ptr_mut(stream))
            .expect("workspace is always initialized");
        // SAFETY: descriptors, scalars, and workspace remain valid until the enqueued call is recorded.
        unsafe {
            sys::cusparseSpMV(
                handle.0,
                operation,
                alpha_ptr,
                matrix.0,
                x_descriptor.0,
                beta_ptr,
                y_descriptor.0,
                T::DATA_TYPE,
                algorithm,
                device_void(workspace_ptr),
            )
        }
        .result()
        .map_err(sparse_error)
    }
}

fn sparse_error(error: result::CusparseError) -> AcceleratorError {
    AcceleratorError::SparseLibraryFailed(format!("{error:?}"))
}

fn device_void(pointer: u64) -> *mut c_void {
    pointer as usize as *mut c_void
}
