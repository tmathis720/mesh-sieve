//! Persistent scalar reductions over device-resident vectors.

use super::{AcceleratorBackend, AcceleratorError, CpuBackend, DeviceBuffer, FvmScalar};

/// Reusable one-scalar output allocation for vector reductions.
pub struct DeviceReduction<T: FvmScalar, B: AcceleratorBackend> {
    /// Maximum accepted input length.
    pub(crate) input_len: usize,
    /// Resident one-element reduction result.
    pub(crate) result: B::Buffer<T>,
    pub(crate) workspace: B::Buffer<T>,
    pub(crate) backend_id: u64,
}

impl<T: FvmScalar, B: AcceleratorBackend> DeviceReduction<T, B> {
    /// Allocate a reduction workspace for vectors of exactly `input_len` values.
    pub fn new(backend: &B, input_len: usize) -> Result<Self, AcceleratorError> {
        let result = backend
            .allocate(1)
            .map_err(|e| AcceleratorError::AllocationFailed {
                bytes: std::mem::size_of::<T>(),
                reason: e.to_string(),
            })?;
        let workspace_len = input_len.div_ceil(256).clamp(1, 4096);
        let workspace =
            backend
                .allocate(workspace_len)
                .map_err(|e| AcceleratorError::AllocationFailed {
                    bytes: workspace_len.saturating_mul(std::mem::size_of::<T>()),
                    reason: e.to_string(),
                })?;
        Ok(Self {
            input_len,
            result,
            workspace,
            backend_id: backend.identity(),
        })
    }

    /// Download the most recently computed scalar.
    pub fn download(&self, backend: &B) -> Result<T, AcceleratorError> {
        let mut value = [T::zeroed()];
        backend
            .download(&self.result, &mut value)
            .map_err(|e| AcceleratorError::DeviceTransferFailed(e.to_string()))?;
        Ok(value[0])
    }

    pub(crate) fn validate(&self, found: usize) -> Result<(), AcceleratorError> {
        if self.input_len == found {
            Ok(())
        } else {
            Err(AcceleratorError::LengthMismatch {
                expected: self.input_len,
                found,
            })
        }
    }
}

impl<T: FvmScalar> DeviceReduction<T, CpuBackend> {
    /// Deterministic sum in input order.
    pub fn sum(
        &mut self,
        input: &<CpuBackend as AcceleratorBackend>::Buffer<T>,
    ) -> Result<T, AcceleratorError> {
        self.validate(input.len())?;
        let value = input
            .as_slice()
            .iter()
            .fold(0.0, |sum, value| sum + value.to_f64());
        self.result.as_mut_slice()[0] = T::from_f64(value);
        Ok(T::from_f64(value))
    }

    /// Deterministic dot product in input order.
    pub fn dot(
        &mut self,
        lhs: &<CpuBackend as AcceleratorBackend>::Buffer<T>,
        rhs: &<CpuBackend as AcceleratorBackend>::Buffer<T>,
    ) -> Result<T, AcceleratorError> {
        self.validate(lhs.len())?;
        self.validate(rhs.len())?;
        let value = lhs
            .as_slice()
            .iter()
            .zip(rhs.as_slice())
            .fold(0.0, |sum, (&a, &b)| sum + a.to_f64() * b.to_f64());
        self.result.as_mut_slice()[0] = T::from_f64(value);
        Ok(T::from_f64(value))
    }

    /// Euclidean norm.
    pub fn l2_norm(
        &mut self,
        input: &<CpuBackend as AcceleratorBackend>::Buffer<T>,
    ) -> Result<T, AcceleratorError> {
        self.validate(input.len())?;
        let value = input
            .as_slice()
            .iter()
            .fold(0.0, |sum, value| sum + value.to_f64() * value.to_f64())
            .sqrt();
        self.result.as_mut_slice()[0] = T::from_f64(value);
        Ok(T::from_f64(value))
    }

    /// Maximum absolute value, or zero for an empty vector.
    pub fn max_abs(
        &mut self,
        input: &<CpuBackend as AcceleratorBackend>::Buffer<T>,
    ) -> Result<T, AcceleratorError> {
        self.validate(input.len())?;
        let value = input
            .as_slice()
            .iter()
            .fold(0.0_f64, |max, value| max.max(value.to_f64().abs()));
        self.result.as_mut_slice()[0] = T::from_f64(value);
        Ok(T::from_f64(value))
    }
}
