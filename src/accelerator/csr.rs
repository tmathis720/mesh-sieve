//! Device-ready sparse matrices for solver interoperability.

use std::collections::HashMap;

use crate::algs::assembly::GlobalCsrPattern;
use crate::discretization::runtime::{ClosureDof, CsrPattern};

use super::plan::{checked_u32, upload};
use super::{AcceleratorBackend, AcceleratorError, CpuBackend, FvmScalar};

/// A validated CSR matrix with device-resident structure and values.
pub struct DeviceCsrMatrix<T: FvmScalar, B: AcceleratorBackend> {
    /// Zero-based row offsets, with `row_count + 1` entries.
    pub row_offsets: B::Buffer<u32>,
    /// Zero-based column indices.
    pub column_indices: B::Buffer<u32>,
    /// Nonzero values in row-major CSR order.
    pub values: B::Buffer<T>,
    /// Number of represented rows.
    pub row_count: usize,
    /// Required input vector length.
    pub column_count: usize,
    /// Reusable implementation-specific SpMV scratch allocation.
    pub workspace: Option<B::Buffer<u8>>,
}

impl<T: FvmScalar, B: AcceleratorBackend> DeviceCsrMatrix<T, B> {
    /// Build a square matrix from a symbolic closure-DOF pattern.
    pub fn from_pattern(
        backend: &B,
        pattern: &CsrPattern,
        values: &[T],
    ) -> Result<Self, AcceleratorError> {
        if pattern.xadj.len() != pattern.rows.len().saturating_add(1) {
            return Err(AcceleratorError::InvalidPlan(format!(
                "CSR has {} row offsets for {} rows",
                pattern.xadj.len(),
                pattern.rows.len()
            )));
        }
        let indices: HashMap<ClosureDof, usize> = pattern
            .rows
            .iter()
            .copied()
            .enumerate()
            .map(|(index, dof)| (dof, index))
            .collect();
        if indices.len() != pattern.rows.len() {
            return Err(AcceleratorError::InvalidPlan(
                "CSR pattern contains duplicate row DOFs".into(),
            ));
        }
        let columns = pattern
            .adjncy
            .iter()
            .map(|dof| {
                indices.get(dof).copied().ok_or_else(|| {
                    AcceleratorError::InvalidPlan(format!(
                        "CSR column {dof:?} has no represented row"
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::from_indices(backend, &pattern.xadj, &columns, values, pattern.rows.len())
    }

    /// Build a compact-row matrix from a globally numbered pattern.
    ///
    /// Output rows follow `pattern.rows`; input columns retain their global
    /// numbers, so `column_count` is one greater than the largest column.
    pub fn from_global_pattern(
        backend: &B,
        pattern: &GlobalCsrPattern,
        values: &[T],
    ) -> Result<Self, AcceleratorError> {
        if pattern.xadj.len() != pattern.rows.len().saturating_add(1) {
            return Err(AcceleratorError::InvalidPlan(format!(
                "global CSR has {} row offsets for {} rows",
                pattern.xadj.len(),
                pattern.rows.len()
            )));
        }
        let column_count = pattern
            .adjncy
            .iter()
            .copied()
            .max()
            .map_or(Ok(0), |value| {
                value.checked_add(1).ok_or(AcceleratorError::IndexOverflow {
                    what: "global CSR column count",
                    value,
                })
            })?;
        Self::from_indices(
            backend,
            &pattern.xadj,
            &pattern.adjncy,
            values,
            column_count,
        )
    }

    fn from_indices(
        backend: &B,
        offsets: &[usize],
        columns: &[usize],
        values: &[T],
        column_count: usize,
    ) -> Result<Self, AcceleratorError> {
        if offsets.is_empty() || offsets[0] != 0 {
            return Err(AcceleratorError::InvalidPlan(
                "CSR row offsets must begin at zero".into(),
            ));
        }
        if offsets.windows(2).any(|pair| pair[0] > pair[1]) {
            return Err(AcceleratorError::InvalidPlan(
                "CSR row offsets must be monotone".into(),
            ));
        }
        if offsets.last().copied() != Some(columns.len()) || values.len() != columns.len() {
            return Err(AcceleratorError::InvalidPlan(format!(
                "CSR terminal offset/column/value lengths disagree: {:?}/{}/{}",
                offsets.last(),
                columns.len(),
                values.len()
            )));
        }
        if let Some(&column) = columns.iter().find(|&&column| column >= column_count) {
            return Err(AcceleratorError::InvalidPlan(format!(
                "CSR column {column} is outside column count {column_count}"
            )));
        }
        checked_u32(offsets.len(), "CSR offset count")?;
        checked_u32(columns.len(), "CSR nonzero count")?;
        // cuSPARSE's 32-bit CSR descriptors interpret these buffers as signed
        // integers. Keep the shared representation valid for both the CPU and
        // CUDA implementations rather than accepting values whose bit pattern
        // would become negative on the device.
        if let Some(&value) = offsets.iter().find(|&&value| value > i32::MAX as usize) {
            return Err(AcceleratorError::IndexOverflow {
                what: "CSR row offset",
                value,
            });
        }
        if let Some(&value) = columns.iter().find(|&&value| value > i32::MAX as usize) {
            return Err(AcceleratorError::IndexOverflow {
                what: "CSR column index",
                value,
            });
        }
        let row_offsets = offsets
            .iter()
            .map(|&value| checked_u32(value, "CSR row offset"))
            .collect::<Result<Vec<_>, _>>()?;
        let column_indices = columns
            .iter()
            .map(|&value| checked_u32(value, "CSR column index"))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            row_offsets: upload(backend, &row_offsets)?,
            column_indices: upload(backend, &column_indices)?,
            values: upload(backend, values)?,
            row_count: offsets.len() - 1,
            column_count,
            workspace: None,
        })
    }
}

impl<T: FvmScalar> DeviceCsrMatrix<T, CpuBackend> {
    /// Compute `y = alpha * A * x + beta * y` in deterministic row order.
    pub fn spmv(
        &self,
        alpha: T,
        x: &<CpuBackend as AcceleratorBackend>::Buffer<T>,
        beta: T,
        y: &mut <CpuBackend as AcceleratorBackend>::Buffer<T>,
    ) -> Result<(), AcceleratorError> {
        if x.as_slice().len() != self.column_count {
            return Err(AcceleratorError::LengthMismatch {
                expected: self.column_count,
                found: x.as_slice().len(),
            });
        }
        if y.as_slice().len() != self.row_count {
            return Err(AcceleratorError::LengthMismatch {
                expected: self.row_count,
                found: y.as_slice().len(),
            });
        }
        for row in 0..self.row_count {
            let mut sum = 0.0;
            for offset in self.row_offsets.as_slice()[row] as usize
                ..self.row_offsets.as_slice()[row + 1] as usize
            {
                let column = self.column_indices.as_slice()[offset] as usize;
                sum += self.values.as_slice()[offset].to_f64() * x.as_slice()[column].to_f64();
            }
            let previous = y.as_slice()[row].to_f64();
            y.as_mut_slice()[row] = T::from_f64(alpha.to_f64() * sum + beta.to_f64() * previous);
        }
        Ok(())
    }
}
