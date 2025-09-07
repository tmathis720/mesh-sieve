//! METIS partitioning wrapper for dual graphs.
//!
//! Provides a safe Rust interface for METIS-based graph partitioning.

use crate::algs::dual_graph::DualGraph;

#[cfg(feature = "metis-support")]
include!("../metis_bindings.rs"); // idx_t, METIS_PartGraphKway, etc.

#[cfg(feature = "metis-support")]
use once_cell::sync::Lazy;
#[cfg(feature = "metis-support")]
use thiserror::Error;

#[cfg(feature = "metis-support")]
const METIS_OK: i32 = 1;
#[cfg(feature = "metis-support")]
const METIS_ERROR_INPUT: i32 = -2;
#[cfg(feature = "metis-support")]
const METIS_ERROR_MEMORY: i32 = -3;
#[cfg(feature = "metis-support")]
const METIS_ERROR: i32 = -4;
#[cfg(feature = "metis-support")]
const METIS_NOPTIONS: usize = 40;

#[cfg(feature = "metis-support")]
#[derive(Debug, Error)]
pub enum MetisError {
    #[error("CSR is invalid: {0}")]
    InvalidCsr(&'static str),
    #[error("index does not fit into METIS idx_t (overflow)")]
    IdxOverflow,
    #[error("METIS: invalid input")]
    ErrorInput,
    #[error("METIS: memory allocation failed")]
    ErrorMemory,
    #[error("METIS: generic failure")]
    ErrorGeneric,
    #[error("METIS returned code {0}")]
    ErrorCode(i32),
}

#[cfg(feature = "metis-support")]
type MetisResult<T> = Result<T, MetisError>;

#[cfg(feature = "metis-support")]
fn translate_ret(ret: i32) -> MetisResult<()> {
    if ret == METIS_OK {
        Ok(())
    } else if ret == METIS_ERROR_INPUT {
        Err(MetisError::ErrorInput)
    } else if ret == METIS_ERROR_MEMORY {
        Err(MetisError::ErrorMemory)
    } else if ret == METIS_ERROR {
        Err(MetisError::ErrorGeneric)
    } else {
        Err(MetisError::ErrorCode(ret))
    }
}

#[cfg(feature = "metis-support")]
fn validate_csr(xadj: &[usize], adjncy_len: usize) -> MetisResult<()> {
    if xadj.is_empty() {
        return Err(MetisError::InvalidCsr("xadj is empty"));
    }
    let n = xadj.len() - 1;
    if xadj[n] != adjncy_len {
        return Err(MetisError::InvalidCsr("xadj[n] != adjncy.len()"));
    }
    if xadj.windows(2).any(|w| w[0] > w[1]) {
        return Err(MetisError::InvalidCsr("xadj must be nondecreasing"));
    }
    Ok(())
}

#[cfg(feature = "metis-support")]
fn to_idx_vec<T: TryInto<idx_t> + Copy>(src: &[T]) -> MetisResult<Vec<idx_t>> {
    src.iter()
        .map(|&x| x.try_into().map_err(|_| MetisError::IdxOverflow))
        .collect()
}

#[cfg(feature = "metis-support")]
fn default_options() -> Vec<idx_t> {
    let mut opts = vec![0 as idx_t; METIS_NOPTIONS];
    unsafe {
        #[allow(non_snake_case)]
        METIS_SetDefaultOptions(opts.as_mut_ptr());
    }
    opts
}

#[cfg(feature = "metis-support")]
static METIS_MUTEX: Lazy<std::sync::Mutex<()>> = Lazy::new(|| std::sync::Mutex::new(()));

/// A wrapper around a METIS partition.
pub struct MetisPartition {
    /// For each vertex i, `part[i]` ∈ [0..nparts)
    pub part: Vec<i32>,
}

impl DualGraph {
    /// Fallible wrapper around `METIS_PartGraphKway`.
    #[cfg(feature = "metis-support")]
    pub fn try_metis_partition(&self, nparts: i32) -> MetisResult<MetisPartition> {
        let n = self.vwgt.len();
        if self.xadj.len() != n + 1 {
            return Err(MetisError::InvalidCsr("xadj.len() != n+1"));
        }
        validate_csr(&self.xadj, self.adjncy.len())?;

        let mut n_idx: idx_t = n.try_into().map_err(|_| MetisError::IdxOverflow)?;
        let mut ncon: idx_t = 1;
        let mut nparts_idx: idx_t = nparts.try_into().map_err(|_| MetisError::IdxOverflow)?;
        let mut xadj = to_idx_vec(&self.xadj)?;
        let mut adjncy = to_idx_vec(&self.adjncy)?;
        let mut vwgt = if !self.vwgt.is_empty() {
            Some(to_idx_vec(&self.vwgt)?)
        } else {
            None
        };

        let mut part = vec![0i32; n];
        let mut objval: idx_t = 0;
        let mut options = default_options();

        let ret = unsafe {
            METIS_PartGraphKway(
                &mut n_idx,
                &mut ncon,
                xadj.as_mut_ptr(),
                adjncy.as_mut_ptr(),
                vwgt.as_mut()
                    .map(|v| v.as_mut_ptr())
                    .unwrap_or(std::ptr::null_mut()),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut nparts_idx,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                options.as_mut_ptr(),
                &mut objval,
                part.as_mut_ptr(),
            )
        };

        translate_ret(ret)?;
        Ok(MetisPartition { part })
    }

    /// Panicking wrapper that preserves legacy behavior.
    #[cfg(feature = "metis-support")]
    pub fn metis_partition(&self, nparts: i32) -> MetisPartition {
        match self.try_metis_partition(nparts) {
            Ok(p) => p,
            Err(e) => panic!("METIS partition failed: {e}"),
        }
    }

    /// Thread-safe wrapper guarding METIS calls with a global mutex.
    #[cfg(feature = "metis-support")]
    pub fn try_metis_partition_threadsafe(&self, nparts: i32) -> MetisResult<MetisPartition> {
        let _g = METIS_MUTEX.lock().unwrap();
        self.try_metis_partition(nparts)
    }
}

#[cfg(all(test, feature = "metis-support"))]
mod tests {
    use super::*;

    #[test]
    fn metis_ok_on_tiny_chain() {
        let g = DualGraph {
            xadj: vec![0, 1, 2, 3, 3],
            adjncy: vec![1, 2, 3],
            vwgt: vec![1, 1, 1, 1],
        };
        let p = g.try_metis_partition(2).expect("metis should succeed");
        assert_eq!(p.part.len(), 4);
        for &pi in &p.part {
            assert!(0 <= pi && pi < 2);
        }
    }
}
