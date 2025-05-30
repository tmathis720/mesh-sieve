use crate::algs::dual_graph::DualGraph;
#[cfg(feature = "metis-support")]
include!("../metis_bindings.rs"); // idx_t, METIS_PartGraphKway, etc.

/// A wrapper around a METIS partition.
pub struct MetisPartition {
    /// for each vertex i, partition[i] ∈ [0..nparts)
    pub part: Vec<i32>,
}

impl DualGraph {
    /// Partition this graph into `nparts` parts using METIS.
    #[cfg(feature = "metis-support")]
    pub fn metis_partition(&self, nparts: i32) -> MetisPartition {
        let n = self.vwgt.len() as idx_t;
        let mut n = n;
        let mut ncon: idx_t = 1;
        let mut nparts = nparts as idx_t;
        let mut xadj: Vec<idx_t> = self.xadj.iter().map(|&u| u as idx_t).collect();
        let mut adjncy: Vec<idx_t> = self.adjncy.iter().map(|&v| v as idx_t).collect();
        let mut vwgt: Vec<idx_t> = self.vwgt.iter().map(|&w| w as idx_t).collect();
        let mut part = vec![0i32; n as usize];

        // METIS options: 0 means “use defaults”
        let mut options = [0; 40];
        options[0] = 1;                   // turn on option processing
        // e.g. options[METIS_OPTION_UFACTOR] = 30;
        let mut objval: idx_t = 0;

        unsafe {
            let ret = METIS_PartGraphKway(
                &mut n,
                &mut ncon,
                xadj.as_mut_ptr(),
                adjncy.as_mut_ptr(),
                vwgt.as_mut_ptr(),
                std::ptr::null_mut(),              // vsize
                std::ptr::null_mut(),              // adjwgt
                &mut nparts,
                std::ptr::null_mut(),              // tpwgts
                std::ptr::null_mut(),              // ubvec
                options.as_mut_ptr(),
                &mut objval,
                part.as_mut_ptr(),
            );
            assert_eq!(ret, 1, "METIS failed");
        }

        MetisPartition { part }
    }
}
