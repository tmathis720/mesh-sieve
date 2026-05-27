//! Compare the focused FVM distributed QA case across communication modes.
//!
//! ```bash
//! # 1) Serial-style baseline and threaded distributed checks
//! cargo test --test fvm_distributed_qa
//!
//! # 2) MPI variant (same tiny case, two ranks)
//! cargo test --features mpi-support --test mpi_fvm_distributed_qa
//! # or with launcher:
//! mpirun -n 2 cargo test --features mpi-support --test mpi_fvm_distributed_qa -- --nocapture
//! ```
//!
//! The two tests validate:
//! - global conservation parity
//! - interface flux cancellation
//! - near-identical per-cell residuals across partitionings

fn main() {
    println!("Run the commands in this file's docs to compare NoComm, RayonComm, and MpiComm modes.");
}
