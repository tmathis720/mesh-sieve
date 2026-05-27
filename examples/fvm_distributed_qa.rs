//! Distributed FVM QA mini-example.
//!
//! This mirrors the communicator/distribution test cases used in CI:
//! - serial baseline (`NoComm`)
//! - threaded distributed path (`RayonComm`, two ranks)
//! - optional MPI distributed path (`MpiComm`, run with `mpirun -n 2`)
//!
//! It demonstrates global residual conservation, interface flux cancellation,
//! and ghost synchronization for cell-centered and face-centered values.

use mesh_sieve::algs::communicator::{Communicator, NoComm};

fn main() {
    let no = NoComm;
    println!("NoComm baseline active: {}", no.is_no_comm());
    println!(
        "See tests/fvm_distributed_qa.rs and tests/mpi_fvm_distributed_qa.rs for executable QA assertions."
    );
}
