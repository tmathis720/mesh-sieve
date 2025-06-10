# sieve-rs

A simple Rust project scaffolded with Cargo.

## Getting Started

To build and run the project:

```powershell
cargo build
cargo run
```

## MPI Integration Tests

Some tests that require true parallel/MPI communication have been migrated from the unit test framework to integration tests in `examples/mpi_complete.rs`. To run these tests, use `mpirun` (or `mpiexec`) with the appropriate number of processes:

```sh
mpirun -n 3 cargo run --example mpi_complete
```

This will execute the MPI-based integration tests, such as `mpi_test_complete_section_no_overlap`, `mpi_test_complete_section_multiple_neighbors`, and `mpi_test_complete_stack_two_ranks`.

## Project Structure
- `src/main.rs`: Entry point of the application.
- `Cargo.toml`: Project manifest.

## License
MIT
