# sieve-rs

sieve-rs is a modular, high-performance Rust library for mesh and data management, designed for scientific computing and PDE codes. It provides abstractions for mesh topology, field data, parallel partitioning, and communication, supporting both serial and MPI-based distributed workflows.

## Features
- **Mesh Topology**: Flexible, generic Sieve data structures for representing mesh connectivity.
- **Field Data**: Atlas and Section types for mapping mesh points to data arrays.
- **Parallel Communication**: Pluggable communication backends (serial, Rayon, MPI) for ghost exchange and mesh distribution.
- **Partitioning**: Built-in support for graph partitioning (Metis, custom algorithms).
- **MPI Integration**: Examples and tests for distributed mesh and data exchange using MPI.
- **Extensive Testing**: Serial, parallel, and property-based tests.

## Getting Started

To build and run the project:

```sh
cargo build
cargo run
```

### Running MPI Examples

Some features require MPI. To run an MPI example (e.g., `mpi_complete.rs`):

```sh
mpirun -n 2 cargo run --example mpi_complete
```

Other examples:
- `mesh_distribute_two_ranks.rs`: Demonstrates mesh distribution across two ranks.
- `mpi_complete_no_overlap.rs`, `mpi_complete_multiple_neighbors.rs`, `mpi_complete_stack.rs`: Test various parallel completion scenarios.

## Project Structure
- `src/`: Library source code
  - `topology/`: Mesh topology and Sieve traits
  - `data/`: Atlas and Section for field data
  - `overlap/`: Overlap and Delta for ghost exchange
  - `algs/`: Algorithms for communication, distribution, completion, partitioning
  - `partitioning/`: Graph partitioning algorithms
- `examples/`: Usage and integration tests (serial and MPI)
- `tests/`: Unit and integration tests
- `benches/`: Benchmarks
- `Cargo.toml`: Project manifest and dependencies

## Optional Features
- `mpi-support`: Enable MPI-based communication and parallel tests
- `metis-support`: Enable Metis-based partitioning (requires Metis and pkg-config)

## License
MIT License. See [LICENSE](LICENSE) for details.
