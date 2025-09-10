# mesh-sieve

mesh-sieve is a modular, high-performance Rust library for mesh and data management, designed for scientific computing and PDE codes. It provides abstractions for mesh topology, field data, parallel partitioning, and communication, supporting both serial and MPI-based distributed workflows.

## Features
- **Mesh Topology**: Flexible, generic Sieve data structures for representing mesh connectivity.
- **Field Data**: Atlas and Section types for mapping mesh points to data arrays.
- **Parallel Communication**: Pluggable communication backends (serial, Rayon, MPI) for ghost exchange and mesh distribution.
- **Partitioning**: Built-in support for graph partitioning (Metis, custom algorithms).
- **MPI Integration**: Examples and tests for distributed mesh and data exchange using MPI.
- **Extensive Testing**: Serial, parallel, and property-based tests.
- **Ergonomic bounds**: `PointLike` and `PayloadLike` marker traits reduce repeated `Copy + Eq + Hash + Ord + Debug` bounds.

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

# Version 1.2.4 — Release Notes

## Highlights

* Add **distributed closures via completion**: call `closure_completed(...)` to traverse across partitions. It fetches remote cones/supports on demand or upfront (policy-driven) and fuses them locally.
* Introduce a minimal **communication layer** (`Communicator`) with **NoComm**, **RayonComm**, and optional **MPI** backends for non-blocking send/recv and barriers.
* Ship a full **section completion pipeline**: `neighbour_links` → `exchange_sizes(_symmetric)` → `exchange_data(_symmetric)` with robust, symmetric handshakes that avoid deadlocks.
* Implement **sieve completion** (`complete_sieve`, `complete_sieve_until_converged`) to synchronize overlap arrows across ranks until convergence.
* Add **stack completion** helpers to mirror section completion for vertical stacks.

## Performance & Determinism

* Replace boxed traversal closures with **concrete iterators**: `closure_iter`, `star_iter`, `closure_both_iter` (zero dyn-dispatch in hot loops).
* Add **point-only adapters** (`cone_points`, `support_points`) to eliminate payload clones during traversal.
* Provide **deterministic traversal order** via a height-major **chart** (index ↔ point) and **bitset** visited sets.
* Make `InMemorySieve` mutators **degree-local** and **prealloc-aware**: `reserve_cone`, `reserve_support`, and incremental mirror updates for `set_cone`/`set_support` (no global rebuilds).
* Extend the same reserve hints to `InMemoryOrientedSieve` and `InMemoryStack`; add bulk helpers (`reserve_from_edges`, `reserve_from_edge_counts`) and `shrink_to_fit` to reclaim memory.

## Correctness Improvements

* Preserve the **remote mesh point id** in overlap completion wires; reconstruct `Remote { rank, remote_point: Some(id) }` exactly on receive.
* Tighten `Overlap::add_link` to respect closure-of-support and avoid duplicate links efficiently.
* Enforce **minimality** in `meet`/`join` (return only minimal shared faces / minimal cofaces).

## API Additions

* `closure_completed(...)` with `CompletionPolicy` (Pre/OnDemand; Cone/Support/Both).
* `Communicator` trait + `NoComm`, `RayonComm`, and (feature-gated) `MpiComm`.
* Completion utilities: `neighbour_links`, `size_exchange`, `data_exchange`, `sieve_completion`, `stack_completion`.
* Deterministic helpers: `chart_points()`, `chart_index()` (height-major chart).

## Optional/Advanced

* Hooks for DAG reachability indices (e.g., GRAIL / chain-labels) to speed redundancy checks and transitive reduction.
* Oriented closure plumbing for assembly into sections (sign-aware accumulation).

## Breaking Changes

* **None.** All new capabilities are additive. Existing calls continue to work; new fast paths are opt-in.

## Upgrade Notes

* Enable MPI backend with `--features mpi-support`.
* Use `closure_local(...)` for legacy behavior; switch to `closure_completed(...)` to traverse globally with overlap-driven completion.
* For best performance, prefer `*_iter` and `cone_points`/`support_points` in custom algorithms.

## Testing & Docs

* Add parity tests for old vs. new traversals.
* Add symmetric comms tests for section/sieve/stack completion.
* Update docs with examples for `closure_completed`, communicator setup, and overlap construction.

## Invariant checks

- By default, internal invariants are verified only in debug builds via `debug_assertions`.
- Continuous integration also exercises these checks in optimized builds by enabling the `strict-invariants` feature.
- End users incur zero overhead in normal release builds where this feature is disabled.

## License
MIT License. See [LICENSE](LICENSE) for details.
