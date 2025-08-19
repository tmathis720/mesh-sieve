# mesh-sieve API User Guide

## 1. Introduction

**mesh-sieve** is a Rust library for representing mesh topologies and field data, supporting refinement, assembly, and distributed exchange. It provides abstractions for mesh connectivity and associated data, enabling efficient algorithms for scientific computing and parallel mesh operations.

**Key concepts:**
- **Points:** Mesh entity identifiers (vertices, edges, faces, cells)
- **Arrows:** Directed incidence relations between points
- **Sieves:** Core topology data structures representing mesh connectivity
- **Stacks:** Vertical composition modeling hierarchical relationships
- **Strata:** Topological levels/dimensions computed from connectivity
- **Atlas/Section:** Maps points to field data with efficient access
- **Bundles:** Combine topology and data for refinement/assembly workflows
- **Overlap:** Shared regions between mesh partitions
- **Communicators:** Abstract parallel communication (MPI, threads, serial)

## 2. Getting Started

Add to `Cargo.toml`:
```toml
[dependencies]
mesh-sieve = "1.0"

# Optional features:
# - mpi-support: MPI backend for distributed runs
# - metis-support: METIS partitioning helpers
# mesh-sieve = { version = "1.0", features = ["mpi-support", "metis-support"] }
````

**Simple example:**

```rust
use mesh_sieve::topology::sieve::InMemorySieve;
use mesh_sieve::topology::sieve::Sieve;
use mesh_sieve::topology::point::PointId;

let mut sieve = InMemorySieve::default();
let p1 = PointId::new(1).unwrap();
let p2 = PointId::new(2).unwrap();

// Add incidence relation p1 -> p2
sieve.add_arrow(p1, p2, ());

// Query topology (point-only iteration avoids payload clones)
for q in sieve.cone_points(p1) {
    println!("{} -> {}", p1.get(), q.get());
}

// Strata queries (computed lazily; errors on cycles)
println!("height({}): {:?}", p2.get(), sieve.height(p2));
println!("depth({}): {:?}", p1.get(), sieve.depth(p1));
println!("diameter():  {:?}", sieve.diameter());
```

## 3. Core Topology: Sieves

### 3.1. The `Sieve` Trait

**Basic incidence operations**

* `cone(p) -> Iterator<Item=(Point, Payload)>`
* `support(p) -> Iterator<Item=(Point, Payload)>`
* `add_arrow(src, dst, payload)`
* `remove_arrow(src, dst) -> Option<Payload>`

**Point-only adapters (preferred in hot paths)**

* `cone_points(p) -> Iterator<Item=Point>`
* `support_points(p) -> Iterator<Item=Point>`

Most algorithms should use the point-only adapters to avoid cloning payloads.

```rust
// Build triangle: face -> edges -> vertices
sieve.add_arrow(face, e1, ());
sieve.add_arrow(face, e2, ());
sieve.add_arrow(e1, v1, ());
sieve.add_arrow(e1, v2, ());

// Point-only iteration
for v in sieve.cone_points(e1) {
    // v1, v2
}
```

**Point iteration**

* `base_points()`, `cap_points()`, `points()`

**Traversal methods**

* Boxed, compatibility layer:

  * `closure(seeds)`, `star(seeds)`, `closure_both(seeds)`
* Concrete, zero dyn-dispatch:

  * `closure_iter(seeds)`, `star_iter(seeds)`, `closure_both_iter(seeds)`

```rust
// Prefer concrete iterators in performance-sensitive code:
let reach: Vec<_> = sieve.closure_iter([p]).collect();
```

**Lattice operations (minimal semantics)**

* `meet(a, b)`: minimal shared sub-entities in `closure(a) ∩ closure(b)`
* `join(a, b)`: minimal cofaces from `star(a) ∪ star(b)`

Both return sorted & deduplicated minimal sets (no element contains another under the respective reachability).

**Covering & updates**

* `add_point(p)`, `remove_point(p)`
* `set_cone(p, iter)`, `add_cone(p, iter)`
* `set_support(q, iter)`, `add_support(q, iter)`
* `restrict_base(iter)`, `restrict_cap(iter)`

**Preallocation hints (new)**

* `reserve_cone(p, additional)`
* `reserve_support(q, additional)`

Implementations may use these to reduce reallocations during bulk edits.

### 3.2. `InMemorySieve<P, Payload>`

Primary implementation storing adjacency in hash maps. Mirrors stay consistent with **degree-local** updates:

* `set_cone` / `set_support` update only mirrors of the touched point
* `add_cone` / `add_support` incrementally update both directions
* `reserve_cone` / `reserve_support` help preallocate

```rust
use mesh_sieve::topology::sieve::{InMemorySieve, Sieve};
use mesh_sieve::topology::point::PointId;

let mut s = InMemorySieve::<PointId, ()>::default();
let a = PointId::new(1).unwrap();
let b = PointId::new(2).unwrap();

s.reserve_cone(a, 4);
s.add_cone(a, [(b, ())]);
```

## 4. Traversal & Completion

### 4.1. Local traversals

* `closure_iter(seeds)` / `star_iter(seeds)` / `closure_both_iter(seeds)`
* `closure()` / `star()` / `closure_both()` keep the legacy boxed API

### 4.2. Distributed closures via completion (new)

Run closure across partitions by fetching missing adjacency over an **Overlap** using a **Communicator** and fusing the result locally.

* `closure_completed(sieve, seeds, overlap, comm, my_rank, policy) -> Vec<Point>`

**Policy**

* Direction: `Cone`, `Support`, or `Both`
* Timing: `Pre` (prefetch up to a depth) or `OnDemand` (fetch when needed)
* Batch size for on-demand requests

```rust
use mesh_sieve::algs::traversal::{closure_completed, CompletionPolicy};
use mesh_sieve::overlap::overlap::Overlap;
use mesh_sieve::algs::communicator::NoComm; // or RayonComm/MpiComm

let mut local = InMemorySieve::<PointId,()>::default();
let overlap: Overlap = InMemorySieve::default();

// ... fill local & overlap ...

let comm = NoComm;
let seeds = [PointId::new(42).unwrap()];
let policy = CompletionPolicy::cone_ondemand();
let global_reach = closure_completed(&mut local, seeds, &overlap, &comm, 0, policy);
```

## 5. Strata & Derived Data

`StrataCache` stores computed topological information:

* **Heights:** distance from sources (downward)
* **Depths:** distance to sinks (upward)
* **Strata layers:** points grouped by height
* **Diameter:** maximum height

**Queries on `Sieve`**

* `height(p) -> Result<u32, MeshSieveError>`
* `depth(p) -> Result<u32, MeshSieveError>`
* `diameter() -> Result<u32, MeshSieveError>`
* `height_stratum(k) -> Iterator<Point>`
* `depth_stratum(k) -> Iterator<Point>`

Caches invalidate on mutation and recompute lazily.

## 6. Stacks: Vertical Composition

The `Stack` trait models vertical relationships between "base" and "cap" sieves.

```rust
use mesh_sieve::topology::stack::InMemoryStack;
use mesh_sieve::topology::point::PointId;

let mut stack = InMemoryStack::new();
let base = PointId::new(1).unwrap();
let cap1 = PointId::new(10).unwrap();

stack.add_arrow(base, cap1, ());
for (cap, _) in stack.lift(base) {
    println!("{} ↑ {}", base.get(), cap.get());
}
```

Key methods: `lift`, `drop`, `add_arrow`, `base()`, `cap()`.
Composed stacks enable multi-level hierarchies.

## 7. Field Data: Atlas & Section

**Atlas** declares layout (sparse, per-point lengths).
**Section<V>** stores values over an Atlas.

```rust
use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::section::Section;
use mesh_sieve::topology::point::PointId;

let p = PointId::new(1).unwrap();
let mut atlas = Atlas::default();
atlas.try_insert(p, 3).unwrap();

let mut sec = Section::<f64>::new(atlas);
sec.try_set(p, &[1.0,2.0,3.0]).unwrap();

let vals = sec.try_restrict(p).unwrap(); // &[f64]
```

All accessors are error-safe (`Result<..>`). Deprecated `restrict/set/insert` variants exist for compatibility.

**Delta<V>** defines how to restrict/fuse values during exchange.
Built-ins: `CopyDelta` (clone/overwrite), others may be provided for additive/assembly semantics.

## 8. Partitioning & METIS Integration

With `metis-support`, you can build a dual graph and run METIS to partition cells.
Use `distribute_mesh` (with a `Communicator`) to construct the local submesh and its `Overlap`.

```rust
#[cfg(feature = "mpi-support")]
{
    use mesh_sieve::algs::distribute::distribute_mesh;
    let (local, overlap) = distribute_mesh(&global, &partition, &comm).unwrap();
}
```

## 9. Parallel Computing

### 9.1. Communicators

`Communicator` abstracts non-blocking send/recv:

* **NoComm**: serial/testing
* **RayonComm**: in-process “ranks” (threads)
* **MpiComm** *(feature `mpi-support`)*: real MPI

```rust
use mesh_sieve::algs::communicator::{Communicator, NoComm};

let comm = NoComm;
let send = comm.isend(0, 0xCAFE, &[1,2,3,4]);
let mut buf = [0u8; 4];
let recv = comm.irecv(0, 0xCAFE, &mut buf);

let _ = send.wait();
let got = recv.wait().unwrap();
```

### 9.2. Overlap

An `Overlap` is a `Sieve<PointId, Remote>` describing sharing:
`local_point ──(Remote{rank, remote_point})──▶ partition_point(rank)`

```rust
use mesh_sieve::overlap::overlap::{Overlap, Remote};
let mut ovlp = Overlap::default();
ovlp.add_arrow(local_p, remote_partition_pt, Remote { rank: nbr, remote_point: remote_p });
```

### 9.3. Completion Algorithms

* **Sieve completion (topology):**

  * `complete_sieve(&mut overlap, &overlap_clone, &comm, my_rank)`
  * `complete_sieve_until_converged(..)`
  * Preserves `Remote { rank, remote_point }` exactly.

* **Section completion (data):**

  * `neighbour_links(&section, &overlap, my_rank) -> map<rank, (send_loc, recv_loc)>`
  * `exchange_sizes_symmetric(..)` then `exchange_data_symmetric::<V, D, _>(..)`
  * `complete_section::<V, D, _>(&mut section, &mut overlap, &comm, &delta, my_rank, n_ranks)`

* **Stack completion (vertical):**

  * `complete_stack(&mut stack, &overlap, &comm, my_rank, n_ranks)`

All routines use symmetric handshakes and return `Result<(), MeshSieveError>`.

## 10. Adjacency & Lattice Helpers

**Neighbor queries**

* `adjacent(&mut sieve, p) -> Vec<Point>`: cell-to-cell by shared face
* `adjacent_with(&mut sieve, p, policy)`: select composition policy (e.g., face-to-volume)

**Meet/Join minimality**
Results are sorted, deduped, and minimal/maximal as appropriate.

## 11. Error Handling & Best Practices

* All fallible operations return `Result<T, MeshSieveError>`.
* Traversals use point-only adapters to avoid payload cloning.
* Mutations are **degree-local**; call `reserve_cone`/`reserve_support` before bulk edits.
* For large traversals, prefer concrete iterators (`*_iter`) over boxed variants.
* For distributed runs, keep tag ranges distinct per algorithm (e.g., sieve/section/closure).

## 12. Examples & Testing

**Run examples**

```bash
# Serial
cargo run --example partition

# MPI (requires feature)
cargo mpirun -n 4 --features mpi-support --example mpi_complete
```

**Key examples**

* `mpi_complete.rs`: section completion across ranks
* `mpi_complete_stack.rs`: stack completion
* `distribute_mpi.rs`: mesh distribution
* `closure_completed.rs`: distributed closures via completion

**Testing**
Unit tests cover topology, traversal parity (boxed vs. concrete), degree-local updates, completion handshakes, and error propagation.

## 13. API Reference (At a Glance)

| Component      | Key Methods                                                                 | Purpose                   |
| -------------- | --------------------------------------------------------------------------- | ------------------------- |
| `Sieve`        | `cone(_), support(_)`, `cone_points(_), support_points(_)`, `*_iter`        | Mesh topology & traversal |
| `Stack`        | `lift(_), drop(_)`                                                          | Vertical composition      |
| `Atlas`        | `try_insert, get`                                                           | Layout                    |
| `Section<V>`   | `try_restrict(_), try_restrict_mut(_), try_set(_)`                          | Field data                |
| `Delta<V>`     | `restrict, fuse`                                                            | Exchange semantics        |
| `Communicator` | `isend, irecv, rank, size, barrier`                                         | Parallel comms            |
| Completion     | `complete_sieve`, `complete_section`, `complete_stack`, `closure_completed` | Sync topology & data      |

**Common signatures**

```rust
fn cone_points(&self, p: PointId) -> impl Iterator<Item=PointId>;
fn closure_iter<'a,I>(&'a self, seeds: I) -> impl Iterator<Item=PointId>
  where I: IntoIterator<Item=PointId>;

fn height(&mut self, p: PointId) -> Result<u32, MeshSieveError>;
fn complete_section<V,D,C>(section: &mut Section<V>, overlap: &mut Overlap, comm: &C,
                           delta: &D, my_rank: usize, n_ranks: usize) -> Result<(), MeshSieveError>;

fn closure_completed<S,C,I>(sieve: &mut S, seeds: I, overlap: &Overlap, comm: &C,
                            my_rank: usize, policy: CompletionPolicy) -> Vec<PointId>
  where S: Sieve<Point=PointId>;
```