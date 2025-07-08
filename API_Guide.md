# mesh-sieve API User Guide Outline

## 1. Introduction

* **What is mesh-sieve?**
  mesh-sieve is a Rust library for representing and manipulating mesh topologies, supporting data refinement, assembly, and distributed mesh exchanges. It provides a flexible, composable abstraction for mesh connectivity and associated data, enabling efficient algorithms for scientific computing, finite element analysis, and parallel mesh operations.

* **Key Concepts**
  - **Points:** Abstract identifiers for mesh entities (vertices, edges, faces, cells).
  - **Arrows (Incidence Relations):** Directed connections between points, encoding mesh topology (e.g., which faces are incident to a cell).
  - **Sieves:** Core data structures representing the incidence graph of a mesh.
  - **Stacks:** Structures for composing multiple sieves, modeling hierarchical or fiber bundle relationships.
  - **Strata:** Layers or levels in the mesh, such as topological dimension or distance from the boundary.
  - **Atlas:** Maps mesh points to data offsets, supporting efficient field data storage.
  - **Sections:** Field data associated with mesh points, supporting restriction and assembly.
  - **Bundles:** Combine topology (stacks) and data (sections) for refinement and assembly workflows.
  - **Overlaps:** Represent shared mesh regions between partitions for parallel computation.
  - **Deltas:** Define how data is transferred or refined across mesh relations.
  - **Communicators:** Abstract parallel communication (MPI, Rayon, or serial).

* **Design Goals**
  - **Zero-cost abstractions:** Overhead-free APIs leveraging Rust's type system and inlining.
  - **Composability:** Modular traits and types for building complex mesh and data structures.
  - **Generic over payloads:** Support for arbitrary user-defined data on mesh relations.
  - **Built-in caching & invalidation:** Automatic management of derived data (e.g., strata) for performance.
  - **MPI/Rayon support:** Seamless integration with parallel and distributed runtimes for scalable mesh operations.

## 2. Getting Started

* **Installation & Cargo Features**
  Add `mesh-sieve` to your `Cargo.toml`:
  ```toml
  [dependencies]
  mesh-sieve = "1.0"
  ```
  Optional features:
  - `mpi-support`: Enables distributed mesh exchange via MPI.
  - `metis-support`: Enables graph partitioning using METIS.
  Enable features as needed:
  ```toml
  mesh-sieve = { version = "1.0", features = ["mpi-support"] }
  ```

* **Your First Mesh**

  1. **Create an `InMemorySieve`:**
     ```rust
     use mesh_sieve::InMemorySieve;
     let mut sieve = InMemorySieve::new();
     ```
  2. **Add arrows (`add_arrow`):**
     ```rust
     sieve.add_arrow(0, 1, ());
     sieve.add_arrow(0, 2, ());
     ```
  3. **Traverse with `cone` / `support`:**
     ```rust
     let cone = sieve.cone(0);      // Points incident to 0
     let support = sieve.support(1); // Points for which 1 is in the cone
     ```

* **Quick Example**
  ```rust
  use mesh_sieve::InMemorySieve;

  fn main() {
      let mut sieve = InMemorySieve::new();
      sieve.add_arrow(0, 1, ());
      sieve.add_arrow(0, 2, ());
      sieve.add_arrow(1, 3, ());
      sieve.add_arrow(2, 3, ());

      // Compute strata (e.g., topological heights)
      let strata = sieve.compute_strata().unwrap();
      println!("Strata: {:?}", strata);

      // Serialize to JSON (requires serde feature)
      let json = serde_json::to_string(&sieve).unwrap();
      println!("Serialized mesh: {}", json);
  }
  ```
  This builds a simple mesh, computes its strata, and serializes it.

## 3. Core Topology API

### 3.1. The `Sieve` Trait

* **Basic Incidence**

  The core of the mesh-sieve API is the ability to express and query incidence relations between mesh points.

  - `cone(p)`: Returns an iterator over the points that are directly incident to point `p` (i.e., the "children" or "downward" incidence).
    ```rust
    for q in sieve.cone(p) {
        // q is incident to p
    }
    ```
  - `support(p)`: Returns an iterator over the points for which `p` is in their cone (i.e., the "parents" or "upward" incidence).
    ```rust
    for r in sieve.support(p) {
        // p is in the cone of r
    }
    ```
  - `add_arrow(src, dst, payload)`: Adds an incidence arrow from `src` to `dst` with the given payload.
    ```rust
    sieve.add_arrow(0, 1, ());
    ```
  - `remove_arrow(src, dst)`: Removes the incidence arrow from `src` to `dst`, if it exists.
    ```rust
    sieve.remove_arrow(0, 1);
    ```
* **Global Point Iteration**

  These methods provide iterators over the points in the sieve, allowing you to traverse the mesh topology at different levels:

  - `base_points()`: Iterates over all base points (typically higher-dimensional entities, e.g., cells).
    ```rust
    for p in sieve.base_points() {
        // p is a base point
    }
    ```
  - `cap_points()`: Iterates over all cap points (typically lower-dimensional entities, e.g., vertices, edges).
    ```rust
    for q in sieve.cap_points() {
        // q is a cap point
    }
    ```
  - `points()`: Iterates over all points in the sieve, regardless of their role.
    ```rust
    for r in sieve.points() {
        // r is any point in the mesh
    }
    ```

* **Graph Traversals**

  These methods provide higher-level traversals of the mesh topology, useful for algorithms that require closure or star neighborhoods.

  - `closure(p)`: Returns an iterator over all points reachable from `p` by following cones recursively (the transitive closure of downward incidence).
    ```rust
    for q in sieve.closure(p) {
        // q is in the closure of p
    }
    ```
  - `star(p)`: Returns an iterator over all points that can reach `p` by following supports recursively (the transitive closure of upward incidence).
    ```rust
    for r in sieve.star(p) {
        // r is in the star of p
    }
    ```
  - `closure_both(p)`: Returns an iterator over all points reachable from `p` by following both cones and supports recursively (bidirectional closure).
    ```rust
    for s in sieve.closure_both(p) {
        // s is in the bidirectional closure of p
    }
    ```

* **Lattice Operations**

  These methods provide set-theoretic operations on mesh neighborhoods, useful for algorithms that require combining or comparing topological regions.

  - `meet(a, b)`: Computes the minimal separator between points `a` and `b` (i.e., the intersection of their closures).
    ```rust
    let sep = sieve.meet(a, b);
    // sep contains points in the intersection of closure(a) and closure(b)
    ```
  - `join(a, b)`: Computes the union of the stars of `a` and `b` (i.e., the smallest set containing all points reachable upward from either).
    ```rust
    let union = sieve.join(a, b);
    // union contains points in the union of star(a) and star(b)
    ```

* **Covering Interface (Sec 2.4)**

  The covering interface provides fine-grained control over the mesh topology, allowing you to add or remove points and their incidence relations, as well as perform bulk and subset operations.

  - `add_point(p)`, `remove_point(p)`: Add or remove a point from the sieve, including all associated arrows.
    ```rust
    sieve.add_point(42);
    sieve.remove_point(42);
    ```
  - `add_base_point(p)`, `remove_base_point(p)`: Explicitly add or remove a point from the set of base points.
    ```rust
    sieve.add_base_point(10);
    sieve.remove_base_point(10);
    ```
  - `add_cap_point(p)`, `remove_cap_point(p)`: Explicitly add or remove a point from the set of cap points.
    ```rust
    sieve.add_cap_point(5);
    sieve.remove_cap_point(5);
    ```
  - Bulk mutations:
    - `set_cone(p, iter)`: Replace the cone of `p` with the given iterator of points.
      ```rust
      sieve.set_cone(0, [1, 2, 3].into_iter());
      ```
    - `add_cone(p, iter)`: Add additional points to the cone of `p`.
      ```rust
      sieve.add_cone(0, [4, 5].into_iter());
      ```
    - `set_support(p, iter)`: Replace the support of `p` with the given iterator of points.
      ```rust
      sieve.set_support(1, [0, 2].into_iter());
      ```
    - `add_support(p, iter)`: Add additional points to the support of `p`.
      ```rust
      sieve.add_support(1, [3].into_iter());
      ```
  - Subsetting:
    - `restrict_base(iter)`: Restrict the sieve to only the specified base points.
      ```rust
      sieve.restrict_base([0, 1].into_iter());
      ```
    - `restrict_cap(iter)`: Restrict the sieve to only the specified cap points.
      ```rust
      sieve.restrict_cap([2, 3].into_iter());
      ```

### 3.2. In-Memory Implementation

The primary in-memory implementation of the `Sieve` trait is `InMemorySieve`, which is generic over point and payload types and provides efficient, thread-safe mesh topology storage.

- `InMemorySieve<P, Payload>`: Stores incidence relations as hash maps from points to sets of arrows, supporting fast queries and mutations.
  ```rust
  use mesh_sieve::InMemorySieve;
  let mut sieve = InMemorySieve::<u32, ()>::new();
  sieve.add_arrow(0, 1, ());
  ```
- `from_arrows`: Construct an `InMemorySieve` from an iterator of arrows (tuples of `(src, dst, payload)`).
  ```rust
  let arrows = vec![(0, 1, ()), (1, 2, ())];
  let sieve = InMemorySieve::from_arrows(arrows);
  ```
- Default trait implementations: Provides default methods for all `Sieve` trait operations, including traversal, mutation, and caching.
- Caching via `OnceCell<StrataCache>`: Expensive derived data (such as strata, heights, and closures) are cached and automatically invalidated on mutation for performance.
- `SieveArcPayload`: A wrapper type to expose `Arc<Payload>` for shared, reference-counted payloads, enabling safe sharing of data across threads or mesh partitions.
  ```rust
  use mesh_sieve::{InMemorySieve, SieveArcPayload};
  let mut sieve = InMemorySieve::<u32, SieveArcPayload<String>>::new();
  sieve.add_arrow(0, 1, SieveArcPayload::new("edge".to_string()));
  ```

## 4. Strata & Caching

* **`StrataCache`**:  
  The `StrataCache` stores derived topological information for a sieve, such as:
  - Heights: The number of downward steps from each point to a cap point.
  - Depths: The number of upward steps from each point to a base point.
  - Strata layers: Groupings of points by height or depth (e.g., all vertices, all cells).
  - Diameter: The maximal distance between any two points in the mesh.
  This cache enables efficient queries and avoids recomputation of expensive traversals.

* **`compute_strata`**:  
  Computes the strata for the sieve using a topological sort and dynamic programming. This method determines the heights, depths, and layers for all points, and populates the `StrataCache`. It is automatically called when needed, but can also be invoked explicitly:
  ```rust
  let strata = sieve.compute_strata().unwrap();
  println!("Height of point 0: {:?}", strata.height(0));
  ```

* **`InvalidateCache`** trait & integration into all mutators  
  Any mutation of the sieve (such as adding/removing arrows or points) automatically invalidates the `StrataCache` and other derived data. This is handled via the `InvalidateCache` trait, ensuring that cached information is always consistent with the current topology.

* **Best Practices**: explicit invalidation vs. automatic on mutation  
  - For most users, automatic cache invalidation is sufficient and ensures correctness.
  - For performance-critical code that performs many mutations in a batch, you may wish to explicitly invalidate or recompute caches at controlled points to avoid repeated recomputation.
  - Always prefer using the provided API methods to mutate the sieve, as direct manipulation may bypass cache management.

## 5. Stacks: Vertical Composition

### 5.1. The `Stack` Trait

The `Stack` trait models vertical composition of sieves, such as fiber bundles or hierarchical mesh relationships. It provides methods to relate points between the "base" and "cap" sieves and to mutate the stack topology.

* **Topology Queries**:
  - `lift(base)`: Returns an iterator over all cap points that lie above the given base point (i.e., the fiber over `base`).
    ```rust
    for cap in stack.lift(base) {
        // cap is in the fiber over base
    }
    ```
  - `drop(cap)`: Returns the base point associated with the given cap point (i.e., the projection from cap to base).
    ```rust
    let base = stack.drop(cap);
    // base is the base point for cap
    ```

* **Mutators**:
  - `add_arrow(base, cap, payload)`: Adds a vertical incidence from `base` to `cap` with the given payload, and invalidates any relevant caches.
    ```rust
    stack.add_arrow(base, cap, ());
    ```
  - `remove_arrow(base, cap)`: Removes the vertical incidence from `base` to `cap`, if it exists, and invalidates caches.
    ```rust
    stack.remove_arrow(base, cap);
    ```

* **Access to underlying sieves**:
  - `base()`: Returns a reference to the underlying base sieve.
    ```rust
    let base_sieve = stack.base();
    ```
  - `cap()`: Returns a reference to the underlying cap sieve.
    ```rust
    let cap_sieve = stack.cap();
    ```

### 5.2. In-Memory Stack

The `InMemoryStack` type provides an efficient, in-memory implementation of the `Stack` trait, supporting fast queries and updates for vertical mesh composition.

- `InMemoryStack<Base, Cap, Payload>`:  
  Stores the vertical incidence relations between base and cap points, with optional payloads for each arrow. Internally, it uses bi-directional hash maps for efficient lookup in both directions.

  ```rust
  use mesh_sieve::InMemoryStack;
  let mut stack = InMemoryStack::<u32, u32, ()>::new();
  stack.add_arrow(0, 10, ());
  stack.add_arrow(0, 11, ());
  stack.add_arrow(1, 12, ());
  ```

- **Bi-directional hash maps**:  
  The stack maintains both base-to-cap and cap-to-base mappings, enabling efficient `lift` and `drop` queries.

- **Cache invalidation**:  
  Any mutation (such as adding or removing arrows) automatically invalidates derived caches, ensuring consistency for strata and other computed data.

- **Tests**:  
  Comprehensive unit tests are provided to verify correctness of stack operations, cache invalidation, and integration with sieves.

### 5.3. Composed Stacks

The `ComposedStack` type allows chaining two stacks together, enabling multi-level or hierarchical mesh relationships (e.g., base → cap → fiber).

- `ComposedStack`:  
  Chains two stacks so that the cap of the first stack becomes the base of the second. This enables traversing or refining data across multiple vertical layers.

  ```rust
  use mesh_sieve::{ComposedStack, InMemoryStack};
  let stack1 = InMemoryStack::<u32, u32, ()>::new();
  let stack2 = InMemoryStack::<u32, u32, ()>::new();
  let composed = ComposedStack::new(stack1, stack2);
  ```

- **Safe buffer management**:  
  Ensures that no memory is leaked when composing or deconstructing stacks, even when using reference-counted (`Arc`) payloads.

- **Arc payloads**:  
  Supports `Arc<Payload>` for safe, shared ownership of data across composed stacks and threads.

- **`lift`/`drop` composition semantics**:  
  - `lift(base)`: Recursively lifts a base point through both stacks, returning all cap points at the top level.
    ```rust
    for cap in composed.lift(base) {
        // cap is in the fiber above base through both stacks
    }
    ```
  - `drop(cap)`: Recursively drops a cap point through both stacks to find the corresponding base point.
    ```rust
    let base = composed.drop(cap);
    // base is the original base point for this cap
    ```

## 6. Field Data: Atlas & Section

### 6.1. `Atlas`

The `Atlas` type provides a mapping from mesh points to slices of data in a flat array, supporting efficient storage and access for field data.

- **Mapping `PointId → (offset, len)`**:  
  Each mesh point is mapped to a tuple `(offset, len)`, indicating where its data lives in the underlying vector. This enables variable-sized data per point (e.g., for vector-valued fields or mixed finite elements).

- **Key methods**:
  - `insert(point, offset, len)`: Adds or updates the mapping for a point.
    ```rust
    atlas.insert(42, 10, 3); // Point 42 gets data at offset 10, length 3
    ```
  - `get(point)`: Returns the `(offset, len)` for a point, if present.
    ```rust
    if let Some((offset, len)) = atlas.get(42) {
        // Use offset and len to access data
    }
    ```
  - `points()`: Iterates over all points in the atlas.
    ```rust
    for p in atlas.points() {
        // p is a point with associated data
    }
    ```
  - `remove_point(point)`: Removes the mapping for a point.
    ```rust
    atlas.remove_point(42);
    ```

- **Cache hooks**:  
  The atlas integrates with the caching system, so changes to the mapping (such as adding or removing points) will invalidate relevant caches in associated sections or bundles.

### 6.2. `Section<V>`

The `Section<V>` type represents field data attached to mesh points, supporting efficient storage, restriction, and assembly operations.

- **Backed by an `Atlas` + flat `Vec<V>`**:  
  The section uses an `Atlas` to map each point to a slice of a contiguous vector of values (`Vec<V>`), enabling fast access and compact storage for per-point data.

- **Key methods**:
  - `restrict(point)`: Returns a slice of data for the given point (read-only).
    ```rust
    let values: &[f64] = section.restrict(point);
    ```
  - `restrict_mut(point)`: Returns a mutable slice of data for the given point.
    ```rust
    let values: &mut [f64] = section.restrict_mut(point);
    ```
  - `set(point, data)`: Sets the data for a point, replacing its current values.
    ```rust
    section.set(point, &[1.0, 2.0, 3.0]);
    ```
  - `iter()`: Iterates over all points and their associated data slices.
    ```rust
    for (p, values) in section.iter() {
        // p is a point, values is its data slice
    }
    ```

- **Dynamic updates**:
  - `add_point(point, len, default)`: Adds a new point with a slice of length `len`, initialized to `default`.
    ```rust
    section.add_point(42, 3, 0.0); // Adds point 42 with 3 zeros
    ```
  - `remove_point(point)`: Removes a point and its data from the section.
    ```rust
    section.remove_point(42);
    ```
  - `scatter_from(other_section)`: Copies data from another section with compatible layout.
    ```rust
    section.scatter_from(&other_section);
    ```

- **Cache integration**:  
  All dynamic updates and mutations automatically invalidate relevant caches, ensuring consistency for derived data and parallel operations.

## 7. Data Refinement & Assembly

### 7.1. The `Delta<V>` Trait

The `Delta<V>` trait defines how field data is refined or assembled across mesh relations, supporting both restriction (splitting data) and fusion (combining data).

- `restrict(v) -> Part`:  
  Given a value `v` (e.g., a field value on a base point), produces a "part" suitable for a cap point (e.g., a subvector or transformed value). Used for pushing data from coarse to fine mesh entities.
  ```rust
  let part = delta.restrict(&value);
  ```

- `fuse(local, incoming)`:  
  Combines a local value and an incoming value (e.g., from another mesh entity or partition), producing a fused result. Used for assembling data from fine to coarse mesh entities.
  ```rust
  let fused = delta.fuse(local, incoming);
  ```

- **Built-in implementations**:
  - `CopyDelta`: Performs a direct copy of data (no transformation).
  - `AddDelta`: Sums values, useful for additive assembly (e.g., finite element residuals).
  - `Orientation`-based deltas: Handle sign or permutation changes when transferring data across oriented mesh relations.

These mechanisms enable flexible and efficient data movement for mesh refinement, assembly, and parallel exchange workflows.

### 7.2. Section-Based Workflows

* **`Bundle<V,D>`**: ties `InMemoryStack`, `Section`, and `Delta`

  The `Bundle<V, D>` type encapsulates a stack, a section, and a delta, providing high-level methods for mesh data refinement and assembly.

  - `refine(bases)`:  
    Pushes data from base points to cap points using the stack and delta. For each base point in `bases`, the bundle applies the delta's `restrict` method to the section data and distributes the result to the corresponding cap points.
    ```rust
    bundle.refine(&[base0, base1]);
    ```

  - `assemble(bases)`:  
    Pulls data from cap points back to base points, combining contributions using the delta's `fuse` method. This is typically used for assembling global data from local contributions.
    ```rust
    bundle.assemble(&[base0, base1]);
    ```

  - `dofs(p)`:  
    Returns the degrees of freedom (data slice) associated with a given cap point, allowing inspection or modification of per-cap data.
    ```rust
    let dofs = bundle.dofs(cap_point);
    ```

  These workflows enable efficient and safe data movement for mesh refinement, restriction, and assembly, abstracting over the details of the underlying topology and data layout.

### 7.3. Refinement Helpers

Refinement helpers provide convenient functions for restricting or assembling data over mesh neighborhoods, supporting both closure and star traversals.

- `restrict_closure(section, sieve, point, delta)`:  
  Restricts data from a section to the closure of a given point, applying the provided delta for each relation. Returns a vector of restricted values.
  ```rust
  let values = restrict_closure(&section, &sieve, point, &delta);
  ```

- `restrict_star(section, sieve, point, delta)`:  
  Restricts data from a section to the star of a given point, useful for assembling contributions from all parents.
  ```rust
  let values = restrict_star(&section, &sieve, point, &delta);
  ```

- `_vec` variants:  
  The `_vec` variants (e.g., `restrict_closure_vec`) return vectors of values for all points in the closure or star, enabling batch operations.

- `ReadOnlyMap` wrapper:  
  Wraps a section or map to provide a read-only interface, ensuring that refinement helpers cannot mutate the underlying data.
  ```rust
  let readonly = ReadOnlyMap::new(&section);
  ```

These helpers simplify common data movement patterns in mesh refinement and assembly, and are designed to work efficiently with both serial and parallel traversals.

## 8. Mesh Distribution & Partitioning

* **`distribute_mesh<S,C>(...)`** helper  
  The `distribute_mesh` function automates the process of partitioning a mesh for parallel or distributed computation. It typically performs the following steps:
  1. Builds an overlap sieve that describes which points are shared between partitions.
  2. Extracts the local submesh for each partition, including all necessary points and arrows.
  3. Completes the sieve by ensuring all required incidence relations are present for local computation.
  4. Returns the local mesh and its associated overlap structure for communication.

  Example usage:
  ```rust
  let (local_sieve, overlap) = distribute_mesh(&global_sieve, partition_info, comm);
  ```

  This enables scalable mesh computations by handling the details of data migration, overlap construction, and partition completion.

* Integration Tests: MPI + NoComm  
  The library includes integration tests for mesh distribution and partitioning, covering both MPI-based and serial (NoComm) scenarios. These tests ensure correctness of overlap construction, submesh extraction, and communication patterns across a variety of mesh topologies and partitioning strategies.

## 9. Overlap & Parallel Exchange

### 9.1. The `Overlap` Sieve

The `Overlap` sieve encodes the relationships between local and remote mesh points across partitions, enabling parallel data exchange and ghost region management.

- `type Overlap = InMemorySieve<PointId, Remote>`:  
  The overlap is represented as a sieve where each arrow connects a local point to a `Remote` (a tuple of `(rank, remote_point)`), indicating that the local point is shared with another process.

- `add_link(local, rank, remote_point)`:  
  Adds a link from a local point to a remote point on the specified rank.
  ```rust
  overlap.add_link(local_point, remote_rank, remote_point);
  ```

- **Neighbor queries**:
  - `neighbours(my_rank)`: Returns an iterator over all ranks that share points with the current process.
    ```rust
    for nbr in overlap.neighbours(my_rank) {
        // nbr is a neighboring rank
    }
    ```
  - `links_to(nbr, my_rank)`: Returns all local points that are linked to the given neighbor rank.
    ```rust
    for (local, remote) in overlap.links_to(nbr, my_rank) {
        // local is a local point, remote is the corresponding remote point on nbr
    }
    ```

The overlap sieve is central to distributed mesh algorithms, supporting ghost exchange, partition completion, and parallel assembly.

### 9.2. Communicators

The communicator abstraction enables mesh data exchange and synchronization across threads or processes, supporting both serial and parallel backends.

- **`Communicator` trait**:  
  Defines the core interface for communication:
  - `isend(rank, data)`: Initiates a non-blocking send of data to the specified rank.
  - `irecv(rank, buffer)`: Initiates a non-blocking receive of data from the specified rank into the buffer.
  - `rank()`: Returns the rank (ID) of the current process or thread.
  - `size()`: Returns the total number of ranks in the communicator.

- **No-op `NoComm` for serial CI**:  
  A dummy communicator that implements the trait but performs no actual communication. Useful for single-process tests and CI environments.

- **In-process `RayonComm`**:  
  A communicator that uses Rayon threads for parallel mesh operations within a single process, enabling shared-memory parallelism.

- **MPI backend `MpiComm`**:  
  A communicator that wraps MPI non-blocking send/receive operations, supporting distributed mesh exchange across multiple processes. Handles message handles and provides a `Wait` mechanism for completion.

These communicator types allow the same mesh code to run efficiently in serial, threaded, or distributed environments with minimal changes.

### 9.3. Completion Algorithms

Completion algorithms synchronize and finalize distributed mesh data structures, ensuring consistency across partitions after communication.

- `complete_sieve`:  
  Completes a distributed sieve by exchanging and merging incidence relations with neighboring partitions. Ensures that all required arrows (including ghosts and overlaps) are present locally.
  ```rust
  complete_sieve(&mut local_sieve, &overlap, &comm);
  ```

- `complete_section`:  
  Completes a distributed section (field data) by exchanging values for shared points, using the overlap and communicator. Handles both restriction (push) and assembly (pull) as needed.
  ```rust
  complete_section(&mut section, &overlap, &comm);
  ```

- `complete_stack`:  
  Completes a distributed stack, synchronizing vertical relations and associated data across partitions.
  ```rust
  complete_stack(&mut stack, &overlap, &comm);
  ```

- **Two‐phase send/receive, cache management**:  
  Completion algorithms typically use a two-phase protocol:
  1. Post all non-blocking sends and receives for required data.
  2. Wait for completion, then merge incoming data into local structures.
  All relevant caches (e.g., strata, overlap) are invalidated and recomputed as needed to ensure correctness after completion.

## 10. Testing & Examples

* Overview of built-in unit tests for each module  
  Each core module in mesh-sieve includes comprehensive unit tests covering:
  - Sieve construction, mutation, and traversal
  - Stack composition and cache invalidation
  - Section and atlas data access and updates
  - Delta refinement and assembly logic
  - Overlap construction and communicator integration
  These tests ensure correctness for both serial and parallel (Rayon/MPI) backends.

* Example applications under `/examples`:

  * `mpi_complete.rs`: Demonstrates distributed mesh completion and ghost exchange using MPI.
  * `mpi_complete_stack.rs`: Shows stack completion and data refinement across partitions.
  * `mpi_complete_no_overlap.rs`: Tests mesh completion in the absence of overlap (single partition).
  * `mpi_complete_multiple_neighbors.rs`: Handles meshes with multiple neighbor partitions and complex overlaps.

* How to run MPI examples  
  To run the MPI-based examples, use `mpirun` or `mpiexec` with the desired number of processes:
  ```sh
  mpirun -n 4 cargo run --example mpi_complete
  ```
  Ensure that the `mpi-support` feature is enabled in your `Cargo.toml` and that your environment has an MPI implementation installed.  
  For serial or Rayon-based examples, simply run with `cargo run --example <example_name>`.

## 11. Best Practices & Performance

* **Caching**: where and when to invalidate  
  - Rely on automatic cache invalidation for most use cases; caches are cleared on any topology or data mutation.
  - For batch updates, consider explicit cache invalidation or recomputation after all changes to avoid repeated recomputation.
  - Avoid direct manipulation of internal data structures; always use API methods to ensure cache consistency.

* **Parallelism**: choosing Rayon vs. MPI  
  - Use Rayon for shared-memory parallelism (multi-threaded, single process); enables fast, lock-free mesh traversals and data refinement.
  - Use MPI for distributed-memory parallelism (multi-process, cluster or HPC); enables mesh partitioning, ghost exchange, and scalable assembly.
  - The API is designed to allow switching between Rayon and MPI with minimal code changes by selecting the appropriate communicator.

* **Memory Safety**: no leaky buffers, zero-cost abstractions  
  - All mesh data structures use safe Rust idioms (e.g., `Arc`, `Vec`, `HashMap`) to prevent memory leaks and ensure thread safety.
  - Zero-cost abstractions: iterators, trait objects, and generics are used to avoid runtime overhead.

* **Payload Choices**: `Arc`, custom payloads, `bytemuck`-compatible types  
  - Use `Arc<Payload>` for shared, reference-counted data (e.g., in parallel or distributed settings).
  - Custom payloads can be attached to arrows for orientation, weights, or user metadata.
  - For high-performance data exchange, use `bytemuck`-compatible types to enable zero-copy serialization and efficient MPI communication.

## 12. Appendix

* **Reference Tables**

  * Trait methods by module  
    | Module         | Trait/Type         | Key Methods                                      |
    |----------------|--------------------|--------------------------------------------------|
    | sieve          | `Sieve`            | `cone`, `support`, `add_arrow`, `remove_arrow`, `base_points`, `cap_points`, `points`, `closure`, `star`, `closure_both`, `meet`, `join`, `add_point`, `remove_point`, `set_cone`, `set_support`, `restrict_base`, `restrict_cap` |
    | stack          | `Stack`            | `lift`, `drop`, `add_arrow`, `remove_arrow`, `base`, `cap` |
    | atlas          | `Atlas`            | `insert`, `get`, `points`, `remove_point`        |
    | section        | `Section`          | `restrict`, `restrict_mut`, `set`, `iter`, `add_point`, `remove_point`, `scatter_from` |
    | delta          | `Delta`            | `restrict`, `fuse`                               |
    | communicator   | `Communicator`     | `isend`, `irecv`, `rank`, `size`                 |
    | overlap        | `Overlap`          | `add_link`, `neighbours`, `links_to`             |
    | completion     | -                  | `complete_sieve`, `complete_section`, `complete_stack` |

  * Typical type signatures  
    | Function/Method         | Signature Example                                                                 |
    |------------------------ |----------------------------------------------------------------------------------|
    | `cone`                  | `fn cone(&self, p: PointId) -> impl Iterator<Item = PointId>`                    |
    | `add_arrow`             | `fn add_arrow(&mut self, src: PointId, dst: PointId, payload: Payload)`          |
    | `restrict` (Section)    | `fn restrict(&self, p: PointId) -> &[V]`                                         |
    | `restrict` (Delta)      | `fn restrict(&self, v: &V) -> Part`                                              |
    | `fuse`                  | `fn fuse(&self, local: &mut V, incoming: &V)`                                    |
    | `lift`                  | `fn lift(&self, base: BaseId) -> impl Iterator<Item = CapId>`                    |
    | `drop`                  | `fn drop(&self, cap: CapId) -> BaseId`                                           |
    | `insert` (Atlas)        | `fn insert(&mut self, p: PointId, offset: usize, len: usize)`                    |
    | `add_link` (Overlap)    | `fn add_link(&mut self, local: PointId, rank: usize, remote: PointId)`           |
    | `isend`                 | `fn isend<T: Serialize>(&self, rank: usize, data: &T)`                           |
    | `irecv`                 | `fn irecv<T: DeserializeOwned>(&self, rank: usize, buffer: &mut T)`              |
    | `complete_sieve`        | `fn complete_sieve<S: Sieve, C: Communicator>(...)`                              |

* **Mapping to Knepley & Karpeev (2009)**

  This section provides a cross-reference between mesh-sieve concepts and the original PETSc Sieve paper by Knepley & Karpeev (2009):

  * **Section 2.4 covering interface**  
    - mesh-sieve's "covering interface" (see `add_point`, `remove_point`, `set_cone`, `set_support`, `restrict_base`, `restrict_cap`) directly corresponds to the covering interface described in Section 2.4 of the paper.
    - The API supports explicit manipulation of base/cap points and bulk incidence mutations, mirroring the mathematical covering operations.

  * **Section 3 fiber-bundle semantics**  
    - The `Stack` trait and its implementations (`InMemoryStack`, `ComposedStack`) model the fiber-bundle semantics described in Section 3.
    - The `lift` and `drop` methods correspond to the fiber and projection operations, enabling vertical composition of mesh topologies.
    - The `Bundle` type combines stacks and sections to support data refinement and assembly workflows, as described in the fiber-bundle context.

  For further details, see the original paper:  
  Knepley, M. G., & Karpeev, D. A. (2009). "Mesh Algorithms for PDE with Sieve I: Mesh Distribution." *Scientific Programming*, 17(3), 215–230.

* **Troubleshooting & FAQ**
