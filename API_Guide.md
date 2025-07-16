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
# mesh-sieve = { version = "1.0", features = ["mpi-support", "metis-support"] }
```

**Simple example:**
```rust
use mesh_sieve::topology::sieve::InMemorySieve;
use mesh_sieve::topology::point::PointId;

let mut sieve = InMemorySieve::default();
let p1 = PointId::new(1).unwrap();
let p2 = PointId::new(2).unwrap();

// Add incidence relation
sieve.add_arrow(p1, p2, ());

// Query topology
for cap in sieve.cone(p1) {
    println!("Point {} connects to {}", p1.get(), cap.get());
}

// Compute strata
if let Ok(strata) = sieve.compute_strata() {
    println!("Height of point {}: {:?}", p1.get(), strata.height(p1));
}
```

## 3. Core Topology: Sieves

### 3.1. The `Sieve` Trait

**Basic incidence operations:**
- `cone(p)`: Iterator over points directly incident to `p` (downward arrows)
- `support(p)`: Iterator over points having `p` in their cone (upward arrows)  
- `add_arrow(src, dst, payload)`: Add incidence relation
- `remove_arrow(src, dst)`: Remove incidence relation

```rust
// Build triangle: vertex → edge → face
sieve.add_arrow(face, edge1, ());
sieve.add_arrow(face, edge2, ());
sieve.add_arrow(edge1, vertex1, ());
sieve.add_arrow(edge1, vertex2, ());
```

**Point iteration:**
- `base_points()`: Higher-dimensional entities (cells, faces)
- `cap_points()`: Lower-dimensional entities (vertices, edges)
- `points()`: All points in the sieve

**Traversal methods:**
- `closure(p)`: All points reachable from `p` by following cones
- `star(p)`: All points that can reach `p` by following supports
- `closure_both(p)`: Bidirectional closure combining cone/support paths

**Lattice operations:**
- `meet(a, b)`: Intersection of closures (minimal separator)
- `join(a, b)`: Union of stars (maximal encompassing set)

**Covering interface:**
- `add_point(p)`, `remove_point(p)`: Manage individual points
- `set_cone(p, iter)`: Replace cone of `p` with new points
- `add_cone(p, iter)`: Add points to existing cone
- `restrict_base(iter)`, `restrict_cap(iter)`: Filter to subset

### 3.2. `InMemorySieve<P, Payload>`

Primary implementation storing adjacency as hash maps:

```rust
use mesh_sieve::topology::sieve::InMemorySieve;
use mesh_sieve::topology::point::PointId;

let mut sieve = InMemorySieve::<PointId, ()>::default();
sieve.add_arrow(PointId::new(1).unwrap(), PointId::new(2).unwrap(), ());

// Construction from arrows
let arrows = vec![
    (PointId::new(1).unwrap(), PointId::new(2).unwrap(), ()),
    (PointId::new(2).unwrap(), PointId::new(3).unwrap(), ()),
];
let sieve = InMemorySieve::from_arrows(arrows);
```

Features automatic caching of derived data (strata, heights) with invalidation on mutation.

## 4. Strata & Derived Data

`StrataCache` stores computed topological information:
- **Heights:** Downward distance to cap points
- **Depths:** Upward distance to base points  
- **Strata layers:** Points grouped by height/depth
- **Diameter:** Maximum distance between any two points

**Computing strata:**
```rust
let strata = sieve.compute_strata()?;
println!("Height of point {}: {:?}", point.get(), strata.height(point));
println!("Depth of point {}: {:?}", point.get(), strata.depth(point));
```

Caches are automatically invalidated on sieve mutations and lazily recomputed when needed.

## 5. Stacks: Vertical Composition

The `Stack` trait models vertical relationships between "base" and "cap" sieves:

```rust
use mesh_sieve::topology::stack::InMemoryStack;
use mesh_sieve::topology::point::PointId;

let mut stack = InMemoryStack::new();
let base = PointId::new(1).unwrap();
let cap1 = PointId::new(10).unwrap();
let cap2 = PointId::new(11).unwrap();

// Add vertical arrows
stack.add_arrow(base, cap1, ());
stack.add_arrow(base, cap2, ());

// Query relationships
for (cap, _payload) in stack.lift(base) {
    println!("Base {} lifts to cap {}", base.get(), cap.get());
}
```

**Key methods:**
- `lift(base)`: Iterator over caps above a base point
- `drop(cap)`: Base point below a cap point
- `add_arrow(base, cap, payload)`: Add vertical relation
- `base()`, `cap()`: Access underlying sieves

**Composed stacks** chain multiple levels:
```rust
use mesh_sieve::topology::stack::ComposedStack;
let composed = ComposedStack::new(stack1, stack2);
// lift/drop operations traverse both levels
```

## 6. Field Data: Atlas & Section

**Atlas** maps points to data layout:
```rust
use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::topology::point::PointId;

let mut atlas = Atlas::default();
let p1 = PointId::new(1).unwrap();
atlas.try_insert(p1, 3).unwrap(); // Point p1 has 3 values

// Query layout
if let Some((offset, len)) = atlas.get(p1) {
    println!("Point {} at offset {}, length {}", p1.get(), offset, len);
}
```

**Section** stores field data:
```rust
use mesh_sieve::data::section::Section;

let mut section = Section::<f64>::new(atlas);
section.try_set(p1, &[1.0, 2.0, 3.0]).unwrap();

// Access data
let values = section.try_restrict(p1).unwrap();
println!("Values: {:?}", values);

// Mutable access
let values_mut = section.try_restrict_mut(p1).unwrap();
values_mut[0] = 42.0;
```

**Error-safe API:**
All methods return `Result` types. Deprecated panic-free versions (marked `#[deprecated]`) are available for compatibility:
- `try_restrict` vs deprecated `restrict`
- `try_set` vs deprecated `set`
- `try_insert` vs deprecated `insert`

## 7. Data Refinement & Assembly

**The `Delta<V>` trait** defines data transformations:
```rust
pub trait Delta<V> {
    type Part;
    fn restrict(&self, v: &V) -> Self::Part;
    fn fuse(local: &mut V, incoming: Self::Part);
}
```

**Built-in implementations:**
- `CopyDelta`: Direct copy without transformation
- `AddDelta`: Sum values for assembly operations
- `Orientation`: Handle sign/permutation changes

**Bundle** combines topology + data for workflows:
```rust
use mesh_sieve::data::bundle::Bundle;
use mesh_sieve::overlap::delta::CopyDelta;

let bundle = Bundle {
    stack,        // InMemoryStack
    section,      // Section<V>  
    delta: CopyDelta,
};

// Push data down (base → cap)
bundle.refine([base_point]).unwrap();

// Pull data up (cap → base)  
bundle.assemble([base_point]).unwrap();
```

**SievedArray** provides refinement-specific data structures:
```rust
use mesh_sieve::data::refine::sieved_array::SievedArray;

let mut coarse = SievedArray::new(coarse_atlas);
let mut fine = SievedArray::new(fine_atlas);

// Refine from coarse to fine
fine.try_refine(&coarse, &refinement_map).unwrap();

// Assemble from fine to coarse  
fine.try_assemble(&mut coarse, &refinement_map).unwrap();
```

## 8. Partitioning & METIS Integration

**Graph partitioning** with METIS support:
```rust
#[cfg(feature = "metis-support")]
{
    use mesh_sieve::algs::metis_partition::DualGraph;
    
    let dual_graph = DualGraph {
        vwgt: vec![1; n_vertices],   // vertex weights
        xadj: vec![...],             // adjacency offsets  
        adjncy: vec![...],           // adjacency list
    };
    
    let partition = dual_graph.metis_partition(n_parts);
    for (vertex, part) in partition.iter() {
        println!("Vertex {} assigned to partition {}", vertex, part);
    }
}
```

**Distribute mesh** across processes:
```rust
#[cfg(feature = "mpi-support")]
{
    use mesh_sieve::algs::distribute::distribute_mesh;
    
    let (local_sieve, overlap) = distribute_mesh(
        &global_sieve,
        &partition_assignment,
        &comm
    ).unwrap();
}
```

## 9. Parallel Computing & MPI

**Overlap** tracks shared points between partitions:
```rust
use mesh_sieve::overlap::overlap::{Overlap, Remote};

let mut overlap = Overlap::default();
overlap.add_arrow(
    local_point,
    remote_partition_point, 
    Remote { rank: remote_rank, remote_point }
);
```

**Communicators** abstract parallel communication:
```rust
use mesh_sieve::algs::communicator::{Communicator, NoComm, MpiComm};

// Serial/testing
let comm = NoComm;

// MPI distributed 
#[cfg(feature = "mpi-support")]
let comm = MpiComm::default();

// Non-blocking send/receive
let send_handle = comm.isend(target_rank, tag, &data);
let recv_handle = comm.irecv(source_rank, tag, &mut buffer);
```

**Completion algorithms** synchronize distributed data:
```rust
use mesh_sieve::algs::completion::{complete_sieve, complete_section, complete_stack};

// Complete topology
complete_sieve(&mut local_sieve, &overlap, &comm, my_rank).unwrap();

// Complete field data  
complete_section(&mut section, &mut overlap, &comm, &delta, my_rank, n_ranks).unwrap();

// Complete vertical composition
complete_stack(&mut stack, &overlap, &comm, my_rank, n_ranks).unwrap();
```

**Error handling:** All completion functions return `Result<(), MeshSieveError>` for robust error handling in distributed environments.

## 10. Error Handling & Best Practices

**Error-safe API design:**
- Modern API uses `Result<T, MeshSieveError>` for all fallible operations
- Deprecated methods marked with `#[deprecated]` may panic (use `try_*` variants)
- Comprehensive error types for topology, communication, and data access failures

**Performance considerations:**
- Automatic cache invalidation on mutations ensures correctness
- For batch operations, explicit cache management may improve performance
- Use `bytemuck`-compatible types for efficient MPI serialization
- Choose appropriate communicator: `NoComm` (serial), `RayonComm` (threads), `MpiComm` (distributed)

**Memory safety:**
- All operations use safe Rust patterns (`Arc`, `Vec`, `HashMap`)
- Zero-cost abstractions through generics and trait objects
- Automatic cleanup prevents buffer leaks in distributed operations

## 11. Examples & Testing

**Running examples:**
```bash
# Serial examples
cargo run --example partition

# MPI examples (requires mpi-support feature)
cargo mpirun -n 4 --features mpi-support --example mpi_complete
cargo mpirun -n 2 --features mpi-support --example mpi_complete_stack
```

**Key examples:**
- `mpi_complete.rs`: Two-rank section completion
- `mpi_complete_stack.rs`: Stack completion across ranks  
- `mpi_partition_exchange.rs`: 4-rank partitioning and data exchange
- `distribute_mpi.rs`: Mesh distribution example

**Testing:** Each module includes comprehensive unit tests covering topology operations, data access, refinement logic, and parallel communication patterns.

## 12. API Reference

**Core traits and types:**

| Component | Key Methods | Purpose |
|-----------|-------------|---------|
| `Sieve` | `cone`, `support`, `add_arrow`, `closure`, `star` | Mesh topology queries/mutations |
| `Stack` | `lift`, `drop`, `add_arrow` | Vertical composition |
| `Atlas` | `try_insert`, `get`, `points` | Point-to-data mapping |
| `Section<V>` | `try_restrict`, `try_restrict_mut`, `try_set` | Field data access |
| `Delta<V>` | `restrict`, `fuse` | Data transformation |
| `Communicator` | `isend`, `irecv`, `rank`, `size` | Parallel communication |

**Typical signatures:**
```rust
fn cone(&self, p: PointId) -> impl Iterator<Item = (PointId, &Payload)>
fn try_restrict(&self, p: PointId) -> Result<&[V], MeshSieveError>  
fn try_insert(&mut self, p: PointId, len: usize) -> Result<(), MeshSieveError>
fn complete_sieve<C: Communicator>(sieve: &mut InMemorySieve<PointId, Remote>, 
                                  overlap: &Overlap, comm: &C, my_rank: usize) 
                                  -> Result<(), MeshSieveError>
```
