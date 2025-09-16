# mesh-sieve

**mesh-sieve** is a modular, high-performance Rust library for mesh topology and field data. It powers refinement/assembly pipelines and overlap-driven exchange for serial, threaded, and MPI-distributed workflows. The APIs are **Result-first** (no hidden panics), invariants are easy to validate (`strict-invariants`), and the data layer is **storage-generic** (CPU `Vec` by default; optional **wgpu** backend).

## Features

* **Mesh Topology**: generic **Sieve** graphs for incidence and traversal (cone/support/closure/star), plus lattice ops (meet/join) and strata (height/depth).
* **Field Data**: **Atlas** (layout) + **Section** (values) with **fallible** accessors and strong invariants. Fast scatter paths for contiguous layouts.
* **Storage Abstraction**: `Section<V, S>` where `S: SliceStorage<V>` (built-in `VecStorage`; optional `WgpuStorage` with compute kernels).
* **Parallel Communication**: pluggable **Communicator** backends (serial `NoComm`, in-process `RayonComm`, feature-gated `MpiComm`).
* **Overlap**: bipartite local↔rank structure with strict mirror validation; helpers to expand along mesh closure and prune detached ranks.
* **Partitioning**: optional METIS helpers and in-tree algorithms.
* **Testing & CI**: property tests, deterministic iterators, and feature-gated deep invariant checks.
* **Performance**: point-only adapters (no payload cloning), degree-local updates, preallocation hints, streaming algorithms, and inline hot paths.

## Getting Started

```sh
cargo build
cargo run
```

### Cargo Features

Enable only what you need:

```toml
[dependencies]
mesh-sieve = { version = "2", features = [
  # safety & determinism
  # "strict-invariants",      # deep invariant checks in debug/CI
  # "deterministic-order",    # stable BTree maps/sets for IO/repro
  # "fast-hash",              # AHash maps/sets for speed (non-deterministic order)

  # parallel & distributed
  # "rayon",                  # parallel refine/assemble utilities
  # "mpi-support",            # MPI communicator backend

  # partitioning
  # "metis-support",          # METIS bindings

  # data adapters
  # "map-adapter",            # legacy infallible helpers (panic-on-miss)

  # GPU
  # "wgpu",                   # WgpuStorage for Section (V: Pod + Zeroable)
] }
```

> **CI tip:** run a lane with `--features strict-invariants` to catch structural and mirror mistakes early, even in optimized builds.

## Quick Examples

### Topology (InMemorySieve)

```rust
use mesh_sieve::topology::sieve::{InMemorySieve, Sieve};
use mesh_sieve::topology::point::PointId;

let mut g = InMemorySieve::<PointId, ()>::default();
let a = PointId::new(1)?; let b = PointId::new(2)?;
g.add_arrow(a, b, ());

for v in g.cone_points(a) { /* point-only: no payload clones */ }
let reach: Vec<_> = g.closure_iter([a]).collect();
```

### Field Data (Atlas + Section over Vec)

```rust
use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::slice_storage::VecStorage;
use mesh_sieve::data::section::Section;
use mesh_sieve::topology::point::PointId;

let mut atlas = Atlas::default();
let p = PointId::new(7)?;
atlas.try_insert(p, 3)?;
let mut sec = Section::<f64, VecStorage<f64>>::new(atlas);

sec.try_set(p, &[1.0, 2.0, 3.0])?;
let s = sec.try_restrict(p)?; // &[f64]
```

### Refine/Assemble (Bundle)

```rust
use mesh_sieve::data::bundle::{Bundle, AverageReducer};
use mesh_sieve::topology::stack::InMemoryStack;

let mut bundle: Bundle<f64> = Bundle {
  stack: InMemoryStack::new(),   // base -> cap (Polarity payload)
  section: /* Section<f64, _> */,
  delta: mesh_sieve::overlap::delta::CopyDelta,
};

// Refinement (base -> cap) with orientation-aware slice transforms:
bundle.refine([/* base points */])?;

// Assembly (cap -> base) with explicit reduction; lengths checked up-front:
bundle.assemble_with([/* base points */], &AverageReducer)?;
```

### Overlap (distributed links)

```rust
use mesh_sieve::overlap::overlap::Overlap;
use mesh_sieve::topology::point::PointId;

let mut ov = Overlap::default();
let p = PointId::new(42)?;
let neighbor = 1usize;
let inserted = ov.add_link_structural_one(p, neighbor); // remote unknown yet
// Later:
ov.resolve_remote_point(p, neighbor, PointId::new(42042)?)?;
```

After removing links or when ranks disappear, call `ov.prune_empty_parts()` or `ov.retain_neighbor_ranks([...])` to drop empty `Part(r)` nodes before iterating `neighbor_ranks()`.

### MPI Examples

```sh
# build with MPI
cargo run --features mpi-support --example mpi_complete

# or:
mpirun -n 2 cargo run --features mpi-support --example mpi_complete
```

More examples:

* `mpi_complete.rs`, `mpi_complete_stack.rs`: section/stack completion
* `mesh_distribute_two_ranks.rs`: distributing a mesh
* `mpi_complete_multiple_neighbors.rs`: multi-neighbor exchange

## Project Structure

```
src/
  topology/        # Sieve & traversal, strata, lattice
  data/            # Atlas, Section, storage backends (Vec/WGPU), deltas, helpers
  overlap/         # Overlap graph, value deltas, perf types
  algs/            # communicators, completion, distribute, partition utilities
  partitioning/    # METIS integration + in-tree algorithms
  ...
```

---

## What’s New in 2.x (highlights)

**Breaking (safer)**

* **Fallible APIs only** in data/atlas: panicking shims (`insert`, `restrict`, `restrict_mut`, `set`, `get`, `get_mut`) are removed or gated. Prefer `try_*`.
* `Atlas::remove_point(p)` now returns `Err(MissingAtlasPoint(p))` if absent.
* `Section::with_atlas_mut` **rejects length changes** for existing points. Use `with_atlas_mut_resize(..., ResizePolicy)` when resizing is intended.
* `Orientation` → **`Polarity`** (renamed). The old alias is deprecated.

**Data/Storage**

* `Section<V, S>` is **generic** over storage (`S: SliceStorage<V>`):

  * `VecStorage` (CPU) for any `V: Clone + Default`
  * `WgpuStorage` (feature `wgpu`) for `V: bytemuck::Pod + Zeroable`
* `ScatterPlan { atlas_version, spans }` is a **stable contract**; plans are refused if versions diverge.
* **Fast-path scatter**: if spans are contiguous and buffer lengths match, we do a single `clone_from_slice`.

**Overlap**

* Stronger invariants in `validate_invariants()`:

  * Bipartite direction and **payload.rank equals Part(r)**
  * No duplicate edges
  * *(opt-in)* no empty Part nodes (`check-empty-part`)
  * **Strict in/out mirror** (under `strict-invariants`): counts, endpoints and payloads must match exactly.

**Performance & Determinism**

* Streaming in `Bundle::{refine, assemble_with}` (no per-base heap churn).
* Preallocation hints (`reserve_cone`, `reserve_support`).
* Inline hot getters (`Atlas::get`, `Section::try_restrict(_mut)`).
* Choose `fast-hash` for speed or `deterministic-order` for reproducible iteration.

**Error Hygiene**

* No synthetic `PointId`s in errors.
* More precise variants (`AtlasPointLengthChanged { point, old_len, new_len }`, `AtlasPlanStale`, strict overlap mirror errors).

---

## API Overview (at a glance)

| Area        | Key APIs                                                                                                                                              |
| ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Sieve**   | `cone(_)/support(_)`, `cone_points(_)/support_points(_)`, `closure_iter`, `star_iter`                                                                 |
| **Atlas**   | `try_insert`, `get`, `remove_point`, `points`, `atlas_map`, `version`, invariants                                                                     |
| **Section** | `new`, `try_restrict(_)/try_restrict_mut(_)`, `try_set`, `with_atlas_mut`, `with_atlas_mut_resize`, `try_scatter_*`, `try_apply_delta_between_points` |
| **Storage** | `VecStorage`, *(opt-in)* `WgpuStorage` (delta via copy/compute)                                                                                       |
| **Deltas**  | `data::refine::SliceDelta` (slice→slice; e.g., `Polarity`), `overlap::ValueDelta` (comm/merge; e.g., `CopyDelta`, `AddDelta`, `ZeroDelta`)            |
| **Overlap** | `add_link_structural_one`, `add_links_structural_bulk`, `resolve_remote_point(s)`, `ensure_closure_of_support`, `prune_empty_parts`, `remove_neighbor_rank`, `retain_neighbor_ranks` |
| **Bundles** | `refine(bases)`, `assemble_with(bases, &Reducer)`                                                                                                     |
| **Algs**    | completion for sieve/section/stack; communicator trait & backends                                                                                     |

> Legacy **infallible** helpers (`restrict_closure`, `restrict_star`, etc.) and the `Map` adapter are gated behind `feature = "map-adapter"`. Prefer `FallibleMap` + `try_*` helpers.

---

## GPU Backend (feature `wgpu`)

* Use `Section<V, WgpuStorage<V>>` with `V: bytemuck::Pod + Zeroable`.
* Delta routing:

  * `Polarity::Forward` → `copy_buffer_to_buffer`
  * `Polarity::Reverse` → tiny `reverse_copy.wgsl` compute kernel (or equivalent)
  * Custom `SliceDelta` → dedicate small compute pipelines
* Scatter validates `ScatterPlan.atlas_version` vs `Atlas::version()` before encoding commands.

```rust
#[cfg(feature = "wgpu")]
{
  use mesh_sieve::data::{atlas::Atlas, section::Section};
  use mesh_sieve::data::wgpu::WgpuStorage;

  let atlas = /* ... */;
  let storage = WgpuStorage::<f32>::new(device.clone(), &atlas)?;
  let mut sec = Section::<f32, WgpuStorage<f32>>::from_storage(atlas, storage);

  // Version-checked scatter:
  let plan = sec.atlas().build_scatter_plan();
  sec.try_scatter_with_plan(&host_values, &plan)?;
}
```

---

## Performance & Determinism

* Prefer **concrete** iterators (`*_iter`) and **point-only** adapters (`cone_points`, `support_points`) in hot paths.
* Use `reserve_cone` / `reserve_support` before bulk updates.
* Turn on `fast-hash` for speed or `deterministic-order` for stable I/O and tests.
* The data layer uses **streaming** and **single-copy fast-paths** whenever layouts are contiguous.

---

## Testing

* Property tests for Atlas/Section coherence (random insert/remove/shuffle + `validate_invariants` + scatter/gather round-trip).
* Delta aliasing tests (overlap/disjoint) to ensure no panics.
* Parallel determinism (with `rayon`) for refine/assemble parity.
* CI lane with:

  ```sh
  cargo test --features strict-invariants
  ```

---

## Breaking Changes & Migration

1. **Panicking data APIs removed/gated**

   * Migrate `insert/restrict/restrict_mut/set/get/get_mut` → `try_*` equivalents.
   * Temporarily enable `map-adapter` to keep legacy helpers while you refactor.

2. **`with_atlas_mut` is strict**

   * Existing points’ lengths may **not** change; you’ll get `AtlasPointLengthChanged`.
   * Use `with_atlas_mut_resize(|atlas| { ... }, ResizePolicy::PreservePrefix|PreserveSuffix|Reinit)` for explicit resizes.

3. **`Orientation` → `Polarity`**

   * Update imports/constructors.

4. **Overlap mirror checks (strict)**

   * If you touched `adjacency_*` directly, switch to API mutators. Strict mode verifies in/out symmetry and payload equality.

---

## License

MIT — see [LICENSE](LICENSE).
