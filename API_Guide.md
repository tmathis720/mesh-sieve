# mesh-sieve API User Guide (Updated)

## 1. Introduction

**mesh-sieve** is a Rust library for representing mesh topologies and field data, with first-class support for refinement, assembly, overlap-driven exchange and (optionally) GPU storage. It centers on:

* **Points** (`PointId`)
* **Arrows** (incidence, typed by payloads)
* **Sieves** (topology graphs)
* **Stacks** (vertical base→cap relations)
* **Strata** (height/depth layers)
* **Atlas & Section** (layout + values)
* **Bundles** (topology + data workflow)
* **Overlap** (bipartite local↔rank structure)

### What changed (high-level)

* **Infallible → fallible**: Panicking shims are removed (or deprecated pending the next major); use `Result` methods.
* **Stricter invariants**: `strict-invariants` (alias: `check-invariants`) runs deep checks in debug/CI, including **in/out mirror validation** for overlaps and pre-checks before slicing in data.
* **Storage abstraction**: `Section` is generic over **storage backends** (CPU vector by default; optional **wgpu** buffer).
* **Determinism & speed**: Streaming patterns (no per-point heap churn), fast-paths for contiguous scatter, and careful inlining.
* **Clearer errors**: No fake PointIds; more precise variants.

---

## 2. Getting Started

**Cargo.toml**

```toml
[dependencies]
mesh-sieve = "2.0"

# Common optional features (opt-in):
# - strict-invariants     : deep invariant checks in debug/CI
# - map-adapter           : exposes deprecated infallible Map adapter & helpers
# - rayon                 : parallel refine/assemble helpers
# - mpi-support           : MPI communicator
# - metis-support         : METIS partitioning
# - fast-hash             : AHash maps/sets in hot paths
# - deterministic-order   : BTree maps/sets for reproducible iteration
# - wgpu                  : GPU storage backend for Section (requires V: Pod + Zeroable)
```

**Hello sieve**

```rust
use mesh_sieve::topology::sieve::{InMemorySieve, Sieve};
use mesh_sieve::topology::point::PointId;

let mut g = InMemorySieve::<PointId, ()>::default();
let a = PointId::new(1)?; let b = PointId::new(2)?;
g.add_arrow(a, b, ());
for v in g.cone_points(a) { /* fast, no payload clones */ }
```

> **Tip:** Enable `strict-invariants` in one CI lane to catch regressions early.

---

## 3. Core Topology: `Sieve`

### 3.1 Trait surface (unchanged ergonomics, better invariants)

* **Incidence**

  * `cone(p) -> impl Iterator<Item=(Point, Payload)>`
  * `support(p) -> ...`
  * point-only adapters: `cone_points`, `support_points` (avoid payload clones)

* **Mutations**

  * `add_arrow(src, dst, payload)`
  * `remove_arrow(src, dst) -> Option<Payload>`
  * **Preallocation**: `reserve_cone(p, additional)`, `reserve_support(q, additional)`

* **Traversal**

  * Concrete: `closure_iter`, `star_iter`, `closure_both_iter` (preferred)
  * Boxed legacy: `closure`, `star`, `closure_both`

* **Derived (lattice/strata)**

  * `height`, `depth`, `diameter`, `height_stratum`, `depth_stratum`
  * Caches auto-invalidate on mutation.

### 3.2 `InMemorySieve`

* Degree-local mirror updates
* Uses **`FastMap/Set`** (configurable via `fast-hash`/`deterministic-order`)

---

## 4. Overlap (distributed sharing)

An `Overlap` is a typed sieve over `OvlId = Local(PointId) | Part(usize)` with payload `Remote{ rank, remote_point: Option<PointId> }`.

**Key ops**

* Build structure:

  * `add_link_structural_one(local, rank) -> bool`
  * `add_links_structural_bulk(iter) -> usize` (dedupes, pre-reserves)
* Resolve mappings:

  * `resolve_remote_point(local, rank, remote) -> Result<()>`
  * `resolve_remote_points(iter) -> Result<()>`
* Expansion from mesh:

  * `ensure_closure_of_support(&mut ovl, &mesh)`
  * `ensure_closure_of_support_from_seeds(...)`
  * `expand_one_layer_mesh(...) -> usize`

**Invariants (robust)**

* Bipartite: `Local(_) -> Part(r)` only
* Payload rank must equal `r`
* No duplicates per `(src,dst)`
* *(opt-in)* `check-empty-part`
* **Strict mirror check** *(under `strict-invariants`)*:

  * `adjacency_out` and `adjacency_in` must **mirror exactly** (identical payloads and counts)
  * Clear errors: `OverlapInOutMirrorMissing`, `OverlapInOutPayloadMismatch`, `OverlapInOutEdgeCountMismatch`

**Queries**

* `neighbor_ranks() -> impl Iterator<Item=usize>`
* `links_to(r) -> impl Iterator<Item=(PointId, Option<PointId>)>`
* `links_to_resolved(r) -> impl Iterator<Item=(PointId, PointId)>`
* Sorted variants for deterministic I/O.

---

## 5. Data: Atlas & Section (storage-generic)

### 5.1 `Atlas` (layout only)

* `try_insert(p, len) -> Result<offset>`
* `get(p) -> Option<(offset, len)>`
* `remove_point(p) -> Result<()>`
  **Breaking**: returns `Err(MissingAtlasPoint(p))` if absent (no silent success).
* `points()`, `atlas_map()`, `iter_entries()`, `version()`, `total_len()`
* **Invariants (robust)**:

  * `order` unique
  * `map.keys == order` (validated always; not only on size mismatch)
  * positive lengths; contiguous offsets; `total_len` matches sum

### 5.2 Storage abstraction (new)

`Section` is now generic over storage:

```rust
use mesh_sieve::data::section::Section;
use mesh_sieve::data::slice_storage::{SliceStorage, VecStorage};

// Default (CPU)
let atlas = /* ... */;
let mut sec = Section::<f64, VecStorage<f64>>::new(atlas);

// Storage-agnostic API uses SliceStorage under the hood
```

**Storage trait (essentials)**

* `ptr/range access` by atlas spans
* `clone_from_slice`/`copy` semantics
* `apply_delta` over (src\_span → dst\_span) without UB, handling overlap safely
* `scatter_from` slices by `(offset,len)` plan
* (wgpu backend handles staging & compute; see §7)

### 5.3 `Section<V, S: SliceStorage<V>>`

* Construct:

  * `new(atlas)` (fills with `V::default()`)
* Access:

  * `try_restrict(p) -> Result<&[V]>`
  * `try_restrict_mut(p) -> Result<&mut [V]>`
  * `try_set(p, &[V]) -> Result<()>`
  * Iteration helpers (`iter`, `for_each_in_order`, etc.)
* Mutation of layout:

  * `try_add_point(p, len) -> Result<()>`
  * `try_remove_point(p) -> Result<()>`
  * `with_atlas_mut(|atlas| { ... }) -> Result<()>`
    **Breaking & robust**: **rejects any length change** for existing points.
  * `with_atlas_mut_resize(|atlas| { ... }, policy) -> Result<()>`
    *(new, recommended when you must change lengths)*
    Policies like:

    * `PreservePrefix`
    * `PreserveSuffix`
    * `Reinit` (fill new/changed extents with `V::default()`)
* Deltas (slice→slice):

  * `try_apply_delta_between_points<D: SliceDelta<V>>(src, dst, &delta)`
    Safe for **overlap & disjoint**; pre-checks lengths and atlas/data invariants (gated by strict).
* Scatter:

  * `try_scatter_in_order(buf)` (atlas order)
  * `try_scatter_from(buf, &spans)` (explicit spans)
  * `try_scatter_with_plan(buf, &ScatterPlan)`
    **Stable contract**: fails with `AtlasPlanStale` if `atlas_version` diverges.
  * **Fast-path** (automatic): if `buf.len() == data.len()` and spans are `[ (0,l0), (l0,l1), ... ]`, we do a single `clone_from_slice`.

> **Migration (breaking)**
> Panicking shims (`insert`, `restrict`, `restrict_mut`, `set`) are removed (or deprecated behind feature gates until the next major). Use `try_*`.

### 5.4 `SliceDelta` vs `ValueDelta` (clear semantics)

* `SliceDelta<V>`: transforms **one slice to another** (e.g., `Polarity::Reverse`) — used by refine and intra-section copy.
* `overlap::ValueDelta<V>`: **communication/merge** semantics for exchange, e.g., `CopyDelta`, `AddDelta`, `ZeroDelta`.

```rust
use mesh_sieve::data::refine::delta::SliceDelta;
use mesh_sieve::overlap::delta::ValueDelta as OverlapDelta;
```

**Naming**
`Orientation` has been renamed to **`Polarity`**; `Delta` alias remains deprecated.

---

## 6. Bundles (topology + data workflow)

```rust
use mesh_sieve::data::bundle::{Bundle, AverageReducer};
use mesh_sieve::topology::stack::InMemoryStack;
use mesh_sieve::topology::arrow::Polarity;

let mut b: Bundle<f64> = Bundle {
    stack: InMemoryStack::new(),
    section: /* Section<V, S> */,
    delta: mesh_sieve::overlap::delta::CopyDelta,
};
```

* **Refine (base→cap)**
  `bundle.refine(bases) -> Result<()>`
  Streams through `stack.lift(base)` and applies `Polarity` as a `SliceDelta`. No per-base heap allocations.

* **Assemble (cap→base)**
  `bundle.assemble_with(bases, &AverageReducer) -> Result<()>`
  **Robust fix**: length validation happens **in the caller** so reducer can stay context-free (no fake `PointId`). Uses a streaming pattern (no `Vec` per base).

---

## 7. GPU storage (feature `wgpu`)

* `Section<V, WgpuStorage<V>>` where `V: bytemuck::Pod + bytemuck::Zeroable`.
* `apply_delta` routing:

  * `Polarity::Forward` → `copy_buffer_to_buffer`
  * `Polarity::Reverse` → a tiny **compute shader** (`reverse_copy.wgsl`) or a dedicated “reverse copy” kernel
  * Custom `SliceDelta` implementations can be dispatched via small compute pipelines.
* `scatter_with_plan`:

  * Uses atlas `version()` + `ScatterPlan` for **command stream validation**; we refuse if versions diverge.
* Data ingress/egress via staging buffers or mapped ranges.
* Keep **host and device** consistency by rebuilding when the atlas changes (and check `plan.atlas_version`).

> **Best practice**
> Gate GPU-safe bounds on `V` only when `wgpu` is enabled; CPU `VecStorage` works with any `V: Clone + Default`.

---

## 8. Optional adapter: `Map` & infallible helpers (feature `map-adapter`)

* The old `Map<V>` trait and helpers like `restrict_closure` / `restrict_star` are **gated**:

  ```rust
  #[cfg(feature = "map-adapter")]
  use mesh_sieve::data::section::Map;

  // Infallible helpers (panic on missing data) are also gated
  #[cfg(feature = "map-adapter")]
  use mesh_sieve::data::refine::helpers::restrict_closure;
  ```
* Preferred path (always on): `FallibleMap<V>` and `try_*` helpers.

---

## 9. Error Model (clear & precise)

All public ops return `Result<_, MeshSieveError>`. Notable variants:

* **Atlas/Section/Scatter**

  * `ZeroLengthSlice`, `DuplicatePoint(PointId)`, `MissingAtlasPoint(PointId)`
  * `ScatterLengthMismatch { expected, found }`
  * `ScatterChunkMismatch { offset, len }`
  * `SliceLengthMismatch { point, expected, found }`
  * `AtlasPlanStale { expected, found }`
  * `AtlasPointLengthChanged { point, old_len, new_len }` *(new; used by `with_atlas_mut` strict rule)*
* **Overlap**

  * `OverlapNonBipartite { src, dst }`
  * `OverlapRankMismatch { expected, found }`
  * `OverlapDuplicateEdge { src, dst }`
  * `OverlapEmptyPart { rank }` *(opt-in feature)*
  * **Strict mirror checks**:

    * `OverlapInOutMirrorMissing { src, dst }`
    * `OverlapInOutPayloadMismatch { src, dst, out, inn }`
    * `OverlapInOutEdgeCountMismatch { out_edges, in_edges }`
* **No synthetic PointIds** anywhere (we removed `new_unchecked(1)` placeholders).

---

## 10. Performance Notes

* `#[inline]` on hot getters: `Atlas::get`, `Section::try_restrict`, `try_restrict_mut`, overlap query adapters.
* **Reserve** before bulk inserts: `reserve_cone`, `reserve_support`; in data rebuilds, compute `total_len_new` and reserve once.
* **Streaming** in `Bundle::{refine, assemble_with}` → avoids per-base `Vec`.
* **Scatter fast-path**: whole-buffer clone when contiguous.
* Choose `fast-hash` for speed or `deterministic-order` for reproducible I/O.

---

## 11. Testing & CI (recommendations)

* **Property tests** for Atlas/Section coherence (random insert/remove/shuffle + invariant checks + scatter/gather round-trip).
* **Mutation tests**: `with_atlas_mut` must **error** if lengths change; verify `with_atlas_mut_resize` honors `ResizePolicy`.
* **Delta aliasing**: overlapping/disjoint spans; first/last points; ensure no panics.
* **Parallel determinism** (feature `rayon`): refine/assemble parity (serial vs parallel) or identical `DuplicateRefinementTarget` error.
* **Strict invariants lane** in CI:

  ```
  cargo test --features strict-invariants
  ```

  (Keep alias `check-invariants` for a deprecation window if you still expose it.)

---

## 12. Migration Guide (breaking but localized)

### Panicking APIs → fallible

* `Atlas::insert` → `try_insert`
* `Section::{restrict, restrict_mut, set}` → `try_restrict`, `try_restrict_mut`, `try_set`
* Infallible helpers → `try_*` variants (or enable `map-adapter` feature temporarily)

### `with_atlas_mut`

* **Old**: allowed silent length changes
* **New**: rejects length changes (`AtlasPointLengthChanged`)
* **If you must resize**: use `with_atlas_mut_resize(|atlas| { ... }, ResizePolicy::Reinit)` (or `PreservePrefix/Suffix`) and handle the `Result`.

### Reducers

* Remove reliance on reducer to report correct point IDs. Do slice length checks **before** calling a reducer; keep reducers stateless.

### Orientation → Polarity

* Replace types/constructors accordingly.

### Overlap invariants

* If you had side-effects touching `adjacency_*` directly, switch to the API mutators; strict mode now checks mirrors and payload equality.

---

## 13. Quick Reference

| Component         | Selected Methods                                                                                     |
| ----------------- | ---------------------------------------------------------------------------------------------------- |
| `Sieve`           | `cone(_), support(_)`, `cone_points(_), support_points(_)`, `closure_iter`, `star_iter`              |
| `InMemorySieve`   | `reserve_cone`, `reserve_support`, degree-local updates                                              |
| `Overlap`         | `add_link_structural_one`, `add_links_structural_bulk`, `resolve_remote_point(s)`, `neighbor_ranks`  |
| `Atlas`           | `try_insert`, `get`, `remove_point`, `points`, `version`, `atlas_map`, invariants                    |
| `Section<V,S>`    | `new`, `try_restrict(_), try_restrict_mut(_), try_set(_)`, `with_atlas_mut`, `with_atlas_mut_resize` |
| `SliceDelta<V>`   | `apply(src,dst)` (e.g., `Polarity::{Forward,Reverse}`)                                               |
| `ValueDelta<V>`   | `restrict(&V)->Part`, `fuse(&mut V, Part)` (e.g., `CopyDelta`, `AddDelta`, `ZeroDelta`)              |
| `Bundle<V,D>`     | `refine(bases)`, `assemble_with(bases,&Reducer)`                                                     |
| Completion (algs) | `complete_sieve`, `complete_section`, `complete_stack`, `closure_completed`                          |

---

## 14. Examples

### Section with default CPU storage

```rust
use mesh_sieve::data::{atlas::Atlas, section::Section};
use mesh_sieve::data::slice_storage::VecStorage;
use mesh_sieve::topology::point::PointId;

let p = PointId::new(7)?;
let mut atlas = Atlas::default();
atlas.try_insert(p, 3)?;
let mut sec = Section::<f64, VecStorage<f64>>::new(atlas);
sec.try_set(p, &[1.0, 2.0, 3.0])?;
let s = sec.try_restrict(p)?; // &[f64]
```

### Safe layout changes

```rust
// Reject length changes for existing points (strict)
let res = sec.with_atlas_mut(|a| {
    // a.try_insert(existing_p, new_len)  // would imply length change → reject at end
});
assert!(res.is_err());

// Explicit resize with policy
use mesh_sieve::data::storage::ResizePolicy;
sec.with_atlas_mut_resize(|a| {
    // add/remove points or change lengths
}, ResizePolicy::PreservePrefix)?;
```

### GPU storage (feature `wgpu`)

```rust
#[cfg(feature = "wgpu")]
{
    use mesh_sieve::data::{atlas::Atlas, section::Section};
    use mesh_sieve::data::wgpu::WgpuStorage;

    let atlas = /* ... */;
    let storage = WgpuStorage::<f32>::new(device, &atlas)?;
    let mut sec = Section::<f32, WgpuStorage<f32>>::from_storage(atlas, storage);

    // Scatter with plan (version-checked)
    let plan = sec.atlas().build_scatter_plan();
    sec.try_scatter_with_plan(&host_buffer, &plan)?;
}
```

---

### Final Notes

* Prefer **fallible** APIs (`try_*`) throughout.
* Turn on `strict-invariants` in CI to catch mirror/layout issues.
* Use storage abstraction to keep your code **backend-agnostic** (CPU today, GPU tomorrow).
* Keep reducers simple; do shape checks at call sites for better errors and easier testing.