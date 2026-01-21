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
* **MultiSection & constraints** (stacked fields + DOF pinning)
* **Mixed sections** (heterogeneous scalar types)
* **Discretization** (region-keyed DOF metadata)
* **CoordinateDM** (coordinate storage + labels + discretization)
* **MeshData/MeshBundle** (I/O containers)

### What changed (high-level)

* **Infallible → fallible**: Panicking shims are removed (or deprecated pending the next major); use `Result` methods.
* **Stricter invariants**: `strict-invariants` (alias: `check-invariants`) runs deep checks in debug/CI, including **in/out mirror validation** for overlaps and pre-checks before slicing in data.
* **Storage abstraction**: `Section` is generic over **storage backends** (CPU vector by default; optional **wgpu** buffer).
* **Multi-field layouts**: `MultiSection` and `FieldSection` stack field DOFs and keep PETSc-style offsets.
* **Mixed scalar sections**: named `MixedSectionStore` for heterogeneous data.
* **Coordinate metadata**: `Discretization` + `CoordinateDM` for region-aware layouts.
* **Geometry wrappers**: `Coordinates` and `HighOrderCoordinates` for fixed-dimension and curved geometry data.
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

**Maintenance**

* `prune_empty_parts() -> usize` — remove dangling `Part(r)` vertices left behind after link removals (keeps `check-empty-part` happy).
* `remove_neighbor_rank(rank) -> bool` — convenience for “rank disappeared”; removes all incident edges and the part node.
* `retain_neighbor_ranks(keep) -> (edges_removed, parts_removed)` — bulk prune to a supplied rank set; deterministic across hash backends.

> **Common step:** After structural deletions or neighborhood recomputations, call `prune_empty_parts()` so optional invariants stay satisfied before iterating `neighbor_ranks()`.

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

### 5.5 Geometry helpers (`Coordinates`, `HighOrderCoordinates`)

`Coordinates` wraps a `Section` with a fixed spatial dimension per point.
Optional `HighOrderCoordinates` can be attached for per-entity geometry DOFs
(for example, curved elements).

```rust
use mesh_sieve::data::coordinates::{Coordinates, HighOrderCoordinates};
use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::storage::VecStorage;
use mesh_sieve::topology::point::PointId;

let mut atlas = Atlas::default();
let p = PointId::new(1)?;
atlas.try_insert(p, 3)?;
let mut coords = Coordinates::<f64, VecStorage<f64>>::try_new(3, atlas)?;

let mut ho_atlas = Atlas::default();
ho_atlas.try_insert(p, 9)?; // multiple of dimension
let high_order = HighOrderCoordinates::<f64, VecStorage<f64>>::try_new(3, ho_atlas)?;
coords.set_high_order(high_order)?;
```

---

### 5.6 Discretization metadata (field layouts by region)

Use `Discretization` to describe per-field DOF layouts keyed by region selectors
(labels or cell types). This lets each field carry its own layout definition
without forcing a single global DOF scheme.

```rust
use mesh_sieve::data::discretization::{Discretization, DofLayout, FieldDiscretization};
use mesh_sieve::topology::cell_type::CellType;

let mut discretization = Discretization::new();

// Velocity: 3 DOFs on fluid-labeled points, higher-order on triangles.
let mut velocity = FieldDiscretization::new();
velocity.set_label_layout("fluid", 1, DofLayout::new(3));
velocity.set_cell_type_layout(CellType::Triangle, DofLayout::new(6));
discretization.insert_field("velocity", velocity);

// Pressure: 1 DOF everywhere in the fluid region.
let mut pressure = FieldDiscretization::new();
pressure.set_label_layout("fluid", 1, DofLayout::new(1));
pressure.set_cell_type_layout(CellType::Triangle, DofLayout::new(1));
discretization.insert_field("pressure", pressure);

// Attach to a mesh container.
// mesh_data.discretization = Some(discretization);
```

---

### 5.7 Constrained fields (`ConstrainedSection`)

`ConstrainedSection` wraps a `Section` with per-point DOF constraints (indices + values).
Use it to enforce Dirichlet-style values after refinement/assembly.

```rust
use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::constrained_section::ConstrainedSection;
use mesh_sieve::data::section::Section;
use mesh_sieve::data::storage::VecStorage;
use mesh_sieve::topology::labels::LabelSet;
use mesh_sieve::topology::point::PointId;

let mut atlas = Atlas::default();
let p = PointId::new(7)?;
atlas.try_insert(p, 3)?;
let section = Section::<f64, VecStorage<f64>>::new(atlas);

let mut labels = LabelSet::new();
labels.set_label(p, "boundary", 1);

let mut constrained = ConstrainedSection::new(section);
for point in labels.points_with_label("boundary", 1) {
    constrained.insert_constraint(point, 2, 0.0)?;
}
constrained.apply_constraints()?;
```

For mesh workflows that refine or assemble, apply constraints afterwards with
`Bundle::refine_with_constraints` and `Bundle::assemble_with_constraints` to keep
boundary conditions enforced.

---

### 5.8 Multi-field layouts (`MultiSection`)

`MultiSection` stacks multiple `FieldSection`s into a single combined atlas,
mirroring PETSc-style field offsets. Each field can carry its own constraints,
and `MultiSection::apply_constraints()` applies them all.

```rust
use mesh_sieve::data::multi_section::{FieldSection, MultiSection};
use mesh_sieve::data::section::Section;
use mesh_sieve::data::storage::VecStorage;
use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::topology::point::PointId;

let mut atlas = Atlas::default();
let p = PointId::new(7)?;
atlas.try_insert(p, 3)?;

let vel = Section::<f64, VecStorage<f64>>::new(atlas.clone());
let pres = Section::<f64, VecStorage<f64>>::new(atlas);

let mut velocity = FieldSection::new("velocity", vel);
velocity.insert_constraint(p, 2, 0.0)?;

let fields = vec![velocity, FieldSection::new("pressure", pres)];
let mut multi = MultiSection::new(fields)?;

let (offset, dof) = multi.field_span(p, 0)?;
multi.apply_constraints()?;
```

---

### 5.9 Mixed scalar sections (`MixedSectionStore`)

Use `MixedSectionStore` to keep named sections with different scalar types.
This is especially useful in I/O contexts where meshes carry mixed-precision
data alongside f64/f32 coordinates.

```rust
use mesh_sieve::data::mixed_section::MixedSectionStore;
use mesh_sieve::data::section::Section;
use mesh_sieve::data::storage::VecStorage;
use mesh_sieve::data::atlas::Atlas;

let atlas = Atlas::default();
let mut store = MixedSectionStore::new();
store.insert("temperature", Section::<f64, VecStorage<f64>>::new(atlas.clone()));
store.insert("material_id", Section::<i32, VecStorage<i32>>::new(atlas));
```

---

### 5.10 Coordinate data management (`CoordinateDM`)

`CoordinateDM` packages `Coordinates` with optional labels and discretization
metadata so you can move coordinate data independently from mesh topology.

```rust
use mesh_sieve::data::coordinate_dm::CoordinateDM;
use mesh_sieve::data::coordinates::Coordinates;
use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::storage::VecStorage;

let atlas = Atlas::default();
let coords = Coordinates::<f64, VecStorage<f64>>::try_new(3, atlas)?;
let dm = CoordinateDM::new(coords);
```

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
  To keep Dirichlet values enforced, use `bundle.refine_with_constraints(bases, &constraints)`.

* **Assemble (cap→base)**
  `bundle.assemble_with(bases, &AverageReducer) -> Result<()>`
  **Robust fix**: length validation happens **in the caller** so reducer can stay context-free (no fake `PointId`). Uses a streaming pattern (no `Vec` per base).
  Use `bundle.assemble_with_constraints(bases, &AverageReducer, &constraints)` to apply constraints after assembly.

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

## 8. Mesh containers & I/O

`MeshData` is the primary container returned by mesh readers. It holds the
topology plus coordinates, named sections, mixed-type sections, labels, cell
types, and optional discretization metadata.

```rust
use mesh_sieve::io::MeshData;
use mesh_sieve::data::section::Section;
use mesh_sieve::data::storage::VecStorage;
use mesh_sieve::topology::cell_type::CellType;
use mesh_sieve::topology::sieve::InMemorySieve;
use mesh_sieve::topology::point::PointId;

let sieve = InMemorySieve::<PointId, ()>::default();
let mut mesh: MeshData<_, f64, VecStorage<f64>, VecStorage<CellType>> = MeshData::new(sieve);
mesh.sections.insert("temperature".to_string(), Section::new(Default::default()));
```

For multi-mesh workflows (e.g., subdomains, partitions, or time slices), use
`MeshBundle`, which can synchronize labels or coordinate values across meshes.
Partitioned mesh I/O also supports bundled `.bundle.json` outputs for
gathered serialization.

---

## 9. Optional adapter: `Map` & infallible helpers (feature `map-adapter`)

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

## 10. Error Model (clear & precise)

All public ops return `Result<_, MeshSieveError>`. Notable variants:

* **Atlas/Section/Scatter**

  * `ZeroLengthSlice`, `DuplicatePoint(PointId)`, `MissingAtlasPoint(PointId)`
  * `ScatterLengthMismatch { expected, found }`
  * `ScatterChunkMismatch { offset, len }`
  * `SliceLengthMismatch { point, expected, found }`
  * `AtlasPlanStale { expected, found }`
  * `AtlasPointLengthChanged { point, old_len, new_len }` *(new; used by `with_atlas_mut` strict rule)*
  * `ConstraintIndexOutOfBounds { point, index, len }` (DOF constraints)
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

## 11. Performance Notes

* `#[inline]` on hot getters: `Atlas::get`, `Section::try_restrict`, `try_restrict_mut`, overlap query adapters.
* **Reserve** before bulk inserts: `reserve_cone`, `reserve_support`; in data rebuilds, compute `total_len_new` and reserve once.
* **Streaming** in `Bundle::{refine, assemble_with}` → avoids per-base `Vec`.
* **Scatter fast-path**: whole-buffer clone when contiguous.
* Choose `fast-hash` for speed or `deterministic-order` for reproducible I/O.

---

## 12. Testing & CI (recommendations)

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

## 13. Migration Guide (breaking but localized)

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

## 14. Quick Reference

| Component         | Selected Methods                                                                                     |
| ----------------- | ---------------------------------------------------------------------------------------------------- |
| `Sieve`           | `cone(_), support(_)`, `cone_points(_), support_points(_)`, `closure_iter`, `star_iter`              |
| `InMemorySieve`   | `reserve_cone`, `reserve_support`, degree-local updates                                              |
| `Overlap`         | `add_link_structural_one`, `add_links_structural_bulk`, `resolve_remote_point(s)`, `neighbor_ranks`  |
| `Atlas`           | `try_insert`, `get`, `remove_point`, `points`, `version`, `atlas_map`, invariants                    |
| `Section<V,S>`    | `new`, `try_restrict(_), try_restrict_mut(_), try_set(_)`, `with_atlas_mut`, `with_atlas_mut_resize` |
| `Coordinates`     | `try_new`, `try_restrict(_mut)`, `set_high_order`, `high_order(_mut)`                                |
| `Discretization`  | `insert_field`, `field`, `field_mut`, `iter`                                                         |
| `MultiSection`    | `field_span`, `field_offset`, `apply_constraints`                                                    |
| `MixedSectionStore` | `insert`, `get`, `get_mut`, `iter`                                                                 |
| `CoordinateDM`    | `new`                                                                                                 |
| `SliceDelta<V>`   | `apply(src,dst)` (e.g., `Polarity::{Forward,Reverse}`)                                               |
| `ValueDelta<V>`   | `restrict(&V)->Part`, `fuse(&mut V, Part)` (e.g., `CopyDelta`, `AddDelta`, `ZeroDelta`)              |
| `Bundle<V,D>`     | `refine(bases)`, `assemble_with(bases,&Reducer)`                                                     |
| `MeshData`/`MeshBundle` | `new`, `sync_labels`, `sync_coordinates`                                                      |
| Completion (algs) | `complete_sieve`, `complete_section`, `complete_stack`, `closure_completed`                          |

---

## 15. Examples

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
