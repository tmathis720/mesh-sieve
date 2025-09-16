# A Physicist’s Guide to **mesh-sieve**

This guide maps common physics workflows (FEM/FVM/DG, multigrid, domain decomposition) onto `mesh-sieve`’s concepts and shows how to use them safely and efficiently. Examples are minimal but realistic; everything compiles with the fallible (`try_*`) APIs.

---

## 0) Core mental model (physics ↔ API)

| Physics thing                                           | `mesh-sieve` piece          | Notes                                                               |
| ------------------------------------------------------- | --------------------------- | ------------------------------------------------------------------- |
| Mesh entities (cells, faces, edges, vertices)           | **Points** (`PointId`)      | Every entity is a point id. Types/dim live in your strata/cache.    |
| Incidence (e.g., “cell has faces”, “edge has vertices”) | **Sieve** arrows            | Directed graph `src → dst` (e.g., cell→face, face→vertex).          |
| Orientation/signs (e.g., face normal, edge direction)   | **Polarity**                | Per-arrow payload that can reverse a slice if needed.               |
| DOFs / field layout                                     | **Atlas**                   | Assign each point a slice length (DOFs per entity).                 |
| Field values                                            | **Section\<V, S>**          | Flat storage over an Atlas. `S` is storage (CPU Vec or GPU).        |
| Prolongation / restriction (h- or p-refine)             | **Stack + Bundle**          | Vertical mapping base→cap with `Polarity` payload; refine/assemble. |
| Ghost/halo exchange                                     | **Overlap** + communicators | Bipartite graph of local points ↔ neighbor rank nodes.              |
| Distributed closure/assembly                            | **Completion**              | Pull missing topology/data over the overlap.                        |

> Safety: All modern data APIs are `Result`-based. In development/CI enable `strict-invariants` to catch structural mistakes early.

---

## 1) Topology: build what your physics needs

Most physics codes need the down- and up-incidence:

```rust
use mesh_sieve::topology::sieve::{InMemorySieve, Sieve};
use mesh_sieve::topology::point::PointId;

let mut g = InMemorySieve::<PointId, ()>::default();

// 2D example: triangle t has edges e1,e2,e3; edge e1 has vertices v1,v2
let (t, e1, e2, e3, v1, v2) = (
    PointId::new(100)?, PointId::new(10)?, PointId::new(11)?,
    PointId::new(12)?, PointId::new(1)?, PointId::new(2)?,
);

g.add_arrow(t, e1, ());
g.add_arrow(t, e2, ());
g.add_arrow(t, e3, ());
g.add_arrow(e1, v1, ());
g.add_arrow(e1, v2, ());

// Traversal without cloning payloads:
for v in g.cone_points(e1) { /* v1, v2 */ }

let closure_of_cell: Vec<_> = g.closure_iter([t]).collect(); // {t, e*, v*}
```

**Tips for performance**

* In hot loops use `cone_points`, `support_points`, and `*_iter` (concrete, no boxing).
* Before bulk edits call `reserve_cone(p, k)` / `reserve_support(q, k)` to avoid reallocs.

---

## 2) Layout and values: Atlas + Section

### 2.1 Declare DOF layout

* **Atlas** decides how many DOFs each point has (scalar fields on vertices: 1, vector on faces: 2 or 3, high-order: bigger per entity).
* Offsets are contiguous in insertion order, making I/O easy.

```rust
use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::topology::point::PointId;

let mut atlas = Atlas::default();
for v in [v1, v2] { atlas.try_insert(v, 1)?; }      // scalar at vertices
for e in [e1, e2, e3] { atlas.try_insert(e, 1)?; }  // e.g., edge multiplier
```

### 2.2 Store field values in a Section

CPU path (default):

```rust
use mesh_sieve::data::slice_storage::VecStorage;
use mesh_sieve::data::section::Section;

let mut u = Section::<f64, VecStorage<f64>>::new(atlas);
u.try_set(v1, &[1.0])?;
u.try_set(v2, &[2.0])?;
let s = u.try_restrict(v1)?; // &[f64]
```

**Robustness**

* All methods are fallible; mismatched slice lengths or unknown points return precise errors.
* `with_atlas_mut` **forbids** changing lengths of existing points (prevents silent corruption). To resize by policy, use `with_atlas_mut_resize` (if you enabled that API).

---

## 3) Orientation & element math (SliceDelta + Polarity)

When mapping slices from one entity to another (e.g., element→face traces, edge direction), you often need to reverse the order depending on local orientation. `Polarity` implements the slice transform:

```rust
use mesh_sieve::data::refine::delta::SliceDelta;
use mesh_sieve::topology::arrow::Polarity;

let src = [10.0, 20.0, 30.0];
let mut dst = [0.0; 3];
Polarity::Reverse.apply(&src, &mut dst)?; // dst = [30,20,10]
```

You can implement your own `SliceDelta` to do permutations/sign flips for vector/tensor DOFs.

---

## 4) Refine & assemble (multilevel, h-refine, averaging/summing)

A **Stack** records vertical arrows from base entities to refined (cap) entities; `Bundle` couples that stack with a `Section` and runs **refine** (push down) and **assemble** (pull up) using per-arrow `Polarity`.

```rust
use mesh_sieve::data::bundle::{Bundle, AverageReducer, SliceReducer};
use mesh_sieve::topology::stack::InMemoryStack;

// Build stack arrows base->cap with Polarity payloads
let mut bundle: Bundle<f64> = Bundle {
  stack: InMemoryStack::new(),
  section: u,                          // your Section<f64, _>
  delta: mesh_sieve::overlap::delta::CopyDelta,
};

// Prolongation (e.g., copy with orientation) base→cap:
bundle.refine([/* base points */])?;

// Assembly (cap→base) with explicit reducer
bundle.assemble_with([/* base points */], &AverageReducer)?;

// Example: sum reducer (typical FEM assembly)
#[derive(Copy, Clone, Default)]
struct SumReducer;
impl SliceReducer<f64> for SumReducer {
  fn make_zero(&self, n: usize) -> Vec<f64> { vec![0.0; n] }
  fn accumulate(&self, acc: &mut [f64], src: &[f64]) -> Result<(), mesh_sieve::mesh_error::MeshSieveError> {
    if acc.len() != src.len() { unreachable!() }
    for (a, s) in acc.iter_mut().zip(src) { *a += *s; }
    Ok(())
  }
}
bundle.assemble_with([/* base */], &SumReducer)?;
```

**Why this design?**

* `Bundle` keeps reducers **context-free**; all shape/length checks happen in `assemble_with`, so error messages reference the real point that failed (no fake IDs).

---

## 5) Distributed runs: Overlap & completion

In domain decomposition you maintain **ghost** relationships to neighbor ranks. The **Overlap** is a bipartite graph `Local(point) → Part(rank)` with payload `{rank, remote_point: Option<PointId>}`.

```rust
use mesh_sieve::overlap::overlap::Overlap;

let mut ov = Overlap::default();
// structure first (remote id unknown):
ov.add_link_structural_one(v1, /* neighbor rank */ 1);
// later, after a neighbor exchange, set the remote id:
ov.resolve_remote_point(v1, 1, /* remote PointId */ PointId::new(5001)?)?;
```

After removing links or rebalancing, run `ov.prune_empty_parts()` or `ov.retain_neighbor_ranks([...])` so empty `Part(r)` nodes disappear before enumerating neighbors.

**Completion** routines (in `algs::completion`) use the overlap to:

* **Complete topology:** pull missing arrows to satisfy closure across ranks.
* **Exchange section data:** size handshake + data transfer using your chosen `ValueDelta` (e.g., `CopyDelta` or additive).

You can run with:

* `NoComm` (serial),
* `RayonComm` (in-process “ranks” for testing),
* `MpiComm` (real MPI, `mpi-support` feature).

---

## 6) GPU path (optional, but practical)

If your field is large and kernels are simple (copy/permutation/reverse), `Section` can store on the GPU:

```rust
#[cfg(feature = "wgpu")]
{
  use mesh_sieve::data::{section::Section, atlas::Atlas};
  use mesh_sieve::data::wgpu::WgpuStorage;

  // V must be bytemuck::Pod + Zeroable under the "wgpu" feature
  let atlas: Atlas = /* construct */;
  let storage = WgpuStorage::<f32>::new(device.clone(), &atlas)?;
  let mut sec = Section::<f32, WgpuStorage<f32>>::from_storage(atlas, storage);

  // Versioned scatter (host→device):
  let plan = sec.atlas().build_scatter_plan();
  sec.try_scatter_with_plan(&host_values, &plan)?;
}
```

* `Polarity::Forward` maps to `copy_buffer_to_buffer`.
* `Polarity::Reverse` uses a tiny compute kernel (`reverse_copy.wgsl`).
* Custom `SliceDelta` can be backed by small compute shaders.
* Plans are **version-checked** (`Atlas::version()`), so command buffers can be safely reused until the layout changes.

---

## 7) Common physics recipes

### 7.1 Poisson (linear FEM) sketch

* **Topology:** build cell→vertex and cell→edge as needed.
* **Atlas:** DOFs at vertices, length 1.
* **Local element:** compute 3×3 (2D tri) contributions and RHS.
* **Assembly:** write into **cap** slices (e.g., per-cell accumulators) and `assemble_with` using a **sum** reducer to the vertex field.
* **Boundary conditions:** keep a `Fixed` mask Section (len 1 per vertex); zero-out rows or project post-assembly.

### 7.2 DG flux across faces

* **Topology:** face→(cell\_minus, cell\_plus), face→trace nodes.
* **Polarity:** use per-arrow `Polarity` so traces on the “plus” side are reversed consistently.
* **Delta:** custom `SliceDelta` for permuting vector/tensor traces.
* **Assembly:** sum into cell fields with a sum reducer.

### 7.3 Finite Volume (conservative updates)

* **Atlas:** one scalar per cell.
* **Flux kernel:** iterate faces, compute flux, and `try_apply_delta_between_points` from face slice to the two adjacent cells with signs encoded in `SliceDelta`.
* **Parallel:** use Overlap completion to keep ghost cells up-to-date each step.

### 7.4 Multigrid / refinement

* **Stack:** base cell → refined cells with `Polarity`.
* **Refine:** `bundle.refine(base_cells)` to prolong.
* **Restrict:** `bundle.assemble_with(base_cells, &SumReducer)` or average.

---

## 8) Safety, errors, and invariants

* Prefer `try_*` everywhere; if you want legacy infallible helpers temporarily, enable `feature = "map-adapter"` while migrating.
* Run tests/CI with `--features strict-invariants`:

  * Atlas vs Section sizes match
  * Contiguous spans
  * Overlap is bipartite with rank-consistent payloads
* No synthetic `PointId` in errors—messages identify the real culprit.

---

## 9) Performance checklist

* Use `cone_points` / `support_points` and `*_iter` traversals.
* Pre-reserve adjacency: `reserve_cone` / `reserve_support`.
* Avoid per-base allocations: `Bundle::assemble_with` is streaming.
* Use scatter **fast-path** when layouts are identical: one `clone_from_slice`.
* Choose `fast-hash` for speed or `deterministic-order` for reproducible IO.

---

## 10) Minimal end-to-end snippet

```rust
use mesh_sieve::{
  topology::{sieve::{InMemorySieve, Sieve}, point::PointId},
  data::{atlas::Atlas, section::Section, slice_storage::VecStorage},
  data::refine::delta::SliceDelta,
  topology::arrow::Polarity,
};

let (a,b,c) = (PointId::new(1)?, PointId::new(2)?, PointId::new(3)?);

// 1) Topology: a line a--b--c as edges→verts
let e1 = PointId::new(10)?; let e2 = PointId::new(11)?;
let mut g = InMemorySieve::<PointId, ()>::default();
g.add_arrow(e1, a, ()); g.add_arrow(e1, b, ());
g.add_arrow(e2, b, ()); g.add_arrow(e2, c, ());

// 2) Layout+values: 1 DOF per vertex
let mut atlas = Atlas::default();
for v in [a,b,c] { atlas.try_insert(v, 1)?; }
let mut u = Section::<f64, VecStorage<f64>>::new(atlas);
u.try_set(a, &[0.0])?; u.try_set(b, &[1.0])?; u.try_set(c, &[0.0])?;

// 3) Example “edge-to-vertex” map with orientation
// (toy: copy edge avg to both ends, reverse on one end)
let edge_val = [2.0];    // pretend 1 dof per edge
let mut tmp = [0.0];
Polarity::Forward.apply(&edge_val, &mut tmp)?; // no-op
let mut tmp_rev = [0.0];
Polarity::Reverse.apply(&edge_val, &mut tmp_rev)?; // same length, reversed (no change for len=1)
// accumulate into vertices:
for (v, src) in [(a, &tmp[..]), (b, &tmp_rev[..])] {
  let cur = u.try_restrict_mut(v)?; cur[0] += src[0] * 0.5;
}
```

---

## 11) When to reach for which feature

* **You just need local meshes & fields (serial):** `InMemorySieve`, `Atlas` + `Section<VecStorage>`, no features.
* **You want MPI:** turn on `mpi-support`, build an `Overlap`, use completion helpers.
* **You need big, simple kernels:** `wgpu` + `Section<WgpuStorage>` with POD types; use plan/versioned scatter.
* **You’re building a solver stack:** model refinement with a `Stack` + `Bundle`, choose a reducer that matches your weak/strong form (sum/average).

---

If you keep these mappings in mind—*points/arrows for topology, atlas/section for fields, stacks/bundles for level transfer, overlap for ghosts*—you’ll be able to slot `mesh-sieve` into FEM/FVM/DG codes with minimal glue while getting solid safety and good performance out of the box.
