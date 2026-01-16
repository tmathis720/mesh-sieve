# `topology` API Guide

This guide explains the design and usage of `mesh-sieve`’s topology layer: points, arrows, sieves, stacks, orientations, caches, and utilities. It follows the Sieve/DAG view of unstructured meshes (Knepley & Karpeev) and mirrors PETSc DMPlex concepts (`cone`, `support`, `closure`, `star`, strata/chart).

---

## Concepts at a glance

* **Point**: opaque handle for a topological entity (cell/face/edge/vertex/DOF).
* **Arrow**: directed incidence `src → dst` with a **payload** (metadata) and optional **orientation**.
* **Sieve**: a bidirectional incidence store with **cone** (outgoing) and **support** (incoming) queries, traversal iterators, and arrow-level mutation. It’s a **simple digraph**: *at most one arrow per `(src,dst)`* (upsert semantics).
* **Stack**: vertical arrows between two sieves (e.g., cell→DOF), with their own payloads/orientations.
* **Strata/Chart**: height/depth layers and a deterministic global order derived from the DAG.

---

## Module map

* `arrow` – arrow type (with payload) and a small `Polarity` enum for stacks.
* `point` – `PointId` newtype over `NonZeroU64`.
* `orientation` – compact orientation groups (bit flip, rotations, dihedral, permutations) implementing `sieve::oriented::Orientation` (group structure).
* `sieve` – core trait and implementations:

  * trait `Sieve`, `SieveRef`, `OrientedSieve`
  * in-memory backends (`InMemorySieve`, deterministic variant, oriented variant)
  * frozen CSR backend
  * traversal iterators
  * helpers: build/bulk reserve/query/strata.
* `stack` – `Stack` trait, `InMemoryStack`, and `ComposedStack`.
* `bounds` – reusable `PointLike`/`PayloadLike` bounds.
* `cache` – `InvalidateCache`.
* `labels` – named integer tags for points (boundary/material IDs).
* `utils` – DAG checker.
* `_debug_invariants` – debug-only mirror/uniqueness checks.

---

## Core types

### `PointId`

Opaque handle for mesh points.

```rust
use mesh_sieve::topology::point::PointId;

let p = PointId::new(42)?;               // rejects 0
let raw: u64 = p.into();                 // infallible conversion
```

* `repr(transparent)` over `NonZeroU64`.
* MPI interop: with `feature = "mpi-support"`, it’s `Equivalence`-compatible with `u64`.
* `TryFrom<u64/usize>` and `From<NonZeroU64>` implemented.

> You can also use your own point types; most APIs are generic over `P: PointLike`.

### `Arrow<P = ()>`

Convenience struct when you want a typed edge object.

```rust
use mesh_sieve::topology::{arrow::Arrow, point::PointId};

let a = Arrow::try_new_from_raw(1, 2, "face")?;
assert_eq!(a.endpoints(), (PointId::new(1)?, PointId::new(2)?));
```

* `map` transforms payloads without changing endpoints.
* `Arrow<()>` has `unit(src,dst)` and a `Default` (1→1) mainly for tests.

### Orientation groups (`topology::orientation`)

Orientation/permutation groups that implement `sieve::oriented::Orientation`:

* `Sign` (`BitFlip`) – 1D edge reversal (`C₂`).
* `Rot<N>` – pure rotations `C_N`.
* `Dihedral<N>` – rotations + reflections `D_N` (triangles `N=3`, quads `N=4`).
* `Perm<K>` – fixed-size permutation on `K` items (`S_K`), with `invert()`.

Sugar:

```rust
use mesh_sieve::topology::orientation::{Sign, D3, D4, S3, S4, accumulate_path};

let total: Sign = accumulate_path([Sign::default(), Sign(true)]);
```

> `compose(a,b)` means “do `a`, then `b`” while following a path; `inverse` is for walking the arrow backwards.

### Bounds & caching

* `PointLike = Copy + Eq + Hash + Ord + Debug`
* `PayloadLike = Clone`
* `InvalidateCache` – every mutating backend implements it and clears its derived structures (strata, etc.).

---

## Point labels (`LabelSet`)

Topology labels attach integer tags to points, grouped by a string name. Use them for
boundary condition IDs, material sets, or other discrete point metadata.

```rust
use mesh_sieve::topology::labels::LabelSet;
use mesh_sieve::topology::point::PointId;

let mut labels = LabelSet::new();
let p = PointId::new(10)?;

labels.set_label(p, "boundary", 2);
assert_eq!(labels.get_label(p, "boundary"), Some(2));

let tagged: Vec<_> = labels.points_with_label("boundary", 2).collect();
```

Each point can carry multiple labels with different names (e.g., `"boundary"` and
`"material"`); within a label name, a point maps to a single integer value.

---

## The `Sieve` trait (core)

```rust
use mesh_sieve::topology::sieve::Sieve;

pub trait Sieve: Default + InvalidateCache {
    type Point: Copy + Eq + Hash + Ord + Debug;
    type Payload;

    // Traversal
    fn cone<'a>(&'a self, p: Self::Point) -> Self::ConeIter<'a>;
    fn support<'a>(&'a self, p: Self::Point) -> Self::SupportIter<'a>;

    // Arrow-level mutation (upsert semantics, mirrors kept in sync)
    fn add_arrow(&mut self, src: Self::Point, dst: Self::Point, payload: Self::Payload);
    fn remove_arrow(&mut self, src: Self::Point, dst: Self::Point) -> Option<Self::Payload>;

    // Point sets
    fn base_points<'a>(&'a self) -> Box<dyn Iterator<Item = Self::Point> + 'a>;
    fn cap_points<'a>(&'a self) -> Box<dyn Iterator<Item = Self::Point> + 'a>;

    // Helpers: closure/star/strata/chart/... (provided; see below)
    /* ... */
}
```

### Invariants

* **Edge uniqueness**: for each `(src,dst)` there is **at most one** arrow; `add_arrow` replaces payload if it already exists.
* **Mirrors**: `cone(src)` and `support(dst)` are perfect mirrors. Debug builds assert this (also in oriented backends).

### Basic usage

```rust
use mesh_sieve::topology::sieve::{InMemorySieve, Sieve};

let (c,f,e,v) = (1,2,3,4);
let mut s = InMemorySieve::<u32, ()>::default();
s.add_arrow(c, f, ());
s.add_arrow(f, e, ());
s.add_arrow(e, v, ());

let cone_f: Vec<_> = s.cone(f).map(|(q,_)| q).collect();    // [e]
let sup_e:  Vec<_> = s.support(e).map(|(p,_)| p).collect(); // [f]
```

### Traversal iterators

* Value-based (clone payloads): `closure_iter`, `star_iter`, `closure_both_iter`.
* Borrow-based (no payload clones; require `SieveRef`): `closure_iter_ref`, `star_iter_ref`, `closure_both_iter_ref`.
* Deterministic variants: `*_sorted` (sorted seeds), and `*_sorted_neighbors` (also sorts neighbor expansion).

```rust
let down: Vec<_> = s.closure(std::iter::once(c)).collect(); // c,f,e,v
let up:   Vec<_> = s.star(std::iter::once(v)).collect();    // v,e,f,c
```

### Strata & chart (DAG analytics)

All default-implemented on the trait; backed by a cache in most backends.

* `height(p)`, `depth(p)`, `diameter()`
* `height_stratum(k)`, `depth_stratum(k)`
* `chart_points()` – height-major then `Ord`; deterministic.
* `points_chart_order()` – same as `chart_points()`.

> Returns an error if a cycle is detected; use `utils::check_dag` to validate first.

### Lattice ops

* `meet(a,b)` – maximal shared **downward** closure.
* `join(a,b)` – maximal shared **upward** star.

### Point/role mutators (via `MutableSieve`)

Available through forwarding methods on `Sieve`:

* `add_point`, `remove_point`
* `add_base_point`, `add_cap_point`
* `remove_base_point`, `remove_cap_point`
* `set_cone`, `add_cone`, `set_support`, `add_support`

---

## `Sieve` implementations

### `InMemorySieve<P, T = ()>`

HashMap + `Vec` adjacency. Fast average-case updates; iteration order follows insertion order unless you call `sort_adjacency()`.

* Cache: `OnceCell<StrataCache<P>>`
* Bulk helpers: `reserve_from_edges`, `reserve_from_edge_counts`, `shrink_to_fit`
* `add_arrow_val` / `add_cone_val` when `T = Arc<U>` (see below)

### `InMemorySieveDeterministic<P, T = ()>`

`BTreeMap` with **sorted neighbor lists**. Deterministic by construction (useful for testing, reproducibility).

### `InMemoryOrientedSieve<P, T = (), O = Sign>`

Stores `(dst, payload, orientation)`, implements **both** `Sieve` (payload-only view) and `OrientedSieve`.

* `add_arrow_o(src,dst,pay,orient)` and mirror maintenance.
* `cone_o`, `support_o` expose `(point, orientation)` where `support_o(p)` reports `orient(src→p)` (invert when walking up).
* Convenience: `add_arrow_val` / `add_arrow_o_val` for `T = Arc<U>`.

### Frozen CSR: `FrozenSieveCsr<P, T>`

Immutable, cache-friendly, deterministic. Build from any `Sieve`:

```rust
use mesh_sieve::topology::sieve::{freeze_csr, Sieve};
let frozen = freeze_csr(s);           // consumes or clones as needed
let deg = frozen.out_degree(c);
```

* All points and neighbor lists are sorted.
* Great for read-heavy traversal; **immutable** (mutators `unreachable!()`).

### Shared payload convenience

When payloads are large or shared, use `Arc<T>`:

```rust
use mesh_sieve::topology::sieve::{InMemorySieveArc, InMemoryOrientedSieveArc, InMemoryStackArc};

type FaceSieve = InMemorySieveArc<u32, MyHeavyPayload>;
type TriSieve  = InMemoryOrientedSieveArc<u32, MyHeavyPayload>; // default Sign; use aliases D3/D4 as needed
type DOFStack  = InMemoryStackArc<u32, u32, MyHeavyPayload>;
```

The iterator clones only the `Arc`, not the heavy payload.

---

## Oriented traversal (`OrientedSieve`)

```rust
use mesh_sieve::topology::sieve::{OrientedSieve, InMemoryOrientedSieve};
use mesh_sieve::topology::orientation::D3;

let mut s = InMemoryOrientedSieve::<u32, (), D3>::default();
s.add_arrow_o(10, 20, (), D3::default());
let down_o = s.closure_o([10]); // Vec<(point, orientation-from-seed)>
let up_o   = s.star_o([20]);    // uses inverse() per step
```

* **Mirror rule**: the orientation stored for `src→dst` must match the one seen in `support_o(dst)`; debug builds check this.
* `accumulate_path` helps compose a path’s orientations explicitly when needed.

---

## Extension traits

* `SieveBuildExt` – bulk insertion: `add_arrows_from`, `add_arrows_dedup_from` (pre-reserves and invalidates once).
* `SieveReserveExt` – forwards to backend preallocation helpers (`reserve_from_edges`, `reserve_from_edge_counts`).
* `SieveQueryExt` – `out_degree`, `in_degree` (can be overridden by backends).
* `SieveRef` – borrow-based cone/support and `*_ref` traversal family (no payload cloning).

---

## Stacks (vertical incidence: mesh ↔ DOF)

### `Stack` trait

Links a **base** sieve to a **cap** sieve with vertical arrows and payload:

```rust
pub trait Stack {
    type Point;    // base point
    type CapPt;    // cap point
    type VerticalPayload: Clone;
    type BaseSieve: Sieve<Point = Self::Point>;
    type CapSieve:  Sieve<Point = Self::CapPt>;

    fn lift(&self, p: Self::Point) -> Box<dyn Iterator<Item=(Self::CapPt, Self::VerticalPayload)> + '_>;
    fn drop(&self, q: Self::CapPt) -> Box<dyn Iterator<Item=(Self::Point, Self::VerticalPayload)> + '_>;

    fn add_arrow(&mut self, base: Self::Point, cap: Self::CapPt, pay: Self::VerticalPayload) -> Result<(), MeshSieveError>;
    fn remove_arrow(&mut self, base: Self::Point, cap: Self::CapPt) -> Result<Option<Self::VerticalPayload>, MeshSieveError>;

    fn base(&self) -> &Self::BaseSieve; // may panic in composed stacks
    fn cap(&self)  -> &Self::CapSieve;

    fn base_mut(&mut self) -> Result<&mut Self::BaseSieve, MeshSieveError>;
    fn cap_mut(&mut self)  -> Result<&mut Self::CapSieve, MeshSieveError>;
}
```

### `InMemoryStack<B,C,V,PB,PC>`

Stores two sieves (`base`, `cap`) with independent horizontal payloads and two vertical maps (`up: B→[(C,V)]`, `down: C→[(B,V)]`).

* Upsert on `(base,cap)`; mirrors and payloads kept consistent.
* Vertical mutations do **not** invalidate base/cap caches; call `invalidate_base_and_cap` if needed.
* Debug invariants ensure no duplicates, exact up/down totals, and membership of vertical points in their respective sieves.

### `ComposedStack<'a, S1, S2, F, VO>`

Stack composition: **lower: `base→mid`**, **upper: `mid→cap`**. Traversal composes vertical payloads using a provided `compose_payload: F(&V1,&V2)->VO`. No mutation or direct base/cap access (panics/`UnsupportedStackOperation`).

> Use this to build multi-level vertical relations (e.g., cell→face→edge→DOF) without materializing an explicit mid layer.

---

## Utilities

### DAG check

`utils::check_dag` / `check_dag_ref` run Kahn’s algorithm via `Sieve`/`SieveRef`:

```rust
use mesh_sieve::topology::{sieve::Sieve, utils::check_dag};

let mut s = InMemorySieve::<u32,()>::default();
// ... add arrows ...
check_dag(&s)?; // Err(CycleDetected) if cyclic
```

### Debug-only invariants

Backends call `debug_invariants!(self)` after mutations (enabled in debug or with `feature = "strict-invariants"`). Checks include:

* no duplicate destinations per source;
* out/in pair counts match;
* mirror entries present;
* in oriented sieves, stored orientations match in both mirrors.

---

## Performance & determinism

* **Updates**: `InMemorySieve` gives amortized *O(1)* upserts; removals scan a single adjacency vec (*O(degree)*).
* **Traversal**: all iterators are DFS with “first seen wins.” Use `*_sorted` or CSR for reproducible traversals.
* **Read-mostly**: freeze to `FrozenSieveCsr` for compact, cache-friendly walks.
* **Payload size**: prefer `Arc<T>` payloads to avoid cloning `T`.
* **Bulk builds**: use `SieveBuildExt::add_arrows_from(_dedup_)` and `SieveReserveExt` to minimize reallocations.
* **Strata**: first query builds the cache (O(|V|+|E|)), subsequent queries are O(1)/O(layer).

---

## Error handling

Most analytical helpers return `Result<_, MeshSieveError>` with variants like `CycleDetected` (DAG violated) or `MissingPointInCone` (backend inconsistency). `Stack` mutators can report `UnsupportedStackOperation` for composed stacks.

---

## End-to-end examples

### 1) Build a 2D cell→face→edge→vertex topology and compute strata

```rust
use mesh_sieve::topology::sieve::{InMemorySieve, Sieve};
let (c,f1,f2,e1,e2,v1,v2,v3) = (1,2,3,4,5,6,7,8);
let mut s = InMemorySieve::<u32, ()>::default();
s.add_arrow(c, f1, ());
s.add_arrow(c, f2, ());
s.add_arrow(f1, e1, ());
s.add_arrow(f2, e2, ());
s.add_arrow(e1, v1, ());
s.add_arrow(e1, v2, ());
s.add_arrow(e2, v2, ());
s.add_arrow(e2, v3, ());
assert_eq!(s.height(v3)?, 3);
assert!(s.closure([c]).any(|p| p == v2));
```

### 2) Oriented faces (triangles) and accumulated orientation

```rust
use mesh_sieve::topology::sieve::{InMemoryOrientedSieve, OrientedSieve};
use mesh_sieve::topology::orientation::D3;

let mut tri = InMemoryOrientedSieve::<u32, (), D3>::default();
tri.add_arrow_o(10, 20, (), D3::default());
tri.add_arrow_o(20, 30, (), D3::default());
let path = tri.closure_o([10]); // [(10,id), (20,id), (30,id)]
```

### 3) Attach DOFs via a stack

```rust
use mesh_sieve::topology::stack::{InMemoryStack, Stack};

let mut stk = InMemoryStack::<u32,u32,()>::new();
// base/cap sieves are available at stk.base / stk.cap; fill them if needed
stk.add_arrow(100, 1000, ())?; // cell 100 → dof 1000
let dofs: Vec<_> = stk.lift(100).map(|(q,_)| q).collect();
```

### 4) Freeze a mutable sieve for read-only kernels

```rust
use mesh_sieve::topology::sieve::{freeze_csr, InMemorySieve, Sieve};

let mut s = InMemorySieve::<u32, ()>::default();
s.add_arrows_from([(1,2,()), (1,3,()), (2,4,())]);
let csr = freeze_csr(s);
let nbrs: Vec<_> = csr.cone(1).map(|(q,_)| q).collect(); // [2,3], sorted
```

---

## Practical tips

* **No parallel edges**: if you “insert” twice, the payload is replaced, not duplicated.
* **Keep mirrors in sync**: backends do this for you; if you write your own, update both out/in maps.
* **Always invalidate caches** after mutations in custom code: call `InvalidateCache::invalidate_cache`.
* **Prefer `SieveRef`** when you only need point IDs; it avoids payload cloning in traversals.
* **Use deterministic paths** (`*_sorted_neighbors` or CSR) for testing or serialization.
