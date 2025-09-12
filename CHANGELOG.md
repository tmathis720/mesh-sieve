# Changelog

## [Unreleased]

- Remove `SieveArcPayload` wrapper. Use `InMemorySieveArc`, `InMemoryOrientedSieveArc`, or `InMemoryStackArc` (store `Arc<T>` directly) for shared payloads.
- Added optional invariant checking for `Atlas` and `Section` via the
  `DebugInvariants` trait. Enable the `strict-invariants` feature (alias
  `check-invariants`) to run these validations in release builds.
- Introduced `FallibleMap` and `try_restrict_*` helpers for error-aware data
  access. Legacy `restrict_*` helpers remain but now document their panicking
  behavior.
- Added `SliceReducer` and `Bundle::assemble_with` to customize how cap slices
  are merged into base slices. `Bundle::assemble` now performs element-wise
  averaging via `AverageReducer`.
- `Bundle::assemble_with` now validates slice lengths up front and returns
  `SliceLengthMismatch` with the exact mismatching cap `PointId`. Reducers such
  as `AverageReducer` assume equal lengths and no longer perform checks.
- Added error variants `AtlasPointLengthChanged`, `ReducerLengthMismatch`,
  and `AtlasContiguityMismatch` for clearer diagnostics.
- Unified invariant checks under a crate-wide `debug_invariants!` macro and
  deprecated the old `data_debug_assert_ok!` shim.
- Renamed `data::refine::delta::Delta` to `SliceDelta`; `Delta` remains as a
  deprecated alias.
- Renamed `overlap::delta::Delta` to `ValueDelta`; `Delta` remains as a
  deprecated alias.
- `Section::with_atlas_mut` now rejects slice length changes with
  `MeshSieveError::AtlasPointLengthChanged`; use the new
  `with_atlas_resize(ResizePolicy, ...)` to explicitly allow resizing.
- **Removed:** deprecated panicking shims `Atlas::insert`, `Section::{restrict, restrict_mut, set}`,
  `SievedArray::{get, get_mut, set, refine_with_sifter, refine, assemble}`, and
  `SievedArray::try_iter` alias.
- **Added:** `map-adapter` feature gating the legacy `Map` trait and
  infallible `restrict_*` helpers. The feature is off by default.
- **Migration:** switch call sites to `try_*` methods or temporarily enable
  `map-adapter` to retain panicking behavior.
