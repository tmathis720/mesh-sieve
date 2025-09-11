# Changelog

## [Unreleased]

- Remove `SieveArcPayload` wrapper. Use `InMemorySieveArc`, `InMemoryOrientedSieveArc`, or `InMemoryStackArc` (store `Arc<T>` directly) for shared payloads.
- Added optional invariant checking for `Atlas` and `Section` via the
  `DebugInvariants` trait. Enable the `check-invariants` feature to run these
  validations in release builds.
- Introduced `FallibleMap` and `try_restrict_*` helpers for error-aware data
  access. Legacy `restrict_*` helpers remain but now document their panicking
  behavior.
- Added `SliceReducer` and `Bundle::assemble_with` to customize how cap slices
  are merged into base slices. `Bundle::assemble` now performs element-wise
  averaging via `AverageReducer`.
- Renamed `data::refine::delta::Delta` to `SliceDelta`; `Delta` remains as a
  deprecated alias.
- Renamed `overlap::delta::Delta` to `ValueDelta`; `Delta` remains as a
  deprecated alias.
- `Section::with_atlas_mut` now rejects slice length changes with
  `MeshSieveError::AtlasSliceLengthChanged`; use the new
  `with_atlas_resize(ResizePolicy, ...)` to explicitly allow resizing.
