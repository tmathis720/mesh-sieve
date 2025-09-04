# Changelog

## [Unreleased]

- Remove `SieveArcPayload` wrapper. Use `InMemorySieveArc`, `InMemoryOrientedSieveArc`, or `InMemoryStackArc` (store `Arc<T>` directly) for shared payloads.
- Added optional invariant checking for `Atlas` and `Section` via the
  `DebugInvariants` trait. Enable the `check-invariants` feature to run these
  validations in release builds.
