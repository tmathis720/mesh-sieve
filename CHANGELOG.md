# Changelog

## [Unreleased]

- Remove `SieveArcPayload` wrapper. Use `InMemorySieveArc`, `InMemoryOrientedSieveArc`, or `InMemoryStackArc` (store `Arc<T>` directly) for shared payloads.
