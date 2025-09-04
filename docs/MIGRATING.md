## Fallible data access

- Old: `restrict_closure(&sieve, &map, seeds)` (panics on missing points).
- New (preferred): `try_restrict_closure(&sieve, &map, seeds)` returning `Result<_, MeshSieveError>`.
- Implement `FallibleMap<V>` for your map types to enable the `try_*` APIs.

## Bundle assembly

- Old: `bundle.assemble(bases)` (previously under-specified).
- New: `bundle.assemble(bases)` now **averages element-wise**.
- Prefer: `bundle.assemble_with(bases, &AverageReducer)` or a custom reducer implementing `Reducer<V>`.

## Delta rename

- Old: `data::refine::delta::Delta`.
- New: `data::refine::delta::SliceDelta` (old `Delta` is a deprecated alias).
- Behavior unchanged.
