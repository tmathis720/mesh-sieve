# Exodus Support

## Overview
`mesh-sieve` ships two Exodus paths:

- **Legacy ASCII**: `ExodusReader`/`ExodusWriter` parse the simple text format that starts with `EXODUS`.
- **Exodus II (HDF5)**: `ExodusIiReader`/`ExodusIiWriter` handle the HDF5-backed Exodus II layout and translate common entities into sieve topology and labels.

## Supported Exodus II entities

### Coordinates
- Reads `coord` (2D) or the `coordx`/`coordy`/`coordz` datasets.
- Uses the optional `num_dim` dataset to set coordinate dimension metadata when present.
- Writes `num_dim` plus `coordx`/`coordy`/`coordz` on export.

### Element blocks
- Reads `elem_blk_id`/`connect*` datasets and maps each block to label entries named `exodus:block:<id>`.
- Writes `elem_blk_id`/`connect*` datasets. Blocks are inferred from labels named `exodus:block:<id>`; otherwise all cells are emitted in block `1`.
- Supported element types:
  - `POINT`
  - `BAR2`, `LINE2`, `EDGE2`
  - `TRI3`
  - `QUAD4`
  - `TET4`
  - `HEX8`
  - `WEDGE6`
  - `PYRAMID5`

### Node sets
- Reads `node_ns*` datasets and labels nodes using `exodus:node_set:<id>`.
- Writes node set datasets from `exodus:node_set:<id>` labels.

### Side sets
- Reads `elem_ss*`/`side_ss*` datasets and labels elements using `exodus:side_set:<id>`.
- Writes side set datasets from `exodus:side_set:<id>` labels. The label value is used as the side ordinal when present.

## Limitations
- Only the **HDF5-backed** Exodus II layout is supported. Classic netCDF Exodus files are not handled.
- Element blocks must be homogeneous: mixed cell types within a block are rejected on export.
- Higher-order Exodus elements (for example, `HEX20`, `TET10`) are not supported.
- Side sets are stored on element points; face/edge points are not created automatically. If multiple sides of the same element appear in one set, the last side ordinal wins.
