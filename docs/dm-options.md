# DMPLEX-style DM option profiles

`mesh-sieve` exposes a serializable configuration layer in `src/dm_options.rs` for downstream PDE applications that already use PETSc DMPLEX-style workflows.  The layer keeps CLI and environment parsing outside the core topology/data structures and lowers profiles into native runtime types:

- `DmplexConfigProfile::mesh_dm_options()` -> `MeshDMOptions`
- `DmplexConfigProfile::distribution_config()` -> `DistributionConfig`
- `DmplexConfigProfile::mesh_check_options()` -> `MeshCheckOptions`
- `DmplexConfigProfile::prepare_for_solve_options()` -> `PrepareForSolveOptions`
- `DmplexConfigProfile::metric_adapt_options()` -> `MeshDMMetricAdaptOptions`

## Common option mappings

| PETSc/DMPLEX-style option | mesh-sieve profile field | Native target |
| --- | --- | --- |
| `-dm_refine_pre <n>` | `refinement.pre_refine` | `MeshDMOptions::pre_refine` |
| `-dm_refine <n>` / `-dm_refine_post <n>` | `refinement.post_refine` | `MeshDMOptions::post_refine` |
| `-dm_distribute <bool>` | `distribution.distribute` | `MeshDMOptions::distribute` |
| `-dm_distribute_overlap <n>` | `overlap.depth` | `DistributionConfig::overlap_depth` |
| `-dm_plex_localize <bool>` | `distribution.localize_coordinates` | `MeshDMOptions::localize_coordinates` |
| `-dm_distribute_synchronize_sections <bool>` | `distribution.synchronize_sections` | `DistributionConfig::synchronize_sections` |
| `-dm_distribute_balance_boundary_ownership <bool>` | `distribution.balance_boundary_ownership` | `DistributionConfig::balance_boundary_ownership` |
| `-dm_plex_reorder_section <vertex_first\|cell_first>` | `ordering` | `MeshDMOptions::reorder_section` |
| `-dm_plex_check_all <bool>` | `checks.check_all` | `MeshCheckOptions::all()` |
| `-dm_plex_check_symmetry <bool>` | `checks.check_symmetry` | `MeshCheckOptions::check_symmetry` |
| `-dm_plex_check_skeleton <bool>` | `checks.check_skeleton` | `MeshCheckOptions::check_skeleton` |
| `-dm_plex_check_faces <bool>` | `checks.check_faces` | `MeshCheckOptions::check_faces` |
| `-dm_plex_check_geometry <bool>` | `checks.check_geometry` | `MeshCheckOptions::check_geometry` |
| `-dm_plex_metric_target_complexity <x>` | `metric.target_complexity` | `MetricNormalizationControls::target_complexity` |
| `-dm_plex_metric_gradation <x>` | `metric.gradation` | `MetricNormalizationControls::gradation` |
| `-dm_plex_metric_h_min <x>` | `metric.min_magnitude` | `MetricNormalizationControls::min_magnitude` |
| `-dm_plex_metric_h_max <x>` | `metric.max_magnitude` | `MetricNormalizationControls::max_magnitude` |
| `-dm_plex_metric_a_min <x>` | `metric.min_anisotropy` | `MetricNormalizationControls::min_anisotropy` |
| `-dm_plex_metric_a_max <x>` | `metric.max_anisotropy` | `MetricNormalizationControls::max_anisotropy` |
| `-dm_plex_metric_hausdorff_number <x>` | `metric.hausdorff_number` | `MetricNormalizationControls::hausdorff_number` |
| `-dm_plex_metric_backend <internal\|triangle\|tetgen\|gmsh\|mmg>` | `metric.backend` | `MetricRemeshingBackend` |
| `-dm_plex_filename <path>` | `io.filename` | application reader/writer plumbing |
| `-dm_plex_format <name>` | `io.format` | application reader/writer plumbing |
| `-dm_plex_interpolate <bool>` | `io.interpolate` | application reader/writer plumbing |

## CLI and environment helpers

`DmplexConfigProfile::from_cli_args()` accepts `-key value`, `-key=value`, and boolean flag forms. Unknown options are ignored, allowing applications to pass larger PETSc-like argument vectors.

`DmplexConfigProfile::from_env()` accepts uppercase option names without leading dashes, such as `DM_DISTRIBUTE=true`, `DM_DISTRIBUTE_OVERLAP=2`, or prefixed forms such as `MESH_SIEVE_DM_PLEX_METRIC_TARGET_COMPLEXITY=1000.0`.
