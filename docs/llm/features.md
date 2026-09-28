# Feature and realm support

Verified against commit `b90eb2a`. Status words:
- `works`: the code path exists and has no known blocker.
- `partial`: the path works but some entries or options fail (see Notes).
- `broken`: the path exists but fails.
- `none`: no code or configuration.

Nothing has been checked end to end in this environment.

## Model x realm

Columns:
- YAML: file name suffix in `data/`, and number of entries.
- In driver map: listed in `cmor_driver.py::REALM_YAML_MAP`.
- Include patterns: frequencies defined in `data/<model>_include_patterns.yaml`.
- Output grid: what `grids` holds in the YAML.

| Model | Realm | YAML (entries) | In driver map | Include patterns | Output grid | Status | Notes |
|---|---|---|---|---|---|---|---|
| CESM | atmos | `atmos` (69) | yes | mon (tavg, tpt), day, 6hr, 3hr | gr | partial | BUG-5 (4 entries), GAP-5 (`-hs-` site vars) |
| CESM | land | `land` (227) | yes | mon (tavg, tpt) | gr | partial | BUG-4 (123 formulas), BUG-5 (44 entries) |
| CESM | ocean | `ocean` (21) | yes | mon, day | gn; 5 vars gn+gr | broken for gn | BUG-13, BUG-8; `gr` needs `--resolution tx2_3v2` |
| CESM | seaIce | `seaIce` (94) | yes | mon, day | gn; 4 global means gm | works | NH/SH variants |
| CESM | aerosol | none | no | mon, day, 6hr, 3hr | none | none | GAP-1 |
| CESM | atmosChem | none | no | mon, day, 6hr, 3hr | none | none | GAP-1 |
| CESM | ocnBgchem | none | no | mon, day | none | none | GAP-1 |
| CESM | landIce | empty file | no | none | none | none | GAP-1 |
| NorESM | atmos | `atmos` (219) | yes | mon, day, 6hr, 3hr, 1hr | gr | partial | BUG-5 (1 entry) |
| NorESM | aerosol | `aerosol` (90) | yes | mon, day, 6hr, 3hr, 1hr | gr | works | |
| NorESM | atmosChem | `atmosChem` (14) | yes | mon, day, 6hr, 3hr, 1hr | gr | works | |
| NorESM | land | `land` (137) | yes | mon, day, 3hr, yr | gr | partial | BUG-4 (2), BUG-5 (1) |
| NorESM | seaIce | `seaIce` (94) | yes | mon, day | gn/gm | works | Copy of the CESM file (Q-4) |
| NorESM | landIce | `landIce` (18) | yes | yr | gn | works | Needs `--ice-sheet` and pyproj (ENV-4) |
| NorESM | ocean, ocnBgchem | none | no | none | none | none | GAP-1 |

Resolutions (`--resolution`), from `data/<model>_regrid_maps.yaml`:

| Model | Resolution | Target grid |
|---|---|---|
| CESM | `ne30` | 1x1 deg |
| CESM | `tx2_3v2` | 1x1 deg |
| CESM | `regular` | none; input is already lat/lon |
| NorESM | `ne30` | 1x1 deg |
| NorESM | `ne16` | 2x2 deg |
| NorESM | `tnx1v4` | 1x1 deg |
| NorESM | `regular` | none; input is already lat/lon |

## Capabilities

| Capability | Status | Where / notes |
|---|---|---|
| Rename a single source | works | `_realize_core` |
| Formula with sources or aliases | works | Only `FORMULA_NAMESPACE` functions (BUG-4) |
| `scale` / `unit_conversion` keys | broken | BUG-3 |
| Per-frequency sources (`freq:`) | works | `_filter_sources` |
| tavg vs tpt file selection | works | `patterns_for_variable` |
| Conservative regrid | works | default |
| Bilinear regrid | works | Needs `regrid_method: bilinear` from `intensive_vars.yaml`. Not used on the plev39 path. |
| Land fraction normalisation | works | `lndgrid` input |
| Ocean fraction handling | partial | Normalizes but does not denormalize (Q-7) |
| plev3/7c/7h/19 interpolation | works | Native grid, before regrid (`vertical.to_plev`, geocat) |
| plev39 zonal mean | works | Regrid, then interpolate, then mean over lon. Labelled `gn` (BUG-6). |
| Hybrid sigma output (full and half levels) | works | atmos, atmosChem and aerosol tables only |
| Soil depth (`sdepth`) | works | `data/depth_bnds.nc` |
| Ocean depth (`z_l`, `zl`) | partial | MOM6 grid needs `ocean_geometry.nc` (BUG-13) |
| FATES pft and fuel axes | works | `fates_levpft`, `fates_levfuel` |
| Native CICE grid | works | `_define_cice_grid` |
| Native CISM grid | works | `_define_cism_grid` + `cism_grid.py` |
| Native MOM6 grid | broken | BUG-13 |
| Site (`hs`) output | none | GAP-5 |
| fx fields (`sftlf`, `sftof`, `areacella`) | partial | Written alongside regridded variables. `table: fx` fails (BUG-8). |
| Non-CMIP7 variables (`--run-all-from-yaml`) | partial | GAP-6 |
| Parallel workers | none | GAP-7 |
| Output validation and HTML report | works | `validate_cmor_output.py`, `build_validation_html.py` |
