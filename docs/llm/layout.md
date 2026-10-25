# Layout: where things live

Verified against commit `b90eb2a`. `file::symbol` references avoid line numbers
because line numbers go stale.

## Tree

```
scripts/                     entry points (not a package; pylint ignores it)
  cmor_driver.py             stage 3: data request -> realize -> regrid -> CMOR write
  gen_timeseries.py          stage 2: history files -> per-variable time series (gents)
  convert_csv_to_yaml.py     stage 1: spreadsheet CSV -> per-realm mapping YAML
  yaml_to_csv.py             reverse of stage 1 (CESM column layout)
  query_missing_vars.py      CSV of mapped + requested-but-unmapped variables
  validate_cmor_output.py    check an output subset; optional plots and HTML
  build_validation_html.py   static HTML site over all validation reports
  find_unmapped_vars.py      broken log parser (BUG-11)
  setup_nird.sh              NIRD module and conda setup
src/cmip7_prep/              library (package name cmip7_prep)
data/                        mapping YAML, lookup tables, dataset JSON, static NetCDF
tests/                       pytest suite; see testing.md
cmip7-cmor-tables/           git submodule (noresm-dev); empty until initialised (ENV-5)
docs/llm/                    these docs
Dockerfile, INSTALL.md, README.md, pyproject.toml, pytest.ini, requirements.txt
```

`CMIP7/` is the default output directory of `CmorSession` (used when no `outdir` is given). The driver's `--outdir` defaults to `.` instead. Local data directories and weight files are not part of the repo; `SESSION.md`, if present, describes them.

## Library modules (`src/cmip7_prep/`)

| Module | Purpose | Key symbols |
|---|---|---|
| `mapping_compat.py` | Load a mapping YAML; build CMIP variables from native data | `Mapping` (`from_packaged_default`, `from_yaml`, `get_cfg`, `realize`, `realize_all`, `timeseries_source_vars`, `iter_variable_names`), `VarConfig`, `FORMULA_NAMESPACE`, `verticalsum`, `sumover_index`, `verticalmean`, `_safe_eval`, `_filter_sources`, `_realize_core`, `STATIC_MODEL_VARS` |
| `pipeline.py` | Select and open time-series files; realize, vertical-interpolate and regrid one variable | `open_native_for_cmip_vars`, `realize_regrid_prepare`, `_collect_required_model_vars`, `_filename_contains_var`, `_apply_vertical_if_needed` |
| `regrid.py` | Apply ESMF weights; land and ocean fraction handling; fx fields; plev39 zonal mean | `regrid_to_latlon_ds`, `regrid_to_latlon`, `_pick_maps`, `MapSpec`, `_regrid_fx_once`, `zonal_mean_on_pressure_grid`, `_attach_time_and_bounds`, `_attach_vertical_metadata`, `_calculate_area_from_bounds` |
| `regrid_maps.py` | Read `data/<model>_regrid_maps.yaml` | `get_map_paths`, `load_regrid_maps` |
| `cache_tools.py` | Cache xESMF regridders and fx fields | `RegridderCache`, `FXCache`, `_make_dummy_grids`, `open_nc` |
| `vertical.py` | Hybrid-sigma to pressure levels | `to_plev`, `_read_requested_levels`, `_resolve_p0`, `remap_isopycnal_to_olevel` (unused, GAP-11) |
| `cmor_writer.py` | CMOR session and variable writing | `CmorSession` (`__enter__`, `write_variable`, `_define_axes`, `_define_mom6_grid`, `_define_cice_grid`, `_define_cism_grid`, `ensure_fx_written_and_cached`, `_write_fx_2d`, `_time_axis_entry`, `_positive_entry`, `table_path`) |
| `cmor_utils.py` | CMOR helpers | `filled_for_cmor` (NaN to 1e20, float32), `encode_time_to_num`, `bounds_from_centers_1d`, `roll_for_monotonic_with_bounds`, `sigma_mid_and_bounds`, `resolve_table_filename`, `load_positive_overrides`, `packaged_dataset_json`, `open_existing_fx` |
| `include_patterns.py` | Read `data/<model>_include_patterns.yaml` | `get_include_patterns`, `all_include_patterns`, `patterns_for_variable`, `sampling_from_branded_name` |
| `variable_selection.py` | Synthesize Data Request variables for `--run-all-from-yaml` | `assemble_yaml_defined_cmip_vars`, `build_synthetic_variable`, `SyntheticVariable` |
| `mom6_static.py` | MOM6 static fx fields | `ocean_fx_fields` (returns `deptho`, `areacello`) |
| `cism_grid.py` | Georeference CISM x/y to lat/lon with pyproj | `project_xy_to_latlon`, `ICE_SHEET_PROJ` (gris EPSG:3413, ais EPSG:3031), `_check_plausible` |
| `unused_functions.py` | Dead unit-conversion helper | `_apply_unit_conversion` (BUG-3) |

Import graph (library):

```
pipeline -> mapping_compat, regrid, vertical
regrid -> cache_tools, regrid_maps, vertical
cmor_writer -> cmor_utils, cism_grid (lazy)
```

`scripts/cmor_driver.py` imports `pipeline`, `regrid`, `cmor_writer`,
`cmor_utils`, `include_patterns`, `mapping_compat`, `mom6_static` and
`variable_selection`. `validate_cmor_output.py` imports
`cmor_driver.REALM_YAML_MAP`, which pulls in `cmor`.

## Task to location

| Task | Edit here |
|---|---|
| Add or change a variable mapping | Spreadsheet, then `convert_csv_to_yaml.py`, then `data/<model>_to_cmip7_<realm>.yaml` (see `mapping.md`) |
| Add a formula function | `mapping_compat.py`: define the function, add it to `FORMULA_NAMESPACE` (the converter's validator uses the same dict) |
| Add a realm for a model | `cmor_driver.py::REALM_YAML_MAP`, `data/<model>_include_patterns.yaml`, `convert_csv_to_yaml.py::MODEL_CONFIGS[...]["realm_outputs"]` and `REALM_GRIDS`, and the `gen_timeseries.py` `--realm` choices |
| Change which history files are read | `data/<model>_include_patterns.yaml` |
| Add a resolution or weight file | `data/<model>_regrid_maps.yaml`, plus `--resolution` choices in `cmor_driver.py::parse_args` |
| Change bilinear vs conservative | `data/intensive_vars.yaml`, then regenerate the YAML (the value is baked in as `regrid_method`). Needs sign-off. |
| Change native vs regridded output | `convert_csv_to_yaml.py::REALM_GRIDS` / `GRIDS_OVERRIDES` (GAP-4). Needs sign-off. |
| Fix a flux sign | `data/<model>_positive.yaml` (Q-2). Needs sign-off. |
| Dataset global attributes | `data/cmor_dataset.json` (CESM) or `data/cmor_dataset_noresm.json` |
| New vertical axis type | `cmor_writer.py::CmorSession._define_axes` (the vertical `elif` chain and `dim_to_axis`) |
| New native horizontal grid | `_define_axes` grid dispatch plus a `_define_<x>_grid` method |
| Time axis choice | `CmorSession._time_axis_entry` (read from the table's `dimensions`) |
| Vertical interpolation | `vertical.to_plev`, `pipeline._apply_vertical_if_needed` |
| Regrid behaviour by input dims | `regrid.regrid_to_latlon` |
| Which variables run | `cmor_driver.main` (Data Request query, 30S-90S de-dup, `--cmip-vars` filter) |
| Per-variable dispatch | `cmor_driver.process_one_var`, `_prepare_native`, `_prepare_seaice_native`, `_prepare_regridded` |
| Output validation | `scripts/validate_cmor_output.py`, `scripts/build_validation_html.py` |
