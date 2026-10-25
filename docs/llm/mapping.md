# Mapping YAML and `data/` configuration

Verified against commit `b90eb2a`. Issue IDs point to `issues.md`.

## Flow of ownership

```
Google Sheet (scientists own it)
  -> CSV export
  -> scripts/convert_csv_to_yaml.py     (engineers run it)
  -> data/<model>_to_cmip7_<realm>.yaml (committed; read by cmor_driver.py)
```

Do not hand-edit the generated YAML for lasting changes. The next regeneration
overwrites it. Fix the spreadsheet, or the converter's rules, instead.
`scripts/yaml_to_csv.py` and `scripts/query_missing_vars.py` go the other way
(YAML to CSV) to seed the spreadsheet.

## Mapping file schema

```yaml
dataset_overrides: {...}        # written by the converter; ignored by all code
variables:
  <branded_name>:               # e.g. tas_tavg-h2m-hxy-u; must match the CMOR table variable_entry key
    table: atmos                # CMOR table key -> tables/CMIP7_<table>.json; a "CMIP7_" prefix is stripped
    units: K                    # written to CMOR; NOT converted (see formula)
    sources:                    # required, non-empty
      - model_var: TREFHT       # native variable; also a formula token
        freq: mon               # optional; picks sources per --frequency
        alias: T                # optional; the formula token name for this source
    formula: "T - 273.15"       # optional; required when more than one source is active (BUG-5)
    grids: [gr]                 # required by the driver: any of gn, gr, gm
    regrid_method: conservative # conservative or bilinear; used on the gr path
    levels: {name: plev19, units: Pa}   # optional vertical handling (see below)
    region: nh                  # optional; set as the CMOR dataset "region"
    variants: [...]             # optional; sea-ice NH/SH split (see below)
    long_name, standard_name, cell_methods, description   # metadata; long_name and standard_name go into attrs
```

| Key | Read by | Notes |
|---|---|---|
| `sources[].model_var` | `pipeline`, `mapping_compat` | A file must exist whose name contains `.<model_var>.`, unless the name is in `STATIC_MODEL_VARS` (`tarea`, `TLAT`, `area`, `landfrac`, `wet`, ...). |
| `sources[].freq` | `mapping_compat._filter_sources` | Sources tagged with the run frequency, plus untagged sources, are used. If no tag matches, the untagged sources are used; if there are none, all sources. |
| `formula` | `mapping_compat._safe_eval` | Python `eval` with no builtins. The namespace is the source tokens plus `FORMULA_NAMESPACE`: `np`, `xr`, `verticalsum`, `sumover_index` (1-based indices), `verticalmean` (CISM sigma-weighted). The result must be a DataArray. |
| `scale`, `unit_conversion` | nothing | Ignored (BUG-3). Put the arithmetic in `formula`. |
| `positive` | nothing | Ignored. Signs come from the table, or from `data/<model>_positive.yaml`. |
| `dims` | nothing | The converter drops it. Old README examples still show it (DOC-1). |
| `grids` | `cmor_driver.process_one_var` | Chooses the code path (`pipeline.md`). It does NOT set the output `grid_label` (BUG-6). |
| `levels.name` | pipeline, writer | Values in use: `plev3`, `plev7c`, `plev7h`, `plev19`, `plev39`, `standard_hybrid_sigma`, `standard_hybrid_sigma_half`. `plevNN` must exist in `CMIP7_coordinate.json`. |
| `levels.src_axis_name` / `src_axis_bnds` | `cmor_utils.sigma_mid_and_bounds` | Fallbacks only; `hyam+hybm` and `hyai+hybi` win. |
| `variants` | `Mapping._load_yaml` | Expanded into keys `<name>:0`, `<name>:1`, ... Each variant overrides `long_name`, `formula` and `region`. Only the seaIce path (`realize_all`) iterates the variants. Other paths use variant 0. |

## Spreadsheet to YAML rules (`convert_csv_to_yaml.py`)

| Rule | CESM (`MODEL_CONFIGS["cesm"]`) | NorESM (`MODEL_CONFIGS["noresm"]`) |
|---|---|---|
| Key column | `CMIP Variable Name` | `Branded Variable Name` |
| Realm column | `Table` | `Modelling Realm - Primary` |
| Source column | `CESM Variable Name` | `NorESM3 name (dependency)` |
| Other columns | `Formula`, `Freq`, `Alias`, `Cell Methods`, `Region`, `Levels Name`, `Levels Units`, `Levels Src Axis Name`, `Levels Src Axis Bnds`, `Long Name`, `Standard Name`, `Units`, `Dimensions` | `Description`, `Units (from Physical Parameter)`, `Dimensions` |
| Rows dropped | Empty source cell | Empty source cell, or the source contains `?`, `n/a`, `#N/A`, `derived`, `IN SURF DATASET` or `can be derived`; or the key contains `_tclm-`, `_tclmdc-` or `_tminavg-` |
| Realms written | atmos, atmosChem, aerosol, land, seaIce (as `seaice`, BUG-7), ocean, fx (merged into ocean) | atmos, land, landice (BUG-7), aerosol, atmosChem |

Derived fields:
- `grids`: `GRIDS_OVERRIDES[name]`, else `REALM_GRIDS[realm]` (GAP-4).
- `regrid_method`: `bilinear` if the root name (the text before the first `_`) is in `data/intensive_vars.yaml`, else `conservative`. This overrides any "Regrid Method" column.
- `levels`:
  - From the Levels columns, if present.
  - Else a `plevNN` dim gives `{name: plevNN, units: Pa}`.
  - Else an `alevhalf` dim gives `standard_hybrid_sigma_half` with `src_axis_name: ilev`.
  - Else a `lev` dim gives `standard_hybrid_sigma`.
- `sources`:
  - A comma list of plain identifiers gives one source per name. `Freq` and `Alias` are matched to sources by position.
  - Anything else is parsed as a formula, and its identifiers become sources.
- Formula rewrites:
  - `VAR(1:3)`, `VAR(1,4)` and `VAR(2)` become `sumover_index(VAR, indexlist=[...], dimname=<guessed from VAR suffix>)`.
  - A formula that mentions `FATES` but not `FATES_FRAC` is wrapped as `(<expr>)*FATES_FRAC`.
  - When the file is written, `FATES_FRAC` is renamed to `FATES_FRACTION`.
- Duplicate keys:
  - For `table == seaIce`, rows become `variants`.
  - Otherwise the first row wins.

## Other files in `data/`

| File | Read by | Content |
|---|---|---|
| `<model>_to_cmip7_<realm>.yaml` | driver via `REALM_YAML_MAP` | Mappings. Realm coverage and known-broken entries: `features.md`, BUG-4, BUG-5. |
| `cesm_to_cmip7_landice.yaml` | nothing | Empty placeholder (GAP-1). |
| `<model>_include_patterns.yaml` | `include_patterns.py` | `realm -> frequency -> sampling (tavg/tpt) -> [filename substrings]`. `{ice_sheet}` is filled from `--ice-sheet`. NorESM 3hr atmos is `cam.h4a`; CESM's is `cam.h3a`. |
| `<model>_regrid_maps.yaml` | `regrid_maps.py` | `inputdata_dir` plus `resolutions.<res>.{conservative, bilinear}` weight file names. `regular` reuses the ocean maps (only opened for fx). |
| `intensive_vars.yaml` | `convert_csv_to_yaml.py` only | Root names that get `regrid_method: bilinear`. Editing it changes published values: needs sign-off. |
| `<model>_positive.yaml` | `cmor_utils.load_positive_overrides` | Branded name to `up`/`down`. Overrides the table; CMOR flips the sign on mismatch (Q-2). Needs sign-off. |
| `cmor_dataset.json`, `cmor_dataset_noresm.json` | `CmorSession.__enter__` | CMOR global attributes and output path and file templates. Each must match its table fork's CV (ENV-2). |
| `depth_bnds.nc` | `_define_axes` (sdepth) | Soil layer bounds. Truncated to the data's level count. |
| `ocean_geometry.nc` | `_define_mom6_grid`, `_write_fx_2d` | Empty file (BUG-13). |
| `CMIP7_data_request_v1.0beta-Variables_v1.2.2.4.csv` | nothing | Reference only (DEBT-6). |

## Checklist: adding a variable

1. The branded name exists in `tables/CMIP7_<table>.json` of the fork you run with. Otherwise CMOR rejects it (GAP-6).
2. Every native source is written by `gen_timeseries.py` as `*.<VAR>.*.nc` in the include-pattern files for its sampling.
3. With more than one source, write a `formula`. Put unit conversions in the formula.
4. The formula only calls names in `FORMULA_NAMESPACE`. Add a function in `mapping_compat.py` if needed, with a doctest.
5. Set `grids` and `regrid_method` through the converter rules. Do not set them by hand.
6. For a flux whose model sign convention differs from the table, add it to `data/<model>_positive.yaml` (with sign-off).
7. Run `pytest tests/test_mapping_compat.py tests/test_convert_csv_to_yaml.py` in Docker.
