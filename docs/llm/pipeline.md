# Pipeline: end-to-end flow

Verified against commit `b90eb2a`. Issue IDs point to `issues.md`.

## Stages

| Stage | Command | Input | Output |
|---|---|---|---|
| 1 | `python scripts/convert_csv_to_yaml.py --model {cesm,noresm} --input X.csv` | Spreadsheet export (CSV) | `<model>_to_cmip7_<realm>.yaml` in the current directory. Move them into `data/` (BUG-7 renames). |
| 2 | `python scripts/gen_timeseries.py --model M --realm R --inputdir D [--outputdir O] [--frequency F..] [--sampling tavg\|tpt] [--ice-sheet gris\|ais] [--years-spec first:last:step]` | History files matched by `*<pattern>*` from the include-pattern YAML | One file per variable, named `...<pattern>.<VAR>.<dates>.nc` |
| 3 | `python scripts/cmor_driver.py --model M --realm R --frequency F --resolution RES --tsdir TS --outdir OUT --tables-root T [--experiment E] [--cmip-vars V..] [--run-all-from-yaml] [--ice-sheet ..] [--ocn-static-file ..] [--custom-yaml Y] [--workers N]` | Time series, mapping YAML, tables, weights | CMOR NetCDF under `OUT/<DRS path>`, CMOR logs in `OUT/logs/` |
| 4 (optional) | `python scripts/validate_cmor_output.py --model M --realm R --experiment E --frequency F --root-output-path OUT [--plot-timeseries] [--plot-maps] [--html]` | Output tree and logs | `OUT/validation_reports/<subset>/`, `index.html` |

Driver defaults: `--model cesm`, `--realm atmos`, `--frequency mon`,
`--resolution ne30`, `--experiment piControl`, `--outdir .`, `--workers 1`, and
`r1i1p1f1` for `--realization-initialization-physics-forcing`.

## Driver: `cmor_driver.main`

1. Parse args and set log levels. `--log-level` applies to the whole `cmip7_prep` logger tree.
2. CESM only, realm ocean or seaIce: `mom6_static.ocean_fx_fields(--ocn-static-file)`.
3. Resolve `TSDIR` from `--tsdir`. CESM can instead derive it from `--caseroot --cimeroot` as `DOUT_S_ROOT/<comp>/proc/tseries/<freq>`.
4. Load the mapping: `--custom-yaml`, or `REALM_YAML_MAP[model][realm]` from `data/`. Set `mapping.default_freq = --frequency`.
5. Load the Data Request (network or cache).
   - Default: `DR.find_variables(cmip7_frequency, modelling_realm, experiment)`. A `30S-90S` regional copy is dropped when a non-regional copy exists.
   - `--run-all-from-yaml`: variables come from the YAML, in YAML order. Names that are not in the DR are synthesized (GAP-6).
   - `--cmip-vars`: filters the list above by branded name. It never adds names.
6. Resolve the tables root. Pass `--tables-root` (BUG-1, BUG-2).
7. Glob `TSDIR/*<pattern>*.nc` over every sampling pattern for (model, realm, frequency).
8. For each variable:
   1. `_collect_required_model_vars` gives the native names. None found means "no mapping in YAML".
   2. `patterns_for_variable` narrows the files to the variable's sampling. The sampling is the branded-name prefix: `tavg` or `tpt`. Other prefixes fall back to `tavg` with a warning.
   3. Keep files whose name contains `.<VAR>.` for a required native var.
   4. Call `process_one_var`. This is serial unless `--workers > 1`, and even then there is no concurrency (GAP-7).
9. The status list is logged at DEBUG only (GAP-8).

## `process_one_var` dispatch

`cfg = mapping.get_cfg(branded_name)`. A missing `grids` key is an error. For
each label in `cfg["grids"]`:

1. `open_native_for_cmip_vars`:
   - Checks that each time-varying source has a file. `STATIC_MODEL_VARS` are exempt.
   - Opens the files with `open_mfdataset(concat_dim="time")` and merges per-variable datasets.
   - Divides `lev` and `ilev` by 1000 (Q-1).
   - Ocean, seaIce and landIce open with `decode_timedelta=False`.
2. Choose the branch:

| Grid label | Realm or config | Helper | What happens |
|---|---|---|---|
| `gn` or `gm` | seaIce | `_prepare_seaice_native` | `mapping.realize_all`, one item per variant (NH/SH). Copies TLAT, TLON, latt_bounds and lont_bounds for (nj, ni) data. |
| `gn` or `gm` | any other | `_prepare_native` | `mapping.realize`. Copies `time_bounds`, plus x0/y0/x1/y1 for CISM. |
| `gr` | `levels.name == plev39` | `_prepare_regridded` | Realize, then regrid the variable and PS to lat/lon (conservative; `regrid_method` is ignored here). Then `zonal_mean_on_pressure_grid`: `to_plev` at each column, then mean over lon. |
| `gr` | anything else | `realize_regrid_prepare` | See the next section. For CESM ocean, `ocn_fx_fields` are merged in. |

3. For each prepared item, open a new `CmorSession`. Set the dataset
   attributes: frequency, r/i/p/f indices, `region` (the variant's `region`,
   default `glb`), and every key in the experiment's CV entry (lists become
   their first element). Build a `vdef`:
   - `name` = the physical parameter
   - `table` = YAML `table`
   - `units` = YAML `units`
   - `positive` = the override from `data/<model>_positive.yaml` (otherwise the table's value is used)
   - `levels`
   - `branded_variable_name`

   Then call `cm.write_variable`.

## `pipeline.realize_regrid_prepare` (the `gr` default path)

1. `mapping.realize` (formula or rename). Chunk time to 12. Attach `landfrac`, `area`, `landmask`, `wet` and `TLAT` if they are present.
2. Hybrid levels (`standard_hybrid_sigma[_half]`, `alev*`): carry PS and the hybrid coefficients.
3. `_apply_vertical_if_needed`: when `levels.name` contains `plev`, run `vertical.to_plev` on the **native** grid, before regridding. Target levels come from `CMIP7_coordinate.json` `axis_entry[<name>].requested`.
4. Rename `levgrnd` or `levsoi` to `sdepth`.
5. `regrid_to_latlon_ds(method=cfg.regrid_method)`, then merge the hybrid coefficients back.

## `regrid.regrid_to_latlon`: dispatch by input dims

| Input dims | Treated as | Pre-step | Post-step |
|---|---|---|---|
| `--resolution regular` | already lat/lon | none; must already have `lat` and `lon` | returned unchanged |
| `ncol` | CAM spectral element | reshape to (lat=1, lon=ncol) | none |
| `lndgrid` | CLM | multiply by `landfrac` | divide by regridded `sftlf/100` |
| `xh`/`yh` or `ni`/`nj` | MOM6 or CICE | multiply by `sftof/100` if `sftof`, `ocnfrac` or `wet` is present | none (Q-7) |

- The weights come from `get_map_paths(model, resolution)[method]`. The method is `conservative` unless the YAML says `bilinear`.
- Output is cast back to `float32`.
- `regrid_to_latlon_ds` also attaches time and its bounds, and fx fields (`sftlf`, `sftof`) regridded conservatively once per map (`FXCache`).
- `areacella` is computed from the destination bounds in the weight file.

## `CmorSession.write_variable`

1. Load the table from `vdef.table`: `tables/CMIP7_<table>.json` (fallbacks in `resolve_table_filename`).
2. `ensure_fx_written_and_cached` writes any fx fields present, such as `sftlf_ti-u-hxy-u` and `sftof_ti-u-hxy-u`.
3. Transpose so time comes first.
4. `_define_axes`:
   - Horizontal, first match wins:

     | Dims | Grid definition |
     |---|---|
     | `xh`/`xq` + `yh`/`yq` | MOM6: 1-D lat/lon from `data/ocean_geometry.nc` (BUG-13) |
     | `nj` + `ni` | CICE `cmor.grid` |
     | `x1`/`y1` or `x0`/`y0` | CISM `cmor.grid` (needs `--ice-sheet` and pyproj) |
     | `lat` + `lon` | latitude/longitude axes |
     | `lat` only | zonal |

   - Time: the axis name (`time`, `time1`, ...) is read from the table entry's `dimensions`. Bounds are synthesized if missing. Units that are not days, hours, minutes or seconds are re-expressed as `days since`.
   - Vertical, first match wins:

     | Match | CMOR axis |
     |---|---|
     | `standard_hybrid_sigma_half` / `alevhalf` | `a_half`, `b_half` |
     | `standard_hybrid_sigma` / `alev` / any `lev` dim | `a`, `b`, `p0`, `ps` |
     | `plev` dim | `plevNN` from `levels.name`; hPa converted to Pa |
     | `sdepth` | bounds from `data/depth_bnds.nc` |
     | `z_l` | `depth_coord` |
     | `zl` | `olevel` |
     | `fates_levpft` | `pft` |
     | `fates_levfuel` | `fuelclass` |

     Hybrid sigma is allowed only for the atmos, atmosChem and aerosol tables.
5. `cmor.variable(branded_name, units, axes, positive=override or table value)`.
6. `grid_label` and `grid` come from the dims (BUG-6).
7. Write in time slabs (dask chunk size, or 512 MB), then write the `ps` z-factor, then close.

The output path follows `output_path_template` in the dataset JSON.
