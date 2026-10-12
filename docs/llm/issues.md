# Issue register

Verified against commit `b90eb2a`. This is the only place where issue details
live. Other docs cite issues by ID only.

ID prefixes:
- `BUG`: code does the wrong thing.
- `GAP`: a feature is missing.
- `DEBT`: dead, duplicated or misleading code.
- `ENV`: environment, build or CI problem.
- `DOC`: human docs are wrong.
- `Q`: an open question for the maintainers. Do not "fix" a Q item without sign-off.

Rules:
- IDs are permanent. Never renumber or reuse an ID.
- When an issue is fixed, set Status to `fixed <commit or date>`. Keep the row.
- A new issue takes the next free number in its prefix.
- `Conf` is how the issue was confirmed:
  - `code`: read in source.
  - `data`: checked by scanning `data/`.
  - `built`: reproduced by a build or run.
  - `inferred`: follows from the code but has not been run.

## BUG

| ID | Location | Problem | Workaround | Conf | Status |
|---|---|---|---|---|---|
| BUG-1 | `scripts/cmor_driver.py::main` | `TABLES_cesm` is commented out but still referenced. `--model cesm` without `--tables-root` raises `NameError`. | Always pass `--tables-root`. | code | open |
| BUG-2 | `scripts/cmor_driver.py::main` | `Path(tables_root).exists` is missing `()`, so it is always truthy. A bad `--tables-root` is not caught until CMOR fails. | Check the path yourself. | code | open |
| BUG-3 | `mapping_compat.py::Mapping.realize`, `::_to_varconfig` | `unit_conversion:` and per-source `scale:` are parsed but never applied. The `_to_varconfig` docstring wrongly says scale is promoted. `unused_functions._apply_unit_conversion` is never called. The converter also drops the spreadsheet "Scale" column. | Put the conversion in `formula`, e.g. `PRECT * 1000`. | code | open |
| BUG-4 | `data/cesm_to_cmip7_land.yaml` | 123 of 227 entries call functions that are not in `FORMULA_NAMESPACE`: `chunits`, `yeartomonth_data`, `yeartomonth_data3D`, `CLM_landunit_to_CMIP6_Lut`, `CLM_pft_to_CMIP6_vegtype`, `burntFraction`, `get_soilpools`, `reduce_lu`, `SUM`, `SSRE_FSRVD`. Each fails with "Error evaluating formula". In NorESM land, `depthsl_ti-u-hxy-u` calls `sum` (builtins are blocked) and `grassFracC3_tavg-u-hxy-u` calls `FATES_CROWNAREA_PF(...)` as a function. | Implement the functions in `mapping_compat.py` and add them to `FORMULA_NAMESPACE`, or fix the spreadsheet. | data | open |
| BUG-5 | `mapping_compat.py::_realize_core` | An entry with more than one source and no `formula` (after freq filtering) raises "Mapping ... is incomplete". Count: CESM atmos 4 (e.g. `ccb_tpt-u-hs-u`), CESM land 44, NorESM atmos 1 (`hur_tpt-al-hs-u`), NorESM land 1 (`cSoilPools_tavg-u-hxy-lnd`). | Add a formula in the spreadsheet. | data | open |
| BUG-6 | `cmor_writer.py::CmorSession.write_variable` | The `grid_label` comes from the data dims only. `lat`+`lon` gives `gr` and "1x1 degree". Anything else gives `gn` and "curvilinear". Results: zonal means (lat only) are labelled `gn`; NorESM ne16 2x2 output says "1x1 degree"; the YAML `grids` value is never passed to the writer. See Q-3. | none | code | open |
| BUG-7 | `scripts/convert_csv_to_yaml.py::MODEL_CONFIGS` | Output file names do not match `cmor_driver.REALM_YAML_MAP`: CESM seaIce is written as `cesm_to_cmip7_seaice.yaml` (the driver reads `..._seaIce.yaml`), and NorESM landice as `noresm_to_cmip7_landice.yaml` (the driver reads `..._landIce.yaml`). NorESM `realm_outputs` has no seaIce entry. | Rename the output files by hand after conversion. | code | open |
| BUG-8 | `data/cesm_to_cmip7_ocean.yaml` (`deptho_ti-u-hxy-sea`) | `table: fx` has no table file `CMIP7_fx.json`. `cmor_utils.resolve_table_filename` falls back to `fx.json` and CMOR cannot load it. The CMIP7 tables have no fx table. | Use the table the variable lives in (ocean). | data | open |
| BUG-9 | `scripts/gen_timeseries.py::main` | Without `--years-spec`, `apply_overwrite("*")` always runs, so existing time series are always overwritten and `--overwrite_timeseries` is ignored. With `--years-spec` the flag is honoured. | none | code | open |
| BUG-10 | `cmor_writer.py::CmorSession` | `dataset_attrs` and `tracking_prefix` are stored and never used. The driver's `{"institution_id": "NCC", "GLOBAL_IS_CMIP7": True}` has no effect. The institution comes from the dataset JSON, falling back to "NCAR". This is harmless today because each model's JSON has the right institution. | Edit the dataset JSON. | code | open |
| BUG-11 | `scripts/find_unmapped_vars.py` | The regexes match log lines the driver no longer writes, so the output is always empty. | Use `scripts/validate_cmor_output.py`. | code | open |
| BUG-12 | `Dockerfile`, all `Path(__file__).parent.parent.parent / "data"` lookups | `pip install --no-deps .` builds a wheel with no `data/` in it (checked by building the wheel). Image `PYTHONPATH=/opt/cmip7-prep` does not contain the package, so the installed copy is imported and the data lookup points outside site-packages. Mapping, include-pattern and regrid-map YAML files are then not found. Reproduced 2026-09-28: `FileNotFoundError: /opt/conda/lib/python3.12/data/noresm_to_cmip7_atmos.yaml`. With `PYTHONPATH=/opt/cmip7-prep/src` it loads. | Run with `-e PYTHONPATH=/opt/cmip7-prep/src`. | built | open |
| BUG-13 | `data/ocean_geometry.nc` | The file is 0 bytes; commit `d1c00a0` ("accidently removed data") emptied it. `_define_mom6_grid` (MOM6 `gn` output) and `_write_fx_2d` (`deptho` on xh/yh) open it and fail. | Restore the file from history or from a MOM6 run. | code | open |

## GAP

| ID | Area | Missing | Status |
|---|---|---|---|
| GAP-1 | Realm coverage | CESM has no YAML for aerosol, atmosChem, ocnBgchem or landIce. `cesm_to_cmip7_landice.yaml` is empty, is not in `REALM_YAML_MAP`, and CESM include patterns have no landIce entry. NorESM has no ocean or ocnBgchem (BLOM). `--realm ocnBgchem` works for neither model. Matrix: `features.md`. | open |
| GAP-2 | `gen_timeseries.py` | `--realm` accepts only atmos, land, seaIce and landIce. Ocean, aerosol, atmosChem and ocnBgchem patterns exist in YAML but cannot be selected. | open |
| GAP-3 | Weight files | `data/*_regrid_maps.yaml` point to absolute HPC paths (`/glade/...`, `/nird/...`). There is no public download, so `gr` output can only be produced where those paths exist or are mounted. | open |
| GAP-4 | Grid policy | The grid choice is hard-coded in `convert_csv_to_yaml.py::REALM_GRIDS` and `GRIDS_OVERRIDES`: seaIce, landIce and ocean are `gn`; all other realms are `gr`; 5 MOM6 fields are `gn`+`gr`; 4 sea-ice global means are `gm`. Changing it means editing Python and regenerating the YAML. | open |
| GAP-5 | Site output | Branded names with `-hs-` (site sampling) have no site extraction. They go down the `gr` regrid path. | open |
| GAP-6 | Non-CMIP7 output | `--run-all-from-yaml` synthesizes variables that are missing from the Data Request (`variable_selection.py`). CMOR still needs a `variable_entry` in the tables, and `get_experiment_info_from_tables` raises unless the experiment is in `tables-cvs/cmor-cvs.json`. The only route for custom tables is `--tables-root` pointing at an edited checkout. | open |
| GAP-7 | Parallelism | With `--workers N`, each variable is submitted and then waited on before the next one starts, so there is no concurrency. | open |
| GAP-8 | Run summary | The per-variable status list is logged only at DEBUG, and no machine-readable summary is written. | open |
| GAP-9 | Dataset metadata | The `parent_*` and `branch_time_*` values in `data/cmor_dataset*.json` are static (parent `piControl-spinup`, branch time 0). The driver overrides only the keys in the experiment's CV entry. `nominal_resolution` is "100 km" even for NorESM ne16 2x2 (Q-6). | open |
| GAP-10 | CLI | `cmor_driver.py` parses `--overwrite`, `--test` and `--debug` but never uses them. `--ocn-static-file` is used only for CESM ocean and seaIce. | open |
| GAP-11 | Unused helpers | These are not called by the pipeline: `vertical.remap_isopycnal_to_olevel` (untested, dims hard-coded), `mom6_static.load_mom6_grid`, `mom6_static.compute_cell_bounds_from_corners`, and `cism_grid.read_esmf_mesh`. | open |

## DEBT

| ID | Location | Problem |
|---|---|---|
| DEBT-1 | `cmor_writer.py::_define_axes` | A second `elif "zl" in var_dims` branch can never run. |
| DEBT-2 | 6 modules | `Path(__file__).parent.parent.parent / "data"` is repeated in `cmor_utils`, `mapping_compat`, `regrid_maps`, `include_patterns` and `cmor_writer` (twice). This only works from a source checkout (BUG-12). |
| DEBT-3 | `pipeline.py::realize_regrid_prepare` | When given file paths instead of a Dataset, it assigns the `(ds, vars)` tuple from `open_native_for_cmip_vars` to `ds_native`. The driver always passes a Dataset, so this path is dead. Step 9 (plev39 zonal mean) is also never reached, because the driver handles plev39 in `_prepare_regridded`. |
| DEBT-4 | several | Stale comments: the `regrid.py` comment "Variables treated as intensive"; the `noresm_regrid_maps.yaml` header says `regrid.py` reads `intensive_vars.yaml` (only the converter does); the `VarConfig.grids` docstring says the default is `["gr"]` (the driver errors instead); the `Mapping.realize` warning says "source variable ... not found in dataset" when the real cause is an unmapped CMIP name; the `cmor_driver.py` module docstring says atm/lnd only. |
| DEBT-5 | `mapping_compat.py::_realize_core` | The `sftlf` and `areacella` special cases compare against short names, but mapping keys are branded (`sftlf_ti-u-hxy-u`), so they never match. |
| DEBT-6 | `data/CMIP7_data_request_v1.0beta-Variables_v1.2.2.4.csv` | 4.6 MB and not read by any code. |
| DEBT-7 | `open_native_for_cmip_vars` | The annotation says it returns `xr.Dataset`. It actually returns `(ds, list)` or `(None, None)`. |

## ENV

| ID | Problem | Workaround |
|---|---|---|
| ENV-1 | `docker build .` sends the whole working tree as build context, including git-ignored data directories. Anything large inside the repo that is not listed in the root `.dockerignore` can fill the Docker disk ("no space left on device"; conf: built). `CMIP7/` (the default `CmorSession` output directory) is not excluded. A `.dockerignore` inside a subdirectory has no effect: Docker reads only the root one. | List the directory in the root `.dockerignore`, or build from tracked files only: `git archive HEAD \| docker build -t cmip7-prep -`. |
| ENV-2 | There are two CMOR table forks with incompatible CVs (conf: checked both on 2026-09-28). `CESM-Development/cmip7-cmor-tables@cesm-dev` has license `CC-BY-4-0` and grid labels `g99 gn gr`. `NorESMhub/cmip7-cmor-tables@noresm-dev` has license `CC-BY-4.0` and grid labels `g100..g185 g999 gn gr`. `data/cmor_dataset.json` matches cesm-dev and `data/cmor_dataset_noresm.json` matches noresm-dev. The Docker and CI default is cesm-dev; the git submodule (`.gitmodules`) is noresm-dev. A NorESM run against cesm-dev tables fails on `license_id`. | For NorESM, build with `--build-arg TABLES_REPO=https://github.com/NorESMhub/cmip7-cmor-tables.git --build-arg TABLES_REF=noresm-dev`, or pass `--tables-root`. See Q-5. |
| ENV-3 | CMOR versions disagree. `INSTALL.md` and `Dockerfile` use CMOR 3.15 with Python 3.12. CI (`.github/workflows/pytest.yaml`) builds CMOR 3.14.0 from source. `environment.yml` pins CMOR 3.14.0 with Python 3.13. | Trust Docker (3.15). |
| ENV-4 | `pyproj` (used by `cism_grid.py` for landIce `gn`) is not listed in `Dockerfile` or `requirements.txt`. It is in `environment.yml`. The image built on 2026-09-28 has pyproj 3.8.0, pulled in by another package (metpy 1.7.1 is also present). A dependency change could drop it. | Add `pyproj` to the `Dockerfile` explicitly (needs sign-off). |
| ENV-5 | The `cmip7-cmor-tables/` submodule directory is empty in a fresh clone. The tests and `cmor_driver.py` default to `<repo>/cmip7-cmor-tables`. | Run `git submodule update --init`, or use the tables baked into the image. |
| ENV-6 | Withdrawn: it described one sandbox's network policy, not the repo. The image build needs network access to `conda.anaconda.org`, PyPI and github.com. | n/a |
| ENV-7 | The lint workflow (`pylint.yml`, job name "pre-commit") uses Python 3.13 and Poetry with no lock file. `.pylintrc` ignores `scripts/`. | none |

## DOC

| ID | File | Stale content |
|---|---|---|
| DOC-1 | `README.md` | Describes a single `cesm_to_cmip7.yaml` or `noresm_to_cmip7.yaml` (the mappings are now split per realm). Shows `convert_csv_to_yaml.py --output` (the flag does not exist; files are written per realm to the current directory). Shows `scale:` and `dims:` in the YAML examples (ignored or dropped). Lists `data/piControl.json` (absent). Implies `ocean_geometry.nc` is usable (BUG-13). The Docker example omits `--model` and `--tables-root`. |
| DOC-2 | `.github/copilot-instructions.md` | Stale: mentions Poetry workflows, a `prepare` CLI, and a `Tables/` directory. |
| DOC-3 | `pyproject.toml` | The `[tool.poetry.scripts]` entry `cmip7_prep.cli:app` points at a module that does not exist. The description says "SE->1deg" only. |

## Q (open questions; do not change behaviour without an answer)

| ID | Question | Context |
|---|---|---|
| Q-1 | Is `lev/1000` and `ilev/1000` in `pipeline.open_native_for_cmip_vars` intended for every dataset? | CAM `lev` is `(a+b)*P0` in hPa, so dividing by 1000 gives sigma if P0 = 1000 hPa. It runs on any dataset that has a `lev` variable. The CMOR sigma axis uses `hyam+hybm` first (`cmor_utils.sigma_mid_and_bounds`), so the effect on published axes may be small. |
| Q-2 | Are the sign overrides in `data/cesm_positive.yaml` and `data/noresm_positive.yaml` right? | They set `rsds`, `rlds`, `rsdt`, `rsdscs` and `rldscs` to `up` where the tables say `down`. CMOR negates the data when the declared `positive` differs from the table. If CAM's FSDS, FLDS and SOLIN are positive-down, the published values are negated. `tauu`/`tauv` set to `up` is plausible, since CAM's TAUX is the stress on the atmosphere. |
| Q-3 | Which `grid_label` should global-mean (`hm`) and zonal-mean (`hy`) output get? | The cesm-dev CV allows only `gn`, `gr` and `g99`. The YAML's `gm` is internal only. See BUG-6. |
| Q-4 | Is `data/noresm_to_cmip7_seaIce.yaml` meant to be a byte-identical copy of the CESM file? | Both models use CICE6. The NorESM converter writes no seaIce file (BUG-7). |
| Q-5 | Which table fork is canonical for Docker and CI? | See ENV-2. |
| Q-6 | Should `nominal_resolution` be "100 km" for NorESM ne16 → 2x2 output? | It is set in `cmor_dataset_noresm.json`. |
| Q-7 | Should ocean and sea-ice regridding also denormalize? | `regrid.regrid_to_latlon` multiplies by `sftof/100` when `sftof`, `ocnfrac` or `wet` is present, but never divides by the destination fraction (land fields do divide). Coastal cells may be diluted. |
