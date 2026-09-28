# Software stack and environment

Verified against commit `b90eb2a`. Issue IDs point to `issues.md`.

## Runtime stack

| Layer | Package | Version source | Used by |
|---|---|---|---|
| Language | Python 3.12 | `Dockerfile`, `INSTALL.md` (`pyproject.toml` allows >=3.10.13,<3.14) | all |
| CMOR writer | `cmor` 3.15 (conda-forge only) | `Dockerfile` | `cmor_writer.py`, `cmor_utils.py`, `scripts/cmor_driver.py` |
| Regridding | `xesmf` + `esmpy` (ESMF) | `Dockerfile` | `cache_tools.py`, `regrid.py` |
| Vertical interpolation | `geocat-comp` (`interp_hybrid_to_pressure`) | `Dockerfile` | `vertical.py` |
| Arrays | `xarray`, `numpy`, `dask`, `netcdf4`, `h5netcdf`, `cftime` | `Dockerfile` | all |
| Config | `pyyaml` | `Dockerfile` | mapping and data tables |
| Data Request | `cmip7-data-request-api` (pip, imported as `data_request_api`) | `Dockerfile` | `cmor_driver.py`, `query_missing_vars.py`, `validate_cmor_output.py` |
| Time series | `gents` (pip; `HFCollection`, `TSCollection`) | `Dockerfile` | `scripts/gen_timeseries.py` |
| Projection | `pyproj` (not listed; in the image indirectly, ENV-4) | `environment.yml` only | `cism_grid.py` (landIce `gn`) |
| Plots | `matplotlib` (optional) | `Dockerfile` | `validate_cmor_output.py` |
| Tests | `pytest`, `pytest-cov` | `Dockerfile` | `tests/` |
| Lint | `black`, `pylint` via pre-commit | `.pre-commit-config.yaml` | CI only |

Packaging uses Poetry metadata (`pyproject.toml`), but dependencies come from
conda. Always install the package with `--no-deps`.

## External inputs (not in git)

| Input | Where it comes from | Needed for |
|---|---|---|
| CMOR tables and CVs | The `cmip7-cmor-tables` fork. Layout: `tables/CMIP7_<table>.json` and `tables-cvs/cmor-cvs.json`. There are two incompatible forks (ENV-2). | every write |
| ESMF weight files | Absolute HPC paths in `data/*_regrid_maps.yaml` (GAP-3) | every `gr` variable |
| CMIP7 Data Request | Downloaded at runtime by `data_request_api` (`dt.get_transformed_content()`). Needs network or a cache. | variable selection |
| Model history files | CESM or NorESM run archive | `gen_timeseries.py` |

Building the image needs network access to `conda.anaconda.org`, PyPI and github.com (the tables are cloned during the build).
| MOM6 static file | `--ocn-static-file` | CESM ocean `deptho` and `areacello` |

Table keys available in both forks: `aerosol`, `atmos`, `atmosChem`,
`cell_measures`, `coordinate`, `formula_terms`, `grids`, `land`, `landIce`,
`long_name_overrides`, `ocean`, `ocnBgchem`, `seaIce`. There is no `fx` table
(BUG-8).

## Environments

### 1. Docker (primary)

```bash
docker build -t cmip7-prep .      # CMIP7/ is still in the context (ENV-1)
```

- The image clones the tables into `/opt/cmip7-prep/cmip7-cmor-tables`. The default is cesm-dev.
- Override the tables with `--build-arg TABLES_REPO=... --build-arg TABLES_REF=...`.
- For NorESM, use the noresm-dev tables (ENV-2).
- Always set `PYTHONPATH=/opt/cmip7-prep/src` (BUG-12).
- Never mount the whole repo over `/opt/cmip7-prep`. That hides the image's tables.
  Mount subdirectories instead.

Run the driver:

```bash
docker run --rm -e PYTHONPATH=/opt/cmip7-prep/src \
  -v /path/ts:/data -v /path/out:/out cmip7-prep \
  python scripts/cmor_driver.py --model noresm --realm atmos --resolution ne16 \
    --tsdir /data --outdir /out --tables-root /opt/cmip7-prep/cmip7-cmor-tables
```

The weight files must be visible at the paths given in
`data/*_regrid_maps.yaml`. Mount them there, or use a custom maps YAML.
There is no CLI flag for the weights path.

HPC: `apptainer build cmip7-prep.sif docker-daemon://cmip7-prep:latest`.

### 2. Conda (secondary; HPC sites)

Follow `INSTALL.md`:
1. Create a conda env with `cmor=3.15` and the science stack.
2. `pip install -e . --no-deps`.
3. `pip install gents dulwich cmip7-data-request-api`.

Site envs:
- Derecho (CESM): `/glade/work/jedwards/conda-envs/CMORDEV`
- NIRD (NorESM): `/projects/NS9560K/diagnostics/cmordev_env_312`, loaded by `scripts/setup_nird.sh`

An editable install keeps the `data/` lookups working.

`environment.yml` is an old NIRD env export (Python 3.13, CMOR 3.14.0). Do not
use it as a reference (ENV-3).

### 3. CI (`.github/workflows/`)

| Workflow | What it does |
|---|---|
| `pytest.yaml` | Ubuntu, Python 3.12. Builds ESMF and CMOR 3.14.0 from source, installs `requirements.txt`, checks out the cesm-dev tables, runs `pytest --doctest-modules`. |
| `pylint.yml` (job "pre-commit") | Python 3.13 + Poetry. Runs `pre-commit run --all-files` (black, pylint, whitespace hooks). |

Both run on push and PR to `main`.
