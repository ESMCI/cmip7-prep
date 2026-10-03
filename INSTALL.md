# Installing cmip7-prep (with cmor 3.15)

This project depends on **CMOR 3.15** and the matching **CMIP7 CMOR tables**.
CMOR 3.15 is the version that validates against the CMIP7 controlled vocabulary
(license IDs, `gNNN` grid labels, etc.), so an older CMOR will fail at write time.

## Requirements

- **Python 3.12.** CMOR 3.15 has no conda build for Python 3.13, and the project
  requires Python < 3.14, so 3.12 is the only workable version.
- The science stack comes from **conda-forge**; a few extras come from **pip**.

## Steps

### 1. Create the conda environment (science stack + cmor 3.15)

```bash
conda create --prefix /projects/NS9560K/diagnostics/cmordev_env_312 -c conda-forge \
  python=3.12 cmor=3.15 xarray numpy dask xesmf cftime pyyaml click pandas geocat-comp
```

When conda prints its plan, confirm the `cmor` line shows `3.15.x` with a `py312`
build string before proceeding.

- `-c conda-forge` — cmor 3.15 and the science packages live on conda-forge.
- `geocat-comp` is conda-only (used by the vertical/regrid code).

### 2. Activate the environment

```bash
conda activate /projects/NS9560K/diagnostics/cmordev_env_312
```

### 3. Install the cmip7-prep project (editable)

```bash
cd /path/to/cmip7-prep
pip install -e . --no-deps
```

`--no-deps` is intentional: the science packages are already installed by conda,
and this stops pip from changing them.

### 4. Install the pip-only packages

```bash
pip install gents dulwich cmip7-data-request-api
```

These three are not on conda-forge, so they must come from pip.

Note that `pip install <pkg>` does nothing when the package is already present.
To move to a newer release, ask for it explicitly:

```bash
pip install -U gents dulwich cmip7-data-request-api
```

## Local development environment (macOS)

The steps above create the shared analysis environment on NIRD.  For running the
test suite and editing the code on a laptop, create a private environment
instead.  Everything the tests need, including CMOR and ESMF, has an
`osx-arm64` build on conda-forge, so no part of the stack has to be skipped.

```bash
conda create -n cmip7-dev -c conda-forge python=3.12 cmor=3.15 \
  numpy xarray netcdf4 cftime pyyaml pytest pytest-cov \
  xesmf=0.8.7 esmpy=8.9.0 geocat-comp=2025.10.01
```

```bash
conda activate cmip7-dev
pip install -e . --no-deps
pip install gents
```

```bash
pytest -q
```

Use **Miniforge** rather than Anaconda for this (see the BLAS entry under
Troubleshooting).  `dulwich` and `cmip7-data-request-api` are only needed for
the data-request and CV tooling, so a test-only environment can leave them out.

## CMIP7 CMOR tables

This recipe only provides the CMOR **library** (3.15). The matching **tables**
live in the `cmip7-cmor-tables/` directory and must be the version that uses the
CMIP7 controlled vocabulary (e.g. `CC-BY-4.0` license IDs and `gNNN` grid
labels). Keep that checkout up to date, or CMOR will reject otherwise-valid runs.

## Troubleshooting

- **`gents` resolves from `~/.local/...` instead of the env:** a personal
  user-folder copy is shadowing the env. Clear it, then reinstall into the env:
  ```bash
  pip uninstall gents      # removes the ~/.local copy
  pip install gents        # reinstalls into the active env
  ```
- **`license_id "..." could not be found` / `grid_label "..." is invalid`:**
  the `cmip7-cmor-tables` checkout is out of date. Update it to the version
  matching CMOR 3.15 / the CMIP7 controlled vocabulary.
- **`zero-size array to reduction operation minimum which has no identity`,
  logged with a traceback while pulling metadata:** one or more history files
  hold no time records, which happens when a run segment opens a history stream
  but ends before a sample is written.  GenTS takes a minimum over each file's
  time coordinate, so an empty file fails.  This is **not fatal** --
  `HFCollection.pull_metadata` defaults to `raise_errors=False`, logs the
  failure with `exc_info=True` and drops the file, so the run continues.
  `gen_timeseries.py` screens such files out up front so they appear as one
  warning each instead of a traceback.
- **`Error importing numpy: you should not try to import numpy from its source
  directory`:** misleading message; numpy prints it whenever its compiled
  extension fails to load for any reason.  Get the real cause with
  ```bash
  python -c "import traceback
  try: import numpy
  except ImportError as e: traceback.print_exception(e.__cause__)"
  ```
  A mixed-channel Anaconda install can leave `libcblas`/`libblas` symlinked to
  an OpenBLAS build that was never installed, in which case the cause names a
  missing `.dylib`.  A conda-forge-only base (Miniforge) avoids this; repairing
  a broken one needs `conda install --force-reinstall openblas blas`.
- **`environment.yml` disagrees with these instructions:** it is an export of
  the older `cmordev_env`, not the `cmordev_env_312` built above, so its pins
  (including `cmor` and `gents`) are not the versions this recipe installs.
  Treat the steps here as authoritative.
