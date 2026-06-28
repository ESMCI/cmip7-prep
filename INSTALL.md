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
