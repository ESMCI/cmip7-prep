# cmip7-prep

A Python library and driver script for preparing CESM and NorESM native model output for submission to CMIP7 via CMOR (Climate Model Output Rewriter).

## What it does

`cmip7-prep` automates the pipeline from raw model timeseries to CMOR-compliant NetCDF:

1. **Variable mapping** — Reads a YAML mapping file (`cesm_to_cmip7.yaml` or `noresm_to_cmip7.yaml`) that describes how native model variables (e.g., `TREFHT`) map to CMIP names (e.g., `tas`), including unit conversions and multi-variable formulas.
2. **File discovery** — Selects only the timeseries files needed for the requested CMIP variables.
3. **Realization** — Evaluates the mapping (direct rename, scaling, or formula) to produce CMIP DataArrays.
4. **Vertical interpolation** — Optionally interpolates hybrid-sigma level variables to standard CMIP pressure grids (e.g., `plev19`, `plev39`) using geocat-comp.
5. **Regridding** — Regrids from native spectral element (SE) or tripolar ocean grids to 1° lat/lon using precomputed ESMF weight files via xESMF.
6. **CMORization** — Writes CMOR-compliant output with correct metadata, bounds, and fill values using the CMOR library.

## Supported models / grids

| Model  | Atmosphere grid | Ocean grid |
|--------|-----------------|------------|
| CESM   | ne30pg3 (SE)    | tx2_3v2 (tripolar) |
| NorESM | ne30pg3 / ne16pg3 (SE) | — |

## Installation

### Prerequisites

A conda environment with the required dependencies:

```bash
conda create -n cmip7-prep python=3.13 \
    xarray numpy dask xesmf cmor cftime pyyaml geocat-comp
conda activate cmip7-prep
```

### Install the package

```bash
pip install -e .
```

### System-specific setup (Derecho / NIRD)

**Derecho (CESM):**
```bash
module load conda
conda activate /glade/work/jedwards/conda-envs/CMORDEV
pip install -e .
```

**NIRD (NorESM):**
```bash
conda activate /projects/NS9560K/diagnostics/cmordev_env/
pip install -e .
```

## Quickstart

Make sure you have generated timeseries files for the run before starting.

**Derecho:**
```bash
qcmd -- bash scripts/fullamon.sh
```

**General usage via `cmor_driver.py`:**
```bash
# Atmosphere variables
python scripts/cmor_driver.py --realm atmos --tsdir /path/to/timeseries/

# Land variables
python scripts/cmor_driver.py --realm land --tsdir /path/to/timeseries/
```

## Variable mapping files

The mapping YAML files live in `data/`. Each entry describes how a native model variable maps to a CMIP variable:

```yaml
# Simple rename with unit conversion
tas:
  table: Amon
  units: K
  source: TREFHT

# Formula combining multiple variables
pr:
  table: Amon
  units: kg m-2 s-1
  raw_variables: [PRECC, PRECL]
  formula: "PRECC + PRECL"

# Pressure-level variable
ta:
  table: Amon
  units: K
  source: T
  levels:
    name: plev19
```

Both CESM (`cesm_to_cmip7.yaml`) and NorESM (`noresm_to_cmip7.yaml`) mappings are included.

## Key modules

| Module | Purpose |
|--------|---------|
| `cmip7_prep.mapping_compat` | Load and evaluate YAML mapping files; `Mapping`, `VarConfig` |
| `cmip7_prep.pipeline` | File discovery, dataset opening, vertical transform dispatch |
| `cmip7_prep.regrid` | Regrid to 1° lat/lon via precomputed ESMF weight files |
| `cmip7_prep.vertical` | Hybrid-sigma → pressure-level interpolation (geocat-comp) |
| `cmip7_prep.cmor_writer` | Write CMOR-compliant output (`CmorSession`) |
| `cmip7_prep.cmor_utils` | Fill values, time encoding, bounds, monotonicity utilities |
| `cmip7_prep.cache_tools` | Regridder and FX field caching (`RegridderCache`, `FXCache`) |
| `cmip7_prep.mom6_static` | Read MOM6 static grid for ocean FX fields |

## Running tests

```bash
pytest
```

Doctests in all source modules are run automatically via `--doctest-modules` (configured in `pytest.ini`).

## Data files

The `data/` directory contains:

| File | Description |
|------|-------------|
| `cesm_to_cmip7.yaml` | CESM → CMIP7 variable mapping |
| `noresm_to_cmip7.yaml` | NorESM → CMIP7 variable mapping |
| `cmor_dataset.json` | Default CMOR dataset attributes |
| `piControl.json` | CMOR experiment metadata for piControl |
| `depth_bnds.nc` | Ocean depth bounds for olevel axis |
| `ocean_geometry.nc` | MOM6 ocean grid geometry |
