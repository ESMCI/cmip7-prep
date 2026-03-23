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

**General usage via `cmor_driver.py`:**
```bash
# Atmosphere variables
python scripts/cmor_driver.py --realm atmos --tsdir /path/to/timeseries/

# Land variables
python scripts/cmor_driver.py --realm land --tsdir /path/to/timeseries/
```

**Derecho:**
```bash
qcmd -- python scripts/cmor_driver.py --realm atmos --tsdir /path/to/timeseries/
```

## Variable mapping files

The mapping YAML files live in `data/`. Each entry describes how a native model variable maps to a CMIP variable.
Keys use the form `<cmip_name>_<frequency>-<level>-<grid>-<realm>`:

```yaml
# Simple source mapping with unit scaling
pr_tavg-u-hxy-u:
  table: atmos
  units: kg m-2 s-1
  sources:
    - model_var: PRECT
      scale: 1000.0   # m/s -> kg m-2 s-1

# Formula combining multiple variables
clt_tavg-u-hxy-u:
  table: atmos
  units: "%"
  formula: CLDTOT * 100
  sources:
    - model_var: CLDTOT

# Pressure-level variable
ta_tavg-p19-hxy-air:
  table: atmos
  units: K
  dims: [time, plev, lat, lon]
  levels:
    name: plev19
    units: Pa
  sources:
    - model_var: T

# Hybrid-sigma level variable
cl_tavg-al-hxy-u:
  table: atmos
  units: "%"
  formula: CLOUD * 100
  dims: [time, lev, lat, lon]
  levels:
    name: standard_hybrid_sigma
    src_axis_name: lev
  sources:
    - model_var: CLOUD
```

Both CESM (`cesm_to_cmip7.yaml`) and NorESM (`noresm_to_cmip7.yaml`) mappings are included.

## Maintaining the CESM variable mapping via Google Sheets

The CESM variable mapping is maintained in a Google Spreadsheet and stored in version control as `data/cesm_to_cmip7.yaml`.

**Spreadsheet:** <https://docs.google.com/spreadsheets/d/1BJV6CLgCTUpuaUlEQoFc-7ATBCsoJezYkyCvU1NTlxw/edit?usp=sharing>

### Column format

Columns A–F and L–S are populated from CMIP7 table metadata.  Columns G–K describe how each CMIP variable is generated from CESM model output and are the ones to fill in:

| Col | Column | Description |
|-----|--------|-------------|
| G | `CESM Variable Name` | The CESM variable(s) needed as input, comma-separated, e.g. `PRECC, PRECL` |
| H | `Formula` | Expression used to compute the CMIP variable from the CESM inputs, e.g. `(PRECC + PRECL) * 1000.0` — leave blank when the input variable needs only a rename or scaling |
| I | `Scale` | Multiplicative scale factor applied to each input variable, comma-separated and positionally aligned with column G, e.g. `1000.0` |
| J | `Freq` | Sampling frequency of each input variable, e.g. `day` for daily fields |
| K | `Alias` | Rename each input variable before use, comma-separated and positionally aligned with column G |

### Workflow

1. Open the spreadsheet and fill in columns G–K for any variables that are missing a `CESM Variable Name`.
2. Export as CSV: **File → Download → Comma Separated Values (.csv)**
3. Save the downloaded file as `data/cesm_data.csv`.
4. Regenerate the YAML:
   ```bash
   python scripts/convert_csv_to_yaml.py --model cesm \
       --input data/cesm_data.csv \
       --output data/cesm_to_cmip7.yaml
   ```

## Key modules

| Module | Purpose |
|--------|---------|
| `cmip7_prep.mapping_compat` | Load and evaluate YAML mapping files; `Mapping`, `VarConfig` |
| `cmip7_prep.pipeline` | File discovery, dataset opening, vertical transform dispatch |
| `cmip7_prep.regrid` | Regrid to 1° lat/lon for CESM/NorESM or 2° lat/lon for NorESM via precomputed ESMF weight files |
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
| `depth_bnds.nc` | Soil level depth bounds for sdepth axis |
| `ocean_geometry.nc` | MOM6 ocean grid geometry |
