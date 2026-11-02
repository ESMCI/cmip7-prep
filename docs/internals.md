# Internals

This page is for changing `cmip7-prep` rather than running it.

## The mapping file format

A mapping file is keyed by branded variable name. Each entry names the model
fields the variable is built from and what has to be done to them.

```yaml
# A rename with a unit scaling.
pr_tavg-u-hxy-u:
  table: atmos
  units: kg m-2 s-1
  sources:
    - model_var: PRECT
      scale: 1000.0   # m/s -> kg m-2 s-1

# A formula over one or more fields.
clt_tavg-u-hxy-u:
  table: atmos
  units: "%"
  formula: CLDTOT * 100
  sources:
    - model_var: CLDTOT

# Interpolated onto pressure levels.
ta_tavg-p19-hxy-air:
  table: atmos
  units: K
  dims: [time, plev, lat, lon]
  levels:
    name: plev19
    units: Pa
  sources:
    - model_var: T

# Left on the model's hybrid-sigma levels.
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

`data/schemas/mapping.schema.yaml` is the schema these files are validated
against.

## Maintaining the mapping

The mappings are maintained in Google Sheets that the scientists fill in, and
converted to yaml. Columns A to F and L to S come from the CMIP7 table
metadata. Columns G to K are the ones to fill in:

| column | name | what it holds |
|---|---|---|
| G | `CESM Variable Name` | the model fields needed, comma separated, as in `PRECC, PRECL` |
| H | `Formula` | how to compute the CMIP7 variable from them, as in `(PRECC + PRECL) * 1000.0`. Leave blank for a rename or a scaling. |
| I | `Scale` | a multiplier per input field, positionally aligned with column G |
| J | `Freq` | the frequency of each input field, as in `day` |
| K | `Alias` | a rename for each input field, positionally aligned with column G |

The CESM spreadsheet is
[here](https://docs.google.com/spreadsheets/d/1BJV6CLgCTUpuaUlEQoFc-7ATBCsoJezYkyCvU1NTlxw/edit?usp=sharing).

To regenerate a mapping file, fill in the columns, export the sheet with
**File → Download → Comma Separated Values (.csv)**, save it under `data/`,
and run:

```bash
python scripts/convert_csv_to_yaml.py --model cesm \
    --input data/cesm_data.csv \
    --output data/cesm_to_cmip7_atmos.yaml
```

For NorESM a variable is kept only when its model field column is filled in,
and every field it depends on has to be in the model output.

## The modules

| module | what it does |
|---|---|
| `mapping_compat` | reads a mapping file and evaluates an entry |
| `pipeline` | finds the input files, opens them, and dispatches the vertical transform |
| `regrid` | regrids to a latitude-longitude grid using the weight files |
| `vertical` | interpolates hybrid-sigma levels onto pressure levels |
| `cmor_writer` | writes the output through CMOR, as `CmorSession` |
| `cmor_utils` | fill values, time encoding, bounds and monotonicity |
| `cache_tools` | caches regridders and fx fields across variables |
| `mom6_static` | reads the MOM6 static grid for the ocean fx fields |
| `include_patterns` | resolves a realm and frequency to its history time slice files |
| `time_windows` | turns `--years` into windows and takes one out of a dataset |
| `output_files` | finds output files whose period another file already covers |
| `worker_lifetime` | asks the kernel to kill a pool worker when the run dies |
| `step_exit` | the exit codes the scripts and the driver agree on |

## The files in `data/`

| file | what it holds |
|---|---|
| `<model>_to_cmip7_<realm>.yaml` | the variable mapping, one per model and realm |
| `<model>_include_patterns.yaml` | the history time slice files per realm, frequency and sampling |
| `<model>_grids.yaml` | the regrid weight files and the grid label per realm |
| `intensive_vars.yaml` | the variables regridded bilinearly rather than conservatively |
| `CMIP7_data_request_*.csv` | the data request, as exported |
| `depth_bnds.nc` | soil level depth bounds, for the `sdepth` axis |
| `ocean_geometry.nc` | the MOM6 ocean grid geometry |
| `schemas/mapping.schema.yaml` | the schema a mapping file is validated against |

## Running the tests

```bash
pytest
```

Doctests in every source module run as part of that, through
`--doctest-modules` in `pytest.ini`.
