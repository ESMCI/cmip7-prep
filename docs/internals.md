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

### Mapping a dimension into vegetation types

Formulas can group a source dimension into a CMIP categorical dimension:

```text
map_multi_dimension(FATES_GPP_PF, fates_levpft)
```

The dimension argument can also be quoted, as `"fates_levpft"`. It is a
dimension name, not an additional source variable. The helper looks it up in
[`dimension_mappings.yaml`](../data/dimension_mappings.yaml), replaces it with
`vegtype`, and sums these 1-based source positions in CMIP coordinate order:

| `vegtype` label | FATES PFTs |
|---|---|
| `broadleaf_deciduous_trees` | 5, 6 |
| `broadleaf_evergreen_trees` | 1, 4 |
| `needleleaf_deciduous_trees` | 3 |
| `needleleaf_evergreen_tree` | 2 |
| `natural_grasses` | 12, 13, 14 |
| `crops` | 15 |
| `shrubs` | 7, 8, 9, 10, 11 |

The input must have 15 PFT entries. Indices are positions even if the input
coordinate labels start at zero. Unknown mappings, missing dimensions,
incorrect source sizes and existing target coordinates raise errors.
Missing data propagates within its group instead of being counted as zero.
Other dimensions, coordinates and variable attributes are preserved, and
Dask-backed input stays lazy. CMOR writes the resulting string labels in the
auxiliary coordinate `sector`, indexed by the output dimension `type`.

Land-use categories can be mapped with
`map_multi_dimension(FIELD, fates_levlanduse)`. The input must have five
entries in FATES order: primary, secondary, rangeland, pasture, crops.
The output dimension is `landuse`, in CMIP requested order:

| `landuse` label | FATES land-use positions |
|---|---|
| `primary_and_secondary_land` | 1, 2 |
| `pastures` | 3, 4 |
| `crops` | 5 |
| `urban` | none (missing values) |

Empty groups in the dimension mapping produce missing values, not zeros.
CMOR writes the land-use labels in the auxiliary coordinate `sector`,
indexed by the output dimension `landuse`.

The helper only groups and sums; it does not apply unit conversions or
`FATES_FRACTION`. Include any scientifically required scaling explicitly in
the surrounding formula.

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
| `dimension_mappings.yaml` | source positions grouped into categorical output dimensions |
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
