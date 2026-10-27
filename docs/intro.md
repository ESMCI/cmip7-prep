# Introduction

`cmip7-prep` processes one archived model case into CMOR-compliant output in
the following three stages.

**Time series generation.** The output from the model is time-slice
files - multiple variables at one time in a file. Time series generation
involves creating files that each contain one variable over a given
period of time. As an example, for a 100 year run you would want to have
the U field from the atmosphere be contained in 10 year segments - so 10
time series files for U. The script `gen_timeseries.py` reads the case's
history files and rewrites them, using
[GenTS](https://github.com/NCAR/gents). Time series files are a
pre-requisite for creating CMORized output.

**CMORization.** `cmor_driver.py` reads the time series files and writes CMIP7
output. It applies the realm's variable mapping, regrids and interpolates where
the data request asks for it, and adds the metadata, bounds and directory
structure CMIP7 requires. It writes a variable only when the data request asks
for that variable for this experiment.

**Validation.** `validate_cmor_output.py` reads the CMORized output
back and reports the variables produced, the variables the request
asked for and did not get and the variables whose CMOR log holds an
error. For each variable produced it records the dimensions and sizes,
which is how a variable written with the wrong shape shows up. It also
optionally plots a mean time series and a time-mean map per variable.
Finally it builds an HTML index summarizing every realm and all
frequencies for that realm that have been validated.

The script [`run_reference_case.py`](tutorial.md#the-arguments) runs all three
stages over every realm and frequency in
`data/<model>_include_patterns.yaml`, unless you specify the subset(s) you
want with `--realms` and `--frequencies`. Every argument it takes is described
in [the tutorial](tutorial.md#the-arguments).

## The branded variable name

A CMIP7 branded variable variable name contains the name of the
variable followed by four additional labels.  For example the time
averaged surface air temperature is represented as follows:

```text
ta_tavg-p19-hxy-air
^^ ^^^^ ^^^ ^^^ ^^^
|  |    |   |   area: over air
|  |    |   horizontal: on an x-y grid
|  |    vertical: on 19 pressure levels
|  temporal: averaged over each reported interval
root variable name
```

| label | says | common values |
|---|---|---|
| temporal | how time was treated | `tavg` averaged, `tpt` at an instant, `tmax`, `tmin`, `tclm` a climatology, `ti` no time dimension |
| vertical | which levels | `u` none, `p19` 19 pressure levels, `al` model levels, `h2m` 2 m height, `d100m` 100 m depth |
| horizontal | how space was treated | `hxy` on a grid, `hm` collapsed to one value, `hs` at sites, `ht` along a transect |
| area | which areas contribute | `u` unmasked, `air`, `sea`, `si` sea ice, `lnd` land, `sn` snow |

Another example is `siextent_tavg-u-hm-u` is sea-ice extent, which
represents sea-ice extent averaged over each reported interval, with
no vertical level and no mask. The `hm` label means the horizontal
dimensions have been collapsed, so this variable is one number per
time step rather than a field on a grid.

The values each label may take come from the CMIP7 controlled vocabulary, which
is a file listing every value CMIP7 accepts. It defines 10 temporal labels, 44
vertical, 6 horizontal and 26 area.

The branded name is used throughout. The mapping files are keyed by it, the
CMOR tables are keyed by it, the suffix is its own level in the output
directory, and the temporal label is what decides whether time-averaged or
instantaneous history files are read.

## Mapping CMIP7 variables to model output variables

A CMIP7 variable is not a model variable. Variable mapping specifies
how to get from one or more model output variables to the required
CMIP7 variable.

The files that specify this mapping are grouped by realms and model
type (i.e. noresm or cesm).  A realm is a part of the Earth system
that CMIP7 groups variables under - the atmosphere, the land, the sea
ice. It groups variables rather than files. As an example, the
`atmos`, `atmosChem` and `aerosol` realms all use data that originate
in CAM time-slice files but differ only in which variables they ask
for out of them.

Naming a realm decides three things: which variable mapping file is read, which
history time slice files are searched, and which frequencies are available.

| realm | covers | history time slice files | frequencies |
|---|---|---|---|
| `atmos` | the atmosphere | `cam.h0a` to `cam.h7i` | mon, day, 6hr, 3hr, 1hr |
| `atmosChem` | atmospheric chemistry |  `cam.h0a` to `cam.h7i` | mon, day, 6hr, 3hr, 1hr |
| `aerosol` | aerosols |  `cam.h0a` to `cam.h7i` | mon, day, 6hr, 3hr, 1hr |
| `land` | the land surface | `clm2.h0a`, `clm2.h0i`, `clm2.h1a`, `clm2.h2a`, `clm2.h3a` | mon, day, 3hr, yr |
| `seaIce` | sea ice | `cice.h`, `cice.h1` | mon, day |
| `landIce` | the ice sheets | `cism.<ice_sheet>.h` | yr |

Every realm and model pair has its own variable mapping yaml file in
[`data/`](https://github.com/ESMCI/cmip7-prep/tree/main/data), named
`<model>_to_cmip7_<realm>.yaml`. For NorESM those are
`noresm_to_cmip7_atmos.yaml`, `noresm_to_cmip7_atmosChem.yaml`,
`noresm_to_cmip7_aerosol.yaml`, `noresm_to_cmip7_land.yaml`,
`noresm_to_cmip7_seaIce.yaml` and `noresm_to_cmip7_landIce.yaml`. Each entry in
the yaml file specifies the relationship between the model's native field or
fields and the target CMIP7 variable.  As an example,
`clt_tavg-u-hxy-u` is obtained from the model variable `CLDTOT`
multiplied by 100 to turn a fraction into a percentage.

Note that the mapping files are generated from Google Sheets that the
scientists fill in, by way of a CSV, using `convert_csv_to_yaml.py`.

## Mapping CMIP7 realm, frequency and sampling to output time series files

A model does not write all of its output to one set of files. As an example, CAM writes
monthly averages to one set of files, daily averages to another, six-hourly
instantaneous values to a third, and so on.

The files `<model>_include_patterns.yaml` specify which history time slice
files to read for a given realm, frequency and sampling type, meaning time
average versus instantaneous:

```yaml
atmos:
  mon:
    tavg: [cam.h0a]
  day:
    tavg: [cam.h1a]
  6hr:
    tavg: [cam.h2a]
    tpt: [cam.h3i]
```

So monthly time-averaged atmosphere output is in the files whose names
contain `cam.h0a`, and six-hourly instantaneous output is in
`cam.h3i`. Note that the entries in this file are substrings, not a
filenames. The files are found by globbing for the substrings (.e.g
globbing `*cam.h0a*`).

The sampling level is there because CMIP7 frequency does not distinguish
time-averaged output from instantaneous output, and both can be monthly. The
branded variable name carries it, as in `tas_tavg-u-hxy-u`, and CAM encodes it
in the file name - `a` for averaged and `i` for instantaneous.

A realm, frequency and sampling combination that is absent from this file is
output the model does not write, and asking for errors with `No include_patterns defined for ...`.

## The grid specification file

Not all NorESM and CESM components run on a logically rectangular grid, and
CMIP7 requires output on one. The atmosphere and land run on a spectral element
grid, the ocean and sea ice on a tripolar grid. Regridding needs a weight file,
and `<model>_grids.yaml` says which one to use.

Each realm gets two entries. `input_grid` names the grid its history files are
on. `grid_label` is the CMIP7 code for the grid its output lands on, and it
appears in the output path and in the filename.

| realm | input_grid | grid_label |
|---|---|---|
| `atmos`, `aerosol`, `atmosChem`, `land` | `ne16` at NorESM3-LM, `ne30` at NorESM3-MM | `g253`, `g106` |
| `ocean`, `ocnBgchem`, `seaIce` | `tnx1v4` | `g202` |
| `landIce` | `custom` | `g194` |

Only the atmosphere and land change with the resolution. `custom` means nothing
is regridded, because those variables already carry latitude and longitude.

The `resolutions` block at the top of the file gives the weight files for each
input grid: a conservative map, and a bilinear map for the variables listed in
`intensive_vars.yaml`. A grid label has to be registered in the controlled
vocabulary, or CMOR refuses to write the variable.

## The CMOR tables

`cmip7-prep` uses the CMIP7 data request to determine which variables are
needed for a given realm and frequency. The CMOR writer then uses the CMOR
JSON tables to determine how to write each variable out - its units, standard
name, dimensions, cell methods and the rest of its metadata.

The per-realm tables are in `cmip7-cmor-tables/tables/`, and the controlled
vocabulary is one file beside them at
`cmip7-cmor-tables/tables-cvs/cmor-cvs.json`. That file also holds the accepted
values for `experiment_id`, `source_id` and `grid_label`. Use `--tables-root`
to read a different checkout.

## What a run needs

- An archived case directory holding the `atm/hist`, `lnd/hist`, `ice/hist`,
  `glc/hist` and `ocn/hist` subdirectories.
- A writable output directory.
- The CMIP7 CMOR tables, which the driver finds beside this repository by
  default.
- The experiment's `experiment_id`, which is checked against the CMIP7
  controlled vocabulary before any work starts.

:::{note}
For NorESM3.0 the ocean realms are produced outside this tool.
:::
