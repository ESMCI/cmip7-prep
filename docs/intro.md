# Introduction

`cmip7-prep` processes one archived model case into CMOR-compliant output in
the following three stages. It works one realm at a time.

A realm is a part of the Earth system that CMIP7 groups variables under, such
as the atmosphere, the land or the sea ice. It groups variables rather than
files. The `atmos`, `atmosChem` and `aerosol` realms all use data that
originate in CAM time-slice files, and differ only in which variables they ask
for out of them.

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
stages over every target realm and frequency and is discussed in more detail below.

## The CMIP7 pieces it uses

Four things come from CMIP7 rather than from this repository. Each one answers a
different question.

The **data request** is a Python package called `cmip7-data-request-api`. It is
queried while a run starts, and it says which variables a given experiment
asks for.

The **CMOR tables** are a set of JSON files, one per realm, in
`cmip7-cmor-tables/tables/`. That directory is a git submodule, and
`--tables-root` reads a different checkout instead. The tables say how each
variable must be written. A table entry gives the units, the standard name,
the dimensions and the cell methods for one variable.

The **controlled vocabulary** is a single file beside those tables, at
`cmip7-cmor-tables/tables-cvs/cmor-cvs.json`. It holds the accepted values for
`experiment_id`, `source_id` and `grid_label`. A run checks its experiment
against this file before it does any work.

The **CMOR library** is used to write the CMOR compliant files. It compares
every variable it is given against that variable's table entry, and it refuses
to write anything that does not match.

Publishing the output to ESGF is done by a separate external tool,
[esgprep](https://esgf.github.io/esgf-prepare/), after this tool has finished.

## The mapping files

None of those four pieces knows anything about NorESM or CESM. The mapping
files in [`data/`](https://github.com/ESMCI/cmip7-prep/tree/main/data) supply
that part. They are written and maintained in this repository rather than
coming from CMIP7.

There is one mapping file for each model and realm, named
`<model>_to_cmip7_<realm>.yaml`:

```text
noresm_to_cmip7_atmos.yaml       cesm_to_cmip7_atmos.yaml
noresm_to_cmip7_atmosChem.yaml   cesm_to_cmip7_atmosChem.yaml
noresm_to_cmip7_aerosol.yaml     cesm_to_cmip7_aerosol.yaml
noresm_to_cmip7_land.yaml        cesm_to_cmip7_land.yaml
noresm_to_cmip7_seaIce.yaml      cesm_to_cmip7_seaIce.yaml
noresm_to_cmip7_landIce.yaml     cesm_to_cmip7_landIce.yaml
                                 cesm_to_cmip7_ocean.yaml
                                 cesm_to_cmip7_ocnBgchem.yaml
```

NorESM has no ocean mapping, because its ocean realms are produced outside
this tool.

Each file is keyed by branded variable name, and each entry says where that
variable comes from in the model. The simplest entry names a single model
field. An entry can also name several fields and the
formula that combines them, a scaling that converts the units, or the vertical
levels and the grid the variable is written on. As an example,
`clt_tavg-u-hxy-u` is obtained from the model variable `CLDTOT` multiplied by
100 to turn a fraction into a percentage.

The mapping files are generated from Google Sheets that the scientists fill
in, by way of a CSV, using `convert_csv_to_yaml.py`.

The three CMIP7 pieces and the mapping files divide the work between them. The
data request says what CMIP7 wants. The tables say what it must look like. The
mapping files say where it comes from in the model. Nothing is written unless
all three agree, which is the subject of the next section.

## How the variable list is decided

Each run of `cmor_driver.py` asks the data request a single question. The
question covers one experiment, one realm and one frequency:

```python
DR.find_variables(
    experiment="piControl",     # --experiment
    modelling_realm="atmos",    # --realm
    cmip7_frequency="mon",      # --frequency
    operation="all",
    skip_if_missing=False,
)
```

The output from the query is a list of variables. Each one carries its branded
variable name, such as `ta_tavg-p19-hxy-air`. That name is the key that the
mapping files and the CMOR tables both use, so it is what ties all three
together.

The query above returns every monthly atmosphere variable that CMIP7 asks for
in the `piControl` experiment, and nothing else.

The driver then makes two changes to the returned list. The data request plays
no part in either one.

Every variable returned by the query carries a region. Most carry `glb`,
meaning the globe, but a variable can also be requested over part of it. The
query can therefore return the same branded variable name more than once, with
a different region each time. For `piControl`, air temperature on pressure
levels is returned twice:

```text
branded variable name     region
ta_tavg-p19-hxy-air       glb
ta_tavg-p19-hxy-air       30S-90S
```

Those two entries carry the same branded variable name, so they are duplicates.
The region decides which one to drop. The `30S-90S` entry is dropped and the
`glb` entry is kept.

Some variables have no second entry. For `piControl`, `cltmodis_tavg-u-hxy-u`
is returned with the region `30S-90S` and nothing else. That entry is kept,
because dropping it would mean not writing the variable at all.

The region from the data request goes no further than this. It is not recorded
in the output.

The hemispheric sea ice variables are the one case where a region does reach
the output. They are returned as `nh` and `sh`, the mapping file gives them the
same two regions, and both are written.

The `--cmip-vars` option to `cmor_driver.py` keeps only the variable names
given on the command line and discards the rest of the list. This is how one
variable is reprocessed without running a whole realm again. Because it filters
what the query returned, it cannot add a variable that the data request did not
ask for.

The driver then looks up each variable from that list in the realm's mapping
file. One of three things happens.

- If the variable is in the data request and in the mapping file, it is
  written.
- If the variable is in the data request but not in the mapping file, nothing
  is written for it. The run reports it as `no mapping in YAML`.
- If the variable is in the mapping file but not in the data request, it is
  ignored.

A run therefore writes only the variables that appear in both the data request
and the mapping file. The second case above is the one to watch. Those are
variables that the model could supply and that nobody has mapped yet.
Validation lists them for the realm that was just run, and
`query_missing_vars.py` reports them across every experiment at once.

The `--run-all-from-yaml` option to `cmor_driver.py` sets the data request
aside. Every variable the mapping file defines is written, whether the data
request asks for it or not. This makes the tool usable for a MIP or a project
run that the CMIP7 data request does not cover.

## What the mapping file does with a selected variable

Once a variable is on the list, its entry in the mapping file drives everything
that follows.

1. The branded name is looked up in `<model>_to_cmip7_<realm>.yaml`. This gives
   one entry.
2. The entry names the model field the variable comes from. Where more than one
   field is needed, it names them all and gives the formula that combines them.
3. The file `<model>_include_patterns.yaml` turns the realm, the frequency and
   the sampling into the list of time series files to open.
4. The entry is evaluated against those files. This may be a rename, a unit
   scaling, or a formula.
5. The branded name and the entry together decide what happens to the vertical
   levels and to the grid. This is where pressure interpolation happens, and
   where the weight files named in `<model>_grids.yaml` are used.
6. CMOR writes the result. It checks the result against the table entry for the
   same branded name.

The sections below describe each of those files in turn.

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

A CMIP7 frequency (e.g. mon, day, 6hr) says how often a value is
reported, but not whether that value is an average over the interval
or a single sample from it. That information is contained in the
branded variable name where `tavg` denotes an average sampling and
`tpt` denotes an instantaneous sampling.
`<model>_include_patterns.yaml` contains the mapping for the realm
(e.g. atmos), the frequency (e.g. day) and the sampling (e.g. tpt) to
the model output files.


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
