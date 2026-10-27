# Tutorial

This runs the whole chain over one case. The example is a fifty-year NorESM
piControl case on NIRD.

## Look before you run

`--dry-run` prints every step it would run, with the exact command, and exits.
A full run is hours long, so check the plan first.

```bash
cd ~/cmip7-prep/scripts
python run_reference_case.py \
  --case-dir <the archived case directory> \
  --outdir <where the output should go> \
  --model noresm --model-res NorESM3-LM \
  --years 1:50:10 \
  --experiment piControl --variant-label r1i1p1f1 \
  --plots --html \
  --dry-run
```

## Run it

The same command without `--dry-run`. A dropped connection kills the run, so
start it under `nohup` or in a `tmux` session.

```bash
cd ~/cmip7-prep/scripts
nohup python run_reference_case.py \
  --case-dir <the archived case directory> \
  --outdir <where the output should go> \
  --model noresm --model-res NorESM3-LM \
  --years 1:50:10 \
  --experiment piControl --variant-label r1i1p1f1 \
  --plots --html \
  > ~/reference_run.log 2>&1 &
```

```bash
tail -f ~/reference_run.log
```

## The arguments

`--case-dir`
: The archived case root, holding `atm/`, `lnd/`, `ice/`, `glc/` and `ocn/`.

`--outdir`
: Where `timeseries/` and `cmor/` are written. Put it on scratch.

`--model`
: `noresm` or `cesm`. This selects the variable mappings and the history file
  patterns. The wrong one produces plausible but wrong output.

`--model-res`
: The resolution as named in `data/<model>_grids.yaml`: `NorESM3-LM`,
  `NorESM3-MM`, `ne30` or `custom`. It decides the grid each realm's input is
  on and the grid label its output carries. A case with `.LM.` in its name is
  `NorESM3-LM`.

`--experiment`
: The CMIP7 `experiment_id`, such as `piControl`. It is checked against the
  controlled vocabulary before any work starts, so a typo fails in seconds
  rather than hours.

`--years FIRST:LAST:INCREMENT`
: The years to process, and how many go in each output file. `1:50:10` gives
  five ten-year files per variable rather than one fifty-year file. Which times
  belong to a window is decided by the data's own time axis, so renamed or
  differently split time series make no difference. Leaving it out puts
  everything in one file.

`--variant-label`
: The ensemble member, such as `r1i1p1f1`. Needed for output you intend to
  submit, since it is what distinguishes one member from another.

`--plots`
: Have validation plot each variable: a mean time series, and a time-mean map
  where the data are on a plottable grid. This is most of validation's running
  time.

`--html`
: Have validation rebuild the browsable index covering every realm validated so
  far.

## Doing less than everything

```bash
--realms seaIce --frequencies mon day     # one realm, two frequencies
--stages validate                         # one stage
--max-plots 20                            # cap the plotting
```

`--ts-workers` and `--cmor-workers` both default to 4. Raising
`--cmor-workers` is usually limited by memory rather than cores: a worker holds
one variable's regridded arrays.
