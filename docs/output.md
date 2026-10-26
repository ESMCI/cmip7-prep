# What a run produces

## The output directory

Everything a run writes goes under `--outdir`:

```text
<outdir>/
  timeseries/<realm>/            one file per variable, from the history files
  cmor/
    CMIP7/MIP-DRS7/...           the CMOR output, in the CMIP7 directory structure
    logs/<realm>/                one CMOR log per variable, region and frequency
    validation_reports/
      index.html                 the browsable index, covering every realm
      <model>_<realm>_<experiment>_<frequency>/
        validation_summary.md    the plain-text summary
        validation_summary.json  the same thing for machines
        missing_variables.csv    variables the request asked for and did not get
        log_records.csv          one row per CMOR log, with its error lines
        dimension_inventory.csv  what dimensions each variable came out with
        redundant_files.csv      files whose period another file already covers
        plots/                   the time series, map and zonal plots
  logs/
    run.log                      the driver's own account of the whole run
    <period>/<step>.log          each step's output, e.g. 1-50/cmor-atmos-day.log
  completed_steps_<period>.json  which steps finished, for resuming
```

The `<period>` comes from `--years` - `1:50:10` gives `1-50`. The output tree
and the time series are shared between periods, because they are one dataset.

## Where the logs go

A run writes logs in four places, from the shortest account to the longest.

**The file you redirect to.** `~/reference_run.log` in the tutorial, or
whatever you named. It holds everything the run printed to the terminal, and is
the one to tail while a run is going.

**`<outdir>/logs/run.log`.** The driver's own account of itself: the command
each step ran, one line per step saying how it ended and how long it took, and
the summary table. It is appended to rather than replaced, so one file tells
the story of every run into that output directory. Each run starts with a
`RUN STARTED` banner.

**`<outdir>/logs/<period>/<step>.log`.** One step's whole output, both standard
output and standard error. As an example, `logs/1-50/cmor-atmos-day.log` is the
CMORization of the atmosphere at daily frequency over years 1 to 50, and
`logs/1-50/timeseries-seaIce.log` is sea-ice time series generation. These are
kept per period, so a run over years 51-100 cannot overwrite the account of
years 1-50. The summary table names the file for each step that failed.

**`<outdir>/cmor/logs/<realm>/`.** CMOR's own logs, one per variable, region,
frequency and window of years. As an example,
`cmor/logs/atmos/cmor_ta_tavg-p19-hxy-air_glb_mon_1-10.log` is what CMOR
reported while writing that variable over years 1 to 10. A units or dimension
error appears here and nowhere else, and validation reads these files to report
which variables have errors.

`<outdir>/completed_steps_<period>.json` is not a log but belongs with them. It
records which steps finished, and a rerun reads it to decide what to skip.

## The CMOR output structure

CMIP7 prescribes the directory structure, and there is no realm level in it:

```text
CMIP7/MIP-DRS7/CMIP/NCC/NorESM3-LM/piControl/r1i1p1f1/glb/mon/tas/tavg-h2m-hxy-u/g202/
    tas_tavg-h2m-hxy-u_mon_glb_g202_NorESM3-LM_piControl_r1i1p1f1_000101-001012.nc
```

Every realm writes into the same tree, and a file's path records its variable
but not which realm produced it. The region level is `glb` for most variables,
and `nh` or `sh` for the hemispheric sea-ice aggregates.

Each filename ends with the period it covers, so `--years 1:50:10` gives five
files per variable whose periods join up with no gap or overlap.

## The end-of-run table

When a run finishes it prints a table of what it did. The table is written to
two files:

- `<outdir>/logs/run.log`, which holds the driver's own lines and nothing else.
- The file you redirected to, which also holds everything the three scripts
  printed.

`run.log` is written as the run goes, so the table is there even if the
connection drops.

```text
==============================================================================
REFERENCE CASE RUN
==============================================================================
  timeseries-seaIce        412.3s  warning
  cmor-seaIce-mon         1831.7s  ok
  validate-seaIce-mon       96.4s  reported problems
  timeseries-land           55.2s  FAILED (1)
------------------------------------------------------------------------------
  1 ok, 1 reported problems, 1 with warnings, 1 failed, 2395.6s
  by stage: cmor 1831.7s, timeseries 467.5s, validate 96.4s

  logs for the failures:
    timeseries-land: logs/1-50/timeseries-land.log
```

The four labels mean different things, and only one of them needs chasing:

`ok`
: The step did what it was asked.

`warning`
: A time series step found no output for a frequency the realm declares,
  because the case wrote no history files at that frequency or none holding
  the variables the realm needs. Rerunning cannot change that, so the run
  carries on and a resume skips the step. The bottom of its log says which
  frequency.

`reported problems`
: Validation ran and found something to report. That is its job: under
  `--strict` it reports whenever a requested variable is missing. Read the
  report rather than rerunning the step.

`FAILED (n)`
: Something went wrong. This is the only label worth chasing, and the table
  lists the log for each one. Without `--keep-going` the run stops at the first
  failure, because everything after it would read input that is not there.

The run's own exit code is 1 only if something failed; warnings and validation
reports leave it at 0.

## Reading a validation report

Start with the plain-text summary:

```bash
cat <outdir>/cmor/validation_reports/noresm_seaIce_piControl_mon/validation_summary.md
```

It gives four counts — expected, produced, with log errors, missing — and then
lists the variables behind each. `missing_variables.csv` is the list on its own,
one per line, which is usually the number worth attention.

For the browsable version, open
`<outdir>/cmor/validation_reports/index.html` in a browser. It needs no web
server, covers every realm and frequency validated so far, and has filters for
experiment, frequency, realm, status and variable.
