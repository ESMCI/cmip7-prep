# Recipes

## Rerun after an interruption

Repeat the same command. Steps recorded as finished are skipped, so a run that
died in the middle continues rather than starting over.

What counts as finished:

- A time series or CMOR step counts if it succeeded, or if its only complaint
  was a frequency with no output.
- Validation counts unless it crashed, because reporting problems is its job.

## Run everything again anyway

```bash
--force
```

This skips nothing and rewrites time series that already exist. It is the right
flag after changing a variable mapping, and the wrong one if you only want
validation redone, because it will regenerate hours of time series.

## Redo only validation

```bash
--stages validate --force
```

`--force` is needed because the previous validation steps are recorded as
finished. With `--stages validate` it touches nothing else.

:::{warning}
Keep `--plots` on a validation rerun. Without it the plot references are
written out of `validation_summary.json`, and the HTML index then shows no
figures for that realm.
:::

## One realm, one frequency

```bash
--realms seaIce --frequencies day
```

## Check the output files join up

Each variable's files should cover its period exactly once. This opens them and
checks the time axes, so it catches a gap or an overlap that filenames alone
would not show:

```bash
python check_window_joins.py <outdir>/cmor
```

It reports one line per variable and exits non-zero if any are wrong. Worth
running after any interrupted run, not only before submission.

## Run one stage by hand

The driver assembles these for you, but each runs on its own.

Time series for one realm:

```bash
python gen_timeseries.py \
  --inputdir <case>/atm/hist \
  --outputdir <outdir>/timeseries/atmos \
  --realm atmos --model noresm \
  --frequency mon day \
  --years-spec 1:50:10 \
  --workers 8
```

CMORize one realm and frequency:

```bash
python cmor_driver.py \
  --realm atmos --frequency mon \
  --model noresm --model-res NorESM3-LM --experiment piControl \
  --tsdir <outdir>/timeseries/atmos \
  --outdir <outdir>/cmor \
  --years 1:50:10 \
  --variant-label r1i1p1f1 \
  --workers 4
```

Validate one realm and frequency:

```bash
python validate_cmor_output.py \
  --model noresm --realm seaIce --experiment piControl --frequency mon \
  --root-output-path <outdir>/cmor \
  --strict --plot-timeseries --plot-maps --html
```

## Publish the validation reports

The `validation_reports/` directory is self-contained, so it can be copied
whole to a web-served location:

```bash
rsync -av --delete <outdir>/cmor/validation_reports \
  /nird/datalake/NS9560K/www/diagnostics/noresm/$USER/<case>/
```

`--delete` makes the published copy match the fresh one exactly, so reports for
a realm you have since removed do not linger. Sync from the output directory
rather than from an earlier copy, or you republish stale reports.

## Limit the memory a run uses

```bash
--cmor-workers 2        # a CMOR worker holds one variable's regridded arrays
--ts-workers 4
```

`gen_timeseries.py` also takes `--memory-budget-gb` (default 16), divided
between its workers. On a shared node, fewer workers is the polite setting as
well as the safer one.
