# When something goes wrong

## A time series step fails on a single history file

```text
ValueError: Expected time array of size 2 or greater, got 1. Group with paths:
[.../n1850esmGaxgGHG.LM.n30b25.529.20261002.cam.h0a.1700-12.nc]
```

A history file belonging to a *different* case is sitting in this case's input
directory. Time series generation matches files by stream, as in `*cam.h0a*`,
which any case's monthly atmosphere files match. Files are then grouped by the
case name in the filename, so the stray file forms a group of its own, and one
file with one time record gives nothing to measure a time step against.

The traceback names the file. Compare its case name with the directory's:

```bash
ls <case>/atm/hist/*cam.h0a* | sed 's|.*/||; s|\.cam\.h0a\..*||' | sort | uniq -c
```

One case name and an even count is what you want. Move any others out of the
input directory altogether — a subdirectory is not far enough, because the
whole tree below the input directory is searched.

A group holding fewer than two time records between them is dropped with a
warning rather than failing the step:

```text
Skipping 1 file(s) of .../n1850esmGaxgGHG...cam.h0a*: 1 time record(s) between
them, too few to measure a time step from.
```

## A step is labelled `warning`

A frequency the realm declares produced no time series at all. The bottom of
that step's log says which:

```text
Time series in .../timeseries for realm seaIce:
  mon    1242 file(s)
  day       0 file(s)  <-- none
```

Either the case wrote no history files at that frequency, or the streams it
wrote hold none of the variables the realm needs. Neither is something a rerun
can fix, so the run carries on and a resume skips the step.

## Validation reports errors you cannot account for

Check which tree the report describes before investigating anything in it:

```bash
grep -o '"root_output_path": "[^"]*"' validation_summary.json
grep -o '"generated_at": "[^"]*"' validation_summary.json
```

A report describes the tree at the path it scanned, at the time it ran. If the
output directory has been renamed or moved since, every number in it —
including the missing count and the plot paths — belongs to the old tree.
Delete the report directory and validate again rather than reading it.

## Workers left running after a killed run

Workers hold their memory until something reaps them, and on a shared node a
week of them can take most of the memory limit. A pool worker asks the kernel to
kill it when the run dies. A pool created inside library code cannot be reached
that way, so check after killing a run:

```bash
pgrep -u $USER -f multiprocessing | while read -r p; do
  ps -o pid=,ppid=,rss=,etime=,cmd= -p "$p"
done
```

Anything whose parent is `1` is orphaned. `kill -9` on those reclaims the
memory.

## A run was killed partway through writing a variable

CMOR names each output file before it has finished filling it, so a file left
behind has a correct name and a short period. The next run writes the full
period under a different name, and both files then sit in the same directory
reporting some of the same times twice.

A variable that is rewritten cleans this up by itself: the driver deletes any
file beside the one it just wrote whose period falls inside it. For a step you
do not intend to rerun, validation reports them in `redundant_files.csv`, and:

```bash
python check_window_joins.py <outdir>/cmor
```

opens the files and checks that each variable's cover its period exactly once.

## `ModuleNotFoundError: No module named 'cmip7_prep'`

`PYTHONPATH` is unset, or set to a relative path that does not resolve from your
current directory. See [Installation](installation.md).

## The HTML index shows no plots for a realm

Validation was rerun for that realm without `--plots`, which writes the plot
references out of the summary. Run it again with `--plots --html`.

## Memory on a shared node

The limit is shared with everyone else on the node, so what is available to you
depends on what they are doing. To see the limit and the current usage inside a
container:

```bash
cat /sys/fs/cgroup/memory/memory.limit_in_bytes
grep -E '^(rss|cache) ' /sys/fs/cgroup/memory/memory.stat
```

`rss` is what processes are actually holding; `cache` is page cache and is
reclaimable. If a CMOR worker is killed with no traceback, it was the kernel's
out-of-memory killer, and fewer `--cmor-workers` is the answer.
