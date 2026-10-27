#!/usr/bin/env python
"""Check that a variable's output files join up with no gap or overlap.

CMORization writes one file per window of years, so a fifty-year request comes
back as five ten-year files.  This checks the thing that splitting can get
wrong: that the files together cover the period exactly once.

    python check_window_joins.py <cmor root>

Every directory holding more than one file is one variable at one frequency on
one grid, so each is checked on its own:

  - times within a file strictly increase
  - the files are consecutive, with the same spacing across a join as within
  - no time appears twice
  - the files' date ranges meet without a gap

Needs no reference output.  Reports one line per variable, and exits non-zero
if any of them is wrong.
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import xarray as xr


def times_of(path: Path):
    """The time coordinate of one file, as cftime objects."""
    coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    with xr.open_dataset(path, decode_times=coder) as dataset:
        if "time" not in dataset.coords:
            return None
        return dataset["time"].values


def typical_spacing(times) -> float:
    """The usual gap between consecutive times, in days."""
    if len(times) < 2:
        return 0.0
    gaps = [(times[i + 1] - times[i]).total_seconds() for i in range(len(times) - 1)]
    return float(np.median(gaps)) / 86400.0


def check_group(paths: list[Path]) -> list[str]:
    """Return what is wrong with one variable's files, or an empty list."""
    problems = []
    series = []
    for path in sorted(paths):
        times = times_of(path)
        if times is None:
            continue
        if len(times) > 1 and not all(
            times[i] < times[i + 1] for i in range(len(times) - 1)
        ):
            problems.append(f"{path.name}: times do not increase")
        series.append((times, path))

    if len(series) < 2:
        return problems

    series.sort(key=lambda pair: pair[0][0])
    spacing = typical_spacing(series[0][0])

    for (earlier, first), (later, second) in zip(series, series[1:]):
        if later[0] <= earlier[-1]:
            problems.append(
                f"{first.name} ends {earlier[-1]} but {second.name} starts "
                f"{later[0]}: they overlap"
            )
            continue
        across = (later[0] - earlier[-1]).total_seconds() / 86400.0
        # Monthly spacing varies with the month, so allow a day either way;
        # anything larger is a missing interval rather than a short February.
        if spacing and abs(across - spacing) > 1.0:
            problems.append(
                f"{first.name} to {second.name}: {across:.1f} days across the "
                f"join where {spacing:.1f} is usual -- a gap"
            )
    return problems


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    root = Path(sys.argv[1]).expanduser().resolve()

    groups: dict[Path, list[Path]] = defaultdict(list)
    for path in root.rglob("*.nc"):
        groups[path.parent].append(path)

    multi = {d: files for d, files in groups.items() if len(files) > 1}
    if not multi:
        print(f"No directory under {root} holds more than one file; nothing to join.")
        return 0

    wrong = 0
    for directory in sorted(multi):
        files = multi[directory]
        problems = check_group(files)
        name = files[0].name.split("_")[0]
        where = directory.relative_to(root)
        if problems:
            wrong += 1
            print(f"WRONG  {name}  ({len(files)} files)  {where}")
            for problem in problems:
                print(f"       {problem}")
        else:
            counts = [times_of(path) for path in files]
            total = sum(len(times) for times in counts if times is not None)
            print(f"ok     {name}  {len(files)} files join up, {total} times")

    print(f"\n{len(multi)} variable(s) checked, {wrong} wrong")
    return 1 if wrong else 0


if __name__ == "__main__":
    sys.exit(main())
