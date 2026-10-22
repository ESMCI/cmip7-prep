"""Splitting a run into windows of years, and taking one out of a dataset.

CMORization writes one output file per window, so ``--years 1:50:10`` gives
five ten-year files rather than one fifty-year file.  The windows are spelled
the same way ``gen_timeseries.py --years-spec`` spells them, so the same
argument can be given to both and they cannot drift apart.

Which times belong to a window is decided by the dataset's own time axis, not
by anything in a filename.  A renamed file, or time series built in pieces of a
different size, make no difference.
"""

from __future__ import annotations

import numpy as np


def year_windows(spec: str) -> list[tuple[int, int]]:
    """Turn ``"first:last:increment"`` into one (first, last) pair per window.

    ``"1:50:10"`` gives ``[(1, 10), (11, 20), (21, 30), (31, 40), (41, 50)]``.
    A final window is short rather than overrunning: ``"1:25:10"`` ends with
    ``(21, 25)``.  An increment covering the whole range gives one window,
    which writes a single file as before.
    """
    parts = str(spec).split(":")
    if len(parts) != 3:
        raise ValueError(
            f"--years takes first:last:increment, for example 1:50:10, not {spec!r}"
        )
    try:
        first, last, increment = (int(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"--years: {spec!r} has a part that is not a number") from exc
    if increment < 1:
        raise ValueError(f"--years: an increment of {increment} covers no years")
    if last < first:
        raise ValueError(f"--years: {last} is before {first}")
    return [
        (start, min(start + increment - 1, last))
        for start in range(first, last + 1, increment)
    ]


def select_years(dataset, first: int, last: int):
    """Return the part of a dataset whose times fall in the years first..last.

    The years come from the time axis itself, so this is independent of the
    calendar and of how the time series files were named or split.  Returns
    None when the window holds nothing, which is the caller's cue to say so
    rather than write an empty file.
    """
    if "time" not in getattr(dataset, "dims", ()):
        return dataset
    years = dataset["time"].dt.year
    keep = np.flatnonzero(((years >= first) & (years <= last)).values)
    if keep.size == 0:
        return None
    return dataset.isel(time=keep)
