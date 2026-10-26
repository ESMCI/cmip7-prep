"""Split a run into windows of years, and take one window out of a dataset.

CMORization writes one output file per window, so ``--years 1:50:10`` gives
five ten-year files rather than one fifty-year file. A window is spelled the
same way ``gen_timeseries.py --years-spec`` spells it, so the same argument
serves both scripts.

The dataset's own time axis decides which times belong to a window. A renamed
file makes no difference, and neither do time series built in pieces of a
different size.
"""

from __future__ import annotations

import numpy as np


def year_windows(spec: str) -> list[tuple[int, int]]:
    """Turn ``"first:last:increment"`` into one (first, last) pair per window.

    ``"1:50:10"`` gives ``[(1, 10), (11, 20), (21, 30), (31, 40), (41, 50)]``.
    A final window is short rather than overrunning, so ``"1:25:10"`` ends
    with ``(21, 25)``. An increment covering the whole range gives one window.
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

    The years come from the time axis itself, so the calendar and the way the
    time series files were split make no difference. The result is None when
    the window holds nothing, and the caller then says so rather than writing
    an empty file.
    """
    if "time" not in getattr(dataset, "dims", ()):
        return dataset
    years = dataset["time"].dt.year
    keep = np.flatnonzero(((years >= first) & (years <= last)).values)
    if keep.size == 0:
        return None
    return dataset.isel(time=keep)
