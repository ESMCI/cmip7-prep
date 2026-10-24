"""Find CMOR output files that duplicate each other, and delete them.

Each output filename ends in the period it covers, as in
``_00010101-00101231.nc`` for years 1 to 10.  One variable is written as one
file per window of years, so fifty years asked for in ten-year windows gives
five files, and no two of those five periods should overlap.

A file whose period lies inside another file's is therefore a duplicate: every
time in it is already in the longer file.  That is what a run killed while
writing a variable leaves behind, because CMOR names the file before it has
finished filling it.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# The trailing period in a CMOR filename.  CMOR writes as many digits as the
# frequency needs, four for a year and up to eight for a day, and the same
# number at both ends.
FILE_PERIOD_RE = re.compile(r"_(?P<first>\d{4,8})-(?P<last>\d{4,8})\.nc$")


def file_period(path: Path | str) -> tuple[str, str] | None:
    """The first and last date in a CMOR filename, as written there.

    Returns None for a name that does not end in a period, such as one of the
    fx files, which cover no time at all.
    """
    match = FILE_PERIOD_RE.search(Path(path).name)
    if match is None:
        return None
    first, last = match.group("first"), match.group("last")
    if len(first) != len(last):
        return None
    return first, last


def covers(outer: tuple[str, str], inner: tuple[str, str]) -> bool:
    """Whether the first period includes every date of the second.

    The dates compare correctly as text because CMOR writes the year first and
    pads each field to a fixed width.  Different widths mean different
    frequencies, which are not comparable.
    """
    if len(outer[0]) != len(inner[0]):
        return False
    return outer[0] <= inner[0] and inner[1] <= outer[1]


def superseded_by(written: Path) -> list[Path]:
    """The files beside this one whose period it already covers.

    One directory of the CMIP7 output holds one variable at one frequency on
    one grid, so every file there is comparable with every other.
    """
    period = file_period(written)
    if period is None:
        return []
    found = []
    for other in sorted(written.parent.glob("*.nc")):
        if other.name == written.name:
            continue
        other_period = file_period(other)
        if other_period is None or other_period == period:
            continue
        if covers(period, other_period):
            found.append(other)
    return found


def remove_superseded(written: Path | str) -> list[Path]:
    """Delete the files this one supersedes, and return what was deleted."""
    written = Path(written)
    removed = []
    for path in superseded_by(written):
        path.unlink()
        logger.warning(
            "Removed %s: it covers %s to %s, which %s now covers in full.",
            path.name,
            *file_period(path),
            written.name,
        )
        removed.append(path)
    return removed


def redundant_pairs(paths: list[Path]) -> list[tuple[Path, Path]]:
    """Pairs of one variable's files where the first's period is inside the second's.

    Reports rather than deletes, for checking output that is already on disk.
    """
    with_period = [(path, file_period(path)) for path in sorted(paths)]
    comparable = [(path, period) for path, period in with_period if period]
    pairs = []
    for inner_path, inner in comparable:
        for outer_path, outer in comparable:
            if (
                outer_path.name != inner_path.name
                and outer != inner
                and covers(outer, inner)
            ):
                pairs.append((inner_path, outer_path))
                break
    return pairs
