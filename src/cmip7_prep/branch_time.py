"""Branch attributes for a run that starts from another run.

A CMORized file names the date in the parent run that its own run started
from.  CIME records the parent and that date in the case's env_run.xml, as
RUN_REFCASE and RUN_REFDATE.  The attributes give the date as a number in the
time units of the parent's files, so those units are read from one of the
parent's history files.

Components need not share time units, so the attributes are worked out for the
component of the realm being written.
"""

from __future__ import annotations

import os
from pathlib import Path
from xml.etree import ElementTree

import cftime
import xarray as xr

from .reference_run import REALM_COMPONENT, env_run_path

# How many files are opened in the search for one with a time axis.
MAX_FILES_TRIED = 20


def read_env_run(case_dir: os.PathLike | str) -> dict[str, str]:
    """Return the value of every entry in a case's env_run.xml, by its id."""
    path = env_run_path(case_dir)
    if not path.is_file():
        raise FileNotFoundError(f"No env_run.xml at {path}")
    root = ElementTree.parse(path).getroot()
    return {
        entry.get("id"): entry.get("value", "")
        for entry in root.iter("entry")
        if entry.get("id")
    }


def time_units_and_calendar(directory: os.PathLike | str) -> tuple[str, str]:
    """Return the units and calendar of the time axis of a directory's files."""
    paths = sorted(Path(directory).glob("*.nc"))
    if not paths:
        raise FileNotFoundError(f"No netCDF files in {directory}")
    for path in paths[:MAX_FILES_TRIED]:
        try:
            with xr.open_dataset(path, decode_cf=False) as ds:
                if "time" in ds.variables and "units" in ds["time"].attrs:
                    attrs = ds["time"].attrs
                    return attrs["units"], attrs.get("calendar", "noleap")
        except OSError:
            continue
    raise ValueError(
        f"None of the first {MAX_FILES_TRIED} files in {directory} has a time "
        "axis with units."
    )


def days_since(units: str) -> str:
    """Return time units counted in days from the same reference date.

    CMOR writes a time axis in days from the reference date it is given.

    >>> days_since("common_years since 0001-01-01 00:00:00")
    'days since 0001-01-01 00:00:00'
    """
    if " since " not in units:
        raise ValueError(f"Time units {units!r} have no reference date.")
    return f"days since {units.split(' since ', 1)[1].strip()}"


def date_to_number(date: str, units: str, calendar: str) -> float:
    """Return a CIME date, yyyy-mm-dd, as a number in the given time units.

    >>> date_to_number("0181-01-01", "days since 0001-01-01", "noleap")
    65700.0
    """
    year, month, day = (int(part) for part in date.split("-"))
    calendar = calendar.lower()
    when = cftime.datetime(year, month, day, calendar=calendar)
    return float(cftime.date2num(when, units, calendar=calendar))


def branch_attributes(
    case_dir: os.PathLike | str,
    realm: str,
    tsdir: os.PathLike | str,
    parent_case_dir: os.PathLike | str | None = None,
) -> dict:
    """Return the branch attributes of one realm's output for a case.

    ``case_dir`` is the case's archive directory and ``tsdir`` holds the
    realm's time series, the files the output's own time units come from.  The
    parent's archive directory is taken to sit beside ``case_dir`` under the
    name RUN_REFCASE gives, unless ``parent_case_dir`` says where it is.

    A startup run has no parent, and an empty dict is returned for it.
    """
    env = read_env_run(case_dir)
    run_type = env.get("RUN_TYPE")
    if run_type not in ("hybrid", "branch"):
        return {}
    missing = [
        name
        for name in ("RUN_REFCASE", "RUN_REFDATE", "RUN_STARTDATE")
        if not env.get(name)
    ]
    if missing:
        raise ValueError(f"{env_run_path(case_dir)} has no value for {missing}.")

    refdate = env["RUN_REFDATE"]
    # A branch run continues the parent's dates, and CIME ignores RUN_STARTDATE.
    startdate = refdate if run_type == "branch" else env["RUN_STARTDATE"]

    if parent_case_dir is None:
        parent_case_dir = Path(case_dir).parent / env["RUN_REFCASE"]
    parent_hist = Path(parent_case_dir) / REALM_COMPONENT[realm] / "hist"
    parent_units, parent_calendar = time_units_and_calendar(parent_hist)
    child_units, child_calendar = time_units_and_calendar(tsdir)
    parent_units = days_since(parent_units)

    return {
        "branch_method": "standard",
        "branch_time_in_parent": date_to_number(refdate, parent_units, parent_calendar),
        "branch_time_in_child": date_to_number(
            startdate, days_since(child_units), child_calendar
        ),
        "parent_time_units": parent_units,
    }
