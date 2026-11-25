"""Tests for the branch attributes read from a case and its parent.

The cases here are a historical run started on 1850-01-01 from 0181-01-01 of
its parent, with the parent counting days from its own year 1.
"""

import pytest
import xarray as xr

from cmip7_prep.branch_time import (
    branch_attributes,
    date_to_number,
    days_since,
    read_env_run,
    time_units_and_calendar,
)

PARENT_UNITS = "days since 0001-01-01 00:00:00"
CHILD_UNITS = "days since 1850-01-01 00:00:00"


def _write_env_run(case_dir, run_type="hybrid"):
    """Write the four env_run.xml entries the branch attributes come from."""
    values = {
        "RUN_TYPE": run_type,
        "RUN_STARTDATE": "1850-01-01",
        "RUN_REFCASE": "parent",
        "RUN_REFDATE": "0181-01-01",
    }
    entries = "".join(
        f'<entry id="{name}" value="{value}"><type>char</type></entry>'
        for name, value in values.items()
    )
    path = case_dir / case_dir.name / "env_run.xml"
    path.parent.mkdir(parents=True)
    path.write_text(f'<file id="env_run.xml"><group id="run">{entries}</group></file>')


def _write_time_file(directory, units, calendar="noleap"):
    """Write a file whose time axis has the given units."""
    directory.mkdir(parents=True, exist_ok=True)
    time = xr.Variable("time", [15.5], {"units": units, "calendar": calendar})
    xr.Dataset({"time": time}).to_netcdf(directory / "case.h0.0001-01.nc")


@pytest.fixture(name="archive")
def archive_fixture(tmp_path):
    """Return an archive holding a child case, its time series and its parent."""
    _write_env_run(tmp_path / "child")
    _write_time_file(tmp_path / "parent" / "ice" / "hist", PARENT_UNITS)
    _write_time_file(tmp_path / "timeseries", CHILD_UNITS)
    return tmp_path


def test_env_run_entries_are_read_by_id(archive):
    """Each entry's value is found under its id."""
    env = read_env_run(archive / "child")

    assert env["RUN_REFCASE"] == "parent"
    assert env["RUN_REFDATE"] == "0181-01-01"


def test_a_case_without_env_run_is_reported(tmp_path):
    """The error names the file that was looked for."""
    with pytest.raises(FileNotFoundError, match="env_run.xml"):
        read_env_run(tmp_path / "child")


def test_time_units_are_read_from_a_file_in_the_directory(archive):
    """The units and calendar are the time variable's attributes."""
    units, calendar = time_units_and_calendar(archive / "parent" / "ice" / "hist")

    assert units == PARENT_UNITS
    assert calendar == "noleap"


def test_a_directory_without_files_is_reported(tmp_path):
    """The error names the directory that held no files."""
    with pytest.raises(FileNotFoundError, match="No netCDF files"):
        time_units_and_calendar(tmp_path)


def test_units_without_a_reference_date_are_rejected():
    """A number of days needs a date to count from."""
    with pytest.raises(ValueError, match="no reference date"):
        days_since("days")


def test_an_upper_case_calendar_is_accepted():
    """MOM6 writes its calendar as NOLEAP."""
    assert date_to_number("0002-01-01", "days since 0001-01-01", "NOLEAP") == 365.0


def test_the_parent_date_is_in_the_parent_units(archive):
    """RUN_REFDATE is counted from the parent's reference date."""
    branch = branch_attributes(archive / "child", "seaIce", archive / "timeseries")

    assert branch["parent_time_units"] == PARENT_UNITS
    assert branch["branch_time_in_parent"] == 180 * 365.0


def test_the_child_date_is_in_the_child_units(archive):
    """RUN_STARTDATE is counted from the reference date of the time series."""
    branch = branch_attributes(archive / "child", "seaIce", archive / "timeseries")

    assert branch["branch_method"] == "standard"
    assert branch["branch_time_in_child"] == 0.0


def test_the_parent_can_be_somewhere_else(archive):
    """A parent archived elsewhere is named instead of being looked for."""
    _write_time_file(archive / "elsewhere" / "ice" / "hist", "days since 0101-01-01")

    branch = branch_attributes(
        archive / "child", "seaIce", archive / "timeseries", archive / "elsewhere"
    )

    assert branch["branch_time_in_parent"] == 80 * 365.0


def test_a_branch_run_starts_on_the_parent_date(tmp_path):
    """CIME ignores RUN_STARTDATE for a branch run."""
    _write_env_run(tmp_path / "child", run_type="branch")
    _write_time_file(tmp_path / "parent" / "ice" / "hist", PARENT_UNITS)
    _write_time_file(tmp_path / "timeseries", PARENT_UNITS)

    branch = branch_attributes(tmp_path / "child", "seaIce", tmp_path / "timeseries")

    assert branch["branch_time_in_child"] == branch["branch_time_in_parent"]


def test_a_startup_run_has_no_branch_attributes(tmp_path):
    """A startup run has no parent, so the defaults are left as they are."""
    _write_env_run(tmp_path / "child", run_type="startup")

    assert not branch_attributes(tmp_path / "child", "seaIce", tmp_path / "timeseries")
