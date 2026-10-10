"""Synthetic end-to-end checks of time series generation, by realm and frequency.

These tests build throwaway history files for every realm and frequency the
include-pattern tables declare, then run the real GenTS collection and time
series steps over them.  Nothing here touches model output on disk, so the whole
sweep runs on a laptop in seconds.

What this is meant to catch:

* an include pattern that selects nothing, which the driver reports only as
  "No input files to process" followed by a clean exit;
* a realm or frequency quietly dropped from a table;
* a GenTS API change, of the kind that deprecated ``include_patterns`` in 1.4.0;
* a stream naming convention that moved on without the tables following.

What it deliberately does not cover: regridding (the ESMF weights live on NIRD),
CMOR table and controlled-vocabulary compliance, and anything about whether the
numbers are physically meaningful.  Those belong to the on-cluster tier.
"""

import glob
import os
import warnings

import numpy as np
import pytest
import xarray as xr
from gents.hfcollection import HFCollection
from gents.timeseries import TSCollection

from cmip7_prep.include_patterns import all_include_patterns, load_include_patterns

MODELS = ["noresm", "cesm"]

# Days between samples, per frequency, so a stream's time axis is at least
# plausible for the frequency it claims to be.
STEP_DAYS = {
    "1hr": 1.0 / 24.0,
    "3hr": 3.0 / 24.0,
    "6hr": 6.0 / 24.0,
    "day": 1.0,
    "mon": 30.0,
    "yr": 365.0,
}

# Neutral variable names: these tests are about file selection and series
# assembly, not about any realm's actual variables.
VARIABLES = ["FIELD_A", "FIELD_B"]

RECORDS_PER_FILE = 2
FILES_PER_STREAM = 2


def _cases():
    """Return every (model, realm, frequency) the tables declare."""
    cases = []
    for model in MODELS:
        for realm, by_frequency in load_include_patterns(model).items():
            for frequency in sorted(by_frequency):
                cases.append((model, realm, frequency))
    return cases


# pytest labels each case from the values themselves, e.g. "noresm-seaIce-day",
# so no explicit ids are needed.
CASES = _cases()


def _patterns(model, realm, frequency):
    """Return the include patterns for one case, as the driver asks for them."""
    ice_sheet = "gris" if realm == "landIce" else None
    return all_include_patterns(model, realm, ice_sheet, None, [frequency])


def _filename(fragment, index):
    """Return a history file name embedding an include-pattern fragment.

    Patterns are stream fragments such as ``cam.h0a``, ``cice.h1.`` or
    ``cism.gris.h``; stripping the dots lets one rule serve all of them.
    """
    return f"case.{fragment.strip('.')}.{1526 + index:04d}-01-01-00000.nc"


def _write_history_file(path, frequency, first_record):
    """Write a minimal history file with RECORDS_PER_FILE samples.

    Time is left as raw offsets with CF attributes set by hand, so the file
    looks like model output rather than something xarray round-tripped: no
    fill value on the coordinates, and an unlimited time dimension.
    """
    step = STEP_DAYS[frequency]
    offsets = np.array(
        [(first_record + n) * step for n in range(RECORDS_PER_FILE)], dtype="f8"
    )
    dataset = xr.Dataset(
        {
            name: (
                ("time", "lat", "lon"),
                np.zeros((RECORDS_PER_FILE, 2, 2), dtype="f4"),
                {"units": "1"},
            )
            for name in VARIABLES
        },
        coords={
            "time": ("time", offsets),
            "lat": ("lat", np.array([-45.0, 45.0])),
            "lon": ("lon", np.array([0.0, 180.0])),
        },
    )
    dataset["time_bnds"] = (
        ("time", "nbnd"),
        np.array([[o, o + step] for o in offsets]),
    )
    dataset["time"].attrs.update(
        units="days since 1850-01-01", calendar="noleap", bounds="time_bnds"
    )
    no_fill = {name: {"_FillValue": None} for name in ("time", "lat", "lon")}
    dataset.to_netcdf(path, unlimited_dims=["time"], encoding=no_fill)


def _build_stream(directory, fragment, frequency):
    """Write FILES_PER_STREAM consecutive history files for one stream."""
    written = []
    for index in range(FILES_PER_STREAM):
        path = os.path.join(directory, _filename(fragment, index))
        _write_history_file(path, frequency, index * RECORDS_PER_FILE)
        written.append(path)
    return written


# ------------------------------------------------------------------ patterns


@pytest.mark.parametrize("model,realm,frequency", CASES)
def test_patterns_are_declared(model, realm, frequency):
    """Every declared realm and frequency yields at least one pattern."""
    patterns = _patterns(model, realm, frequency)
    assert patterns, f"{model}/{realm}/{frequency} declares no patterns"
    assert all(isinstance(p, str) and p for p in patterns)
    assert not any("{" in p for p in patterns), "unsubstituted placeholder"


@pytest.mark.parametrize("model,realm,frequency", CASES)
def test_patterns_select_their_own_files(tmp_path, model, realm, frequency):
    """Files named after a pattern are found by the glob the driver builds.

    The driver wraps each pattern as ``*<pattern>*`` and globs the input
    directory (see gen_timeseries.py), so that exact form is what is tested.
    """
    patterns = _patterns(model, realm, frequency)
    for fragment in patterns:
        _build_stream(tmp_path, fragment, frequency)

    for fragment in patterns:
        found = glob.glob(os.path.join(tmp_path, f"*{fragment}*"))
        assert found, f"pattern {fragment!r} matched nothing it should have"


# ------------------------------------------------------------- series output


@pytest.mark.parametrize("model,realm,frequency", CASES)
def test_timeseries_generated_per_variable(tmp_path, model, realm, frequency):
    """GenTS turns each synthetic stream into one time series per variable."""
    inputdir = tmp_path / "hist"
    outputdir = tmp_path / "timeseries"
    inputdir.mkdir()
    outputdir.mkdir()

    patterns = _patterns(model, realm, frequency)
    fragment = patterns[0]
    written = _build_stream(inputdir, fragment, frequency)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        collection = HFCollection(str(inputdir), num_processes=1)
        collection = collection.include([f"*{fragment}*"])
        assert len(collection) == len(written)
        collection.pull_metadata(show_progress=False)

        series = TSCollection(collection, str(outputdir), num_processes=1)
        series = series.apply_overwrite("*")
        series.execute()

    produced = [
        os.path.basename(p)
        for p in glob.glob(str(outputdir / "**" / "*.nc"), recursive=True)
    ]
    assert produced, f"{model}/{realm}/{frequency}: no time series written"
    for variable in VARIABLES:
        matching = [name for name in produced if variable in name]
        assert len(matching) == 1, (
            f"{model}/{realm}/{frequency}: expected one series for {variable}, "
            f"got {matching}"
        )


# -------------------------------------------------------------- API contract


def test_unknown_realm_is_rejected():
    """A realm no table defines raises rather than returning nothing."""
    with pytest.raises(ValueError, match="No include_patterns"):
        all_include_patterns("noresm", "no_such_realm")


def test_unknown_frequency_is_rejected():
    """A frequency a realm does not define raises and lists what is available."""
    with pytest.raises(ValueError, match="No include_patterns"):
        all_include_patterns("noresm", "seaIce", None, None, ["1hr"])


def test_ice_sheet_placeholder_is_substituted():
    """The landIce patterns carry the ice sheet given, not a placeholder."""
    for sheet in ("gris", "ais"):
        patterns = all_include_patterns("noresm", "landIce", sheet, None, ["yr"])
        assert patterns
        assert all(sheet in p for p in patterns), patterns
