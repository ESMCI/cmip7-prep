"""Tests for splitting a run into windows of years."""

import cftime
import numpy as np
import pytest
import xarray as xr

from cmip7_prep.time_windows import select_years, year_windows


def _dataset(years, calendar="noleap"):
    """A dataset with one time per year given, plus a field and a static one."""
    times = [cftime.datetime(year, 7, 1, calendar=calendar) for year in years]
    return xr.Dataset(
        {
            "T": ("time", np.arange(len(times), dtype="f4")),
            "area": ((), 1.0),
        },
        coords={"time": times},
    )


class TestYearWindows:
    """Tests for reading the window specification."""

    def test_an_even_split(self):
        """Fifty years in tens is five windows."""
        assert year_windows("1:50:10") == [
            (1, 10),
            (11, 20),
            (21, 30),
            (31, 40),
            (41, 50),
        ]

    def test_a_short_final_window(self):
        """The last window stops at the last year rather than overrunning."""
        assert year_windows("1:25:10") == [(1, 10), (11, 20), (21, 25)]

    def test_an_increment_covering_everything_gives_one_window(self):
        """Which writes a single file, as before windows existed."""
        assert year_windows("1526:1535:10") == [(1526, 1535)]

    def test_one_year_at_a_time(self):
        """An increment of one is a window per year."""
        assert year_windows("5:7:1") == [(5, 5), (6, 6), (7, 7)]

    @pytest.mark.parametrize(
        "spec", ["1:50", "1:50:10:2", "", "a:50:10", "1:b:10", "1:50:c"]
    )
    def test_a_malformed_specification_is_refused(self, spec):
        """Better to refuse than to guess at what was meant."""
        with pytest.raises(ValueError):
            year_windows(spec)

    def test_an_increment_of_zero_is_refused(self):
        """It would cover no years and loop forever."""
        with pytest.raises(ValueError, match="covers no years"):
            year_windows("1:50:0")

    def test_a_backwards_range_is_refused(self):
        """50 to 1 is a mistake, not an empty run."""
        with pytest.raises(ValueError, match="before"):
            year_windows("50:1:10")


class TestSelectYears:
    """Tests for taking one window out of a dataset."""

    def test_only_the_window_comes_back(self):
        """The years asked for, by the dataset's own time axis."""
        chosen = select_years(_dataset(range(1, 21)), 1, 10)
        assert chosen["time"].dt.year.values.tolist() == list(range(1, 11))

    def test_both_ends_are_included(self):
        """1 to 10 means 1 and 10 are in it."""
        chosen = select_years(_dataset(range(1, 21)), 11, 20)
        years = chosen["time"].dt.year.values
        assert years.min() == 11 and years.max() == 20

    def test_an_empty_window_gives_none(self):
        """Asking for years the case does not have says so."""
        assert select_years(_dataset(range(1, 11)), 41, 50) is None

    def test_a_partly_covered_window_gives_what_there_is(self):
        """A case ending mid-window is not an error."""
        chosen = select_years(_dataset(range(1, 16)), 11, 20)
        assert chosen["time"].dt.year.values.tolist() == [11, 12, 13, 14, 15]

    def test_a_dataset_without_time_is_passed_through(self):
        """Nothing to select; a time-invariant field is whole already."""
        static = xr.Dataset({"area": ((), 1.0)})
        assert select_years(static, 1, 10) is static

    def test_a_static_variable_survives_the_selection(self):
        """Selecting on time must not drop what has no time."""
        chosen = select_years(_dataset(range(1, 21)), 1, 10)
        assert "area" in chosen.data_vars

    @pytest.mark.parametrize("calendar", ["noleap", "360_day", "standard"])
    def test_any_calendar(self, calendar):
        """The years come from the axis, so the calendar does not matter."""
        chosen = select_years(_dataset(range(1, 21), calendar=calendar), 5, 7)
        assert chosen["time"].dt.year.values.tolist() == [5, 6, 7]
