"""Tests for deriving an averaging period from a file's own attributes.

Model output without time bounds has to have them synthesized, since CMOR
refuses a time axis without bounds when the table expects a mean.  With two or
more time steps the spacing gives the period; with one step it can only come
from the file, via the ``time_period_freq`` attribute the output
carries.
"""

import numpy as np
import xarray as xr

from cmip7_prep.cmor_writer import period_in_axis_units

YEAR_AXIS = "common_year since 0000-01-01 0:0:0"
DAY_AXIS = "days since 1850-01-01"


def _dataset(period=None):
    """Return a dataset carrying one time value and an optional period."""
    attrs = {"time_period_freq": period} if period else {}
    return xr.Dataset(
        {"acab": ("time", np.array([1.0]))},
        coords={"time": ("time", np.array([1527.0]))},
        attrs=attrs,
    )


def _bounds_from(times, period=None):
    """Return the bounds the writer builds for a set of time values.

    Mirrors the arithmetic in cmor_writer: each period ends at its own time
    value, because that is how the output is labelled.
    """
    times = np.asarray(times, dtype="f8")
    if times.size >= 2:
        left = np.empty_like(times)
        left[1:] = times[:-1]
        left[0] = times[0] - (times[1] - times[0])
    else:
        left = np.array([times[0] - period], dtype="f8")
    return np.column_stack([left, times]).tolist()


class TestEndStampedBounds:
    """Each period ends at its time value, not centred on it.

    A CISM file stamped 1527 holds the annual mean for 1526, so its bounds run
    1526 to 1527.  Centring would give 1526.5 to 1527.5: the right length, the
    wrong year.
    """

    def test_one_annual_step(self):
        """One step uses the period the file records."""
        assert _bounds_from([1527.0], period=1.0) == [[1526.0, 1527.0]]

    def test_two_annual_steps(self):
        """Consecutive periods meet at the time values, on calendar years."""
        assert _bounds_from([1527.0, 1528.0]) == [
            [1526.0, 1527.0],
            [1527.0, 1528.0],
        ]

    def test_periods_are_contiguous(self):
        """No gaps or overlaps between consecutive periods."""
        bounds = _bounds_from([1527.0, 1528.0, 1529.0])
        for earlier, later in zip(bounds, bounds[1:]):
            assert earlier[1] == later[0]

    def test_one_step_and_many_steps_agree(self):
        """The first period is the same whichever path produced it."""
        alone = _bounds_from([1527.0], period=1.0)[0]
        together = _bounds_from([1527.0, 1528.0])[0]
        assert alone == together


class TestMatchingUnits:
    """Periods expressed in the unit the axis already counts in."""

    def test_annual_period_on_a_year_axis(self):
        """The land-ice case: one annual step on an axis counted in years."""
        assert period_in_axis_units(_dataset("year_1"), YEAR_AXIS) == 1.0

    def test_multi_year_period(self):
        """A five-year mean is five years long."""
        assert period_in_axis_units(_dataset("year_5"), YEAR_AXIS) == 5.0

    def test_daily_period_on_a_day_axis(self):
        """A daily mean is one day long on an axis counted in days."""
        assert period_in_axis_units(_dataset("day_1"), DAY_AXIS) == 1.0


class TestRefusals:
    """Cases with no exact answer, where guessing would misstate the period."""

    def test_years_against_a_day_axis(self):
        """A year is 360, 365 or 365.25 days depending on the calendar."""
        assert period_in_axis_units(_dataset("year_1"), DAY_AXIS) is None

    def test_months_against_a_day_axis(self):
        """A month has no fixed length at all."""
        assert period_in_axis_units(_dataset("month_1"), DAY_AXIS) is None

    def test_hours_against_a_day_axis(self):
        """Exact conversions are not attempted either, only matching units.

        Six hours is a quarter of a day in every calendar, but no output this
        is used on needs that, and converting is where mistakes would hide.
        """
        assert period_in_axis_units(_dataset("hour_6"), DAY_AXIS) is None

    def test_no_period_attribute(self):
        """Output that does not record its period yields nothing."""
        assert period_in_axis_units(_dataset(), YEAR_AXIS) is None

    def test_unparseable_period(self):
        """An attribute that is not <unit>_<count> is not guessed at."""
        assert period_in_axis_units(_dataset("annual"), YEAR_AXIS) is None

    def test_unknown_unit(self):
        """A unit not in the table is refused rather than assumed."""
        assert period_in_axis_units(_dataset("fortnight_1"), YEAR_AXIS) is None

    def test_empty_units_string(self):
        """An axis with no units cannot be converted to."""
        assert period_in_axis_units(_dataset("year_1"), "") is None
