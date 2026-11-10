"""Tests for the iceband axis built from CICE's ice thickness categories.

siitdconc, siitdsnconc, siitdsnthick and siitdthick are written per ice
thickness category, which the CMOR tables call iceband.  CICE writes them on
the dimension nc and records the categories in NCAT, as issue #131 reports.
"""

import numpy as np

from cmip7_prep.cmor_writer import iceband_values_and_bounds

# NCAT from a NorESM cice.h1 file: the maximum thickness of each of the five
# categories, the last of which CICE leaves open.
NCAT = [0.6445072, 1.391433, 2.470179, 4.567288, 1e08]


def test_bounds_start_at_zero_and_end_at_the_categories():
    """The cell edges are 0 followed by each category's maximum."""
    _, bounds = iceband_values_and_bounds(NCAT)

    assert bounds.shape == (5, 2)
    assert bounds[0, 0] == 0.0
    np.testing.assert_allclose(bounds[:, 1], NCAT)


def test_bounds_join_with_no_gap():
    """Each category begins where the one below it ends."""
    _, bounds = iceband_values_and_bounds(NCAT)

    np.testing.assert_array_equal(bounds[1:, 0], bounds[:-1, 1])


def test_a_value_is_the_middle_of_its_bounds():
    """A closed category takes the middle of its bounds."""
    values, bounds = iceband_values_and_bounds(NCAT)

    np.testing.assert_allclose(values[:4], bounds[:4].mean(axis=1))


def test_the_open_category_takes_its_lower_bound():
    """The open category's middle is 5e+07, so its lower bound is used."""
    values, bounds = iceband_values_and_bounds(NCAT)

    assert values[4] == bounds[4, 0]


def test_the_values_increase_and_lie_within_their_bounds():
    """CMOR requires both of these of an axis."""
    values, bounds = iceband_values_and_bounds(NCAT)

    assert np.all(np.diff(values) > 0)
    assert np.all(values >= bounds[:, 0])
    assert np.all(values <= bounds[:, 1])
