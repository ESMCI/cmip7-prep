"""The vYYYYMMDD data version a run writes under."""

import pytest

from cmip7_prep.data_version import checked_data_version


def test_a_dated_version_is_returned_unchanged():
    """A well-formed version passes through."""
    assert checked_data_version("v20200101") == "v20200101"


def test_surrounding_space_is_stripped():
    """A value copied from a file still works."""
    assert checked_data_version(" v20200101\n") == "v20200101"


@pytest.mark.parametrize("value", ["20261102", "v2026112", "v2026-11-02", "latest"])
def test_a_version_that_is_not_vyyyymmdd_is_refused(value):
    """The message names the form it wants."""
    with pytest.raises(ValueError, match="vYYYYMMDD"):
        checked_data_version(value)


def test_a_version_that_is_not_a_date_is_refused():
    """Eight digits are not enough; they have to be a date."""
    with pytest.raises(ValueError, match="not a date"):
        checked_data_version("v20261332")
