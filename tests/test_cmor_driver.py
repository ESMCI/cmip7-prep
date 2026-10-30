"""Tests for CMOR driver result handling."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
# pylint: disable=wrong-import-position
from cmor_driver import _cmor_exit_code
from cmip7_prep.mapping_ownership import MappingOwnership
from cmip7_prep.step_exit import NOTHING_TO_WRITE


@pytest.mark.parametrize(
    "ownership,expected",
    [
        (MappingOwnership(set(), {}, {"unmapped"}), NOTHING_TO_WRITE),
        (MappingOwnership(set(), {"elsewhere": ("land",)}, set()), 0),
        (
            MappingOwnership(set(), {"elsewhere": ("land",)}, {"unmapped"}),
            NOTHING_TO_WRITE,
        ),
    ],
)
def test_no_owned_variables_are_not_treated_as_write_failures(ownership, expected):
    """Unmapped requests return no-work unless every request belongs elsewhere."""
    assert _cmor_exit_code([], [], ownership) == expected


def test_owned_variable_with_no_output_still_fails():
    """An eligible variable that was attempted but produced nothing is fatal."""
    ownership = MappingOwnership({"owned"}, {}, {"unmapped"})
    assert _cmor_exit_code([object()], [], ownership) == 1


def test_any_written_variable_succeeds():
    """A successful write keeps the run successful despite other diagnostics."""
    ownership = MappingOwnership({"written"}, {}, {"unmapped"})
    assert _cmor_exit_code([object()], ["written"], ownership) == 0
