"""Tests for CMOR driver result handling."""

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
# pylint: disable=wrong-import-position
from cmor_driver import (
    _attempted_variable_count,
    _cmor_exit_code,
    _job_name,
    _job_variable_name,
)
from cmip7_prep.mapping_ownership import MappingOwnership
from cmip7_prep.step_exit import NOTHING_TO_WRITE
from cmip7_prep.variable_ref import VariableRef


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


def test_year_windows_count_as_one_attempted_variable():
    """Multiple failed jobs for one variable count once in the summary."""
    variable = VariableRef.from_data_request(
        SimpleNamespace(branded_variable_name="tas_tavg-u-hxy-u")
    )
    jobs = [
        {"cmip_var": variable, "window": (1, 1)},
        {"cmip_var": variable, "window": (2, 2)},
    ]

    assert _attempted_variable_count([job["cmip_var"] for job in jobs]) == 1
    assert _job_variable_name(variable) == "tas_tavg-u-hxy-u"
    assert _job_name(jobs[0]) == "tas_tavg-u-hxy-u years 1-1"
    assert _job_name(jobs[1]) == "tas_tavg-u-hxy-u years 2-2"
