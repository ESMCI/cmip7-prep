"""Tests for reading the variable and frequency out of a CMOR log filename."""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
# pylint: disable=wrong-import-position
from validate_cmor_output import parse_log_name, parse_log_variable


class TestParseLogVariable:
    """Tests for reading a CMOR log name."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("cmor_siextent_tavg-u-hm-u_nh_day_1-50.log", "siextent_tavg-u-hm-u"),
            ("cmor_ta_tavg-p19-hxy-air_glb_mon_11-20.log", "ta_tavg-p19-hxy-air"),
            # This name has no window, because the run was given no --years.
            ("cmor_tas_tavg-h2m-hxy-u_glb_3hr.log", "tas_tavg-h2m-hxy-u"),
        ],
    )
    def test_the_variable_is_read(self, name, expected):
        """The name gives the branded variable name."""
        assert parse_log_variable(Path(name)) == expected

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("cmor_siextent_tavg-u-hm-u_nh_day_1-50.log", "day"),
            ("cmor_ta_tavg-p19-hxy-air_glb_mon_11-20.log", "mon"),
            ("cmor_tas_tavg-h2m-hxy-u_glb_3hr.log", "3hr"),
            ("cmor_siextent_tavg-u-hm-u_nh_1hr_1-50.log", "1hr"),
        ],
    )
    def test_the_frequency_is_read(self, name, expected):
        """A variable written at two frequencies has a log for each."""
        assert parse_log_name(Path(name))[1] == expected

    @pytest.mark.parametrize(
        "name",
        [
            "something_else.log",
            "cmor.log",
            "run.log",
            "cmor-atmos-day.log",
            # This name records no frequency.
            "cmor_tas_tavg-h2m-hxy-u_glb.log",
            # This name records the time instead.
            "cmor_20261022T225943Z_tas.log",
        ],
    )
    def test_other_files_are_not_mistaken_for_cmor_logs(self, name):
        """The log directory holds the run's own logs as well."""
        assert parse_log_variable(Path(name)) is None
        assert parse_log_name(Path(name)) is None
