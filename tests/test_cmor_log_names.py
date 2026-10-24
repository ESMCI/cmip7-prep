"""Tests for reading the variable out of a CMOR log filename."""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
# pylint: disable=wrong-import-position
from validate_cmor_output import parse_log_variable


class TestParseLogVariable:
    """Tests for both forms of CMOR log name."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            # Named after the variable, region and window, as written now.
            ("cmor_siextent_tavg-u-hm-u_nh_1-50.log", "siextent_tavg-u-hm-u"),
            ("cmor_siextent_tavg-u-hm-u_sh_41-50.log", "siextent_tavg-u-hm-u"),
            ("cmor_ta_tavg-p19-hxy-air_glb_11-20.log", "ta_tavg-p19-hxy-air"),
            # No window, for a run given no --years.
            ("cmor_tas_tavg-h2m-hxy-u_glb.log", "tas_tavg-h2m-hxy-u"),
            # Timestamped, as earlier versions wrote.
            (
                "cmor_20261022T225943Z_siextent_tavg-u-hm-u.log",
                "siextent_tavg-u-hm-u",
            ),
        ],
    )
    def test_the_variable_is_read(self, name, expected):
        """Both naming schemes give the branded variable name."""
        assert parse_log_variable(Path(name)) == expected

    @pytest.mark.parametrize(
        "name",
        [
            "something_else.log",
            "cmor.log",
            "run.log",
            "cmor-atmos-day.log",
        ],
    )
    def test_other_files_are_not_mistaken_for_cmor_logs(self, name):
        """The log directory holds the run's own logs as well."""
        assert parse_log_variable(Path(name)) is None
