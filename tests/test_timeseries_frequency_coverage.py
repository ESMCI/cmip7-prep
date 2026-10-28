"""Tests for counting a time series step's output per frequency."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
# pylint: disable=wrong-import-position
from gen_timeseries import timeseries_per_frequency


def touch_stream(outputdir, stream, variable="TREFHT"):
    """Write an empty time series file for one history stream."""
    path = outputdir / f"case.{stream}.{variable}.000101-001012.nc"
    path.touch()
    return path


class TestTimeseriesPerFrequency:
    """How many files each requested frequency produced."""

    def test_counts_the_files_of_one_frequency(self, tmp_path):
        """Two monthly sea ice files are counted as two."""
        touch_stream(tmp_path, "cice.h", "siconc")
        touch_stream(tmp_path, "cice.h", "sithick")
        assert timeseries_per_frequency(
            tmp_path, "noresm", "seaIce", None, ["mon"]
        ) == {"mon": 2}

    def test_a_frequency_with_no_output_counts_zero(self, tmp_path):
        """Daily is zero when only the monthly stream produced files."""
        touch_stream(tmp_path, "cice.h")
        assert timeseries_per_frequency(
            tmp_path, "noresm", "seaIce", None, ["mon", "day"]
        ) == {"mon": 1, "day": 0}

    def test_one_of_a_frequency_s_streams_is_enough(self, tmp_path):
        """Land monthly is covered by clm2.h0a alone, without clm2.h0i."""
        touch_stream(tmp_path, "clm2.h0a")
        assert timeseries_per_frequency(tmp_path, "noresm", "land", None, ["mon"]) == {
            "mon": 1
        }

    def test_an_empty_directory_counts_zero_everywhere(self, tmp_path):
        """Nothing produced means nothing is covered."""
        assert timeseries_per_frequency(
            tmp_path, "noresm", "seaIce", None, ["mon", "day"]
        ) == {"mon": 0, "day": 0}

    def test_output_for_another_realm_does_not_count(self, tmp_path):
        """Atmosphere files do not satisfy a sea ice request."""
        touch_stream(tmp_path, "cam.h0a")
        assert timeseries_per_frequency(
            tmp_path, "noresm", "seaIce", None, ["mon"]
        ) == {"mon": 0}
