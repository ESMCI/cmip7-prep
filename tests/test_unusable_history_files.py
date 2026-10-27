"""Tests for dropping history files GenTS cannot derive a time step from."""

import logging
import os
import sys

import numpy as np
import pytest
import xarray as xr
from gents.hfcollection import HFCollection

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
# pylint: disable=wrong-import-position
from gen_timeseries import find_unusable_files

LOGGER = logging.getLogger("test")
CASE = "n1850esmGHGGaxg.LM.nor30b27.001.20261016"
OTHER_CASE = "n1850esmGaxgGHG.LM.n30b25.529.20261002"


def write_history_file(directory, case, stamp, records, stream="cam.h0a"):
    """Write one history file holding the given number of time records."""
    path = directory / f"{case}.{stream}.{stamp}.nc"
    dataset = xr.Dataset(
        {"TREFHT": ("time", np.zeros(records, dtype="f4"))},
        coords={"time": np.arange(records, dtype="f8")},
    )
    dataset["time"].attrs = {"units": "days since 0001-01-01", "calendar": "noleap"}
    dataset.to_netcdf(path)
    return str(path)


class TestFindUnusableFiles:
    """Which files are kept back from GenTS, and which are not."""

    def test_a_whole_month_series_is_usable(self, tmp_path):
        """Twelve files of one record each give eleven gaps to measure."""
        for month in range(1, 13):
            write_history_file(tmp_path, CASE, f"0001-{month:02d}", 1)
        assert not find_unusable_files(tmp_path, ["*cam.h0a*"], 1, LOGGER)

    def test_one_file_from_another_case_is_dropped(self, tmp_path):
        """It groups on its own name, so its single record is all it has."""
        for month in range(1, 13):
            write_history_file(tmp_path, CASE, f"0001-{month:02d}", 1)
        stray = write_history_file(tmp_path, OTHER_CASE, "1700-12", 1)

        assert find_unusable_files(tmp_path, ["*cam.h0a*"], 1, LOGGER) == [stray]

    def test_a_file_from_another_case_with_two_records_is_kept(self, tmp_path):
        """Two records in one file are two time values, which is enough."""
        write_history_file(tmp_path, CASE, "0001-01", 1)
        write_history_file(tmp_path, CASE, "0001-02", 1)
        write_history_file(tmp_path, OTHER_CASE, "1700-12", 2)

        assert not find_unusable_files(tmp_path, ["*cam.h0a*"], 1, LOGGER)

    def test_a_file_with_no_time_records_is_dropped(self, tmp_path):
        """A stream opened by a segment that ended before any sample reached it."""
        for month in range(1, 13):
            write_history_file(tmp_path, CASE, f"0001-{month:02d}", 1)
        empty = write_history_file(tmp_path, CASE, "0002-01", 0)

        assert find_unusable_files(tmp_path, ["*cam.h0a*"], 1, LOGGER) == [empty]

    def test_an_empty_directory_has_nothing_to_drop(self, tmp_path):
        """Nothing matched, so there is nothing to keep back."""
        assert not find_unusable_files(tmp_path, ["*cam.h0a*"], 1, LOGGER)

    def test_a_stray_file_in_a_subdirectory_is_dropped(self, tmp_path):
        """GenTS walks the whole tree, so a subdirectory is not out of reach."""
        for month in range(1, 13):
            write_history_file(tmp_path, CASE, f"0001-{month:02d}", 1)
        misc = tmp_path / "misc_files"
        misc.mkdir()
        stray = write_history_file(misc, OTHER_CASE, "1700-12", 1)

        assert find_unusable_files(tmp_path, ["*cam.h0a*"], 1, LOGGER) == [stray]


class TestTheWholeChain:
    """What GenTS does with the files, once the unusable ones are held back."""

    @staticmethod
    def collection(root, unusable):
        """One HFCollection narrowed the way gen_timeseries.py narrows it."""
        collection = HFCollection(root, num_processes=1)
        if unusable:
            collection = collection.exclude(unusable)
        return collection.include(["*cam.h0a*"])

    def test_a_stray_file_breaks_gents_when_it_is_not_held_back(self, tmp_path):
        """The group it forms has one time value, and no gap to measure."""
        for month in range(1, 13):
            write_history_file(tmp_path, CASE, f"0001-{month:02d}", 1)
        misc = tmp_path / "misc_files"
        misc.mkdir()
        write_history_file(misc, OTHER_CASE, "1700-12", 1)

        with pytest.raises(ValueError, match="size 2 or greater"):
            self.collection(tmp_path, []).include_years(1, 10)

    def test_holding_it_back_leaves_the_rest_usable(self, tmp_path):
        """GenTS still walks the subdirectory; the file is simply not there to
        read by the time it does."""
        for month in range(1, 13):
            write_history_file(tmp_path, CASE, f"0001-{month:02d}", 1)
        misc = tmp_path / "misc_files"
        misc.mkdir()
        write_history_file(misc, OTHER_CASE, "1700-12", 1)

        unusable = find_unusable_files(tmp_path, ["*cam.h0a*"], 1, LOGGER)
        kept = self.collection(tmp_path, unusable).include_years(1, 10)

        assert len(list(kept)) == 12
