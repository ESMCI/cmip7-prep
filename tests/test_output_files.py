"""Tests for a CMOR file's period and the files it makes redundant."""

from pathlib import Path

from cmip7_prep.output_files import (
    covers,
    file_period,
    redundant_pairs,
    remove_superseded,
    superseded_by,
)

STEM = "siextent_SImon_NorESM3-LM_n1850_r1i1p1f1_gn"
AREA = "areacella_fx_NorESM3-LM_n1850_r1i1p1f1_gn.nc"


def name(first: str, last: str) -> str:
    """A CMOR filename covering the given period."""
    return f"{STEM}_{first}-{last}.nc"


class TestFilePeriod:
    """Reading the period out of a filename."""

    def test_reads_daily_dates(self):
        """A daily file's dates are eight digits each."""
        assert file_period(Path("/out") / name("00010101", "00101231")) == (
            "00010101",
            "00101231",
        )

    def test_reads_monthly_dates(self):
        """A monthly file's dates are six digits each."""
        assert file_period(name("000101", "001012")) == ("000101", "001012")

    def test_is_none_for_a_file_covering_no_time(self):
        """An fx file's name ends in the grid, with no period."""
        assert file_period(AREA) is None

    def test_is_none_when_the_dates_differ_in_width(self):
        """Dates of different widths cannot be a period."""
        assert file_period(name("0001", "00101231")) is None


class TestCovers:
    """Comparing two periods."""

    def test_includes_an_inner_period(self):
        """A decade covers the first ten months of it."""
        assert covers(("00010101", "00101231"), ("00010101", "00011031"))

    def test_rejects_a_later_period(self):
        """The next decade is outside this one."""
        assert not covers(("00010101", "00101231"), ("00110101", "00201231"))

    def test_rejects_periods_of_different_width(self):
        """Annual and daily dates are not comparable."""
        assert not covers(("0001", "0010"), ("00010101", "00011031"))


class TestRedundantPairs:
    """Reporting a variable's files that repeat another's times."""

    def test_finds_a_partial_file(self):
        """The ten-month file is covered by the ten-year one."""
        full = Path("/out") / name("00010101", "00101231")
        partial = Path("/out") / name("00010101", "00011031")
        assert redundant_pairs([full, partial]) == [(partial, full)]

    def test_accepts_a_clean_series(self):
        """Consecutive decades and an fx file leave nothing to report."""
        paths = [
            Path("/out") / name("00010101", "00101231"),
            Path("/out") / name("00110101", "00201231"),
            Path("/out") / AREA,
        ]
        assert not redundant_pairs(paths)


class TestSupersededBy:
    """Finding what a newly written file replaces on disk."""

    def test_finds_the_partial_file_beside_it(self, tmp_path):
        """Only the file inside the written period is superseded."""
        full = tmp_path / name("00010101", "00101231")
        partial = tmp_path / name("00010101", "00011031")
        later = tmp_path / name("00110101", "00201231")
        for path in (full, partial, later):
            path.touch()
        assert superseded_by(full) == [partial]

    def test_is_empty_for_the_only_file(self, tmp_path):
        """A variable with one file supersedes nothing."""
        full = tmp_path / name("00010101", "00101231")
        full.touch()
        assert not superseded_by(full)


class TestRemoveSuperseded:
    """Deleting what a newly written file replaces."""

    def test_deletes_only_the_partial_file(self, tmp_path):
        """The written file and the next period are both left alone."""
        full = tmp_path / name("00010101", "00101231")
        partial = tmp_path / name("00010101", "00011031")
        later = tmp_path / name("00110101", "00201231")
        for path in (full, partial, later):
            path.touch()

        assert remove_superseded(full) == [partial]
        assert not partial.exists()
        assert full.exists()
        assert later.exists()

    def test_leaves_a_lone_file_alone(self, tmp_path):
        """Nothing is deleted when there is nothing redundant beside it."""
        full = tmp_path / name("00010101", "00101231")
        full.touch()

        assert not remove_superseded(full)
        assert full.exists()
