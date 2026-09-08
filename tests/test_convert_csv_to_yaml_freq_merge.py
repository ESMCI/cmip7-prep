"""Tests for the per-frequency row merging in convert_csv_to_yaml.py.

Rows that share a branded name but differ only in output frequency are merged
into one entry carrying per-source ``freq`` tags, instead of being discarded as
duplicates.  These tests pin that behaviour (and its guards) so the grouping
code can be refactored safely.

The parsing-helper unit tests live in ``test_convert_csv_to_yaml``, the
``read_csv`` integration tests in ``test_convert_csv_to_yaml_read_csv``, and the
warning tests in ``test_convert_csv_to_yaml_diagnostics``.
"""

import os
import sys

# Allow importing the script directly from the scripts/ directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
# pylint: disable=wrong-import-position
from convert_csv_to_yaml import (
    MODEL_CONFIGS,
    _freq_mergeable,
    _group_entries,
    _union_sources,
    read_csv,
)

from tests.csv_helpers import CESM_FIELDNAMES, write_temp_csv as _write_temp_csv


def _entry(sources, **fields):
    """Build a minimal entry dict with the given sources and extra fields."""
    return {"sources": list(sources), **fields}


class TestUnionSources:
    """Tests for _union_sources()."""

    def test_dedups_by_key_preserving_order(self):
        """Identical (model_var, freq, alias) sources collapse; order is kept."""
        a = _entry([{"model_var": "tarea"}, {"model_var": "x", "freq": "mon"}])
        b = _entry([{"model_var": "tarea"}, {"model_var": "y", "freq": "day"}])
        assert _union_sources([a, b]) == [
            {"model_var": "tarea"},
            {"model_var": "x", "freq": "mon"},
            {"model_var": "y", "freq": "day"},
        ]

    def test_same_var_different_freq_kept_separate(self):
        """The same model_var at two freqs is two distinct sources."""
        a = _entry([{"model_var": "siconc", "freq": "mon"}])
        b = _entry([{"model_var": "siconc", "freq": "day"}])
        assert _union_sources([a, b]) == [
            {"model_var": "siconc", "freq": "mon"},
            {"model_var": "siconc", "freq": "day"},
        ]


class TestFreqMergeable:
    """Tests for _freq_mergeable()."""

    def test_mergeable_when_only_sources_differ(self):
        """Same non-source fields + freq-tagged sources are mergeable."""
        a = _entry([{"model_var": "x", "freq": "mon"}], table="seaIce")
        b = _entry([{"model_var": "y", "freq": "day"}], table="seaIce")
        assert _freq_mergeable([a, b], _union_sources([a, b])) is True

    def test_ignorable_freetext_may_differ(self):
        """description/long_name may differ without blocking a merge."""
        a = _entry([{"model_var": "x", "freq": "mon"}], description="foo")
        b = _entry([{"model_var": "y", "freq": "day"}], description="bar")
        assert _freq_mergeable([a, b], _union_sources([a, b])) is True

    def test_blocked_when_real_field_differs(self):
        """A differing non-ignorable field (units) blocks the merge."""
        a = _entry([{"model_var": "x", "freq": "mon"}], units="K")
        b = _entry([{"model_var": "y", "freq": "day"}], units="degC")
        assert _freq_mergeable([a, b], _union_sources([a, b])) is False

    def test_shared_untagged_source_allowed(self):
        """A source present in every row (e.g. tarea) may stay untagged."""
        a = _entry([{"model_var": "tarea"}, {"model_var": "x", "freq": "mon"}])
        b = _entry([{"model_var": "tarea"}, {"model_var": "y", "freq": "day"}])
        assert _freq_mergeable([a, b], _union_sources([a, b])) is True

    def test_nonshared_untagged_source_blocks(self):
        """A source in only one row without a freq tag blocks the merge.

        Otherwise the pipeline could not tell which frequency to pull it at.
        """
        a = _entry([{"model_var": "x"}])
        b = _entry([{"model_var": "y"}])
        assert _freq_mergeable([a, b], _union_sources([a, b])) is False


class TestGroupEntriesFreqMerge:
    """Tests for _group_entries()'s merge/discard decision."""

    def test_freq_rows_merged(self):
        """Two freq rows for one name merge into one freq-tagged entry."""
        a = _entry([{"model_var": "siconc", "freq": "mon"}], table="seaIce")
        b = _entry([{"model_var": "siconc_d", "freq": "day"}], table="seaIce")
        data = _group_entries([("siconc", a, 2), ("siconc", b, 3)])
        assert data["siconc"]["sources"] == [
            {"model_var": "siconc", "freq": "mon"},
            {"model_var": "siconc_d", "freq": "day"},
        ]

    def test_true_duplicate_discarded_not_merged(self):
        """Identical rows are discarded (union does not grow), and reported."""
        a = _entry([{"model_var": "x", "freq": "mon"}], table="atmos")
        b = _entry([{"model_var": "x", "freq": "mon"}], table="atmos")
        collapsed = []
        data = _group_entries([("v", a, 2), ("v", b, 3)], collapsed=collapsed)
        assert data["v"]["sources"] == [{"model_var": "x", "freq": "mon"}]
        assert collapsed == ["v"]

    def test_unmergeable_rows_discarded(self):
        """Rows that differ in a real field are discarded, not merged."""
        a = _entry([{"model_var": "x", "freq": "mon"}], table="atmos", units="K")
        b = _entry([{"model_var": "y", "freq": "day"}], table="atmos", units="degC")
        collapsed = []
        data = _group_entries([("v", a, 2), ("v", b, 3)], collapsed=collapsed)
        assert data["v"]["units"] == "K"  # first row kept
        assert collapsed == ["v"]


class TestFreqMergeIntegration:
    """End-to-end: two CSV rows for one name at different Freq merge."""

    def _row(self, **kwargs):
        base = {f: "" for f in CESM_FIELDNAMES}
        base.update(kwargs)
        return base

    def test_two_freq_rows_merge_via_read_csv(self, tmp_path):
        """A monthly and a daily row for one branded name merge into one entry."""
        rows = [
            self._row(
                **{
                    "Branded Variable Name": "siconc",
                    "Modelling Realm - Primary": "seaIce",
                    "CESM Variable Name": "siconc",
                    "CMIP7 Frequency": "mon",
                }
            ),
            self._row(
                **{
                    "Branded Variable Name": "siconc",
                    "Modelling Realm - Primary": "seaIce",
                    "CESM Variable Name": "siconc_d",
                    "CMIP7 Frequency": "day",
                }
            ),
        ]
        data = read_csv(
            _write_temp_csv(tmp_path, CESM_FIELDNAMES, rows), MODEL_CONFIGS["cesm"]
        )
        var = data["seaIce"]["variables"]["siconc"]
        assert var["sources"] == [
            {"model_var": "siconc", "freq": "mon"},
            {"model_var": "siconc_d", "freq": "day"},
        ]

    def _region_freq_rows(self, table):
        """Four rows for one name: two regions (ata, grl) x two frequencies."""
        return [
            self._row(
                **{
                    "Branded Variable Name": "v",
                    "Modelling Realm - Primary": table,
                    "Units (from Physical Parameter)": "W m-2",
                    "CESM Variable Name": var,
                    "CMIP7 Frequency": freq,
                    "Region": region,
                }
            )
            for region, var, freq in (
                ("ata", "A_mon", "mon"),
                ("ata", "A_day", "day"),
                ("grl", "G_mon", "mon"),
                ("grl", "G_day", "day"),
            )
        ]

    def _read(self, tmp_path, table):
        """Run the four region x frequency rows through read_csv."""
        data = read_csv(
            _write_temp_csv(tmp_path, CESM_FIELDNAMES, self._region_freq_rows(table)),
            MODEL_CONFIGS["cesm"],
        )
        return data[table]["variables"]["v"]

    def test_region_and_freq_non_seaice_keeps_one_region(self, tmp_path, capsys):
        """Frequencies merge within a region; then all but the first region go.

        ``_group_entries`` buckets by region first and freq-merges inside each
        bucket, so both regions do merge -- and then the non-seaIce branch keeps
        ``region_entries[0]`` and discards the rest.  The Antarctic pair
        survives as one freq-tagged entry and the Greenland pair is dropped
        outright.

        This pins today's lossy behaviour, not the desired one: on the real
        spreadsheet it is how 51 ``grl`` rows vanish (TASKS.md D14).  When D14
        lands this test should fail, and the new expectation belongs here.
        """
        var = self._read(tmp_path, "landIce")

        assert var["region"] == "ata"
        assert var["sources"] == [
            {"model_var": "A_mon", "freq": "mon"},
            {"model_var": "A_day", "freq": "day"},
        ]
        assert "G_mon" not in str(var) and "G_day" not in str(var)

        err = capsys.readouterr().err
        assert "duplicate row discarded" in err
        assert "region: kept 'ata' / discarded 'grl'" in err

    def test_region_and_freq_seaice_keeps_regions_but_not_their_sources(
        self, tmp_path, capsys
    ):
        """seaIce turns regions into variants -- carrying only region, not sources.

        The variant machinery copies ``_VARIANT_FIELDS`` (long_name, formula,
        region) into each variant and takes everything else, ``sources``
        included, from the first region.  So both regions are named, but only
        the Antarctic sources survive, and unlike the non-seaIce path this
        collapse emits no WARN.
        """
        var = self._read(tmp_path, "seaIce")

        assert var["variants"] == [{"region": "ata"}, {"region": "grl"}]
        assert var["sources"] == [
            {"model_var": "A_mon", "freq": "mon"},
            {"model_var": "A_day", "freq": "day"},
        ]
        assert "G_mon" not in str(var) and "G_day" not in str(var)
        assert "duplicate row discarded" not in capsys.readouterr().err
