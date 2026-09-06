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
        data = _group_entries(
            [("v", a, 2), ("v", b, 3)], collapsed=collapsed
        )
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
                    "CMIP Branded Variable Name": "siconc",
                    "Table": "seaIce",
                    "CESM Variable Name": "siconc",
                    "Freq": "mon",
                }
            ),
            self._row(
                **{
                    "CMIP Branded Variable Name": "siconc",
                    "Table": "seaIce",
                    "CESM Variable Name": "siconc_d",
                    "Freq": "day",
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
