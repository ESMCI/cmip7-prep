"""Tests for the diagnostics convert_csv_to_yaml.py emits about bad rows.

The converter never drops a suspect row; it writes it and reports the problem on
stderr.  These tests cover what it reports and how the offending row is
identified -- check_entry's problem list, the duplicate-collapse warning, and the
spreadsheet row number attached to both.
"""

import os
import sys

# Allow importing the script directly from the scripts/ directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
# pylint: disable=wrong-import-position
from convert_csv_to_yaml import MODEL_CONFIGS, _group_entries, check_entry, read_csv

from tests.csv_helpers import (
    CESM_FIELDNAMES,
    write_temp_csv as _write_temp_csv,
)

# ── check_entry ───────────────────────────────────────────────────────────────


class TestCheckEntry:
    """Tests for check_entry()."""

    def test_valid_formula_is_silent(self):
        """A formula using only available functions produces no problems."""
        assert (
            check_entry("v", {"formula": "verticalsum(SOILICE, capped_at=5000)"}) == []
        )

    def test_undefined_function_is_flagged(self):
        """A syntactically valid call to a missing function is reported."""
        problems = check_entry("v", {"formula": "chunits(QICE, units='kg m-2 s-1')"})
        assert problems == ["v: formula calls undefined function 'chunits'"]

    def test_each_undefined_function_reported_once(self):
        """Repeated calls to the same missing function collapse to one problem."""
        problems = check_entry("v", {"formula": "chunits(A) + chunits(B)"})
        assert problems == ["v: formula calls undefined function 'chunits'"]

    def test_method_calls_are_not_flagged(self):
        """DataArray method calls have no resolvable name and are left alone."""
        assert (
            check_entry("v", {"formula": "(a * b).where(a > 0).sum(dim=['nj'])"}) == []
        )

    def test_syntax_error_is_flagged(self):
        """Prose in the formula column is reported as an invalid expression."""
        problems = check_entry("v", {"formula": "ask for max in history"})
        assert problems == [
            "v: formula 'ask for max in history' is not a valid expression"
        ]

    def test_cam_history_notation_is_flagged(self):
        """CAM history-field selectors are not arithmetic."""
        problems = check_entry("v", {"formula": "PBLH:X"})
        assert "CAM history-field notation" in problems[0]

    def test_plain_source_list_is_silent(self):
        """A comma-separated list of identifiers is a well-formed source."""
        assert check_entry("v", {}, raw_source="TGCLDLWP, TGCLDIWP") == []

    def test_annotated_source_is_flagged(self):
        """A source carrying an annotation is not a plain variable list."""
        problems = check_entry("v", {}, raw_source="IWPMODIS [COSP]")
        assert problems == [
            "v: source 'IWPMODIS [COSP]' is not a plain list of variable names"
        ]


# ── _group_entries ────────────────────────────────────────────────────────────


class TestGroupEntriesDuplicates:
    """Tests for the duplicate-collapse warning in _group_entries().

    A non-seaIce variable with several CSV rows keeps only the first.  That is
    sometimes right (an exact re-entry) and sometimes real data loss (two rows
    offering different source models, as landIce ``acabf_tavg-u-hxy-is`` does),
    so the collapse must never happen in silence.

    Entries are ``(name, entry, row)`` triples; *row* is the spreadsheet row the
    entry came from.  Both rows carry the same variable name, so the row number
    is the only thing that tells the two apart.
    """

    def test_single_entry_is_silent(self, capsys):
        """One row per name is the normal case and warns about nothing."""
        data = _group_entries(
            [("v", {"table": "land", "sources": [{"model_var": "A"}]}, 2)]
        )
        assert set(data) == {"v"}
        assert capsys.readouterr().err == ""

    def test_seaice_variants_are_not_duplicates(self, capsys):
        """seaIce rows are merged into variants, which is not a discard."""
        entries = [
            ("si", {"table": "seaIce", "region": "nh", "long_name": "N"}, 2),
            ("si", {"table": "seaIce", "region": "sh", "long_name": "S"}, 3),
        ]
        data = _group_entries(entries)
        assert len(data["si"]["variants"]) == 2
        assert capsys.readouterr().err == ""

    def test_discarded_duplicate_warns(self, capsys):
        """A collapsed non-seaIce duplicate names the variable on stderr."""
        entries = [
            ("acabf", {"table": "landIce", "sources": [{"model_var": "QICE"}]}, 412),
            (
                "acabf",
                {"table": "landIce", "sources": [{"model_var": "acab_applied"}]},
                858,
            ),
        ]
        _group_entries(entries)
        err = capsys.readouterr().err
        assert "WARN acabf (row 858): duplicate row discarded" in err

    def test_warning_names_both_rows(self, capsys):
        """The row kept is named too -- the variable name matches both rows."""
        entries = [
            ("acabf", {"table": "landIce", "sources": [{"model_var": "QICE"}]}, 412),
            (
                "acabf",
                {"table": "landIce", "sources": [{"model_var": "acab_applied"}]},
                858,
            ),
        ]
        _group_entries(entries)
        err = capsys.readouterr().err
        assert "row 858" in err and "keeping row 412" in err

    def test_warning_reports_differing_fields(self, capsys):
        """The warning shows what was lost, not just that something was."""
        entries = [
            (
                "acabf",
                {
                    "table": "landIce",
                    "formula": 'chunits(QICE, units="kg m-2 s-1")',
                    "sources": [{"model_var": "QICE"}],
                },
                412,
            ),
            (
                "acabf",
                {"table": "landIce", "sources": [{"model_var": "acab_applied"}]},
                858,
            ),
        ]
        _group_entries(entries)
        err = capsys.readouterr().err
        assert "sources" in err
        assert "QICE" in err and "acab_applied" in err
        # A field present on the kept row but absent from the dropped one is
        # still a difference worth reporting.
        assert "formula" in err

    def test_identical_duplicate_is_called_out_as_harmless(self, capsys):
        """An exact re-entry still warns, but says no fields differ."""
        entry = {"table": "land", "sources": [{"model_var": "A"}]}
        _group_entries([("v", dict(entry), 2), ("v", dict(entry), 9)])
        err = capsys.readouterr().err
        assert "WARN v (row 9): duplicate row discarded" in err
        assert "identical" in err

    def test_collapsed_rows_are_collected(self, capsys):
        """The optional collector receives one record per discarded row."""
        collapsed = []
        entries = [
            ("v", {"table": "land", "sources": [{"model_var": "A"}]}, 2),
            ("v", {"table": "land", "sources": [{"model_var": "B"}]}, 3),
            ("v", {"table": "land", "sources": [{"model_var": "C"}]}, 4),
        ]
        _group_entries(entries, collapsed=collapsed)
        assert collapsed == ["v", "v"]
        capsys.readouterr()

    def test_first_entry_still_wins(self, capsys):
        """Warning is additive: the collapse behaviour itself is unchanged."""
        entries = [
            ("v", {"table": "land", "sources": [{"model_var": "A"}]}, 2),
            ("v", {"table": "land", "sources": [{"model_var": "B"}]}, 3),
        ]
        data = _group_entries(entries)
        assert data["v"]["sources"] == [{"model_var": "A"}]
        capsys.readouterr()


# ── row numbers in warnings ───────────────────────────────────────────────────


class TestCheckEntryRowNumbers:
    """check_entry() prefixes the spreadsheet row when given one."""

    def test_row_is_included_when_given(self):
        """The row number lands between the name and the problem."""
        problems = check_entry("v", {"formula": "chunits(X)"}, row=858)
        assert problems == ["v (row 858): formula calls undefined function 'chunits'"]

    def test_row_is_omitted_when_not_given(self):
        """Without a row the message keeps its original shape."""
        problems = check_entry("v", {"formula": "chunits(X)"})
        assert problems == ["v: formula calls undefined function 'chunits'"]

    def test_every_problem_on_a_row_is_labelled(self):
        """A row tripping several checks gets the row on each line."""
        problems = check_entry(
            "v", {"formula": "chunits(X)"}, raw_source="IWPMODIS [COSP]", row=7
        )
        assert len(problems) == 2
        assert all(p.startswith("v (row 7): ") for p in problems)


class TestReadCsvRowNumbers:
    """read_csv() reports the spreadsheet row a flagged entry came from."""

    FIELDNAMES = CESM_FIELDNAMES
    CFG = MODEL_CONFIGS["cesm"]

    def _row(self, **kwargs):
        base = {f: "" for f in self.FIELDNAMES}
        base.update(kwargs)
        return base

    def _bad(self, name, formula="chunits(X)"):
        return self._row(
            **{
                "CMIP Branded Variable Name": name,
                "Table": "atmos",
                "Dimensions": "time, lat, lon",
                "CESM Variable Name": "X",
                "Formula": formula,
            }
        )

    def test_first_data_row_is_row_2(self, tmp_path, capsys):
        """The header is row 1, so the first record is row 2 -- as in a sheet."""
        path = _write_temp_csv(tmp_path, self.FIELDNAMES, [self._bad("a")])
        read_csv(path, self.CFG)
        assert "WARN a (row 2):" in capsys.readouterr().err

    def test_rows_count_records_not_file_lines(self, tmp_path, capsys):
        """A cell containing a newline must not shift later row numbers."""
        rows = [
            self._bad("a"),  # row 2
            self._bad("b", formula="chunits(X)\nsecond physical line"),  # row 3
            self._bad("c"),  # row 4, but file line 5
        ]
        path = _write_temp_csv(tmp_path, self.FIELDNAMES, rows)
        read_csv(path, self.CFG)
        err = capsys.readouterr().err
        assert "WARN a (row 2):" in err
        assert "WARN c (row 4):" in err
        assert "row 5" not in err

    def test_skipped_rows_still_advance_the_count(self, tmp_path, capsys):
        """Rows dropped by should_keep must not renumber the rows after them."""
        rows = [
            self._row(
                **{
                    "CMIP Branded Variable Name": "skipped",
                    "Table": "atmos",
                    "CESM Variable Name": "N/A",
                }
            ),  # row 2, dropped by source_skip_phrases
            self._bad("a"),  # row 3
        ]
        path = _write_temp_csv(tmp_path, self.FIELDNAMES, rows)
        read_csv(path, self.CFG)
        assert "WARN a (row 3):" in capsys.readouterr().err
