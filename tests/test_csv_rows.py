"""Tests for cmip7_prep.csv_rows.

Run with:
    pytest tests/test_csv_rows.py -v
or to also run the embedded doctests:
    pytest --doctest-modules src/cmip7_prep/csv_rows.py tests/test_csv_rows.py -v
"""

from cmip7_prep.csv_rows import (
    CESM_COLUMNS,
    sources_to_freq_alias,
    variable_to_rows,
)

# ── sources_to_freq_alias ───────────────────────────────────────────────


class TestSourcesToFreqAlias:
    """Tests for sources_to_freq_alias()."""

    def test_empty(self):
        """Empty sources list returns three empty strings."""
        assert sources_to_freq_alias([]) == ("", "")

    def test_single_no_extras(self):
        """A single source with no extra attrs returns three empty strings."""
        assert sources_to_freq_alias([{"model_var": "TREFHT"}]) == ("", "")

    def test_single_with_freq(self):
        """A single source with freq is reflected in the Freq string."""
        freq, alias = sources_to_freq_alias([{"model_var": "siconc", "freq": "day"}])
        assert freq == "day"
        assert alias == ""

    def test_two_sources_first_has_freq(self):
        """Positional alignment: first source has freq, second does not."""
        freq, alias = sources_to_freq_alias(
            [{"model_var": "siconc", "freq": "day"}, {"model_var": "tarea"}]
        )
        assert freq == "day, "
        assert alias == ""

    def test_two_sources_both_have_scale(self):
        """Both sources with identical scales."""
        freq, alias = sources_to_freq_alias([{"model_var": "A"}, {"model_var": "B"}])
        assert freq == ""
        assert alias == ""

    def test_all_three_attrs(self):
        """A source with all three extra attrs populates all three columns."""
        freq, alias = sources_to_freq_alias(
            [{"model_var": "siconc_d", "freq": "day", "alias": "siconc"}]
        )
        assert freq == "day"
        assert alias == "siconc"

    def test_mixed_attrs_positional(self):
        """Mixed attrs across three sources are aligned positionally."""
        sources = [
            {"model_var": "siconc_d", "freq": "day", "alias": "siconc"},
            {"model_var": "siconc", "freq": "mon"},
            {"model_var": "tarea"},
        ]
        freq, alias = sources_to_freq_alias(sources)
        assert freq == "day, mon, "
        assert alias == "siconc, , "


# ── variable_to_rows ──────────────────────────────────────────────────────────


class TestVariableToRow:
    """Tests for variable_to_rows()."""

    def test_basic_fields(self):
        """Core fields are mapped to the correct CSV columns."""
        row = variable_to_rows(
            "tas",
            {
                "table": "atmos",
                "long_name": "Near-Surface Air Temperature",
                "units": "K",
                "sources": [{"model_var": "TREFHT"}],
            },
        )
        assert len(row) == 1
        row = row[0]
        assert row["CMIP Variable Name"] == "tas"
        assert row["Table"] == "atmos"
        assert row["Long Name"] == "Near-Surface Air Temperature"
        assert row["Units"] == "K"
        assert row["CESM Variable Name"] == "TREFHT"

    def test_cesm_var_name_uses_source_names_not_formula(self):
        """CESM Variable Name is always source names, never the formula string."""
        row = variable_to_rows(
            "cl",
            {
                "table": "atmos",
                "units": "%",
                "formula": "CLOUD * 100",
                "sources": [{"model_var": "CLOUD"}],
            },
        )[0]
        assert row["CESM Variable Name"] == "CLOUD"
        assert row["Formula"] == "CLOUD * 100"

    def test_no_formula_uses_source_names(self):
        """Without a formula, CESM Variable Name is a comma-separated list of source names."""
        row = variable_to_rows(
            "pr",
            {
                "table": "atmos",
                "units": "kg m-2 s-1",
                "sources": [{"model_var": "PRECC"}, {"model_var": "PRECL"}],
            },
        )[0]
        assert row["CESM Variable Name"] == "PRECC, PRECL"

    def test_freq_in_freq_column(self):
        """Freq attributes appear in the Freq column, aligned with CESM Variable Name."""
        row = variable_to_rows(
            "siarea",
            {
                "table": "seaIce",
                "units": "m2",
                "sources": [
                    {"model_var": "siconc", "freq": "day"},
                    {"model_var": "tarea"},
                ],
            },
        )[0]
        assert row["CESM Variable Name"] == "siconc, tarea"
        assert row["Freq"] == "day, "

    def test_optional_fields_empty_when_absent(self):
        """Optional columns are empty strings when not present in the variable dict."""
        row = variable_to_rows(
            "tas",
            {
                "table": "atmos",
                "units": "K",
                "sources": [{"model_var": "TREFHT"}],
            },
        )[0]
        assert row["Standard Name"] == ""
        assert row["Cell Methods"] == ""
        assert row["Regrid Method"] == ""

    def test_optional_fields_populated(self):
        """Optional columns are populated when present in the variable dict."""
        row = variable_to_rows(
            "pr",
            {
                "table": "atmos",
                "standard_name": "precipitation_flux",
                "cell_methods": "time: mean",
                "regrid_method": "conservative",
                "units": "kg m-2 s-1",
                "sources": [{"model_var": "PRECT"}],
            },
        )[0]
        assert row["Standard Name"] == "precipitation_flux"
        assert row["Cell Methods"] == "time: mean"
        assert row["Regrid Method"] == "conservative"

    def test_levels_written_to_their_own_columns(self):
        """A levels block is written to the Levels * columns."""
        row = variable_to_rows(
            "ta",
            {
                "table": "atmos",
                "units": "K",
                "sources": [{"model_var": "T"}],
                "levels": {
                    "name": "standard_hybrid_sigma",
                    "units": "1",
                    "src_axis_name": "lev",
                    "src_axis_bnds": "ilev",
                },
            },
        )[0]
        assert row["Levels Name"] == "standard_hybrid_sigma"
        assert row["Levels Units"] == "1"
        assert row["Levels Src Axis Name"] == "lev"
        assert row["Levels Src Axis Bnds"] == "ilev"

    def test_variants_produce_one_row_each(self):
        """A variable with variants returns one row per variant."""
        rows = variable_to_rows(
            "siarea",
            {
                "table": "seaIce",
                "units": "m2",
                "sources": [{"model_var": "siconc"}, {"model_var": "tarea"}],
                "variants": [
                    {"long_name": "NH", "formula": "formula_nh"},
                    {"long_name": "SH", "formula": "formula_sh"},
                ],
            },
        )
        assert len(rows) == 2
        assert rows[0]["CESM Variable Name"] == "siconc, tarea"
        assert rows[1]["CESM Variable Name"] == "siconc, tarea"
        assert rows[0]["Formula"] == "formula_nh"
        assert rows[1]["Formula"] == "formula_sh"

    def test_no_sources_or_formula_gives_empty(self):
        """A variable with neither sources nor formula gets an empty expression."""
        print(variable_to_rows("mystery", {"table": "atmos", "units": "1"}))
        row = variable_to_rows("mystery", {"table": "atmos", "units": "1"})[0]
        assert row["CESM Variable Name"] == ""

    def test_all_columns_present(self):
        """All CESM_COLUMNS are present in the returned row."""
        row = variable_to_rows(
            "tas",
            {
                "table": "atmos",
                "units": "K",
                "sources": [{"model_var": "TREFHT"}],
            },
        )[0]
        assert set(row.keys()) == set(CESM_COLUMNS)
