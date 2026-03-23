"""Tests for scripts/yaml_to_csv.py.

Run with:
    pytest tests/test_yaml_to_csv.py -v
or to also run the embedded doctests:
    pytest --doctest-modules scripts/yaml_to_csv.py tests/test_yaml_to_csv.py -v
"""

import csv
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
# pylint: disable=wrong-import-position
from yaml_to_csv import (
    CESM_COLUMNS,
    sources_to_expr,
    variable_to_rows,
    yaml_to_csv,
)

# ── sources_to_expr ───────────────────────────────────────────────────────────


class TestSourcesToExpr:
    """Tests for sources_to_expr()."""

    def test_single_source_no_scale(self):
        """A single source without scale returns just the variable name."""
        assert sources_to_expr([{"model_var": "TREFHT"}]) == "TREFHT"

    def test_single_source_with_scale(self):
        """A single source with a scale factor includes the multiplication."""
        assert sources_to_expr([{"model_var": "QFLX", "scale": -1.0}]) == "QFLX * -1.0"

    def test_two_sources_no_scale(self):
        """Two sources without scale are joined with '+'."""
        assert (
            sources_to_expr([{"model_var": "PRECC"}, {"model_var": "PRECL"}])
            == "PRECC + PRECL"
        )

    def test_three_sources(self):
        """Three sources are joined with '+'."""
        expr = sources_to_expr(
            [
                {"model_var": "SOIL1C"},
                {"model_var": "SOIL2C"},
                {"model_var": "SOIL3C"},
            ]
        )
        assert expr == "SOIL1C + SOIL2C + SOIL3C"

    def test_multiple_sources_with_scale(self):
        """Multiple sources each with a scale factor are represented correctly."""
        expr = sources_to_expr(
            [
                {"model_var": "SOIL1C", "scale": 0.001},
                {"model_var": "SOIL2C", "scale": 0.001},
            ]
        )
        assert expr == "SOIL1C * 0.001 + SOIL2C * 0.001"

    def test_empty_sources(self):
        """An empty sources list returns an empty string."""
        assert sources_to_expr([]) == ""

    def test_source_with_freq_ignored(self):
        """The freq field is informational only and does not appear in the expression."""
        expr = sources_to_expr([{"model_var": "siage_d", "freq": "day"}])
        assert expr == "siage_d"
        assert "freq" not in expr
        assert "day" not in expr


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
                "dims": ["time", "lat", "lon"],
                "sources": [{"model_var": "TREFHT"}],
            },
        )
        assert len(row) == 1
        row = row[0]
        assert row["CMIP Variable Name"] == "tas"
        assert row["Table"] == "atmos"
        assert row["Long Name"] == "Near-Surface Air Temperature"
        assert row["Units"] == "K"
        assert row["Dimensions"] == '["time", "lat", "lon"]'
        assert row["CESM Variable Name"] == "TREFHT"

    def test_cesm_var_name_uses_source_names_not_formula(self):
        """CESM Variable Name is always source names, never the formula string."""
        row = variable_to_rows(
            "cl",
            {
                "table": "atmos",
                "units": "%",
                "dims": ["time", "lev", "lat", "lon"],
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
                "dims": ["time", "lat", "lon"],
                "sources": [{"model_var": "PRECC"}, {"model_var": "PRECL"}],
            },
        )[0]
        assert row["CESM Variable Name"] == "PRECC, PRECL"

    def test_scale_excluded_from_cesm_variable_name(self):
        """Scale factors are not included in CESM Variable Name — only the variable name."""
        row = variable_to_rows(
            "evspsbl",
            {
                "table": "atmos",
                "units": "kg m-2 s-1",
                "dims": ["time", "lat", "lon"],
                "sources": [{"model_var": "QFLX", "scale": -1.0}],
            },
        )[0]
        assert row["CESM Variable Name"] == "QFLX"

    def test_optional_fields_empty_when_absent(self):
        """Optional columns are empty strings when not present in the variable dict."""
        row = variable_to_rows(
            "tas",
            {
                "table": "atmos",
                "units": "K",
                "dims": ["time", "lat", "lon"],
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
                "dims": ["time", "lat", "lon"],
                "sources": [{"model_var": "PRECT"}],
            },
        )[0]
        assert row["Standard Name"] == "precipitation_flux"
        assert row["Cell Methods"] == "time: mean"
        assert row["Regrid Method"] == "conservative"

    def test_dims_as_json(self):
        """Dims list is serialised as JSON in the output row."""
        row = variable_to_rows(
            "ta",
            {
                "table": "atmos",
                "units": "K",
                "dims": ["time", "lev", "lat", "lon"],
                "sources": [{"model_var": "T"}],
            },
        )[0]
        assert row["Dimensions"] == '["time", "lev", "lat", "lon"]'

    def test_empty_dims(self):
        """Missing dims key results in an empty JSON list in Dimensions field."""
        row = variable_to_rows(
            "tas",
            {
                "table": "atmos",
                "units": "K",
                "sources": [{"model_var": "TREFHT"}],
            },
        )[0]
        assert row["Dimensions"] == "[]"

    def test_variants_produce_one_row_each(self):
        """A variable with variants returns one row per variant."""
        rows = variable_to_rows(
            "siarea",
            {
                "table": "seaIce",
                "units": "m2",
                "dims": ["time"],
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
        row = variable_to_rows("mystery", {"table": "atmos", "units": "1"})[0]
        assert row["CESM Variable Name"] == ""

    def test_all_columns_present(self):
        """All CESM_COLUMNS are present in the returned row."""
        row = variable_to_rows(
            "tas",
            {
                "table": "atmos",
                "units": "K",
                "dims": ["time", "lat", "lon"],
                "sources": [{"model_var": "TREFHT"}],
            },
        )[0]
        assert set(row.keys()) == set(CESM_COLUMNS)


# ── yaml_to_csv integration ───────────────────────────────────────────────────


def _make_yaml(tmp_path, variables):
    """Write a minimal YAML file and return its path."""
    data = {
        "dataset_overrides": {"institution_id": "NCAR", "source_id": "CESM3"},
        "variables": variables,
    }
    path = tmp_path / "input.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f)
    return str(path)


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class TestYamlToCsv:
    """Integration tests for yaml_to_csv()."""

    def test_returns_row_count(self, tmp_path):
        """yaml_to_csv returns the number of rows written."""
        ypath = _make_yaml(
            tmp_path,
            {
                "tas": {
                    "table": "atmos",
                    "units": "K",
                    "dims": ["time", "lat", "lon"],
                    "sources": [{"model_var": "TREFHT"}],
                },
                "pr": {
                    "table": "atmos",
                    "units": "kg m-2 s-1",
                    "dims": ["time", "lat", "lon"],
                    "sources": [{"model_var": "PRECT"}],
                },
            },
        )
        cpath = str(tmp_path / "out.csv")
        n = yaml_to_csv(ypath, cpath)
        assert n == 2

    def test_csv_has_correct_columns(self, tmp_path):
        """The output CSV has exactly the CESM_COLUMNS headers."""
        ypath = _make_yaml(
            tmp_path,
            {
                "tas": {
                    "table": "atmos",
                    "units": "K",
                    "sources": [{"model_var": "TREFHT"}],
                },
            },
        )
        cpath = str(tmp_path / "out.csv")
        yaml_to_csv(ypath, cpath)
        rows = _read_csv(cpath)
        assert set(rows[0].keys()) == set(CESM_COLUMNS)

    def test_simple_variable_round_trip_fields(self, tmp_path):
        """A simple variable's fields survive a YAML→CSV round-trip."""
        ypath = _make_yaml(
            tmp_path,
            {
                "tas": {
                    "table": "atmos",
                    "long_name": "Near-Surface Air Temperature",
                    "units": "K",
                    "dims": ["time", "lat", "lon"],
                    "sources": [{"model_var": "TREFHT"}],
                }
            },
        )
        cpath = str(tmp_path / "out.csv")
        yaml_to_csv(ypath, cpath)
        rows = _read_csv(cpath)
        assert len(rows) == 1
        r = rows[0]
        assert r["CMIP Variable Name"] == "tas"
        assert r["Table"] == "atmos"
        assert r["Long Name"] == "Near-Surface Air Temperature"
        assert r["Units"] == "K"
        assert r["Dimensions"] == '["time", "lat", "lon"]'
        assert r["CESM Variable Name"] == "TREFHT"

    def test_cesm_var_name_is_source_names_not_formula(self, tmp_path):
        """CESM Variable Name is the source variable name, not the formula string."""
        ypath = _make_yaml(
            tmp_path,
            {
                "clt": {
                    "table": "atmos",
                    "units": "%",
                    "dims": ["time", "lat", "lon"],
                    "formula": "CLDTOT * 100",
                    "sources": [{"model_var": "CLDTOT"}],
                }
            },
        )
        cpath = str(tmp_path / "out.csv")
        yaml_to_csv(ypath, cpath)
        rows = _read_csv(cpath)
        assert rows[0]["CESM Variable Name"] == "CLDTOT"
        assert rows[0]["Formula"] == "CLDTOT * 100"

    def test_multi_source_names_comma_separated(self, tmp_path):
        """Multiple sources produce a comma-separated list in CESM Variable Name."""
        ypath = _make_yaml(
            tmp_path,
            {
                "cSoil": {
                    "table": "land",
                    "units": "kg m-2",
                    "dims": ["time", "lat", "lon"],
                    "formula": "(SOIL1C + SOIL2C + SOIL3C)/1000.0",
                    "sources": [
                        {"model_var": "SOIL1C"},
                        {"model_var": "SOIL2C"},
                        {"model_var": "SOIL3C"},
                    ],
                }
            },
        )
        cpath = str(tmp_path / "out.csv")
        yaml_to_csv(ypath, cpath)
        rows = _read_csv(cpath)
        assert rows[0]["CESM Variable Name"] == "SOIL1C, SOIL2C, SOIL3C"
        assert rows[0]["Formula"] == "(SOIL1C + SOIL2C + SOIL3C)/1000.0"

    def test_scale_excluded_from_cesm_variable_name(self, tmp_path):
        """Scale factors are not included in CESM Variable Name."""
        ypath = _make_yaml(
            tmp_path,
            {
                "evspsbl": {
                    "table": "atmos",
                    "units": "kg m-2 s-1",
                    "dims": ["time", "lat", "lon"],
                    "sources": [{"model_var": "QFLX", "scale": -1.0}],
                }
            },
        )
        cpath = str(tmp_path / "out.csv")
        yaml_to_csv(ypath, cpath)
        rows = _read_csv(cpath)
        assert rows[0]["CESM Variable Name"] == "QFLX"

    def test_optional_fields_empty_when_absent(self, tmp_path):
        """Optional columns are empty when not present in the YAML."""
        ypath = _make_yaml(
            tmp_path,
            {
                "tas": {
                    "table": "atmos",
                    "units": "K",
                    "sources": [{"model_var": "TREFHT"}],
                },
            },
        )
        cpath = str(tmp_path / "out.csv")
        yaml_to_csv(ypath, cpath)
        rows = _read_csv(cpath)
        assert rows[0]["Standard Name"] == ""
        assert rows[0]["Cell Methods"] == ""
        assert rows[0]["Regrid Method"] == ""

    def test_empty_variables_section(self, tmp_path):
        """An empty variables section produces a CSV with no data rows."""
        ypath = _make_yaml(tmp_path, {})
        cpath = str(tmp_path / "out.csv")
        n = yaml_to_csv(ypath, cpath)
        assert n == 0
        rows = _read_csv(cpath)
        assert not rows

    def test_csv_compatible_with_convert_csv_to_yaml(self, tmp_path):
        """Full pipeline: YAML → CSV → YAML and check key fields survive."""
        # Import here to avoid circular dependency at module level
        # pylint: disable=import-outside-toplevel
        from convert_csv_to_yaml import MODEL_CONFIGS, read_csv

        variables = {
            "tas": {
                "table": "atmos",
                "long_name": "Near-Surface Air Temperature",
                "standard_name": "air_temperature",
                "units": "K",
                "dims": ["time", "lat", "lon"],
                "sources": [{"model_var": "TREFHT"}],
            },
            "cl": {
                "table": "atmos",
                "units": "%",
                "dims": ["time", "lev", "lat", "lon"],
                "formula": "CLOUD * 100",
                "sources": [{"model_var": "CLOUD"}],
                "cell_methods": "time: mean",
            },
        }
        ypath = _make_yaml(tmp_path, variables)
        cpath = str(tmp_path / "cesm.csv")
        yaml_to_csv(ypath, cpath)

        cfg = MODEL_CONFIGS["cesm"]
        result = read_csv(cpath, cfg)

        tas = result["variables"]["tas"]
        assert tas["table"] == "atmos"
        assert tas["units"] == "K"
        assert tas["sources"] == [{"model_var": "TREFHT"}]

        cl = result["variables"]["cl"]
        assert cl["formula"] == "CLOUD * 100"
        assert "levels" in cl  # lev dim → levels block added
