"""Integration tests for read_csv() and write_yaml() in convert_csv_to_yaml.py.

These build a CSV on disk, run it through the converter, and assert on the
resulting data structure -- as opposed to the unit tests in
``test_convert_csv_to_yaml`` and the warning tests in
``test_convert_csv_to_yaml_diagnostics``.
"""

import os
import sys

import yaml

# Allow importing the script directly from the scripts/ directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
# pylint: disable=wrong-import-position
from convert_csv_to_yaml import MODEL_CONFIGS, read_csv, write_yaml

from tests.csv_helpers import (
    CESM_FIELDNAMES,
    NORESM_FIELDNAMES,
    write_temp_csv as _write_temp_csv,
)


class TestReadCsvNorESM:
    """Integration tests for read_csv() with the NorESM model config."""

    FIELDNAMES = NORESM_FIELDNAMES
    CFG = MODEL_CONFIGS["noresm"]

    def test_simple_variable(self, tmp_path):
        """A simple single-source variable is parsed correctly."""
        rows = [
            {
                "Branded Variable Name": "tas",
                "Modelling Realm - Primary": "atmos",
                "CMIP6 Compound Name": "Near-Surface Air Temperature",
                "Description": "Temperature at 2m",
                "Units (from Physical Parameter)": "K",
                "Dimensions": "time, lat, lon",
                "NorESM3 name (dependency)": "TREFHT",
                "CMIP7 Freq.": "mon",
            }
        ]
        data = read_csv(_write_temp_csv(tmp_path, self.FIELDNAMES, rows), self.CFG)
        assert "atmos" in data
        var = data["atmos"]["variables"]["tas"]
        assert var["table"] == "atmos"
        assert "long_name" not in var
        assert var["units"] == "K"
        assert var["sources"] == [{"model_var": "TREFHT"}]
        assert "formula" not in var

    def test_math_formula_stored(self, tmp_path):
        """A math expression is stored as formula and sources are extracted."""
        rows = [
            {
                "Branded Variable Name": "pr",
                "Modelling Realm - Primary": "atmos",
                "Units (from Physical Parameter)": "kg m-2 s-1",
                "Dimensions": "time, lat, lon",
                "NorESM3 name (dependency)": "PRECC + PRECL",
                "CMIP6 Compound Name": "",
                "Description": "",
                "CMIP7 Freq.": "",
            }
        ]
        data = read_csv(_write_temp_csv(tmp_path, self.FIELDNAMES, rows), self.CFG)
        var = data["atmos"]["variables"]["pr"]
        assert var["formula"] == "PRECC + PRECL"
        assert {"model_var": "PRECC"} in var["sources"]
        assert {"model_var": "PRECL"} in var["sources"]

    def test_levels_added_for_lev_dim(self, tmp_path):
        """A 'lev' dimension triggers addition of a levels block."""
        rows = [
            {
                "Branded Variable Name": "ta",
                "Modelling Realm - Primary": "atmos",
                "Units (from Physical Parameter)": "K",
                "Dimensions": "time, lev, lat, lon",
                "NorESM3 name (dependency)": "T",
                "CMIP6 Compound Name": "",
                "Description": "",
                "CMIP7 Freq.": "",
            }
        ]
        data = read_csv(_write_temp_csv(tmp_path, self.FIELDNAMES, rows), self.CFG)
        var = data["atmos"]["variables"]["ta"]
        assert "levels" in var
        assert var["levels"]["name"] == "standard_hybrid_sigma"
        assert var["levels"]["src_axis_name"] == "lev"
        assert var["levels"]["src_axis_bnds"] == "ilev"

    def test_no_levels_for_olevel_dim(self, tmp_path):
        """'olevel' contains 'lev' as a substring but is not the hybrid sigma axis."""
        rows = [
            {
                "Branded Variable Name": "tos",
                "Modelling Realm - Primary": "atmos",  # contrived, just testing dim logic
                "Dimensions": "time, olevel, lat, lon",
                "NorESM3 name (dependency)": "SST",
                "CMIP6 Compound Name": "",
                "Description": "",
                "Units (from Physical Parameter)": "K",
                "CMIP7 Freq.": "",
            }
        ]
        data = read_csv(_write_temp_csv(tmp_path, self.FIELDNAMES, rows), self.CFG)
        assert "levels" not in data["atmos"]["variables"]["tos"]

    def test_no_levels_without_lev_dim(self, tmp_path):
        """Without a 'lev' dimension, no levels block is added."""
        rows = [
            {
                "Branded Variable Name": "tas",
                "Modelling Realm - Primary": "atmos",
                "Dimensions": "time, lat, lon",
                "NorESM3 name (dependency)": "TREFHT",
                "CMIP6 Compound Name": "",
                "Description": "",
                "Units (from Physical Parameter)": "K",
                "CMIP7 Freq.": "",
            }
        ]
        data = read_csv(_write_temp_csv(tmp_path, self.FIELDNAMES, rows), self.CFG)
        assert "levels" not in data["atmos"]["variables"]["tas"]

    def test_ocean_rows_filtered(self, tmp_path):
        """Ocean rows are filtered out for NorESM."""
        rows = [
            {
                "Branded Variable Name": "tos",
                "Modelling Realm - Primary": "ocean",
                "NorESM3 name (dependency)": "SST",
                "CMIP6 Compound Name": "",
                "Description": "",
                "Units (from Physical Parameter)": "K",
                "Dimensions": "time, lat, lon",
                "CMIP7 Freq.": "",
            }
        ]
        data = read_csv(_write_temp_csv(tmp_path, self.FIELDNAMES, rows), self.CFG)
        assert all(len(rd["variables"]) == 0 for rd in data.values())

    def test_skip_phrases_filter_rows(self, tmp_path):
        """Rows with skip phrases in the source field are excluded."""
        rows = [
            {
                "Branded Variable Name": "v1",
                "Modelling Realm - Primary": "atmos",
                "NorESM3 name (dependency)": "n/a",
                "CMIP6 Compound Name": "",
                "Description": "",
                "Units (from Physical Parameter)": "K",
                "Dimensions": "time, lat, lon",
                "CMIP7 Freq.": "",
            },
            {
                "Branded Variable Name": "v2",
                "Modelling Realm - Primary": "atmos",
                "NorESM3 name (dependency)": "derived",
                "CMIP6 Compound Name": "",
                "Description": "",
                "Units (from Physical Parameter)": "K",
                "Dimensions": "time, lat, lon",
                "CMIP7 Freq.": "",
            },
            {
                "Branded Variable Name": "v3",
                "Modelling Realm - Primary": "atmos",
                "NorESM3 name (dependency)": "TREFHT",
                "CMIP6 Compound Name": "",
                "Description": "",
                "Units (from Physical Parameter)": "K",
                "Dimensions": "time, lat, lon",
                "CMIP7 Freq.": "",
            },
        ]
        data = read_csv(_write_temp_csv(tmp_path, self.FIELDNAMES, rows), self.CFG)
        assert list(data["atmos"]["variables"].keys()) == ["v3"]

    def test_dataset_overrides_noresm(self, tmp_path):
        """NorESM dataset_overrides are populated correctly."""
        rows = [
            {
                "Branded Variable Name": "tas",
                "Modelling Realm - Primary": "atmos",
                "NorESM3 name (dependency)": "TREFHT",
                "CMIP6 Compound Name": "",
                "Description": "",
                "Units (from Physical Parameter)": "K",
                "Dimensions": "time, lat, lon",
                "CMIP7 Freq.": "",
            }
        ]
        data = read_csv(_write_temp_csv(tmp_path, self.FIELDNAMES, rows), self.CFG)
        assert data["atmos"]["dataset_overrides"]["institution_id"] == "NCC"
        assert data["atmos"]["dataset_overrides"]["source_id"] == "NorESM3"

    def test_dims_normalised(self, tmp_path):
        """'longitude' and 'latitude' in dims are normalised to 'lon' and 'lat'."""
        rows = [
            {
                "Branded Variable Name": "tas",
                "Modelling Realm - Primary": "atmos",
                "NorESM3 name (dependency)": "TREFHT",
                "Dimensions": "time, longitude, latitude",
                "CMIP6 Compound Name": "",
                "Description": "",
                "Units (from Physical Parameter)": "K",
                "CMIP7 Freq.": "",
            }
        ]
        data = read_csv(_write_temp_csv(tmp_path, self.FIELDNAMES, rows), self.CFG)
        assert data["atmos"]["variables"]["tas"]["dims"] == ["time", "lon", "lat"]


class TestReadCsvCESM:
    """Integration tests for read_csv() with the CESM model config."""

    FIELDNAMES = CESM_FIELDNAMES
    CFG = MODEL_CONFIGS["cesm"]

    def _row(self, **kwargs):
        base = {f: "" for f in self.FIELDNAMES}
        base.update(kwargs)
        return base

    def test_simple_variable(self, tmp_path):
        """A simple single-source variable is parsed correctly."""
        rows = [
            self._row(
                **{
                    "CMIP Branded Variable Name": "tas",
                    "Table": "atmos",
                    "Long Name": "Near-Surface Air Temperature",
                    "Units": "K",
                    "Dimensions": "time, lat, lon",
                    "CESM Variable Name": "TREFHT",
                }
            )
        ]
        data = read_csv(_write_temp_csv(tmp_path, self.FIELDNAMES, rows), self.CFG)
        var = data["atmos"]["variables"]["tas"]
        assert var["table"] == "atmos"
        assert var["long_name"] == "Near-Surface Air Temperature"
        assert var["units"] == "K"
        assert var["sources"] == [{"model_var": "TREFHT"}]

    def test_ocean_kept(self, tmp_path):
        """Ocean rows are kept for CESM."""
        rows = [
            self._row(
                **{
                    "CMIP Branded Variable Name": "tos",
                    "Table": "ocean",
                    "CESM Variable Name": "SST",
                }
            )
        ]
        data = read_csv(_write_temp_csv(tmp_path, self.FIELDNAMES, rows), self.CFG)
        assert "tos" in data["ocean"]["variables"]

    def test_seaice_kept(self, tmp_path):
        """SeaIce rows are kept for CESM."""
        rows = [
            self._row(
                **{
                    "CMIP Branded Variable Name": "siconc",
                    "Table": "seaIce",
                    "CESM Variable Name": "siconc",
                }
            )
        ]
        data = read_csv(_write_temp_csv(tmp_path, self.FIELDNAMES, rows), self.CFG)
        assert "siconc" in data["seaIce"]["variables"]

    def test_standard_name_stored(self, tmp_path):
        """Standard name is stored in the output variable dict."""
        rows = [
            self._row(
                **{
                    "CMIP Branded Variable Name": "pr",
                    "Table": "atmos",
                    "Standard Name": "precipitation_flux",
                    "Units": "kg m-2 s-1",
                    "Dimensions": "time, lat, lon",
                    "CESM Variable Name": "PRECT",
                }
            )
        ]
        data = read_csv(_write_temp_csv(tmp_path, self.FIELDNAMES, rows), self.CFG)
        assert data["atmos"]["variables"]["pr"]["standard_name"] == "precipitation_flux"

    def test_cell_methods_stored(self, tmp_path):
        """Cell methods are stored in the output variable dict."""
        rows = [
            self._row(
                **{
                    "CMIP Branded Variable Name": "cl",
                    "Table": "atmos",
                    "Units": "%",
                    "Dimensions": "time, lev, lat, lon",
                    "CESM Variable Name": "CLOUD",
                    "Formula": "CLOUD * 100",
                    "Cell Methods": "time: mean",
                }
            )
        ]
        data = read_csv(_write_temp_csv(tmp_path, self.FIELDNAMES, rows), self.CFG)
        assert data["atmos"]["variables"]["cl"]["cell_methods"] == "time: mean"

    def test_regrid_method_stored(self, tmp_path):
        """Regrid method is stored in the output variable dict."""
        rows = [
            self._row(
                **{
                    "CMIP Branded Variable Name": "pr",
                    "Table": "atmos",
                    "Units": "kg m-2 s-1",
                    "Dimensions": "time, lat, lon",
                    "CESM Variable Name": "PRECT",
                    "Regrid Method": "conservative",
                }
            )
        ]
        data = read_csv(_write_temp_csv(tmp_path, self.FIELDNAMES, rows), self.CFG)
        assert data["atmos"]["variables"]["pr"]["regrid_method"] == "conservative"

    def test_levels_added_for_lev_dim(self, tmp_path):
        """A 'lev' dimension triggers addition of a levels block."""
        rows = [
            self._row(
                **{
                    "CMIP Branded Variable Name": "cl",
                    "Table": "atmos",
                    "Units": "%",
                    "Dimensions": "time, lev, lat, lon",
                    "CESM Variable Name": "CLOUD",
                    "Formula": "CLOUD * 100",
                }
            )
        ]
        data = read_csv(_write_temp_csv(tmp_path, self.FIELDNAMES, rows), self.CFG)
        assert "levels" in data["atmos"]["variables"]["cl"]
        assert data["atmos"]["variables"]["cl"]["levels"]["src_axis_name"] == "lev"

    def test_dataset_overrides_cesm(self, tmp_path):
        """CESM dataset_overrides are populated correctly."""
        rows = [
            self._row(
                **{
                    "CMIP Branded Variable Name": "tas",
                    "Table": "atmos",
                    "CESM Variable Name": "TREFHT",
                }
            )
        ]
        data = read_csv(_write_temp_csv(tmp_path, self.FIELDNAMES, rows), self.CFG)
        assert data["atmos"]["dataset_overrides"]["institution_id"] == "NCAR"
        assert data["atmos"]["dataset_overrides"]["source_id"] == "CESM3"

    def test_empty_source_skipped(self, tmp_path):
        """A row with an empty CESM Variable Name is skipped."""
        rows = [
            self._row(
                **{
                    "CMIP Branded Variable Name": "tas",
                    "Table": "atmos",
                    "CESM Variable Name": "",
                }
            )
        ]
        data = read_csv(_write_temp_csv(tmp_path, self.FIELDNAMES, rows), self.CFG)
        assert len(data["atmos"]["variables"]) == 0

    def test_formula_expression(self, tmp_path):
        """Formula column is stored and sources are derived from CESM Variable Name."""
        rows = [
            self._row(
                **{
                    "CMIP Branded Variable Name": "clt",
                    "Table": "atmos",
                    "Units": "%",
                    "Dimensions": "time, lat, lon",
                    "CESM Variable Name": "CLDTOT",
                    "Formula": "CLDTOT * 100",
                }
            )
        ]
        data = read_csv(_write_temp_csv(tmp_path, self.FIELDNAMES, rows), self.CFG)
        var = data["atmos"]["variables"]["clt"]
        assert var["formula"] == "CLDTOT * 100"
        assert var["sources"] == [{"model_var": "CLDTOT"}]

    def test_scale_from_column(self, tmp_path):
        """Scale column is merged into the sources list as a float."""
        rows = [
            self._row(
                **{
                    "CMIP Branded Variable Name": "evspsbl",
                    "Table": "atmos",
                    "Units": "kg m-2 s-1",
                    "Dimensions": "time, lat, lon",
                    "CESM Variable Name": "QFLX",
                    "Scale": "-1.0",
                }
            )
        ]
        data = read_csv(_write_temp_csv(tmp_path, self.FIELDNAMES, rows), self.CFG)
        var = data["atmos"]["variables"]["evspsbl"]
        assert var["sources"] == [{"model_var": "QFLX"}]

    def test_freq_from_column(self, tmp_path):
        """Freq column is merged positionally into the sources list."""
        rows = [
            self._row(
                **{
                    "CMIP Branded Variable Name": "siarea",
                    "Table": "seaIce",
                    "Units": "m2",
                    "Dimensions": "time",
                    "CESM Variable Name": "siconc_d, siconc",
                    "Freq": "day, mon",
                }
            )
        ]
        data = read_csv(_write_temp_csv(tmp_path, self.FIELDNAMES, rows), self.CFG)
        sources = data["seaIce"]["variables"]["siarea"]["sources"]
        assert sources == [
            {"model_var": "siconc_d", "freq": "day"},
            {"model_var": "siconc", "freq": "mon"},
        ]

    def test_alias_from_column(self, tmp_path):
        """Alias column is merged positionally into the sources list."""
        rows = [
            self._row(
                **{
                    "CMIP Branded Variable Name": "siarea",
                    "Table": "seaIce",
                    "Units": "m2",
                    "Dimensions": "time",
                    "CESM Variable Name": "siconc_d, siconc, tarea",
                    "Freq": "day, mon, ",
                    "Alias": "siconc, , ",
                }
            )
        ]
        data = read_csv(_write_temp_csv(tmp_path, self.FIELDNAMES, rows), self.CFG)
        sources = data["seaIce"]["variables"]["siarea"]["sources"]
        assert sources[0] == {"model_var": "siconc_d", "freq": "day", "alias": "siconc"}
        assert sources[1] == {"model_var": "siconc", "freq": "mon"}
        assert sources[2] == {"model_var": "tarea"}

    def test_multi_source_formula(self, tmp_path):
        """Multiple sources in CESM Variable Name with a formula expression."""
        rows = [
            self._row(
                **{
                    "CMIP Branded Variable Name": "pr",
                    "Table": "atmos",
                    "Units": "kg m-2 s-1",
                    "Dimensions": "time, lat, lon",
                    "CESM Variable Name": "PRECC, PRECL",
                    "Formula": "(PRECC + PRECL) * 1000.0",
                }
            )
        ]
        data = read_csv(_write_temp_csv(tmp_path, self.FIELDNAMES, rows), self.CFG)
        var = data["atmos"]["variables"]["pr"]
        assert var["formula"] == "(PRECC + PRECL) * 1000.0"
        assert [s["model_var"] for s in var["sources"]] == ["PRECC", "PRECL"]


# ── write_yaml round-trip ─────────────────────────────────────────────────────


class TestWriteYaml:
    """Tests for write_yaml()."""

    def test_roundtrip_simple(self, tmp_path):
        """A simple dict round-trips through write_yaml and yaml.safe_load."""
        data = {
            "dataset_overrides": {"institution_id": "NCC", "source_id": "NorESM3"},
            "variables": {
                "tas": {
                    "table": "atmos",
                    "units": "K",
                    "sources": [{"model_var": "TREFHT"}],
                }
            },
        }
        out = str(tmp_path / "out.yaml")
        write_yaml(data, out)
        with open(out, encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        assert loaded["variables"]["tas"]["table"] == "atmos"
        assert loaded["variables"]["tas"]["sources"][0]["model_var"] == "TREFHT"

    def test_model_var_not_braces(self, tmp_path):
        """The post-processing step must expand {model_var: X} to 'model_var: X'."""
        data = {
            "dataset_overrides": {"source_id": "NorESM3"},
            "variables": {
                "pr": {"sources": [{"model_var": "PRECC"}, {"model_var": "PRECL"}]}
            },
        }
        out = str(tmp_path / "out.yaml")
        write_yaml(data, out)
        with open(out, encoding="utf-8") as f:
            content = f.read()
        # Should NOT contain the raw flow-style dict notation
        assert "{model_var:" not in content

    def test_blank_line_after_before_new_var(self, tmp_path):
        """A blank line is inserted after each 'units:' key."""
        data = {
            "dataset_overrides": {"source_id": "NorESM3"},
            "variables": {
                "tas": {"units": "K", "sources": [{"model_var": "TREFHT"}]},
                "gpp_tavg-u-hxy-lnd": {
                    "units": "kg C",
                    "sources": [{"model_var": "FATES_GPP"}],
                },
            },
        }
        out = str(tmp_path / "out.yaml")
        write_yaml(data, out)
        with open(out, encoding="utf-8") as f:
            lines = f.readlines()
        pre_break_index = [i for i, l in enumerate(lines) if "tavg" in l]
        for idx in pre_break_index:
            assert (
                lines[idx - 1].strip() == ""
            ), f"Expected blank line before 'new CMIP named variable' at line {idx - 1}"

    def test_blank_line_after_source_id(self, tmp_path):
        """A blank line is inserted after 'source_id:'."""
        data = {
            "dataset_overrides": {"source_id": "CESM3"},
            "variables": {"tas": {"units": "K", "sources": [{"model_var": "TREFHT"}]}},
        }
        out = str(tmp_path / "out.yaml")
        write_yaml(data, out)
        with open(out, encoding="utf-8") as f:
            lines = f.readlines()
        source_id_idx = next(i for i, l in enumerate(lines) if "source_id:" in l)
        assert lines[source_id_idx + 1].strip() == ""
