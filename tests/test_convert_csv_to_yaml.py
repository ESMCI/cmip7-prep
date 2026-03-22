"""Tests for scripts/convert_csv_to_yaml.py.

Run with:
    pytest tests/test_convert_csv_to_yaml.py -v
or to also run the embedded doctests:
    pytest --doctest-modules scripts/convert_csv_to_yaml.py tests/test_convert_csv_to_yaml.py -v
"""

import csv
import os
import sys

import yaml

# Allow importing the script directly from the scripts/ directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
# pylint: disable=wrong-import-position
from convert_csv_to_yaml import (
    MODEL_CONFIGS,
    analyse_expression,
    clean_string,
    clean_strings,
    extract_variables,
    fix_number_norwegian_format,
    is_math_expression,
    read_csv,
    should_keep,
    write_yaml,
)


# ── is_math_expression ────────────────────────────────────────────────────────


class TestIsMathExpression:
    """Tests for is_math_expression()."""

    def test_addition(self):
        """Addition operator is detected as math."""
        assert is_math_expression("PRECC + PRECL") is True

    def test_subtraction(self):
        """Subtraction operator is detected as math."""
        assert is_math_expression("A - B") is True

    def test_multiplication_with_number(self):
        """Multiplication with a numeric constant is detected as math."""
        assert is_math_expression("PRECC * 0.001") is True

    def test_division(self):
        """Division operator is detected as math."""
        assert is_math_expression("A / B") is True

    def test_power(self):
        """Power operator is detected as math."""
        assert is_math_expression("U**2") is True

    def test_function_call(self):
        """A function call is detected as math."""
        assert is_math_expression("verticalsum(SOILWATER)") is True

    def test_numpy_function(self):
        """A numpy function call is detected as math."""
        assert is_math_expression("np.sqrt(U**2 + V**2)") is True

    def test_single_variable(self):
        """A single variable name is not a math expression."""
        assert is_math_expression("PRECC") is False

    def test_single_variable_with_numbers_in_name(self):
        """A variable name containing digits is not a math expression."""
        assert is_math_expression("T2M") is False

    def test_single_uppercase(self):
        """A plain uppercase token is not a math expression."""
        assert is_math_expression("TREFHT") is False

    def test_empty_string(self):
        """An empty or whitespace string has no math operators → False."""
        assert is_math_expression("") is False

    def test_two_words_space_separated(self):
        """'A B' matches the \\w+\\s+\\w+ pattern → treated as an expression."""
        assert is_math_expression("A B") is True


# ── extract_variables ─────────────────────────────────────────────────────────


class TestExtractVariables:
    """Tests for extract_variables()."""

    def test_addition(self):
        """Both variables in an addition are extracted."""
        result = extract_variables("PRECC + PRECL")
        model_vars = [v["model_var"] for v in result]
        assert model_vars == ["PRECC", "PRECL"]

    def test_multiplication_with_constant(self):
        """Only the variable, not the numeric constant, is extracted."""
        result = extract_variables("PRECC * 0.001")
        model_vars = [v["model_var"] for v in result]
        assert model_vars == ["PRECC"]

    def test_function_with_kwargs(self):
        """Keyword argument names are not extracted as variables."""
        result = extract_variables("verticalsum(SOILWATER, capped_at=1000)")
        model_vars = [v["model_var"] for v in result]
        assert model_vars == ["SOILWATER"]

    def test_numpy_expression(self):
        """Variables inside numpy expressions are extracted."""
        result = extract_variables("np.sqrt(U**2 + V**2)")
        model_vars = [v["model_var"] for v in result]
        assert model_vars == ["U", "V"]

    def test_returns_list_of_dicts(self):
        """Result is a list of dicts each containing model_var."""
        result = extract_variables("A + B")
        assert isinstance(result, list)
        assert all("model_var" in v for v in result)

    def test_three_variables(self):
        """Three variables in a sum are all extracted."""
        result = extract_variables("SOIL1C + SOIL2C + SOIL3C")
        model_vars = [v["model_var"] for v in result]
        assert model_vars == ["SOIL1C", "SOIL2C", "SOIL3C"]

    def test_ignores_known_modules(self):
        """Module names like 'np' and function names like 'sqrt' are ignored."""
        result = extract_variables("np.sqrt(X)")
        model_vars = [v["model_var"] for v in result]
        assert "np" not in model_vars
        assert "sqrt" not in model_vars
        assert "X" in model_vars

    def test_ignores_python_keywords(self):
        """Python keywords are not extracted as model variables."""
        result = extract_variables("True + False")
        assert not result

    def test_formula_with_division(self):
        """Variables in a division formula are extracted."""
        result = extract_variables("(TOTLITC + CWD_C)/1000.0")
        model_vars = [v["model_var"] for v in result]
        assert "TOTLITC" in model_vars
        assert "CWD_C" in model_vars


# ── analyse_expression ────────────────────────────────────────────────────────


class TestAnalyseExpression:
    """Tests for analyse_expression()."""

    def test_math_expression(self):
        """A math expression is flagged and variables extracted."""
        result = analyse_expression("PRECC + PRECL")
        assert result["is_math"] is True
        assert result["variables"] == [{"model_var": "PRECC"}, {"model_var": "PRECL"}]

    def test_single_variable(self):
        """A single variable is not flagged as math."""
        result = analyse_expression("T2M")
        assert result["is_math"] is False
        assert result["variables"] == [{"model_var": "T2M"}]

    def test_whitespace_stripped(self):
        """Leading/trailing whitespace is stripped before analysis."""
        result = analyse_expression("  T2M  ")
        assert result["is_math"] is False
        assert result["variables"] == [{"model_var": "T2M"}]

    def test_formula_with_constant(self):
        """A formula multiplied by a constant is detected as math."""
        result = analyse_expression("CLDTOT * 100")
        assert result["is_math"] is True
        vars_ = [v["model_var"] for v in result["variables"]]
        assert "CLDTOT" in vars_


# ── should_keep ───────────────────────────────────────────────────────────────


class TestShouldKeepNorESM:
    """Tests for should_keep() with the NorESM model config."""

    CFG = MODEL_CONFIGS["noresm"]

    def _row(self, realm, source):
        return {
            "Modelling Realm - Primary": realm,
            "NorESM3 name (dependency)": source,
        }

    def test_keep_atmos(self):
        """Atmos rows with a valid source are kept."""
        assert should_keep(self._row("atmos", "T2M"), self.CFG) is True

    def test_keep_land(self):
        """Land rows with a valid source are kept."""
        assert should_keep(self._row("land", "SOILWATER"), self.CFG) is True

    def test_skip_ocean(self):
        """Ocean rows are skipped for NorESM."""
        assert should_keep(self._row("ocean", "SST"), self.CFG) is False

    def test_skip_seaice(self):
        """SeaIce rows are skipped for NorESM."""
        assert should_keep(self._row("seaIce", "siconc"), self.CFG) is False

    def test_skip_empty_source(self):
        """Rows with an empty source field are skipped."""
        assert should_keep(self._row("atmos", ""), self.CFG) is False

    def test_skip_whitespace_only_source(self):
        """Rows with a whitespace-only source field are skipped."""
        assert should_keep(self._row("atmos", "   "), self.CFG) is False

    def test_skip_question_mark(self):
        """Rows whose source is '?' are skipped."""
        assert should_keep(self._row("atmos", "?"), self.CFG) is False

    def test_skip_na(self):
        """Rows whose source is 'n/a' are skipped."""
        assert should_keep(self._row("atmos", "n/a"), self.CFG) is False

    def test_skip_derived(self):
        """Rows whose source is 'derived' are skipped."""
        assert should_keep(self._row("atmos", "derived"), self.CFG) is False

    def test_skip_can_be_derived(self):
        """Rows whose source is 'can be derived' are skipped."""
        assert should_keep(self._row("atmos", "can be derived"), self.CFG) is False

    def test_skip_in_surf_dataset(self):
        """Rows whose source is 'IN SURF DATASET' are skipped."""
        assert should_keep(self._row("land", "IN SURF DATASET"), self.CFG) is False

    def test_keep_math_expression(self):
        """Rows with a math expression as the source are kept."""
        assert should_keep(self._row("atmos", "PRECC + PRECL"), self.CFG) is True


class TestShouldKeepCESM:
    """Tests for should_keep() with the CESM model config."""

    CFG = MODEL_CONFIGS["cesm"]

    def _row(self, realm, source):
        return {"Table": realm, "CESM Variable Name": source}

    def test_keep_atmos(self):
        """Atmos rows with a valid source are kept."""
        assert should_keep(self._row("atmos", "TREFHT"), self.CFG) is True

    def test_keep_land(self):
        """Land rows with a valid source are kept."""
        assert should_keep(self._row("land", "SOILWATER"), self.CFG) is True

    def test_keep_ocean(self):
        """Ocean rows are kept for CESM (unlike NorESM)."""
        assert should_keep(self._row("ocean", "SST"), self.CFG) is True

    def test_keep_seaice(self):
        """SeaIce rows are kept for CESM."""
        assert should_keep(self._row("seaIce", "siconc"), self.CFG) is True

    def test_keep_fx(self):
        """FX rows are kept for CESM."""
        assert should_keep(self._row("fx", "deptho"), self.CFG) is True

    def test_skip_empty_source(self):
        """Rows with an empty source field are skipped."""
        assert should_keep(self._row("atmos", ""), self.CFG) is False

    def test_keep_math_expression(self):
        """Rows with a math expression as the source are kept."""
        assert should_keep(self._row("atmos", "CLDTOT * 100"), self.CFG) is True


# ── clean_string / clean_strings ──────────────────────────────────────────────


class TestCleanString:
    """Tests for clean_string()."""

    def test_longitude(self):
        """'longitude' is normalised to 'lon' when normalize_dim_names=True."""
        assert clean_string("longitude", normalize_dim_names=True) == "lon"

    def test_latitude(self):
        """'latitude' is normalised to 'lat' when normalize_dim_names=True."""
        assert clean_string("latitude", normalize_dim_names=True) == "lat"

    def test_strip_whitespace(self):
        """Leading/trailing whitespace is stripped."""
        assert clean_string("  time  ") == "time"

    def test_remove_single_quotes(self):
        """Single quotes are removed."""
        assert clean_string("'lev'") == "lev"

    def test_remove_double_quotes(self):
        """Double quotes are removed."""
        assert clean_string('"lev"') == "lev"

    def test_passthrough_lev(self):
        """'lev' passes through unchanged."""
        assert clean_string("lev") == "lev"

    def test_alevel_to_lev(self):
        """'lev' passes through unchanged."""
        assert clean_string("alevel", normalize_dim_names=True) == "lev"

    def test_passthrough_time(self):
        """'time' passes through unchanged."""
        assert clean_string("time") == "time"


class TestCleanStrings:
    """Tests for clean_strings()."""

    def test_list_of_dims_normalize(self):
        """Each element in a list is cleaned."""
        result = clean_strings(["time", "longitude", "latitude"], normalize_dim_names=True)
        assert result == ["time", "lon", "lat"]

    def test_list_of_dims(self):
        """Each element in a list is cleaned."""
        result = clean_strings(["time", "longitude", "latitude"])
        assert result == ["time", "longitude", "latitude"]

    def test_single_string(self):
        """A plain string is cleaned as a single value."""
        assert clean_strings("longitude", normalize_dim_names=True) == "lon"

    def test_passthrough_non_string(self):
        """Non-string values pass through unchanged."""
        assert clean_strings(42, normalize_dim_names=True) == 42


# ── fix_number_norwegian_format ───────────────────────────────────────────────


class TestFixNumberNorwegianFormat:
    """Tests for fix_number_norwegian_format()."""

    def test_comma_decimal(self):
        """Comma decimal separator is converted to period."""
        assert fix_number_norwegian_format("1,5") == "1.5"

    def test_period_thousands_separator(self):
        """Period thousands separators combined with comma decimal are fixed."""
        assert fix_number_norwegian_format("1.000,5") == "1000.5"

    def test_unicode_minus(self):
        """Unicode minus sign is replaced with ASCII hyphen."""
        result = fix_number_norwegian_format("\u22121,5")
        assert result == "-1.5"

    def test_plain_string_passthrough(self):
        """A plain unit string without Norwegian formatting survives intact.

        (periods are stripped by the thousands-separator logic, so this tests
        that pure unit strings with no commas are not mangled in a way that
        breaks things — the current function does strip periods, which is a
        known limitation for unit strings like "kg m-2 s-1".)
        """
        result = fix_number_norwegian_format("K")
        assert result == "K"

    def test_non_string_passthrough(self):
        """Non-string values pass through unchanged."""
        assert fix_number_norwegian_format(1.5) == 1.5

    def test_integer_passthrough(self):
        """Integer values pass through unchanged."""
        assert fix_number_norwegian_format(42) == 42


# ── read_csv integration tests ────────────────────────────────────────────────


def _write_temp_csv(tmp_path, fieldnames, rows):
    """Helper: write rows to a CSV and return its path as a string."""
    path = tmp_path / "test.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


class TestReadCsvNorESM:
    """Integration tests for read_csv() with the NorESM model config."""

    FIELDNAMES = [
        "Branded Variable Name",
        "Modelling Realm - Primary",
        "CMIP6 Compound Name",
        "Description",
        "Units (from Physical Parameter)",
        "Dimensions",
        "NorESM3 name (dependency)",
        "CMIP7 Freq.",
    ]
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
        assert "variables" in data
        var = data["variables"]["tas"]
        assert var["table"] == "atmos"
        assert var["long_name"] == "Near-Surface Air Temperature"
        assert var["units"] == "K"
        assert var["sources"] == [{"model_var": "TREFHT"}]
        assert "formula" not in var
        assert var["freq"] == "mon"

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
        var = data["variables"]["pr"]
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
        var = data["variables"]["ta"]
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
        assert "levels" not in data["variables"]["tos"]

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
        assert "levels" not in data["variables"]["tas"]

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
        assert len(data["variables"]) == 0

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
        assert list(data["variables"].keys()) == ["v3"]

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
        assert data["dataset_overrides"]["institution_id"] == "NCC"
        assert data["dataset_overrides"]["source_id"] == "NorESM3"

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
        assert data["variables"]["tas"]["dims"] == ["time", "lon", "lat"]


class TestReadCsvCESM:
    """Integration tests for read_csv() with the CESM model config."""

    FIELDNAMES = [
        "CMIP Variable Name",
        "Table",
        "Long Name",
        "Standard Name",
        "Units",
        "Dimensions",
        "CESM Variable Name",
        "Cell Methods",
        "Regrid Method",
    ]
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
                    "CMIP Variable Name": "tas",
                    "Table": "atmos",
                    "Long Name": "Near-Surface Air Temperature",
                    "Units": "K",
                    "Dimensions": "time, lat, lon",
                    "CESM Variable Name": "TREFHT",
                }
            )
        ]
        data = read_csv(_write_temp_csv(tmp_path, self.FIELDNAMES, rows), self.CFG)
        var = data["variables"]["tas"]
        assert var["table"] == "atmos"
        assert var["long_name"] == "Near-Surface Air Temperature"
        assert var["units"] == "K"
        assert var["sources"] == [{"model_var": "TREFHT"}]

    def test_ocean_kept(self, tmp_path):
        """Ocean rows are kept for CESM."""
        rows = [
            self._row(
                **{
                    "CMIP Variable Name": "tos",
                    "Table": "ocean",
                    "CESM Variable Name": "SST",
                }
            )
        ]
        data = read_csv(_write_temp_csv(tmp_path, self.FIELDNAMES, rows), self.CFG)
        assert "tos" in data["variables"]

    def test_seaice_kept(self, tmp_path):
        """SeaIce rows are kept for CESM."""
        rows = [
            self._row(
                **{
                    "CMIP Variable Name": "siconc",
                    "Table": "seaIce",
                    "CESM Variable Name": "siconc",
                }
            )
        ]
        data = read_csv(_write_temp_csv(tmp_path, self.FIELDNAMES, rows), self.CFG)
        assert "siconc" in data["variables"]

    def test_standard_name_stored(self, tmp_path):
        """Standard name is stored in the output variable dict."""
        rows = [
            self._row(
                **{
                    "CMIP Variable Name": "pr",
                    "Table": "atmos",
                    "Standard Name": "precipitation_flux",
                    "Units": "kg m-2 s-1",
                    "Dimensions": "time, lat, lon",
                    "CESM Variable Name": "PRECT",
                }
            )
        ]
        data = read_csv(_write_temp_csv(tmp_path, self.FIELDNAMES, rows), self.CFG)
        assert data["variables"]["pr"]["standard_name"] == "precipitation_flux"

    def test_cell_methods_stored(self, tmp_path):
        """Cell methods are stored in the output variable dict."""
        rows = [
            self._row(
                **{
                    "CMIP Variable Name": "cl",
                    "Table": "atmos",
                    "Units": "%",
                    "Dimensions": "time, lev, lat, lon",
                    "CESM Variable Name": "CLOUD * 100",
                    "Cell Methods": "time: mean",
                }
            )
        ]
        data = read_csv(_write_temp_csv(tmp_path, self.FIELDNAMES, rows), self.CFG)
        assert data["variables"]["cl"]["cell_methods"] == "time: mean"

    def test_regrid_method_stored(self, tmp_path):
        """Regrid method is stored in the output variable dict."""
        rows = [
            self._row(
                **{
                    "CMIP Variable Name": "pr",
                    "Table": "atmos",
                    "Units": "kg m-2 s-1",
                    "Dimensions": "time, lat, lon",
                    "CESM Variable Name": "PRECT",
                    "Regrid Method": "conservative",
                }
            )
        ]
        data = read_csv(_write_temp_csv(tmp_path, self.FIELDNAMES, rows), self.CFG)
        assert data["variables"]["pr"]["regrid_method"] == "conservative"

    def test_levels_added_for_lev_dim(self, tmp_path):
        """A 'lev' dimension triggers addition of a levels block."""
        rows = [
            self._row(
                **{
                    "CMIP Variable Name": "cl",
                    "Table": "atmos",
                    "Units": "%",
                    "Dimensions": "time, lev, lat, lon",
                    "CESM Variable Name": "CLOUD * 100",
                }
            )
        ]
        data = read_csv(_write_temp_csv(tmp_path, self.FIELDNAMES, rows), self.CFG)
        assert "levels" in data["variables"]["cl"]
        assert data["variables"]["cl"]["levels"]["src_axis_name"] == "lev"

    def test_dataset_overrides_cesm(self, tmp_path):
        """CESM dataset_overrides are populated correctly."""
        rows = [
            self._row(
                **{
                    "CMIP Variable Name": "tas",
                    "Table": "atmos",
                    "CESM Variable Name": "TREFHT",
                }
            )
        ]
        data = read_csv(_write_temp_csv(tmp_path, self.FIELDNAMES, rows), self.CFG)
        assert data["dataset_overrides"]["institution_id"] == "NCAR"
        assert data["dataset_overrides"]["source_id"] == "CESM3"

    def test_empty_source_skipped(self, tmp_path):
        """A row with an empty CESM Variable Name is skipped."""
        rows = [
            self._row(
                **{
                    "CMIP Variable Name": "tas",
                    "Table": "atmos",
                    "CESM Variable Name": "",
                }
            )
        ]
        data = read_csv(_write_temp_csv(tmp_path, self.FIELDNAMES, rows), self.CFG)
        assert len(data["variables"]) == 0

    def test_formula_expression(self, tmp_path):
        """A math expression in CESM Variable Name is stored as formula."""
        rows = [
            self._row(
                **{
                    "CMIP Variable Name": "clt",
                    "Table": "atmos",
                    "Units": "%",
                    "Dimensions": "time, lat, lon",
                    "CESM Variable Name": "CLDTOT * 100",
                }
            )
        ]
        data = read_csv(_write_temp_csv(tmp_path, self.FIELDNAMES, rows), self.CFG)
        var = data["variables"]["clt"]
        print(var)
        assert var["formula"] == "CLDTOT * 100"
        assert [v["model_var"] for v in var["sources"]] == ["CLDTOT"]


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
