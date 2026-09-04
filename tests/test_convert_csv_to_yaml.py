"""Unit tests for the parsing helpers in scripts/convert_csv_to_yaml.py.

Tests for ``read_csv`` itself live in ``test_convert_csv_to_yaml_read_csv``, and
tests for the warning machinery in ``test_convert_csv_to_yaml_diagnostics``.

Run with:
    pytest tests/test_convert_csv_to_yaml.py -v
or to also run the script's embedded doctests:
    pytest --doctest-modules scripts/convert_csv_to_yaml.py -v
"""

import os
import sys

# Allow importing the script directly from the scripts/ directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
# pylint: disable=wrong-import-position
from convert_csv_to_yaml import (
    MODEL_CONFIGS,
    _parse_csv_identifiers,
    _split_positional,
    analyse_expression,
    clean_string,
    clean_strings,
    extract_variables,
    fix_number_norwegian_format,
    is_math_expression,
    should_keep,
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
        """FX rows are kept for CESM (written into the ocean yaml)."""
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
        result = clean_strings(
            ["time", "longitude", "latitude"], normalize_dim_names=True
        )
        assert result == ["time", "lon", "lat"]

    def test_list_of_dims_no_normalize(self):
        """Each element in a list is cleaned without dim name normalization."""
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


# ── _parse_csv_identifiers / _split_positional ────────────────────────────────


class TestCsvHelpers:
    """Tests for _parse_csv_identifiers() and _split_positional()."""

    def test_valid_identifiers(self):
        """Single, comma-separated, underscored, and digit-containing names all parse."""
        assert _parse_csv_identifiers("TREFHT") == ["TREFHT"]
        assert _parse_csv_identifiers("siconc, tarea") == ["siconc", "tarea"]
        assert _parse_csv_identifiers("SOIL1C, SOIL2C, SOIL3C") == [
            "SOIL1C",
            "SOIL2C",
            "SOIL3C",
        ]
        assert _parse_csv_identifiers("FATES_FRAC, GPP") == ["FATES_FRAC", "GPP"]
        assert _parse_csv_identifiers("T2M") == ["T2M"]

    def test_expressions_return_none(self):
        """Formula and arithmetic expressions return None (fallback to analyse_expression)."""
        assert _parse_csv_identifiers("CLDTOT * 100") is None
        assert _parse_csv_identifiers("PRECC + PRECL") is None

    def test_split_padding_and_trimming(self):
        """Short lists are padded; long lists are trimmed; exact length is unchanged."""
        assert _split_positional("day, mon, ", 3) == ["day", "mon", ""]
        assert _split_positional("a, b", 3) == ["a", "b", ""]
        assert _split_positional("a, b, c, d", 2) == ["a", "b"]
        assert _split_positional("", 2) == ["", ""]
        assert _split_positional("-1.0", 1) == ["-1.0"]
