"""Tests for pipeline.py: file discovery and model-var collection."""

# pylint: disable=duplicate-code

import logging
import tempfile
from pathlib import Path

from cmip7_prep.mapping_compat import Mapping
from cmip7_prep.pipeline import _collect_required_model_vars, _filename_contains_var


# ---------------------------------------------------------------------------
# _filename_contains_var
# ---------------------------------------------------------------------------


class TestFilenameContainsVar:
    """Tests for _filename_contains_var dot-delimited variable token matching."""

    def test_matches_dot_delimited_token(self):
        """Returns True when the filename contains '.VAR.' as a token."""
        assert _filename_contains_var("cam.h0.TS.0001-01.nc", "TS")

    def test_no_match_for_different_var(self):
        """Returns False when a different variable token is present."""
        assert not _filename_contains_var("cam.h0.PS.0001-01.nc", "TS")

    def test_no_partial_match(self):
        """'TA' does not match inside 'TAS' because '.TA.' != '.TAS.'."""
        assert not _filename_contains_var("cam.h0.TAS.001.nc", "TA")

    def test_path_object_input(self):
        """Accepts a pathlib.Path as input."""
        assert _filename_contains_var(Path("run.cam.h1.TS.001.nc"), "TS")

    def test_with_fname_pattern_both_must_match(self):
        """Returns True only when both the pattern and variable token appear."""
        assert _filename_contains_var("file.TS.001.nc", "TS", fname_pattern=".001.")

    def test_with_fname_pattern_no_var_match(self):
        """Returns False when the pattern matches but the variable token does not."""
        assert not _filename_contains_var("file.PS.001.nc", "TS", fname_pattern=".001.")

    def test_with_fname_pattern_no_pattern_match(self):
        """Returns False when the variable token matches but the pattern does not."""
        assert not _filename_contains_var("file.TS.002.nc", "TS", fname_pattern=".001.")


# ---------------------------------------------------------------------------
# _collect_required_model_vars
# ---------------------------------------------------------------------------

_SIMPLE_YAML = """
variables:
    tas:
        sources:
            - {model_var: TS}
        table: Amon
        units: K
    pr:
        sources:
            - {model_var: PRECT}
        table: Amon
        units: kg m-2 s-1
    ta:
        sources:
            - {model_var: T}
        raw_variables: [T]
        formula: "T"
        table: Amon
        units: K
        levels:
            name: plev19
"""

_SIGMA_YAML = """
variables:
    ua:
        sources:
            - {model_var: U}
        table: Amon
        units: m s-1
        levels:
            name: standard_hybrid_sigma
"""


def _mapping_from_yaml(yaml_str: str) -> Mapping:
    """Write yaml_str to a temp file and return the resulting Mapping."""
    with tempfile.NamedTemporaryFile("w+", suffix=".yaml", delete=False) as f:
        f.write(yaml_str)
        name = f.name
    return Mapping(name)


class TestCollectRequiredModelVars:
    """Tests for _collect_required_model_vars model-variable gathering."""

    def test_source_variable_collected(self):
        """The source variable for a simple mapping is included."""
        m = _mapping_from_yaml(_SIMPLE_YAML)
        needed = _collect_required_model_vars(m, ["tas"])
        assert "TS" in needed

    def test_multiple_vars(self):
        """Source variables for all requested CMIP variables are collected."""
        m = _mapping_from_yaml(_SIMPLE_YAML)
        needed = _collect_required_model_vars(m, ["tas", "pr"])
        assert "TS" in needed
        assert "PRECT" in needed

    def test_plev_adds_hybrid_coeffs(self):
        """Pressure-level variables trigger collection of PS, hyam, hybm, P0."""
        m = _mapping_from_yaml(_SIMPLE_YAML)
        needed = _collect_required_model_vars(m, ["ta"])
        for v in ("PS", "hyam", "hybm", "P0"):
            assert v in needed, f"Expected '{v}' in needed vars for plev variable"

    def test_sigma_var_adds_extended_coeffs(self):
        """Hybrid-sigma variables trigger collection of all sigma coefficients."""
        m = _mapping_from_yaml(_SIGMA_YAML)
        needed = _collect_required_model_vars(m, ["ua"])
        for v in ("PS", "hyam", "hybm", "hyai", "hybi", "P0"):
            assert v in needed

    def test_unknown_var_skipped_with_warning(self, caplog):
        """Unknown CMIP variable names are skipped and a warning is logged."""
        m = _mapping_from_yaml(_SIMPLE_YAML)
        with caplog.at_level(logging.WARNING, logger="cmip7_prep.pipeline"):
            needed = _collect_required_model_vars(m, ["nonexistent_var"])
        assert isinstance(needed, list)
        # Verify that a WARNING from cmip7_prep.pipeline was logged with the expected message.
        assert any(
            record.levelno == logging.WARNING
            and record.name == "cmip7_prep.pipeline"
            and "Skipping" in record.getMessage()
            and "no mapping found" in record.getMessage()
            for record in caplog.records
        )

    def test_result_is_sorted(self):
        """Returned list of model variables is sorted alphabetically."""
        m = _mapping_from_yaml(_SIMPLE_YAML)
        needed = _collect_required_model_vars(m, ["tas", "pr"])
        assert needed == sorted(needed)
