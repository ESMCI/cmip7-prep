"""Additional tests for mapping_compat.py: list-style YAML, realize, unit conversions."""

import tempfile

import numpy as np
import pytest
import xarray as xr

from cmip7_prep.mapping_compat import (
    Mapping,
    VarConfig,
    _apply_unit_conversion,
    _to_varconfig,
)


# ---------------------------------------------------------------------------
# VarConfig.as_cfg — None fields are excluded
# ---------------------------------------------------------------------------


class TestVarConfigAsCfg:
    """Tests for VarConfig.as_cfg() dict representation."""

    def test_none_fields_excluded(self):
        """Fields with value None are omitted from the dict."""
        vc = VarConfig(name="pr", table="Amon", units="kg m-2 s-1")
        cfg = vc.as_cfg()
        assert "raw_variables" not in cfg
        assert "formula" not in cfg
        assert cfg["name"] == "pr"

    def test_all_set_fields_present(self):
        """All non-None fields appear in the dict."""
        vc = VarConfig(
            name="ta",
            table="Amon",
            units="K",
            source="T",
            regrid_method="bilinear",
        )
        cfg = vc.as_cfg()
        assert cfg["source"] == "T"
        assert cfg["regrid_method"] == "bilinear"


# ---------------------------------------------------------------------------
# Mapping — list-style YAML (CMIP6 format)
# ---------------------------------------------------------------------------


class TestMappingListStyle:
    """Tests for loading CMIP6-style list YAML mappings."""

    _LIST_YAML = """
- name: tas
  source: T2
  table: Amon
  units: K
- name: pr
  source: PRECT
  table: Amon
  units: kg m-2 s-1
"""

    def _make_mapping(self, tmp_path):
        """Write the list-style YAML to a temp file and return a Mapping."""
        mapping_path = tmp_path / "mapping_list.yaml"
        mapping_path.write_text(self._LIST_YAML)
        return Mapping(str(mapping_path))

    def test_list_style_loads_all_vars(self, tmp_path):
        """All variables in the list are accessible via get_cfg."""
        m = self._make_mapping(tmp_path)
        assert m.get_cfg("tas")["source"] == "T2"
        assert m.get_cfg("pr")["source"] == "PRECT"

    def test_unknown_var_raises_key_error(self, tmp_path):
        """get_cfg raises KeyError for a variable not present in the mapping."""
        m = self._make_mapping(tmp_path)
        with pytest.raises(KeyError, match="No mapping"):
            m.get_cfg("winds")


# ---------------------------------------------------------------------------
# Mapping — dict-style YAML with formula
# ---------------------------------------------------------------------------


class TestMappingFormula:
    """Tests for formula-based variable realization."""

    _FORMULA_YAML = """
variables:
  clisccp:
    raw_variables: [CLD_A, CLD_B]
    formula: "CLD_A + CLD_B"
    units: "1"
    table: CFmon
"""

    def _make_mapping(self):
        """Write the formula YAML to a temp file and return a Mapping."""
        with tempfile.NamedTemporaryFile("w+", suffix=".yaml", delete=False) as f:
            f.write(self._FORMULA_YAML)
            name = f.name
        return Mapping(name)

    def test_realize_formula(self):
        """Formula 'CLD_A + CLD_B' produces the elementwise sum."""
        m = self._make_mapping()
        data = np.ones((3, 4))
        ds = xr.Dataset(
            {
                "CLD_A": xr.DataArray(data, dims=["lat", "lon"]),
                "CLD_B": xr.DataArray(data * 2, dims=["lat", "lon"]),
            }
        )
        result = m.realize(ds, "clisccp")
        assert result is not None
        np.testing.assert_allclose(result.values, 3.0)

    def test_formula_missing_raw_var_raises(self):
        """realize() raises KeyError when a raw variable required by the formula is absent."""
        m = self._make_mapping()
        ds = xr.Dataset({"CLD_A": xr.DataArray(np.ones((2, 2)), dims=["lat", "lon"])})
        with pytest.raises(KeyError, match="CLD_B"):
            m.realize(ds, "clisccp")


# ---------------------------------------------------------------------------
# Mapping — source mapping with realize
# ---------------------------------------------------------------------------


class TestMappingRealize:
    """Tests for source-based variable realization and error handling."""

    _SRC_YAML = """
variables:
  tas:
    source: TS
    table: Amon
    units: K
"""

    def test_realize_source_returns_correct_data(self):
        """realize() returns the source DataArray unchanged."""
        with tempfile.NamedTemporaryFile("w+", suffix=".yaml", delete=True) as f:
            f.write(self._SRC_YAML)
            f.flush()
            m = Mapping(f.name)

        ts_data = np.array([[280.0, 290.0], [300.0, 310.0]])
        ds = xr.Dataset({"TS": xr.DataArray(ts_data, dims=["lat", "lon"])})
        result = m.realize(ds, "tas")
        np.testing.assert_array_equal(result.values, ts_data)

    def test_realize_missing_source_raises(self):
        """realize() raises KeyError when the source variable is absent."""
        with tempfile.NamedTemporaryFile("w+", suffix=".yaml", delete=True) as f:
            f.write(self._SRC_YAML)
            f.flush()
            m = Mapping(f.name)

        ds = xr.Dataset({"T": xr.DataArray([1.0, 2.0])})  # 'TS' missing
        with pytest.raises(KeyError, match="TS"):
            m.realize(ds, "tas")

    def test_realize_unknown_cmip_var_warns(self):
        """realize() emits a RuntimeWarning and returns None for unknown CMIP variables."""
        with tempfile.NamedTemporaryFile("w+", suffix=".yaml", delete=True) as f:
            f.write(self._SRC_YAML)
            f.flush()
            m = Mapping(f.name)

        ds = xr.Dataset({"TS": xr.DataArray([1.0])})
        with pytest.warns(RuntimeWarning, match="not found"):
            result = m.realize(ds, "ta")
        assert result is None


# ---------------------------------------------------------------------------
# _apply_unit_conversion
# ---------------------------------------------------------------------------


class TestApplyUnitConversion:
    """Tests for _apply_unit_conversion dict and string-expression rules."""

    def test_dict_scale_only(self):
        """scale-only dict multiplies each value by scale."""
        da = xr.DataArray([1.0, 2.0, 3.0])
        result = _apply_unit_conversion(da, {"scale": 2.0})
        np.testing.assert_allclose(result.values, [2.0, 4.0, 6.0])

    def test_dict_scale_and_offset(self):
        """scale+offset dict applies linear transformation."""
        da = xr.DataArray([0.0, 100.0])
        result = _apply_unit_conversion(da, {"scale": 1.0, "offset": 273.15})
        np.testing.assert_allclose(result.values, [273.15, 373.15])

    def test_string_expression(self):
        """String expression 'x * 1000.0' scales by 1000."""
        da = xr.DataArray([1.0, 2.0, 3.0])
        result = _apply_unit_conversion(da, "x * 1000.0")
        np.testing.assert_allclose(result.values, [1000.0, 2000.0, 3000.0])

    def test_unsupported_type_raises(self):
        """Non-string, non-dict rule raises TypeError."""
        da = xr.DataArray([1.0])
        with pytest.raises(TypeError, match="unit_conversion"):
            _apply_unit_conversion(da, 42)

    def test_bad_string_expr_raises(self):
        """Invalid expression string raises ValueError."""
        da = xr.DataArray([1.0])
        with pytest.raises(ValueError, match="unit_conversion"):
            _apply_unit_conversion(da, "undefined_func(x)")


# ---------------------------------------------------------------------------
# _to_varconfig — edge cases
# ---------------------------------------------------------------------------


class TestToVarconfig:
    """Tests for _to_varconfig normalization of raw YAML entries."""

    def test_single_raw_variable_becomes_source(self):
        """A single raw_variable with no formula is promoted to source."""
        cfg = {"raw_variables": ["TS"], "table": "Amon"}
        vc = _to_varconfig("tas", cfg)
        assert vc.source == "TS"
        assert vc.raw_variables is None

    def test_multiple_raw_variables_kept(self):
        """Multiple raw_variables with a formula are retained as raw_variables."""
        cfg = {"raw_variables": ["A", "B"], "formula": "A + B", "table": "Amon"}
        vc = _to_varconfig("derived", cfg)
        assert vc.raw_variables == ["A", "B"]
        assert vc.source is None
