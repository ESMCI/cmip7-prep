"""Additional tests for mapping_compat.py: list-style YAML, realize, unit conversions."""

import tempfile
import pytest
import numpy as np
import xarray as xr


from cmip7_prep.mapping_compat import (
    Mapping,
    VarConfig,
)

_FORMULA_YAML = """
variables:
    clisccp:
        sources:
            - {model_var: CLD_A}
            - {model_var: CLD_B}
        formula: "CLD_A + CLD_B"
        units: "1"
        table: CFmon
"""

_SRC_YAML = """
variables:
  tas:
    sources:
      - {model_var: TS}
    table: Amon
    units: K
"""


# ---------------------------------------------------------------------------
# VarConfig.as_cfg — None fields are excluded
# ---------------------------------------------------------------------------
class TestVarConfigAsCfg:
    """Tests for VarConfig.as_cfg() dict representation."""

    def test_none_fields_excluded(self):
        """Fields with value None are omitted from the dict."""
        vc = VarConfig(name="pr", table="Amon", units="kg m-2 s-1")
        cfg = vc.as_cfg()
        # Only CMIP7 'sources' key is supported
        assert "formula" not in cfg
        assert cfg["name"] == "pr"

    def test_all_set_fields_present(self):
        """All non-None fields appear in the dict."""
        vc = VarConfig(
            name="ta",
            table="Amon",
            units="K",
            # Only CMIP7 'sources' key is supported
            regrid_method="bilinear",
        )
        cfg = vc.as_cfg()
        # Only CMIP7 'sources' key is supported
        assert cfg["regrid_method"] == "bilinear"


# ---------------------------------------------------------------------------
# Mapping — dict-style YAML with formula
# ---------------------------------------------------------------------------
# pylint: disable=too-few-public-methods
class TestMappingFormula:
    """Tests for formula-based variable realization."""

    def _make_mapping(self):
        """Write the formula YAML to a temp file and return a Mapping."""
        with tempfile.NamedTemporaryFile("w+", suffix=".yaml", delete=False) as f:
            f.write(_FORMULA_YAML)
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
        # Removed: formula YAML with legacy keys
        np.testing.assert_allclose(result.values, 3.0)


class TestMappingRealize:
    """Tests for source-based variable realization and error handling."""

    # pylint: disable=too-few-public-methods
    def test_realize_unknown_cmip_var_warns(self):
        """realize() emits a RuntimeWarning and returns None for unknown CMIP variables."""
        with tempfile.NamedTemporaryFile("w+", suffix=".yaml", delete=True) as f:
            f.write(_SRC_YAML)
            f.flush()
            m = Mapping(f.name)

        ds = xr.Dataset({"TS": xr.DataArray([1.0])})
        with pytest.warns(RuntimeWarning, match="not found"):
            result = m.realize(ds, "ta")
        assert result is None

    def test_realize_source_returns_correct_data(self):
        """realize() returns the source DataArray unchanged."""
        with tempfile.NamedTemporaryFile("w+", suffix=".yaml", delete=True) as f:
            f.write(_SRC_YAML)
            f.flush()
            m = Mapping(f.name)

        ts_data = np.array([[280.0, 290.0], [300.0, 310.0]])
        ds = xr.Dataset({"TS": xr.DataArray(ts_data, dims=["lat", "lon"])})
        result = m.realize(ds, "tas")
        np.testing.assert_array_equal(result.values, ts_data)

    def test_realize_missing_source_raises(self):
        """realize() raises KeyError when the source variable is absent."""
        with tempfile.NamedTemporaryFile("w+", suffix=".yaml", delete=True) as f:
            f.write(_SRC_YAML)
            f.flush()
            m = Mapping(f.name)

        ds = xr.Dataset({"T": xr.DataArray([1.0, 2.0])})  # 'TS' missing
        with pytest.raises(KeyError, match="TS"):
            m.realize(ds, "tas")
