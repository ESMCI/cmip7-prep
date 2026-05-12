"""Tests for currently unused functions like _apply_unit_conversion."""

import pytest
import numpy as np
import xarray as xr

from cmip7_prep.unused_functions import _apply_unit_conversion


class TestApplyUnitConversion:
    """Tests for _apply_unit_conversion dict and string-expression rules."""

    # pylint: disable=too-few-public-methods

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
