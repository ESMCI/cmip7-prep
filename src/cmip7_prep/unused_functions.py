"""Utility functions for unit conversion and related transformations."""
from typing import Any

import xarray as xr

from cmip7_prep.mapping_compat import _safe_eval


def _apply_unit_conversion(da: xr.DataArray, rule: Any) -> xr.DataArray:
    """Apply a unit conversion rule to a DataArray.

    >>> import xarray as xr
    >>> da = xr.DataArray([1.0, 2.0, 3.0])
    >>> _apply_unit_conversion(da, {"scale": 2.0}).values.tolist()
    [2.0, 4.0, 6.0]
    >>> _apply_unit_conversion(da, {"offset": 10.0}).values.tolist()
    [11.0, 12.0, 13.0]
    >>> _apply_unit_conversion(da, "x * 2").values.tolist()
    [2.0, 4.0, 6.0]
    """
    if isinstance(rule, str):
        try:
            out = _safe_eval(rule, {"x": da})
        except Exception as exc:
            raise ValueError(
                f"Error evaluating unit_conversion expression: {exc}"
            ) from exc
        if not isinstance(out, xr.DataArray):
            raise ValueError("unit_conversion expression did not return a DataArray")
        return out

    if isinstance(rule, dict):
        scale = rule.get("scale", 1.0)
        offset = rule.get("offset", 0.0)
        return da * float(scale) + float(offset)

    raise TypeError(
        "unit_conversion must be a string expression or a dict with 'scale'/'offset'"
    )
