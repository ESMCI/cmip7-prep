"""Vertical coordinate handling for CESM → CMIP7.

This module provides utilities to convert hybrid-sigma model levels to requested
pressure levels (e.g., CMIP plev19) prior to CMORization.

Primary entry point:
    to_plev19(ds, var, tables_path, ...)

Dependencies:
    - geocat-comp (preferred): uses `interp_hybrid_to_pressure`
      If not available, this function raises ImportError (fallback can be added later).
"""

from __future__ import annotations

import json
import os

import numpy as np
import xarray as xr
from geocat.comp import interp_hybrid_to_pressure


def _read_requested_levels(
    tables_path: str | os.PathLike, axis_name: str = "plev19"
) -> np.ndarray:
    """Read requested target pressure levels from CMIP coordinate JSON.

    Parameters
    ----------
    tables_path : str or Path
        Directory containing CMIPx coordinate/table JSON files.
    axis_name : str
        Axis entry name to read (e.g., 'plev19').

    Returns
    -------
    np.ndarray
        1-D array of requested pressure levels in Pa.
    """
    coord_json_candidates = [
        "CMIP7_coordinate.json",
        "CMIP6_coordinate.json",
        "CMIP_coordinate.json",
    ]
    coord_json = None
    for name in coord_json_candidates:
        candidate = os.path.join(str(tables_path), name)
        if os.path.exists(candidate):
            coord_json = candidate
            break
    if coord_json is None:
        raise FileNotFoundError(
            f"Could not find a coordinate table JSON under {tables_path!s}; "
            f"tried {coord_json_candidates}"
        )

    with open(coord_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    try:
        req = data["axis_entry"][axis_name]["requested"]
    except Exception as exc:  # pragma: no cover
        raise KeyError(f"Axis entry '{axis_name}' not found in {coord_json}") from exc

    levels = np.asarray(req, dtype="f8")
    return levels


def _resolve_p0(ds: xr.Dataset, p0_name: str = "P0") -> float:
    """Return reference pressure P0 (Pa) from dataset or default to 100000 Pa.

    Tries, in order:
      1) Variable `P0` inside the dataset (scalar or size-1 array)
      2) Global attribute `P0`
      3) Default 100000.0 Pa
    """
    if p0_name in ds:
        da = ds[p0_name]
        # Handle scalar DataArray or size-1 arrays robustly
        try:
            if isinstance(da, xr.DataArray):
                if getattr(da, "ndim", None) == 0 or getattr(da, "size", None) == 1:
                    return float(np.asarray(da.values).reshape(()).item())
            # Fallback: attempt direct float conversion (covers plain scalars)
            return float(da)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            # Continue to other sources
            pass

        # Last resort: try via .values if size-1
        arr = np.asarray(getattr(da, "values", da))
        if getattr(arr, "size", None) == 1:
            try:
                return float(arr.reshape(()).item())
            except (TypeError, ValueError):
                pass

    if p0_name in ds.attrs:
        try:
            return float(ds.attrs[p0_name])
        except (TypeError, ValueError):
            pass

    return 100000.0  # Pa


def to_plev19(
    ds: xr.Dataset,
    var: str,
    tables_path: str | os.PathLike,
    *,
    lev_dim: str = "lev",
    ps_name: str = "PS",
    hyam_name: str = "hyam",
    hybm_name: str = "hybm",
    p0_name: str = "P0",
) -> xr.Dataset:
    """Interpolate a hybrid-level variable to CMIP plev19 pressure levels (Pa).

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing the variable and required hybrid inputs.
        Expected fields: `var`, `PS`, `hyam`, `hybm` (names configurable).
    var : str
        Name of the variable to be interpolated along the vertical.
    tables_path : str or Path
        Path to CMIPx Tables directory (for coordinate JSON).
    lev_dim : str, default "lev"
        Name of the hybrid-level dimension in `var`.
    ps_name, hyam_name, hybm_name, p0_name : str
        Names for surface pressure, hybrid A/B (midpoint) coefficients, and reference pressure.

    Returns
    -------
    xr.Dataset
        A new dataset with `var` replaced by a pressure-level version with dimension `plev`
        and coordinate `plev` set to the requested values (Pa). Drops `hyam`, `hybm`, and `P0`
        if they were present.

    Notes
    -----
    This function prefers geocat-comp's `interp_hybrid_to_pressure`. If geocat-comp
    is not importable in the environment, it will raise ImportError with guidance.
    """

    required = [var, ps_name, hyam_name, hybm_name]
    missing = [name for name in required if name not in ds]
    if missing:
        raise KeyError(f"Missing required variables in dataset: {missing}")

    p0 = _resolve_p0(ds, p0_name=p0_name)
    new_levels = _read_requested_levels(tables_path, axis_name="plev19")

    # geocat-comp performs log-pressure interpolation internally
    out_da = interp_hybrid_to_pressure(
        ds[var],
        ds[ps_name],
        ds[hyam_name],
        ds[hybm_name],
        p0=p0,
        new_levels=new_levels,
        lev_dim=lev_dim,
    )

    # Ensure dimension is named 'plev' and coordinate is present
    if "plev" not in out_da.dims:
        out_da = out_da.rename({lev_dim: "plev"})
    out_da = out_da.assign_coords(plev=("plev", new_levels))
    out_da["plev"].attrs.update(
        {"units": "Pa", "standard_name": "air_pressure", "positive": "down"}
    )

    # Assemble return dataset
    ds_out = ds.copy()
    ds_out[var] = out_da

    # Optionally drop hybrid coefficients and P0 if present
    drop = [n for n in (hyam_name, hybm_name, p0_name) if n in ds_out.variables]
    if drop:
        ds_out = ds_out.drop_vars(drop)

    return ds_out
