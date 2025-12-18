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
import logging
import numpy as np
import xarray as xr
from geocat.comp import interp_hybrid_to_pressure


logger = logging.getLogger(__name__)


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
    >>> import tempfile, json, os
    >>> with tempfile.TemporaryDirectory() as d:
    ...     path = os.path.join(d, "CMIP7_coordinate.json")
    ...     with open(path, "w") as f:
    ...         json.dump({"axis_entry": {"plev19": {"requested": [1000, 500, 100]}}}, f)
    ...     _read_requested_levels(d, axis_name="plev19")
    array([1000.,  500.,  100.])
    """
    coord_json_candidates = [
        "CMIP7_coordinate.json",
        "CMIP6_coordinate.json",
        "CMIP_coordinate.json",
    ]
    coord_json = None
    for name in coord_json_candidates:
        candidate = os.path.join(str(tables_path), name)
        logger.info("Checking for coordinate JSON: %s", candidate)
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
    >>> import xarray as xr
    >>> ds = xr.Dataset({"P0": ((), 95000.0)})
    >>> _resolve_p0(ds)
    95000.0
    >>> ds = xr.Dataset(attrs={"P0": 98000.0})
    >>> _resolve_p0(ds)
    98000.0
    >>> ds = xr.Dataset()
    >>> _resolve_p0(ds)
    100000.0
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

    required = [str(var), ps_name, hyam_name, hybm_name]
    missing = [name for name in required if name not in ds]
    if missing:
        raise KeyError(f"Missing required variables in dataset: {missing}")

    p0 = _resolve_p0(ds, p0_name=p0_name)
    new_levels = _read_requested_levels(tables_path, axis_name="plev19")

    # geocat-comp performs log-pressure interpolation internally
    out_da = interp_hybrid_to_pressure(
        ds[str(var)],
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
    ds_out[str(var)] = out_da

    # Optionally drop hybrid coefficients and P0 if present
    drop = [n for n in (hyam_name, hybm_name, p0_name) if n in ds_out.variables]
    if drop:
        ds_out = ds_out.drop_vars(drop)

    return ds_out


def remap_isopycnal_to_olevel(so_native, z_iface, olevel_bnds, fill_value=np.nan):
    """
    so_native: DataArray [time, k_native, y, x]
    z_iface:   DataArray [time, k_native+1, y, x]  (depth m, positive down)
    olevel_bnds: 1D ndarray [k_target, 2]          (depth bounds m, positive down)
    returns: so_olevel [time, k_target, y, x]
    """

    k_t = olevel_bnds.shape[0]
    # native layer bounds, thickness
    zn0 = z_iface.isel(k_iface=slice(0, -1))
    zn1 = z_iface.isel(k_iface=slice(1, None))
    dz_native = (zn1 - zn0).clip(min=0.0)

    # allocate output
    out = xr.full_like(
        so_native.isel(k_native=slice(0, k_t)).rename(k_native="olevel"), fill_value
    ).isel(olevel=slice(0, k_t))
    out = out.assign_coords(olevel=np.arange(k_t))

    # vectorized overlap (broadcast to [time, k_native, k_target, y, x])
    # target bounds to arrays we can broadcast
    zb0 = xr.DataArray(olevel_bnds[:, 0], dims=["olevel"])
    zb1 = xr.DataArray(olevel_bnds[:, 1], dims=["olevel"])

    # expand dims for broadcasting
    zn0e = zn0.expand_dims({"olevel": k_t}, axis=1)
    zn1e = zn1.expand_dims({"olevel": k_t}, axis=1)
    dz_n = dz_native.expand_dims({"olevel": k_t}, axis=1)
    zb0e = zb0.reshape((1, k_t, 1, 1)).broadcast_like(zn0e)
    zb1e = zb1.reshape((1, k_t, 1, 1)).broadcast_like(zn0e)

    # overlap thickness between native [zn0,zn1] and target [zb0,zb1]
    top = xr.ufuncs.maximum(zn0e, zb0e)
    bottom = xr.ufuncs.minimum(zn1e, zb1e)
    overlap = (bottom - top).clip(min=0.0)

    weights = overlap / dz_n.where(dz_n > 0)
    # normalize per target bin: sum_k (w_k) may be < 1 near boundaries
    wsum = weights.sum(dim="k_native")
    so_wsum = (weights * so_native.expand_dims({"olevel": k_t}, axis=1)).sum(
        dim="k_native"
    )

    out_vals = so_wsum.where(wsum > 0) / wsum.where(wsum > 0)
    out[:] = out_vals.transpose("time", "olevel", "y", "x")
    return out
