"""
mom6_static.py: Utilities for reading and extracting grid information from MOM6 static files.
"""

import xarray as xr
import numpy as np
from cmip7_prep.regrid import _sftof_from_native


def ocean_fx_fields(static_path):
    """
    Read MOM6 static grid and write ocean-related fx fields (sftof, areacello, etc).
    Returns a dict of DataArrays for use in regridding normalization/denormalization.
    If out_path is given, writes a NetCDF with these fields.
    """

    ds = xr.open_dataset(static_path)
    fx = {}
    # Extract sftof (sea fraction) using the helper
    sftof = _sftof_from_native(ds)
    if sftof is not None:
        fx["sftof"] = sftof
    # Extract deptho if present
    if "deptho" in ds:
        fx["deptho"] = ds["deptho"]
    # Ensure areacello is included (already present above, but keep for clarity)
    if "areacello" in ds:
        fx["areacello"] = ds["areacello"]
    # Optionally add other ocean mask/area fields as needed

    return fx


def compute_cell_bounds_from_corners(corner_array):
    """
    Given a 2D array of cell corners (shape [ny+1, nx+1]),
    compute bounds for each cell as [min, max].
    Returns a (ny*nx, 2) array for CMOR.
    """
    # For each cell, get the 4 corners and compute min/max
    ny, nx = corner_array.shape[0] - 1, corner_array.shape[1] - 1
    bounds = np.empty((ny * nx, 2), dtype=corner_array.dtype)
    idx = 0
    for j in range(ny):
        for i in range(nx):
            cell_corners = [
                corner_array[j, i],
                corner_array[j, i + 1],
                corner_array[j + 1, i],
                corner_array[j + 1, i + 1],
            ]
            bounds[idx, 0] = np.nanmin(cell_corners)
            bounds[idx, 1] = np.nanmax(cell_corners)
            idx += 1
    return bounds


def load_mom6_grid(static_path):
    """Load MOM6 static file and return geolat, geolon, geolat_c, geolon_c arrays."""
    ds = xr.open_dataset(static_path)
    # For a supergrid, centers are every other point, bounds are full array
    # Target: centers (480, 540), corners (481, 541)
    # For a supergrid, centers are every other point starting at 1,
    # bounds are every other point starting at 0
    geolat = ds["y"].values[1::2, 1::2]  # (480, 540)
    geolon = ds["x"].values[1::2, 1::2]  # (480, 540)
    geolat_c = ds["y"].values[0::2, 0::2]  # (481, 541)
    geolon_c = ds["x"].values[0::2, 0::2]  # (481, 541)
    # Normalize longitudes to 0-360
    geolon = np.mod(geolon, 360.0)
    geolon_c = np.mod(geolon_c, 360.0)
    # Compute bounds arrays using corners

    return geolat, geolon, geolat_c, geolon_c
