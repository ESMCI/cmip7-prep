# cmip7_prep/cache_tools.py
"""Tools for caching and reuse in regridding."""
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging
import xesmf as xe
import xarray as xr
import numpy as np

from cmip7_prep.cmor_utils import bounds_from_centers_1d

logger = logging.getLogger(__name__)


# -------------------------
# NetCDF opener (backends)
# -------------------------
def open_nc(path: Path) -> xr.Dataset:
    """Open NetCDF with explicit engines and narrow exception handling.

    Tries 'netcdf4' then 'scipy'. Collects the failure reasons and raises a
    single RuntimeError if neither works.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Weight file not found: {path}")

    errors: dict[str, Exception] = {}
    for engine in ("netcdf4", "scipy"):
        try:
            return xr.open_dataset(str(path), engine=engine)
        except (ValueError, OSError, ImportError, ModuleNotFoundError) as exc:
            # ValueError: invalid/unavailable engine or decode issue
            # OSError: low-level file I/O/HDF5 issues
            # ImportError/ModuleNotFoundError: backend not installed
            errors[engine] = exc

    details = "; ".join(
        f"{eng}: {type(err).__name__}: {err}" for eng, err in errors.items()
    )
    raise RuntimeError(
        f"Could not open {path} with xarray engines ['netcdf4', 'scipy']. "
        f"Tried both; reasons: {details}"
    )


def _make_dummy_grids(mapfile: Path) -> tuple[xr.Dataset, xr.Dataset]:
    """Construct minimal ds_in/ds_out satisfying xESMF when reusing weights.
    Adds CF-style bounds for both lat and lon so conservative methods don’t
    trigger cf-xarray’s bounds inference on size-1 dimensions."""
    with open_nc(mapfile) as m:
        nlon_in, nlat_in = _get_src_shape(m)
    lat_out_1d, lon_out_1d = _get_dst_latlon_1d()

    # --- Dummy INPUT grid (unstructured → represent as 2D with length-1 lat) ---
    lat_in = np.arange(
        -180.0, 180.0, 360.0 / nlat_in, dtype="f8"
    )  # e.g., [0], length can be 1
    lon_in = np.arange(0.5, 360.5, 360.0 / nlon_in, dtype="f8")
    ds_in = xr.Dataset(
        data_vars={
            "lat_bnds": (
                ("lat", "nbnds"),
                bounds_from_centers_1d(lat_in, "lat"),
            ),
            "lon_bnds": (
                ("lon", "nbnds"),
                bounds_from_centers_1d(lon_in, "lon"),
            ),
        },
        coords={
            "lat": (
                "lat",
                lat_in,
                {
                    "units": "degrees_north",
                    "standard_name": "latitude",
                    "bounds": "lat_bnds",
                },
            ),
            "lon": (
                "lon",
                lon_in,
                {
                    "units": "degrees_east",
                    "standard_name": "longitude",
                    "bounds": "lon_bnds",
                },
            ),
            "nbnds": ("nbnds", np.array([0, 1], dtype="i4")),
        },
    )

    # --- OUTPUT grid from weights (canonical 1° lat/lon) ---
    lat_out_bnds = bounds_from_centers_1d(lat_out_1d, "lat")
    lon_out_bnds = bounds_from_centers_1d(lon_out_1d, "lon")

    ds_out = xr.Dataset(
        data_vars={
            "lat_bnds": (("lat", "nbnds"), lat_out_bnds),
            "lon_bnds": (("lon", "nbnds"), lon_out_bnds),
        },
        coords={
            "lat": (
                "lat",
                lat_out_1d,
                {
                    "units": "degrees_north",
                    "standard_name": "latitude",
                    "bounds": "lat_bnds",
                },
            ),
            "lon": (
                "lon",
                lon_out_1d,
                {
                    "units": "degrees_east",
                    "standard_name": "longitude",
                    "bounds": "lon_bnds",
                },
            ),
            "nbnds": ("nbnds", np.array([0, 1], dtype="i4")),
        },
    )
    return ds_in, ds_out


# -------------------------
# Minimal dummy grids from the weight file (based on your approach)
# -------------------------


def read_array(m: xr.Dataset, *names: str) -> Optional[xr.DataArray]:
    """helper tool to read array from xarray datasets"""
    for n in names:
        if n in m:
            return m[n]
    return None


def _get_src_shape(m: xr.Dataset) -> Tuple[int, int]:
    """Infer the source grid 'shape' expected by xESMF's ds_to_ESMFgrid.

    We provide a dummy 2D shape even when the true source is unstructured.
    """
    a = read_array(m, "src_grid_dims")
    if a is not None:
        vals = np.asarray(a).ravel().astype(int)
        if vals.size == 1:
            return (1, int(vals[0]))
        if vals.size >= 2:
            return (int(vals[-2]), int(vals[-1]))
    # fallbacks for unstructured
    for n in ("src_grid_size", "n_a"):
        if n in m:
            size = int(np.asarray(m[n]).ravel()[0])
            return (1, size)
    # very last resort: infer from max index of sparse matrix rows
    if "row" in m:
        size = int(np.asarray(m["row"]).max())
        return (1, size)
    raise ValueError("Cannot infer source grid size from weight file.")


def _get_dst_latlon_1d() -> Tuple[np.ndarray, np.ndarray]:
    """Return 1D dest lat, lon arrays from weight file.

    Prefers 2D center lat/lon (yc_b/xc_b or lat_b/lon_b), reshaped to (ny, nx),
    then converts to 1D centers by taking first column/row, which is valid for
    regular 1° lat/lon weights.
    """
    # Final fallback: fabricate a 1° grid
    ny, nx = 180, 360
    lat = np.linspace(-89.5, 89.5, ny, dtype="f8")
    lon = (np.arange(nx, dtype="f8") + 0.5) * (360.0 / nx)

    return lat, lon


# -------------------------
# Cache
# -------------------------
class FXCache:
    """Cache of regridded FX fields (sftlf, areacella) keyed by mapfile."""

    _cache: Dict[Path, xr.Dataset] = {}

    @classmethod
    def get(cls, key: Path) -> xr.Dataset | None:
        """get cached variable"""
        return cls._cache.get(key)

    @classmethod
    def put(cls, key: Path, ds_fx: xr.Dataset) -> None:
        """put variable into cache"""
        cls._cache[key] = ds_fx

    @classmethod
    def clear(cls) -> None:
        """clear cache"""
        cls._cache.clear()


class RegridderCache:
    """Cache of xESMF Regridders constructed from weight files.

    We build minimal `ds_in`/`ds_out` from the weight file to satisfy CF checks,
    then reuse the weight file for the actual mapping.
    """

    _cache: Dict[Path, xe.Regridder] = {}

    @classmethod
    def get(cls, mapfile: Path, method_label: str) -> xe.Regridder:
        """Return a cached regridder for the given weight file and method."""
        mapfile = mapfile.expanduser().resolve()
        if mapfile not in cls._cache:
            if not mapfile.exists():
                raise FileNotFoundError(f"Regrid weights not found: {mapfile}")
            ds_in, ds_out = _make_dummy_grids(mapfile)
            logger.info("Creating xESMF Regridder from weights: %s", mapfile)
            cls._cache[mapfile] = xe.Regridder(
                ds_in,
                ds_out,
                method=method_label,
                filename=str(mapfile),  # reuse the ESMF weight file on disk
                reuse_weights=True,
                periodic=True,  # 0..360 longitudes
            )
        return cls._cache[mapfile]

    @classmethod
    def clear(cls) -> None:
        """Clear all cached regridders (useful for tests or releasing resources)."""
        cls._cache.clear()
