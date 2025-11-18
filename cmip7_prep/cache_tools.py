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
    lat_out_1d, lon_out_1d = _get_dst_latlon_1d(mapfile=mapfile)

    # --- Dummy INPUT grid (unstructured → represent as 2D with length-1 lat) ---
    lat_in = np.arange(
        -90.0, 90.0, 180.0 / nlat_in, dtype="f8"
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
            return (int(vals[0]), 1)
        if vals.size >= 2:
            return (int(vals[-2]), int(vals[-1]))
    # fallbacks for unstructured
    for n in ("src_grid_size", "n_a"):
        if n in m:
            size = int(np.asarray(m[n]).ravel()[0])
            return (size, 1)
    # very last resort: infer from max index of sparse matrix rows
    if "row" in m:
        size = int(np.asarray(m["row"]).max())
        return (size, 1)
    raise ValueError("Cannot infer source grid size from weight file.")


def _get_dst_latlon_1d(mapfile: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return 1D dest lat, lon arrays from mapping (weight) file if available,
    else fabricate a 1° grid."""
    try:
        ds = open_nc(mapfile)
        # Try common variable names for 2D center lat/lon
        for lat_name, lon_name in [
            ("yc_b", "xc_b"),
            ("lat_b", "lon_b"),
            ("lat", "lon"),
        ]:
            if lat_name in ds and lon_name in ds:
                lat2d = ds[lat_name].values
                lon2d = ds[lon_name].values
                # If 2D, take first column/row for 1D
                if lat2d.ndim == 2 and lon2d.ndim == 2:
                    lat1d = lat2d[:, 0]
                    lon1d = lon2d[0, :]
                    return lat1d, lon1d
                # If already 1D
                if lat2d.ndim == 1 and lon2d.ndim == 1:
                    logger.info(
                        "Destination lat/lon read as 1D from %s/%s", lat_name, lon_name
                    )
                    lat1d = lat2d[::360]
                    lon1d = lon2d[:360]
                    return lat1d, lon1d
        # Fallback: try to infer from dimensions
        if "lat" in ds.dims and "lon" in ds.dims:
            lat = ds["lat"].values
            lon = ds["lon"].values
            return lat, lon
    except Exception as e:
        logger.warning(f"Could not read lat/lon from mapping file {mapfile}: {e}")
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

    _cache: Dict[tuple, xe.Regridder] = {}

    @classmethod
    def get(
        cls,
        mapfile: Path,
        method_label: str,
        src_mask: xr.DataArray | None = None,
        dst_mask: xr.DataArray | None = None,
    ) -> xe.Regridder:
        """Return a cached regridder for the given weight file, method, and masks."""
        mapfile = mapfile.expanduser().resolve()

        # Build a cache key that includes src_mask and dst_mask presence and identity
        def mask_id(mask):
            if mask is None:
                return None
            try:
                return (mask.shape, str(mask.dtype), hash(mask.values.tobytes()))
            except Exception:
                return (mask.shape, str(mask.dtype), id(mask))

        src_mask_id = mask_id(src_mask)
        dst_mask_id = mask_id(dst_mask)
        cache_key = (str(mapfile), method_label, src_mask_id, dst_mask_id)

        if cache_key not in cls._cache:
            if not mapfile.exists():
                raise FileNotFoundError(f"Regrid weights not found: {mapfile}")
            ds_in, ds_out = _make_dummy_grids(mapfile)

            # Attach masks to dummy grids if provided
            if src_mask is not None:
                ds_in["mask"] = src_mask
            # if dst_mask is not None:
            #    ds_out["mask"] = dst_mask

            logger.info(
                "Creating xESMF Regridder from weights: %s (with masks)", mapfile
            )
            # import pdb; pdb.set_trace()
            cls._cache[cache_key] = xe.Regridder(
                ds_in,
                ds_out,
                weights=str(mapfile),
                # results seem insensitive to this method choice
                method=method_label,
                filename=str(mapfile),  # reuse the ESMF weight file on disk
                reuse_weights=True,
                periodic=True,  # 0..360 longitudes
            )
        return cls._cache[cache_key]

    @classmethod
    def clear(cls) -> None:
        """Clear all cached regridders (useful for tests or releasing resources)."""
        cls._cache.clear()
