# cmip7_prep/cache_tools.py
"""Tools for caching and reuse in regridding."""
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging
import xesmf as xe
import xarray as xr
import numpy as np

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

    weights = xr.open_dataset(mapfile)
    in_shape = weights.src_grid_dims.load().data

    # Since xESMF expects 2D vars, we'll insert a dummy dimension of size-1
    if len(in_shape) == 1:
        in_shape = [1, in_shape.item()]

    # output variable shape
    out_shape = weights.dst_grid_dims.load().data.tolist()[::-1]

    # Some prep to get the bounds:
    # Note that bounds are needed for conservative regridding and not for bilinear
    lat_b_out = np.zeros(out_shape[0] + 1)
    lon_b_out = weights.xv_b.data[: out_shape[1] + 1, 0]
    lat_b_out[:-1] = weights.yv_b.data[np.arange(out_shape[0]) * out_shape[1], 0]
    lat_b_out[-1] = weights.yv_b.data[-1, -1]

    dummy_in = xr.Dataset(
        {
            "lat": ("lat", np.empty((in_shape[0],))),
            "lon": ("lon", np.empty((in_shape[1],))),
            "lat_b": ("lat_b", np.empty((in_shape[0] + 1,))),
            "lon_b": ("lon_b", np.empty((in_shape[1] + 1,))),
        }
    )
    dummy_out = xr.Dataset(
        {
            "lat": ("lat", weights.yc_b.data.reshape(out_shape)[:, 0]),
            "lon": ("lon", weights.xc_b.data.reshape(out_shape)[0, :]),
            "lat_b": ("lat_b", lat_b_out),
            "lon_b": ("lon_b", lon_b_out),
        }
    )

    return dummy_in, dummy_out


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
    except (AttributeError, KeyError, IndexError, ValueError) as e:
        logger.warning("Could not read lat/lon from mapping file %s: %s", mapfile, e)
    # Final fallback: fabricate a 1° grid
    ny, nx = 180, 360
    lat = np.linspace(-89.5, 89.5, ny, dtype="f8")
    lon = (np.arange(nx, dtype="f8") + 0.5) * (360.0 / nx)
    return lat, lon


# -------------------------
# Cache
# -------------------------
class FXCache:
    """Cache of regridded FX fields (sftlf, areacella,
    areacello, sftof, deptho) keyed by mapfile."""

    _cache: Dict[Path, xr.Dataset] = {}

    @classmethod
    def get(cls, key: Path) -> xr.Dataset | None:
        """get cached variable"""
        return cls._cache.get(key)

    @classmethod
    def put(cls, key: Path, ds_fx: xr.Dataset) -> None:
        """put variable into cache"""
        logger.info("Caching FX fields : %s", ds_fx.data_vars.keys())
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
    ) -> xe.Regridder:
        """Return a cached regridder for the given weight file, method, and masks."""
        mapfile = mapfile.expanduser().resolve()

        # Build a cache key

        cache_key = (str(mapfile), method_label)

        if cache_key not in cls._cache:
            if not mapfile.exists():
                raise FileNotFoundError(f"Regrid weights not found: {mapfile}")
            ds_in, ds_out = _make_dummy_grids(mapfile)

            logger.info("Creating xESMF Regridder from weights: %s", mapfile)
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
