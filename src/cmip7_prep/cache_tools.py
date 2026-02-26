# cmip7_prep/cache_tools.py
"""Tools for caching and reuse in regridding."""
from pathlib import Path
from typing import Dict
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
    dumbdim = False
    # Since xESMF expects 2D vars, we'll insert a dummy dimension of size-1
    if len(in_shape) == 1:
        dumbdim = True
        in_shape = [1, in_shape.item()]

    # output variable shape
    out_shape = weights.dst_grid_dims.load().data.tolist()[::-1]
    logger.info("in_shape  from weights: %s", in_shape)
    # Some prep to get the bounds:
    # Note that bounds are needed for conservative regridding and not for bilinear
    lat_b_out = np.zeros(out_shape[0] + 1)
    lon_b_out = weights.xv_b.data[: out_shape[1] + 1, 0]
    lat_b_out[:-1] = weights.yv_b.data[np.arange(out_shape[0]) * out_shape[1], 0]
    lat_b_out[-1] = weights.yv_b.data[-1, -1]
    if dumbdim:
        dummy_in = xr.Dataset(
            {
                "lat": ("lat", np.empty((in_shape[0],))),
                "lon": ("lon", np.empty((in_shape[1],))),
                "lat_b": ("lat_b", np.empty((in_shape[0] + 1,))),
                "lon_b": ("lon_b", np.empty((in_shape[1] + 1,))),
            }
        )
    else:
        dummy_in = xr.Dataset(
            {
                "lat": ("lat", np.empty((in_shape[1],))),
                "lon": ("lon", np.empty((in_shape[0],))),
                "lat_b": ("lat_b", np.empty((in_shape[1] + 1,))),
                "lon_b": ("lon_b", np.empty((in_shape[0] + 1,))),
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
