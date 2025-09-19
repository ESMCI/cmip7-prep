# cmip7_prep/regrid.py
"""Regridding utilities for CESM -> 1° lat/lon using precomputed ESMF weights."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple

import numpy as np
import xarray as xr
import xesmf as xe

# Default weight maps; override via function args.
DEFAULT_CONS_MAP = Path(
    "/glade/campaign/cesm/cesmdata/inputdata/cpl/gridmaps/ne30pg3/map_ne30pg3_to_1x1d_aave.nc"
)
DEFAULT_BILIN_MAP = Path(
    "/glade/campaign/cesm/cesmdata/inputdata/cpl/gridmaps/ne30pg3/map_ne30pg3_to_1x1d_bilin.nc"
)  # optional bilinear map

# Variables treated as "intensive" → prefer bilinear when available.
INTENSIVE_VARS = {
    "tas",
    "tasmin",
    "tasmax",
    "psl",
    "ps",
    "huss",
    "uas",
    "vas",
    "sfcWind",
    "ts",
    "prsn",
    "clt",
    "ta",
    "ua",
    "va",
    "zg",
    "hus",
    "thetao",
    "uo",
    "vo",
    "so",
}


@dataclass(frozen=True)
class MapSpec:
    """Specification of which weight map to use for a variable."""

    method_label: str  # "conservative" or "bilinear"
    path: Path


# -------------------------
# NetCDF opener (backends)
# -------------------------
def _open_nc(path: Path) -> xr.Dataset:
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


# -------------------------
# Minimal dummy grids from the weight file (based on your approach)
# -------------------------


def _read_array(m: xr.Dataset, *names: str) -> Optional[xr.DataArray]:
    for n in names:
        if n in m:
            return m[n]
    return None


def _get_src_shape(m: xr.Dataset) -> Tuple[int, int]:
    """Infer the source grid 'shape' expected by xESMF's ds_to_ESMFgrid.

    We provide a dummy 2D shape even when the true source is unstructured.
    """
    a = _read_array(m, "src_grid_dims")
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


def _get_dst_latlon_1d(m: xr.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """Return 1D dest lat, lon arrays from weight file.

    Prefers 2D center lat/lon (yc_b/xc_b or lat_b/lon_b), reshaped to (ny, nx),
    then converts to 1D centers by taking first column/row, which is valid for
    regular 1° lat/lon weights.
    """
    lat2d = _read_array(m, "yc_b", "lat_b", "dst_grid_center_lat", "yc", "lat")
    lon2d = _read_array(m, "xc_b", "lon_b", "dst_grid_center_lon", "xc", "lon")
    if lat2d is not None and lon2d is not None:
        # figure out (ny, nx)
        if "dst_grid_dims" in m:
            ny, nx = [int(x) for x in np.asarray(m["dst_grid_dims"]).ravel()][-2:]
        else:
            # try to infer directly from array size
            size = int(np.asarray(lat2d).size)
            # default 1x1 grid size is 180*360
            if size == 180 * 360:
                ny, nx = 180, 360
            else:
                # fallback: assume square-ish
                ny = int(round(np.sqrt(size)))
                nx = size // ny
        lat2d = np.asarray(lat2d).reshape(ny, nx)
        lon2d = np.asarray(lon2d).reshape(ny, nx)
        return lat2d[:, 0].astype("f8"), lon2d[0, :].astype("f8")

    # If 1D lat/lon already present
    lat1d = _read_array(m, "lat", "yc")
    lon1d = _read_array(m, "lon", "xc")
    if lat1d is not None and lon1d is not None and lat1d.ndim == 1 and lon1d.ndim == 1:
        return np.asarray(lat1d, dtype="f8"), np.asarray(lon1d, dtype="f8")

    # Final fallback: fabricate a 1° grid
    ny, nx = 180, 360
    lat = np.linspace(-89.5, 89.5, ny, dtype="f8")
    lon = (np.arange(nx, dtype="f8") + 0.5) * (360.0 / nx)
    return lat, lon


def _make_dummy_grids(mapfile: Path) -> Tuple[xr.Dataset, xr.Dataset]:
    """Construct minimal ds_in/ds_out satisfying xESMF when reusing weights."""
    with _open_nc(mapfile) as m:
        nlat_in, nlon_in = _get_src_shape(m)
        lat_out_1d, lon_out_1d = _get_dst_latlon_1d(m)

    # Dummy input: arbitrary 2D indices, only shapes matter when weights are provided.
    ds_in = xr.Dataset(
        {
            "lat": ("lat", np.arange(nlat_in, dtype="f8")),
            "lon": ("lon", np.arange(nlon_in, dtype="f8")),
        }
    )
    ds_in["lat"].attrs.update({"units": "degrees_north", "standard_name": "latitude"})
    ds_in["lon"].attrs.update({"units": "degrees_east", "standard_name": "longitude"})

    # Output: 1D regular lat/lon extracted from weights
    ds_out = xr.Dataset({"lat": ("lat", lat_out_1d), "lon": ("lon", lon_out_1d)})
    ds_out["lat"].attrs.update({"units": "degrees_north", "standard_name": "latitude"})
    ds_out["lon"].attrs.update({"units": "degrees_east", "standard_name": "longitude"})

    return ds_in, ds_out


# -------------------------
# Cache
# -------------------------


class _RegridderCache:
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


# -------------------------
# Selection & utilities
# -------------------------


def _pick_maps(
    varname: str,
    conservative_map: Optional[Path] = None,
    bilinear_map: Optional[Path] = None,
    force_method: Optional[str] = None,
) -> MapSpec:
    """Choose which precomputed map file to use for a variable."""
    cons = Path(conservative_map) if conservative_map else DEFAULT_CONS_MAP
    bilin = Path(bilinear_map) if bilinear_map else DEFAULT_BILIN_MAP

    if force_method:
        if force_method not in {"conservative", "bilinear"}:
            raise ValueError("force_method must be 'conservative' or 'bilinear'")
        if force_method == "bilinear":
            if not bilin or not str(bilin):
                raise FileNotFoundError("Bilinear map requested but not provided.")
            return MapSpec("bilinear", bilin)
        return MapSpec("conservative", cons)

    if varname in INTENSIVE_VARS and bilin and str(bilin):
        return MapSpec("bilinear", bilin)
    return MapSpec("conservative", cons)


def _ensure_ncol_last(da: xr.DataArray) -> Tuple[xr.DataArray, Tuple[str, ...]]:
    """Move 'ncol' to the last position; return (da, non_spatial_dims)."""
    if "ncol" not in da.dims:
        raise ValueError(f"Expected 'ncol' in dims; got {da.dims}")
    non_spatial = tuple(d for d in da.dims if d != "ncol")
    return da.transpose(*non_spatial, "ncol"), non_spatial


def _rename_xy_to_latlon(da: xr.DataArray) -> xr.DataArray:
    """Normalize 2-D dims to ('lat','lon') if they came out as ('y','x')."""
    dim_map = {}
    if "y" in da.dims:
        dim_map["y"] = "lat"
    if "x" in da.dims:
        dim_map["x"] = "lon"
    return da.rename(dim_map) if dim_map else da


# -------------------------
# Public API
# -------------------------


def regrid_to_1deg(
    ds_in: xr.Dataset,
    varname: str,
    *,
    method: Optional[str] = None,
    conservative_map: Optional[Path] = None,
    bilinear_map: Optional[Path] = None,
    keep_attrs: bool = True,
    dtype: str | None = "float32",
    output_time_chunk: int | None = 12,
) -> xr.DataArray:
    """Regrid a field on (time, ncol[, ...]) to (time, [lev,] lat, lon).

    Parameters
    ----------
    ds_in : Dataset with var on (..., ncol)
    varname : str
        Variable name to regrid.
    method : Optional[str]
        Force "conservative" or "bilinear". If None, choose based on var type.
    conservative_map, bilinear_map : Path
        ESMF weight files. If missing, defaults are used.
    keep_attrs : bool
        Copy attrs from input variable to output.
    dtype : str or None
        Cast input to this dtype before regridding (default float32). Set None to disable.
    output_time_chunk : int or None
        If set and 'time' present, make xESMF return chunked output along 'time'.
    """
    if varname not in ds_in:
        raise KeyError(f"{varname!r} not in dataset.")

    da = ds_in[varname]
    da2, non_spatial = _ensure_ncol_last(da)

    # cast to save memory
    if dtype is not None and str(da2.dtype) != dtype:
        da2 = da2.astype(dtype)

    # keep dask lazy and chunk along time if present
    if "time" in da2.dims and output_time_chunk:
        da2 = da2.chunk({"time": output_time_chunk})

    spec = _pick_maps(
        varname,
        conservative_map=conservative_map,
        bilinear_map=bilinear_map,
        force_method=method,
    )
    regridder = _RegridderCache.get(spec.path, spec.method_label)

    # tell xESMF to produce chunked output
    kwargs = {}
    if "time" in da2.dims and output_time_chunk:
        kwargs["output_chunks"] = {"time": output_time_chunk}

    da2_2d = (
        da2.rename({"ncol": "lon"})
        .expand_dims({"lat": 1})  # add a dummy 'lat' of length 1
        .transpose(*non_spatial, "lat", "lon")  # ensure last two dims are ('lat','lon')
    )
    if "time" in da2_2d.dims and output_time_chunk:
        kwargs["output_chunks"] = {"time": output_time_chunk}

    out = regridder(da2_2d, **kwargs)  # -> (*non_spatial, y/x or lat/lon)
    out = _rename_xy_to_latlon(out)

    if keep_attrs:
        out.attrs.update(da.attrs)

    for c in non_spatial:
        if c in ds_in.coords and c in out.dims:
            out = out.assign_coords({c: ds_in[c]})

    if "lat" in out.coords:
        out["lat"].attrs.setdefault("units", "degrees_north")
        out["lat"].attrs.setdefault("standard_name", "latitude")
    if "lon" in out.coords:
        out["lon"].attrs.setdefault("units", "degrees_east")
        out["lon"].attrs.setdefault("standard_name", "longitude")

    return out


def regrid_mask_or_area(
    da_in: xr.DataArray,
    *,
    conservative_map: Optional[Path] = None,
) -> xr.DataArray:
    """Regrid a mask or cell-area field using conservative weights."""
    if "ncol" not in da_in.dims:
        raise ValueError("Expected 'ncol' in dims for mask/area regridding.")
    if "time" in da_in.dims:
        da_in = da_in.transpose("time", "ncol", ...)

    spec = MapSpec(
        "conservative", Path(conservative_map) if conservative_map else DEFAULT_CONS_MAP
    )
    regridder = _RegridderCache.get(spec.path, spec.method_label)

    out = regridder(da_in)
    out = _rename_xy_to_latlon(out)
    return out
