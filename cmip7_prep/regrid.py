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
DEFAULT_CONS_MAP = Path("map_ne30pg3_to_1x1d_aave.nc")
DEFAULT_BILIN_MAP = Path("")  # optional bilinear map

# Variables treated as "intensive" → prefer bilinear when available.
INTENSIVE_VARS = {
    "tas", "tasmin", "tasmax", "psl", "ps", "huss", "uas", "vas", "sfcWind",
    "ts", "prsn", "clt", "ta", "ua", "va", "zg", "hus", "thetao", "uo", "vo", "so",
}

@dataclass(frozen=True)
class MapSpec:
    """Specification of which weight map to use for a variable."""
    method_label: str  # "conservative" or "bilinear"
    path: Path

# -------------------------
# helpers to build minimal grids from weight file
# -------------------------

def _first_var(m: xr.Dataset, *names: str) -> Optional[xr.DataArray]:
    for n in names:
        if n in m:
            return m[n]
    return None

def _src_coords_from_map(mapfile: Path) -> Tuple[xr.DataArray, xr.DataArray]:
    """Return (lat_in, lon_in) as 1-D arrays for the source (unstructured) grid."""
    with xr.open_dataset(mapfile) as m:
        lat = _first_var(m, "lat_a", "yc_a", "src_grid_center_lat", "y_a", "y_A")
        lon = _first_var(m, "lon_a", "xc_a", "src_grid_center_lon", "x_a", "x_A")
        if lat is not None and lon is not None:
            lat = lat.rename("lat").astype("f8").reset_coords(drop=True)
            lon = lon.rename("lon").astype("f8").reset_coords(drop=True)
            # ensure 1-D
            lat = lat if lat.ndim == 1 else lat.stack(points=lat.dims).rename("lat")
            lon = lon if lon.ndim == 1 else lon.stack(points=lon.dims).rename("lon")
            return lat, lon

        # fallback: size only
        size = None
        for c in ("src_grid_size", "n_a"):
            if c in m:
                try:
                    size = int(np.asarray(m[c]).ravel()[0])
                    break
                except Exception:
                    pass
        if size is None:
            # try dims on sparse index variables
            for v in ("row", "col"):
                if v in m and m[v].dims:
                    # ESMF uses 1-based indices; not directly size, but better than nothing
                    size = int(np.asarray(m[v]).max())
                    break
        if size is None:
            raise ValueError("Cannot infer source grid size from weight file.")

    lat = xr.DataArray(np.zeros(size, dtype="f8"), dims=("points",), name="lat")
    lon = xr.DataArray(np.zeros(size, dtype="f8"), dims=("points",), name="lon")
    return lat, lon

def _dst_coords_from_map(mapfile: Path) -> Dict[str, xr.DataArray]:
    """Extract dest lat/lon (+bounds if present) from an ESMF map file."""
    with xr.open_dataset(mapfile) as m:
        # centers
        lat = _first_var(m, "lat_b", "yc_b", "lat", "yc", "dst_grid_center_lat")
        lon = _first_var(m, "lon_b", "xc_b", "lon", "xc", "dst_grid_center_lon")

        if lat is not None and lon is not None:
            lat = lat.rename("lat").astype("f8")
            lon = lon.rename("lon").astype("f8")
            # If 2-D curvilinear, keep dims; if 1-D, leave as-is
        else:
            # fallback from dims
            dims = None
            for name in ("dst_grid_dims", "dst_grid_size"):
                if name in m:
                    dims = np.asarray(m[name]).ravel()
                    break
            if dims is None or dims.size < 2:
                # assume 1° regular
                ny, nx = 180, 360
            else:
                if dims.size == 1:
                    ny, nx = int(dims[0]), 1
                else:
                    ny, nx = int(dims[-2]), int(dims[-1])
            lat = xr.DataArray(np.linspace(-89.5, 89.5, ny), dims=("lat",), name="lat")
            lon = xr.DataArray((np.arange(nx) + 0.5) * (360.0 / nx), dims=("lon",), name="lon")

        # bounds (optional)
        lat_b = _first_var(m, "lat_bnds", "lat_b", "bounds_lat", "lat_bounds", "y_bnds", "yb")
        lon_b = _first_var(m, "lon_bnds", "lon_b", "bounds_lon", "lon_bounds", "x_bnds", "xb")

    coords = {"lat": lat, "lon": lon}
    if lat_b is not None:
        coords["lat_bnds"] = lat_b.astype("f8")
    if lon_b is not None:
        coords["lon_bnds"] = lon_b.astype("f8")
    return coords

def _make_ds_in_out_from_map(mapfile: Path) -> Tuple[xr.Dataset, xr.Dataset]:
    """Construct minimal CF-like ds_in (unstructured) and ds_out (structured/curvilinear) from weight file."""
    lat_in, lon_in = _src_coords_from_map(mapfile)
    dst = _dst_coords_from_map(mapfile)

    # ds_in: unstructured → 1-D lat/lon on 'points'
    if lat_in.dims != ("points",):
        lat_in = lat_in.rename({lat_in.dims[0]: "points"})
    if lon_in.dims != ("points",):
        lon_in = lon_in.rename({lon_in.dims[0]: "points"})
    ds_in = xr.Dataset({"lat": lat_in, "lon": lon_in})
    ds_in["lat"].attrs.update({"units": "degrees_north", "standard_name": "latitude"})
    ds_in["lon"].attrs.update({"units": "degrees_east", "standard_name": "longitude"})

    # ds_out: accept 1-D lat/lon (regular) or 2-D (curvilinear) from weights
    ds_out = xr.Dataset({k: v for k, v in dst.items() if k in {"lat", "lon"}})
    for k in ("lat", "lon"):
        if k in ds_out:
            ds_out[k].attrs.update(
                {"units": f"degrees_{'north' if k == 'lat' else 'east'}",
                 "standard_name": "latitude" if k == "lat" else "longitude"}
            )
    return ds_in, ds_out

class _RegridderCache:
    """Cache of xESMF Regridders constructed from weight files.

    This avoids reconstructing regridders for the same weight file multiple times
    and provides a small API to fetch or clear cached instances.
    """
    _cache: Dict[Path, xe.Regridder] = {}

    @classmethod
    def get(cls, mapfile: Path, method_label: str) -> xe.Regridder:
        """Return a cached regridder for the given weight file and method.

        We build minimal `ds_in`/`ds_out` by reading lon/lat from the weight file.
        This satisfies xESMF's CF checks even when we reuse weights.
        """
        mapfile = mapfile.expanduser().resolve()
        if mapfile not in cls._cache:
            if not mapfile.exists():
                raise FileNotFoundError(f"Regrid weights not found: {mapfile}")
            ds_in, ds_out = _make_ds_in_out_from_map(mapfile)
            cls._cache[mapfile] = xe.Regridder(
                ds_in, ds_out,
                method=method_label,
                filename=str(mapfile),
                reuse_weights=True,
            )
        return cls._cache[mapfile]

    @classmethod
    def clear(cls) -> None:
        """Clear all cached regridders (useful for tests or releasing resources)."""
        cls._cache.clear()

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

def regrid_to_1deg(
    ds_in: xr.Dataset,
    varname: str,
    *,
    method: Optional[str] = None,
    conservative_map: Optional[Path] = None,
    bilinear_map: Optional[Path] = None,
    keep_attrs: bool = True,
) -> xr.DataArray:
    """Regrid a field on (time, ncol[, ...]) to (time, [lev,] lat, lon)."""
    if varname not in ds_in:
        raise KeyError(f"{varname!r} not in dataset.")

    da = ds_in[varname]
    da2, non_spatial = _ensure_ncol_last(da)

    spec = _pick_maps(
        varname,
        conservative_map=conservative_map,
        bilinear_map=bilinear_map,
        force_method=method,
    )
    regridder = _RegridderCache.get(spec.path, spec.method_label)

    out = regridder(da2)  # -> (*non_spatial, y/x or lat/lon)
    out = _rename_xy_to_latlon(out)

    # Try to attach standard attrs
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

    spec = MapSpec("conservative", Path(conservative_map) if conservative_map else DEFAULT_CONS_MAP)
    regridder = _RegridderCache.get(spec.path, spec.method_label)

    out = regridder(da_in)
    out = _rename_xy_to_latlon(out)
    return out

