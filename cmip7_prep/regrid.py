"""Regridding utilities for CESM -> 1° lat/lon using precomputed ESMF weights."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple

import xarray as xr
import xesmf as xe
import numpy as np

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


class _RegridderCache:
    """Cache of xESMF Regridders constructed from weight files.

    This avoids reconstructing regridders for the same weight file multiple times
    and provides a small API to fetch or clear cached instances.
    """

    _cache: Dict[Path, xe.Regridder] = {}

    @classmethod
    def get(cls, mapfile: Path, method_label: str) -> xe.Regridder:
        """Return a cached regridder for the given weight file and method.

        If no regridder exists yet for `mapfile`, it is created using xESMF with
        `filename=mapfile` (so source/destination grids are read from the weight
        file) and stored in the cache. Subsequent calls reuse the same instance.

        Parameters
        ----------
        mapfile : Path
            Path to an ESMF weight file.
        method_label : str
            xESMF method label; used only for constructor parity.

        Returns
        -------
        xe.Regridder
            Cached or newly created regridder.
        """
        mapfile = mapfile.expanduser().resolve()
        if mapfile not in cls._cache:
            if not mapfile.exists():
                raise FileNotFoundError(f"Regrid weights not found: {mapfile}")
            cls._cache[mapfile] = xe.Regridder(
                xr.Dataset(),
                xr.Dataset(),
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


def _dst_coords_from_map(mapfile: Path) -> Dict[str, xr.DataArray]:
    """Extract dest lat/lon (+bounds if present) from an ESMF map file."""
    with xr.open_dataset(mapfile) as m:
        # centers
        if "lat" in m and "lon" in m:
            lat = m["lat"]
            lon = m["lon"]
        elif "yc" in m and "xc" in m:
            lat = m["yc"].rename("lat")
            lon = m["xc"].rename("lon")
        else:
            dims = np.asarray(m.get("dst_grid_dims", [180, 360])).ravel()
            ny = int(dims[-2]) if dims.size >= 2 else 180
            nx = int(dims[-1]) if dims.size >= 2 else 360
            lat = xr.DataArray(np.linspace(-89.5, 89.5, ny), dims=("lat",), name="lat")
            lon = xr.DataArray(
                (np.arange(nx) + 0.5) * (360.0 / nx), dims=("lon",), name="lon"
            )

        # bounds
        lat_b = None
        lon_b = None
        for cand in ("lat_bnds", "lat_b", "bounds_lat", "lat_bounds", "y_bnds", "yb"):
            if cand in m:
                lat_b = m[cand]
                break
        for cand in ("lon_bnds", "lon_b", "bounds_lon", "lon_bounds", "x_bnds", "xb"):
            if cand in m:
                lon_b = m[cand]
                break

    coords = {"lat": lat, "lon": lon}
    if lat_b is not None:
        coords["lat_bnds"] = lat_b
    if lon_b is not None:
        coords["lon_bnds"] = lon_b
    return coords


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

    # Attach lat/lon (+bounds) from map if available
    try:
        dst_coords = _dst_coords_from_map(spec.path)
        if {"lat", "lon"}.issubset(out.dims):
            out = out.assign_coords(
                {k: v for k, v in dst_coords.items() if k in {"lat", "lon"}}
            )
            for bname in ("lat_bnds", "lon_bnds"):
                if bname in dst_coords:
                    out = out.assign_coords({bname: dst_coords[bname]})
    except (OSError, ValueError, KeyError):
        # Non-fatal: keep whatever coords xESMF provided
        pass

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

    try:
        dst_coords = _dst_coords_from_map(spec.path)
        if {"lat", "lon"}.issubset(out.dims):
            out = out.assign_coords(
                {k: v for k, v in dst_coords.items() if k in {"lat", "lon"}}
            )
            for bname in ("lat_bnds", "lon_bnds"):
                if bname in dst_coords:
                    out = out.assign_coords({bname: dst_coords[bname]})
    except (OSError, ValueError, KeyError):
        pass

    return out
