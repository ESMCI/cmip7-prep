# Example usage (in docstring):
# ds = xr.open_dataset("input.nc")
# zonal_p39 = zonal_mean_on_p39(ds, "ta", tables_path="cmip7-cmor-tables/tables")
# cmip7_prep/regrid.py
"""Regridding utilities for CESM -> 1° lat/lon using precomputed ESMF weights."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import logging
import os
import xarray as xr

# import warnings
import numpy as np
from cmip7_prep.cmor_utils import bounds_from_centers_1d
from cmip7_prep.cache_tools import FXCache, RegridderCache, read_array, open_nc
from cmip7_prep import vertical

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# try:
#    import xesmf as xe
# except ModuleNotFoundError as e:
#    _HAS_XESMF = False
try:
    import dask.array as _da  # noqa: F401

    _HAS_DASK = True
except ModuleNotFoundError as e:
    _HAS_DASK = False

# Default weight maps; override via function args.
INPUTDATA_DIR = Path("/glade/campaign/cesm/cesmdata/inputdata/")
DEFAULT_CONS_MAP_NE30 = Path(
    INPUTDATA_DIR / "cpl/gridmaps/ne30pg3/map_ne30pg3_to_1x1d_aave.nc"
)
DEFAULT_BILIN_MAP_NE30 = Path(
    INPUTDATA_DIR / "cpl/gridmaps/ne30pg3/map_ne30pg3_to_1x1d_bilin.nc"
)  # optional bilinear map

DEFAULT_CONS_MAP_T232 = Path(
    INPUTDATA_DIR / "cpl/gridmaps/tx2_3v2/map_t232_TO_1x1d_aave.251023.nc"
)
DEFAULT_BILIN_MAP_T232 = Path(
    INPUTDATA_DIR / "cpl/gridmaps/tx2_3v2/map_t232_TO_1x1d_blin.251023.nc"
)  # optional bilinear map
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
    "so",
    "uo",
    "vo",
}


@dataclass(frozen=True)
class MapSpec:
    """Specification of which weight map to use for a variable."""

    method_label: str  # "conservative" or "bilinear"
    path: Path


def _attach_vertical_metadata(ds_out: xr.Dataset, ds_src: xr.Dataset) -> xr.Dataset:
    """
    Pass-through vertical metadata needed for hybrid-sigma:
      - level midpoints: lev (1D)
      - level interfaces: ilev (1D)  -> bounds for lev
      - hybrid coefficients: hyam, hybm (mid), hyai, hybi (interfaces)
      - p0/P0 scalar
    Does not regrid any of these (they are non-horizontal).
    """
    # carry 'lev' coord if the field uses it

    if "lev" in ds_src and "lev" not in ds_out.coords:
        ds_out = ds_out.assign_coords(lev=ds_src["lev"])

    # carry interface levels and hybrid coeffs if present
    for name in ("ilev", "hyam", "hybm", "hyai", "hybi", "P0", "p0"):
        if name in ds_src and name not in ds_out:
            ds_out[name] = ds_src[name]

    # ensure lev points to ilev as bounds (what CMOR expects)
    if "lev" in ds_out and "ilev" in ds_out:
        attrs = dict(ds_out["lev"].attrs)
        attrs.setdefault("units", "1")
        attrs["bounds"] = "ilev"
        ds_out["lev"].attrs = attrs

    return ds_out


# Variables treated as "intensive" → prefer bilinear when available.
def zonal_mean_on_pressure_grid(
    ds: xr.Dataset,
    var: str,
    *,
    tables_path: str | os.PathLike = "cmip7-cmor-tables/tables",
    target: str = "plev39",
    lev_dim: str = "lev",
    lon_dim: str = "lon",
    ps_name: str = "PS",
    hyam_name: str = "hyam",
    hybm_name: str = "hybm",
    p0_name: str = "P0",
    keep_attrs: bool = True,
) -> xr.DataArray:
    """
    Compute zonal mean (average over longitude) and interpolate
    to a target CMIP pressure grid (e.g., plev19, plev39).

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing the variable and required hybrid inputs.
    var : str
        Name of the variable to process.
    tables_path : str or Path
        Path to CMIP7 Tables directory (for coordinate JSON).
    target : str, default "plev39"
        Name of the target pressure grid (e.g., 'plev19', 'plev39').
    lev_dim, lon_dim, lat_dim : str
        Names for vertical, longitude, and latitude dimensions.
    ps_name, hyam_name, hybm_name, p0_name : str
        Names for surface pressure, hybrid A/B (midpoint) coefficients, and reference pressure.
    keep_attrs : bool
        Whether to keep variable attributes in the output.

    Returns
    -------
    xr.DataArray
        Zonal mean of the variable, interpolated to the target
        pressure levels, dims: (plev, lat, [time]).
    """
    # 1. Zonal mean (average over longitude)
    if lon_dim not in ds[var].dims:
        raise ValueError(
            f"Longitude dimension '{lon_dim}' not found in variable '{var}'"
        )
    zonal = ds[var].mean(dim=lon_dim, keep_attrs=keep_attrs)

    # 2. Build a new dataset for vertical interpolation
    required = [ps_name, hyam_name, hybm_name]
    missing = [name for name in required if name not in ds]
    if missing:
        raise KeyError(f"Missing required variables in dataset: {missing}")

    ds_zonal = ds.copy()
    ds_zonal[var] = zonal

    # 3. Interpolate to the requested pressure grid using vertical.to_plev
    out_ds = vertical.to_plev(
        ds_zonal,
        var,
        tables_path,
        target=target,
        lev_dim=lev_dim,
        ps_name=ps_name,
        hyam_name=hyam_name,
        hybm_name=hybm_name,
        p0_name=p0_name,
    )
    # Output dims: (plev, lat, [time])
    return out_ds[var]


# -------------------------
# Selection & utilities
# -------------------------
def _sftof_from_native(ds: xr.Dataset) -> xr.DataArray | None:
    """Return sea fraction (sftof) from ocean static grid, using 'wet' mask if present."""
    # Try common names for ocean fraction
    for name in ["sftof", "ocnfrac", "wet"]:
        if name in ds:
            v = ds[name]
            # If 'wet', convert 0/1 to percent
            if name == "wet":
                vmax = float(np.nanmax(np.asarray(v)))
                # If 0/1, convert to percent
                if vmax <= 1.0 + 1e-6:
                    out = v * 100.0
                else:
                    out = v
            else:
                out = v
            out = out.clip(min=0.0, max=100.0)
            out = out.astype("f8")
            attrs = dict(out.attrs)
            attrs["units"] = "%"
            attrs.setdefault("standard_name", "sea_area_fraction")
            attrs.setdefault("long_name", "Percentage of sea area")
            out.attrs = attrs
            return out
    return None


def _hybrid_support_names(cfg: dict) -> set[str]:
    """Return names needed for hybrid-sigma CMOR axis/z-factors, based on mapping cfg."""
    levels = (cfg or {}).get("levels") or {}
    if (levels.get("name") or "").lower() != "standard_hybrid_sigma":
        return set()
    # mapping keys with sensible defaults for CESM
    return {
        levels.get("src_axis_name", "lev"),
        levels.get("src_axis_bnds", "ilev"),
        levels.get("hyam", "hyam"),
        levels.get("hybm", "hybm"),
        levels.get("hyai", "hyai"),
        levels.get("hybi", "hybi"),
        levels.get("p0", "P0"),
        levels.get("ps", "PS"),  # PS needs regridding (handled separately below)
    }


def _pick_maps(
    varname: str,
    conservative_map: Optional[Path] = None,
    bilinear_map: Optional[Path] = None,
    force_method: Optional[str] = None,
    realm: Optional[str] = None,
) -> MapSpec:
    """Choose which precomputed map file to use for a variable."""
    if realm == "ocn":
        cons = Path(conservative_map) if conservative_map else DEFAULT_CONS_MAP_T232
        bilin = Path(bilinear_map) if bilinear_map else DEFAULT_BILIN_MAP_T232
    else:
        cons = Path(conservative_map) if conservative_map else DEFAULT_CONS_MAP_NE30
        bilin = Path(bilinear_map) if bilinear_map else DEFAULT_BILIN_MAP_NE30

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
    if "ncol" in da.dims:
        hdim = "ncol"
    elif "lndgrid" in da.dims:
        hdim = "lndgrid"
    else:
        raise ValueError(f"Expected 'ncol' or 'lndgrid' in dims; got {da.dims}")
    non_spatial = tuple(d for d in da.dims if d != hdim)
    return da.transpose(*non_spatial, hdim), non_spatial


# -------------------------
# Public API
# -------------------------
# regrid.py


def regrid_to_1deg_ds(
    ds_in: xr.Dataset,
    varnames: str | list[str],
    *,
    time_from: xr.Dataset | None = None,
    method: Optional[str] = None,
    conservative_map: Optional[Path] = None,
    bilinear_map: Optional[Path] = None,
    keep_attrs: bool = True,
    dtype: str | None = "float32",
    output_time_chunk: int | None = 12,
    sftlf_path: Optional[Path] = None,
) -> xr.Dataset:
    """Regrid var(s) and return a Dataset."""

    names = [varnames] if isinstance(varnames, str) else list(varnames)
    out_vars: dict[str, xr.DataArray] = {}

    for name in names:
        out_vars[name] = regrid_to_1deg(
            ds_in,
            name,
            method=method,
            conservative_map=conservative_map,
            bilinear_map=bilinear_map,
            keep_attrs=keep_attrs,
            dtype=dtype,
            output_time_chunk=output_time_chunk,
        )

    ds_out = xr.Dataset(out_vars)

    # Attach time (and bounds) from the original dataset if requested
    if time_from is not None:
        ds_out = _attach_time_and_bounds(ds_out, time_from)
    if "ncol" in ds_in.dims:
        realm = "atm"
    elif "lndgrid" in ds_in.dims:
        realm = "lnd"
    else:
        realm = "ocn"

    # Pick the mapfile you used for conservative/bilinear selection
    spec = _pick_maps(
        varnames[0] if isinstance(varnames, list) else varnames,
        conservative_map=conservative_map,
        bilinear_map=bilinear_map,
        force_method="conservative",
        realm=realm,
    )  # fx always conservative
    logger.info("using fx map: %s", spec.path)
    ds_fx = _regrid_fx_once(spec.path, ds_in, sftlf_path)  # ← uses cache

    if ds_fx:
        # Don’t overwrite if user already computed and passed them in
        for name in (
            "sftlf",
            "sftof",
            "areacella",
            "orog",
            "lat",
            "lon",
            "lat_bnds",
            "lon_bnds",
        ):
            if name in ds_fx and name not in ds_out:
                ds_out[name] = ds_fx[name]

    # --- NEW: carry hybrid metadata (or any requested 1-D fields) unchanged ---
    ds_out = _attach_vertical_metadata(ds_out, ds_in)
    return ds_out


def _normalize_land_field(da2: xr.DataArray, ds_in: xr.Dataset) -> xr.DataArray:
    """Normalize source field by source landfrac."""
    logger.info("Normalizing land field by source landfrac")
    if "landfrac" not in ds_in:
        raise ValueError("Missing required variables for land normalization: landfrac")
    landfrac = ds_in["landfrac"].fillna(0)
    norm_src = da2.fillna(0) * landfrac
    return norm_src


def _denormalize_land_field(
    out_norm: xr.DataArray, ds_in: xr.Dataset, mapfile: Path
) -> xr.DataArray:
    """Denormalize field by destination landfrac."""
    logger.info("Denormalizing land field ")
    ds_fx = _regrid_fx_once(mapfile, ds_in)
    if "sftlf" not in ds_fx:
        raise ValueError(
            "Missing required variables for land denormalization: landfrac"
        )
    landfrac = ds_fx["sftlf"] / 100.0  # percent to fraction
    logger.debug("sum of regridded landfrac: %s", float(landfrac.sum().values))

    out = out_norm / landfrac.where(landfrac > 0)
    return out


def _denormalize_ocn_field(
    out_norm: xr.DataArray, ds_in: xr.Dataset, mapfile: Path
) -> xr.DataArray:
    """Denormalize field by destination sftof (sea fraction)."""
    logger.info("Denormalizing ocean field by destination sftof (sea fraction)")
    # Try to find an existing regridded sftof file first
    sftof_dst = None
    outdir = Path(mapfile).parent.parent.parent / "Ofx" / "sftof" / "gr"
    # This path logic may need adjustment for your output structure
    if outdir.exists():
        files = list(outdir.glob("sftof_Ofx_*.nc"))
        if files:
            try:
                ds_fx_file = xr.open_dataset(files[0])
                if "sftof" in ds_fx_file:
                    sftof_dst = ds_fx_file["sftof"]

            # pylint: disable=broad-exception-caught
            except Exception as err:
                logger.warning("Failed to read regridded sftof file: %s", err)
    if sftof_dst is None:
        ds_fx = _regrid_fx_once(mapfile, ds_in)
        if "sftof" in ds_fx:
            sftof_dst = ds_fx["sftof"]
        else:
            logger.warning(
                "Destination sftof not found; falling back to source sftof if available."
            )
    logger.info("sftof_dst dims: %s", sftof_dst.dims if sftof_dst is not None else None)
    if sftof_dst is not None:
        frac_dst = sftof_dst / 100.0
        out = out_norm / frac_dst.where(frac_dst > 0)
    else:
        out = out_norm
    return out


def _dst_latlon_1d_from_map(mapfile: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return canonical 1-D (lat, lon) for the destination grid.

    Robust to map files whose stored dst_grid_dims or reshape order is swapped.
    We derive 1-D axes by taking UNIQUE values from the 2-D center fields.
    """
    with open_nc(mapfile) as m:
        lat2d = read_array(m, "yc_b", "lat_b", "dst_grid_center_lat", "yc", "lat")
        lon2d = read_array(m, "xc_b", "lon_b", "dst_grid_center_lon", "xc", "lon")

        if lat2d is not None and lon2d is not None:
            lat2 = np.asarray(lat2d).reshape(-1)  # flatten safely
            lon2 = np.asarray(lon2d).reshape(-1)

            # Take unique values (rounded to avoid tiny FP noise)
            lat_unique = np.unique(lat2.round(6))
            lon_unique = np.unique(lon2.round(6))

            # If either came out descending, sort ascending
            lat1d = np.sort(lat_unique).astype("f8")
            lon1d = np.sort(lon_unique).astype("f8")

            # If longitudes are in [-180,180], convert to [0,360)
            if lon1d.min() < 0.0 or lon1d.max() <= 180.0:
                lon1d = (lon1d % 360.0).astype("f8")
                lon1d.sort()

            # Prefer the classic 180/360 lengths if present
            if lat1d.size == 360 and lon1d.size == 180:
                # swapped: pick every other for lat (=180) and expand lon to 360 if needed
                # However, for standard 1° grids stored swapped, lat values repeat; use stride 2.
                lat1d = lat1d[::2]
                # For lon 180 centers, mapfile likely only stored 0.5..179.5.
                # We'll fabricate 0.5..359.5 to be safe.
                if lon1d.size == 180:
                    lon1d = np.arange(360, dtype="f8") + 0.5

            # sanity bounds
            if not (
                -91.0 <= float(lat1d.min()) <= -89.0
                and 89.0 <= float(lat1d.max()) <= 91.0
            ):
                # If still odd, try the alternative:
                # extract along the other axis by reshaping via dst_grid_dims
                # Fallback to canonical 1°
                lat1d = np.linspace(-89.5, 89.5, 180, dtype="f8")
            if lon1d.size != 360:
                lon1d = np.arange(360, dtype="f8") + 0.5

            return lat1d, lon1d

        # 1-D fields present already
        lat1d = read_array(m, "lat", "yc")
        lon1d = read_array(m, "lon", "xc")
        if lat1d is not None and lon1d is not None:
            lat1 = np.sort(np.asarray(lat1d, dtype="f8"))
            lon1 = np.sort(np.asarray(lon1d, dtype="f8"))
            if lon1.min() < 0.0 or lon1.max() <= 180.0:
                lon1 = lon1 % 360.0
                lon1.sort()
            if lat1.size == 360:
                lat1 = lat1[::2]
            if lon1.size != 360:
                lon1 = np.arange(360, dtype="f8") + 0.5
            return lat1, lon1

    # Last resort: fabricate standard 1°
    lat = np.linspace(-89.5, 89.5, 180, dtype="f8")
    lon = np.arange(360, dtype="f8") + 0.5
    return lat, lon


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
        raise KeyError(f"{varname!r} not in dataset: variables {list(ds_in.variables)}")

    realm = None
    var_da = ds_in[varname]  # always a DataArray
    if "ncol" not in var_da.dims and "lndgrid" not in var_da.dims:
        logger.info("Variable has no 'ncol' or 'lndgrid' dim; assuming ocn variable.")
        hdim = "tripolar"
        da2 = var_da  # Use the DataArray, not the whole Dataset
        non_spatial = [d for d in da2.dims if d not in ("yh", "xh")]
        realm = "ocn"
        method = method or "conservative"  # force conservative for ocn
        # --- OCEAN: Normalize by sftof (sea fraction) if present ---
        sftof = _sftof_from_native(ds_in)
        if sftof is not None:
            logger.info("Normalizing ocean field by source sftof (sea fraction)")
            frac = sftof / 100.0
            da2 = da2.fillna(0) * frac
    else:
        da2, non_spatial = _ensure_ncol_last(var_da)
        # For land/atm, set realm and hdim
        if "ncol" in var_da.dims:
            realm = "atm"
            hdim = "ncol"
        elif "lndgrid" in var_da.dims:
            realm = "lnd"
            hdim = "lndgrid"
            da2 = _normalize_land_field(da2, ds_in)
        else:
            # Fallback: use last dim
            hdim = da2.dims[-1]

    # cast to save memory
    if dtype is not None and str(da2.dtype) != dtype:
        da2 = da2.astype(dtype)

    # keep dask lazy and chunk along time if present
    if "time" in da2.dims and output_time_chunk and _HAS_DASK:
        da2 = da2.chunk({"time": output_time_chunk})

    spec = _pick_maps(
        varname,
        conservative_map=conservative_map,
        bilinear_map=bilinear_map,
        force_method=method,
        realm=realm,
    )
    logger.info(
        "Regridding %s using %s map: %s for realm %s",
        varname,
        spec.method_label,
        spec.path,
        realm,
    )
    regridder = RegridderCache.get(spec.path, spec.method_label)

    # tell xESMF to produce chunked output
    kwargs = {}
    if "time" in da2.dims and output_time_chunk:
        kwargs["output_chunks"] = {"time": output_time_chunk}
    if realm in ("atm", "lnd"):
        da2_2d = (
            da2.rename({hdim: "lon"})
            .expand_dims({"lat": 1})  # add a dummy 'lat' of length 1
            .transpose(
                *non_spatial, "lat", "lon"
            )  # ensure last two dims are ('lat','lon')
        )
    else:
        da2_2d = da2.rename({"xh": "lon", "yh": "lat"}).transpose(
            *non_spatial, "lat", "lon"
        )

        da2_2d = da2_2d.assign_coords(lon=((da2_2d.lon % 360)))
    logger.info(
        "da2_2d range: %f to %f lat, %f to %f lon",
        da2_2d["lat"].min().item(),
        da2_2d["lat"].max().item(),
        da2_2d["lon"].min().item(),
        da2_2d["lon"].max().item(),
    )

    out_norm = regridder(da2_2d, skipna=True, na_thres=1.0, **kwargs)
    logger.info("Regridding complete. out_norms dims: %s", out_norm.dims)
    if realm == "lnd":
        out = _denormalize_land_field(out_norm, ds_in, spec.path)
    elif realm == "ocn":
        out = _denormalize_ocn_field(out_norm, ds_in, spec.path)
        logger.info("Denormalized ocean field. out dims: %s", out.dims)
    else:
        out = out_norm

    # --- NEW: robust lat/lon assignment based on destination grid lengths ---
    lat1d, lon1d = _dst_latlon_1d_from_map(spec.path)
    ny, nx = len(lat1d), len(lon1d)

    # find the last two dims that came from xESMF
    spatial_dims = [d for d in out.dims if d not in non_spatial]
    if len(spatial_dims) < 2:
        raise ValueError(f"Unexpected output dims {out.dims}; need two spatial dims.")

    if len(spatial_dims) > 2:
        logger.warning(
            "More than two spatial dims found in output: %s; using last two.",
            spatial_dims,
        )
        spatial_dims = spatial_dims[-2:]
    da, db = spatial_dims[-2], spatial_dims[-1]
    na, nb = out.sizes[da], out.sizes[db]
    logger.debug(
        "Output spatial dims: %s (%d), %s (%d); target (lat %d, lon %d)",
        da,
        na,
        db,
        nb,
        ny,
        nx,
    )
    # Decide mapping by comparing lengths to (ny, nx)
    if na == ny and nb == nx:
        out = out.rename({da: "lat", db: "lon"})
    elif na == nx and nb == ny:
        out = out.rename({da: "lon", db: "lat"})
    else:
        # Heuristic fallback: pick the dim whose size matches 180 as lat
        if {na, nb} == {ny, nx}:
            # covered above; should not reach here
            pass
        else:
            # choose the one closer to 180 as lat
            choose_lat = da if abs(na - 180) <= abs(nb - 180) else db
            choose_lon = db if choose_lat == da else da
            out = out.rename({choose_lat: "lat", choose_lon: "lon"})
    logger.debug("Final output dims: %s", out.dims)
    # assign canonical 1-D coords
    out = out.assign_coords(lat=("lat", lat1d), lon=("lon", lon1d))

    try:
        out = out.transpose(*non_spatial, "lat", "lon")
    except ValueError:
        # fallback if non_spatial is empty
        out = out.transpose("lat", "lon")
    if keep_attrs and hasattr(var_da, "attrs"):
        out.attrs.update(var_da.attrs)

    return out


def _attach_time_and_bounds(ds_out: xr.Dataset, time_from: xr.Dataset) -> xr.Dataset:
    """Copy 'time' coord and its bounds from time_from into ds_out, unchanged.

    - Requires that time_from already has valid bounds.
    - No reindexing or synthesis: fail fast if lengths mismatch.
    """
    # 1) get time coord safely (no boolean evaluation!)
    if "time" in time_from.coords:
        time_da = time_from.coords["time"]
    elif "time" in time_from:
        time_da = time_from["time"]
    else:
        return ds_out  # nothing to copy

    # 2) attach exact time coord (preserves dtype/attrs/chunks)
    ds_out = ds_out.assign_coords(time=time_da)

    # 3) find bounds variable name
    bounds_name = None
    b = time_da.attrs.get("bounds")
    if isinstance(b, str) and b in time_from:
        bounds_name = b
    elif "time_bounds" in time_from:
        bounds_name = "time_bounds"
    elif "time_bnds" in time_from:
        bounds_name = "time_bnds"
    else:
        return ds_out  # no bounds to copy

    tb = time_from[bounds_name]
    if not isinstance(tb, xr.DataArray):
        return ds_out

    # 4) ensure dims are exactly (time, nbnd) without changing values
    if tb.dims[0] != "time":
        other = next((d for d in tb.dims if d != "time"), None)
        if other is not None:
            tb = tb.transpose("time", other)

    # 5) strict length check (no reindex)
    if tb.sizes.get("time") != ds_out.sizes.get("time"):
        raise ValueError(
            f"time_bounds length {tb.sizes.get('time')} != time length {ds_out.sizes.get('time')}"
        )

    # 6) attach as data variable (creates nbnd dim if needed)
    ds_out[bounds_name] = tb.copy(deep=False)

    # 7) ensure the time coord points to the correct bounds var name
    ta = dict(time_da.attrs) if hasattr(time_da, "attrs") else {}
    ta["bounds"] = bounds_name
    ds_out["time"].attrs = ta

    return ds_out


def _first_present(ds: xr.Dataset, names: list[str]) -> str | None:
    for n in names:
        if n in ds:
            return n
    return None


def _sftlf_from_native(ds: xr.Dataset) -> xr.DataArray | None:
    name = _first_present(ds, ["LANDFRAC", "landfrac", "landmask", "frac_lnd"])
    if name is None:
        return None
    v = ds[name]
    # If in 0..1, convert to percent; if already 0..100, leave it
    vmax = float(np.nanmax(np.asarray(v)))
    out = v * 100.0 if vmax <= 1.0 + 1e-6 else v
    out = out.clip(min=0.0, max=100.0)
    out = out.astype("f8")
    attrs = dict(out.attrs)
    attrs["units"] = "%"
    attrs.setdefault("standard_name", "land_area_fraction")
    attrs.setdefault("long_name", "Percentage of land area")
    out.attrs = attrs
    return out


def _build_fx_native(ds_native: xr.Dataset) -> xr.Dataset:
    pieces = {}
    sftlf = _sftlf_from_native(ds_native)
    if sftlf is not None:
        pieces["sftlf"] = sftlf
    # Also extract sftof (sea fraction) if present
    sftof = None
    for name in ["sftof", "ocnfrac", "wet"]:
        if name in ds_native:
            logger.info("Extracting sftof from native variable %s", name)
            v = ds_native[name]
            # If 'wet', convert 0/1 to percent
            if name == "wet":
                vmax = float(np.nanmax(np.asarray(v)))
                if vmax <= 1.0 + 1e-6:
                    sftof = v * 100.0
                else:
                    sftof = v
            else:
                sftof = v
            sftof = sftof.clip(min=0.0, max=100.0)
            sftof = sftof.astype("f8")
            attrs = dict(sftof.attrs)
            attrs["units"] = "%"
            attrs.setdefault("standard_name", "sea_area_fraction")
            attrs.setdefault("long_name", "Percentage of sea area")
            sftof.attrs = attrs
            break
    if sftof is not None:
        pieces["sftof"] = sftof
    if not pieces:
        return xr.Dataset()
    ds_fx = xr.Dataset(pieces)
    # normalize horizontal dim name to what your regrid code expects
    #    ds_fx = ds_fx.rename({"lndgrid": "ncol"})
    logger.info("sftlf in ds_fx: %s", "sftlf" in ds_fx)
    return ds_fx


def compute_areacella_from_bounds(
    ds: xr.Dataset, *, radius_m: float = 6_371_220.0
) -> xr.DataArray:
    """
    Compute areacella (m^2) from 1x1 lat/lon bounds.
    Earth radius matches that in CESM shr_const_mod.F90
    Requires 1D coords 'lat','lon' and bounds 'lat_bnds','lon_bnds' with shape (N,2).
    """
    logger.info("computing areacella from lat/lon bounds")
    if "lat_bnds" not in ds:
        lat_bnds = bounds_from_centers_1d(ds["lat"].values, kind="lat")
        ds["lat_bnds"] = xr.DataArray(
            lat_bnds,
            dims=("lat", "bnds"),
            attrs={"long_name": "latitude cell boundaries", "units": "degrees_north"},
        )
    if "lon_bnds" not in ds:
        lon_bnds = bounds_from_centers_1d(ds["lon"].values, kind="lon")
        ds["lon_bnds"] = xr.DataArray(
            lon_bnds,
            dims=("lon", "bnds"),
            attrs={"long_name": "longitude cell boundaries", "units": "degrees_east"},
        )
    lat_b = np.asarray(ds["lat_bnds"].values, dtype="f8")  # (nlat, 2)
    lon_b = np.asarray(ds["lon_bnds"].values, dtype="f8")  # (nlon, 2)

    # Validation: check shape
    if lat_b.ndim != 2 or lat_b.shape[1] != 2:
        raise ValueError(f"lat_bnds must have shape (nlat, 2), got {lat_b.shape}")
    if lon_b.ndim != 2 or lon_b.shape[1] != 2:
        raise ValueError(f"lon_bnds must have shape (nlon, 2), got {lon_b.shape}")

    # Check for non-monotonic bounds across grid
    if not np.all(np.diff(lat_b[:, 0]) >= 0):
        logger.warning("lat_bnds[:, 0] is not monotonic increasing.")
    if not np.all(np.diff(lat_b[:, 1]) >= 0):
        logger.warning("lat_bnds[:, 1] is not monotonic increasing.")
    if not np.all(np.diff(lon_b[:, 0]) >= 0):
        logger.warning("lon_bnds[:, 0] is not monotonic increasing.")
    if not np.all(np.diff(lon_b[:, 1]) >= 0):
        logger.warning("lon_bnds[:, 1] is not monotonic increasing.")

    # radians
    lat_b_rad = np.deg2rad(lat_b)
    lon_b_rad = np.deg2rad(lon_b % 360.0)  # ensure [0,360)
    dlam = lon_b_rad[:, 1] - lon_b_rad[:, 0]  # (nlon,)
    dlam = np.where(dlam < 0, dlam + 2 * np.pi, dlam)
    # Δ(sin φ)
    sin_phi2_minus_phi1 = np.sin(lat_b_rad[:, 1]) - np.sin(lat_b_rad[:, 0])  # (nlat,)

    # broadcast to 2D (lat,lon)
    area = (radius_m**2) * sin_phi2_minus_phi1[:, None] * dlam[None, :]

    da = xr.DataArray(
        area,
        dims=("lat", "lon"),
        coords={"lat": ds["lat"], "lon": ds["lon"]},
        name="areacella",
        attrs={
            "standard_name": "cell_area",
            "long_name": "Grid-Cell Area",
            "units": "m2",
        },
    )
    return da


def _regrid_fx_once(
    mapfile: Path, ds_native: xr.Dataset, sftlf_path: Path | None = None
) -> xr.Dataset:
    """Compute & regrid sftlf/areacella once for a given mapfile; cache result.

    sftlf is regridded from source; areacella is computed on the destination grid.
    """
    cached = FXCache.get(mapfile)
    if cached is not None:
        logger.info("Getting cached fx variables")
        return cached
    out_vars = {}
    if sftlf_path:
        logger.info("Getting sftlf from output path %s", sftlf_path)
        out_vars["sftlf"] = xr.open_mfdataset(sftlf_path)["sftlf"]

    ds_fx_native = _build_fx_native(ds_native)

    regridder = RegridderCache.get(
        mapfile,
        "conservative",
    )

    # Add native grid fields to FXCache
    native_fx = {}
    for key in ["sftof", "deptho", "areacello"]:
        if key in ds_fx_native:
            native_fx[f"{key}_native"] = ds_fx_native[key]
    # Always cache sftof_native if present
    if "sftof" in ds_fx_native:
        native_fx["sftof_native"] = ds_fx_native["sftof"]
    if native_fx:
        FXCache.put(mapfile, xr.Dataset(native_fx))
    # Regrid sftlf from source if present
    if "sftlf" not in out_vars and "sftlf" in ds_fx_native:
        da = ds_fx_native["sftlf"].fillna(0)
        da2 = (
            da.rename({"lndgrid": "lon"})
            .expand_dims({"lat": 1})
            .transpose(..., "lat", "lon")
        )
        lndarea = (ds_native["landfrac"] * ds_native["area"] * 1.0e6).sum(
            dim=("lndgrid")
        )
        logger.info("Total land area on source grid: %.3e m^2", lndarea.values)
        out = regridder(da2, skipna=True, na_thres=1.0)
        spatial = [d for d in out.dims if d in ("lat", "lon")]
        out = out.transpose(*spatial)
        out.name = "sftlf"
        out.attrs.update(da.attrs)
        out_vars["sftlf"] = out
    # For regridded grid, set sftof = 1 - sftlf
    if "sftlf" in out_vars:
        logger.info("Computing regridded sftof as 1 - sftlf")
        sftlf = out_vars["sftlf"]
        sftof = (1.0 - sftlf / 100.0) * 100.0
        sftof = sftof.clip(min=0.0, max=100.0)
        sftof.name = "sftof"
        sftof.attrs["units"] = "%"
        sftof.attrs.setdefault("standard_name", "sea_area_fraction")
        sftof.attrs.setdefault("long_name", "Percentage of sea area")
        out_vars["sftof"] = sftof

    # Regrid deptho from source if present
    if "deptho" in ds_fx_native:
        logger.info("Regridding deptho from native")
        da = ds_fx_native["deptho"]
        da2 = da.rename({"xh": "lon", "yh": "lat"}).transpose(..., "lat", "lon")
        out = regridder(da2, skipna=True, na_thres=1.0)
        spatial = [d for d in out.dims if d in ("lat", "lon")]
        out = out.transpose(*spatial)
        out.name = "deptho"
        out.attrs.update(da.attrs)
        out_vars["deptho"] = out

    # Regrid areacello from source if present
    if "areacello" in ds_fx_native:
        logger.info("Regridding areacello from native")
        da = ds_fx_native["areacello"]
        da2 = da.rename({"xh": "lon", "yh": "lat"}).transpose(..., "lat", "lon")
        out = regridder(da2, skipna=True, na_thres=1.0)
        spatial = [d for d in out.dims if d in ("lat", "lon")]
        out = out.transpose(*spatial)
        out.name = "areacello"
        out.attrs.update(da.attrs)
        out_vars["areacello"] = out
    # Always compute areacella on the destination grid, not by regridding
    # Use the destination grid from the mapfile
    lat1d, lon1d = _dst_latlon_1d_from_map(mapfile)
    ds_grid = xr.Dataset(
        coords={
            "lat": ("lat", lat1d, {"units": "degrees_north"}),
            "lon": ("lon", lon1d, {"units": "degrees_east"}),
        }
    )
    areacella = compute_areacella_from_bounds(ds_grid)
    out_vars["areacella"] = areacella
    if "sftlf" in out_vars:
        lndarea = (areacella * out_vars["sftlf"] / 100.0).sum(dim=("lat", "lon"))
        logger.info(
            "Total land area on destination grid: %.3e m^2", float(lndarea.values)
        )
    ds_fx = xr.Dataset(out_vars)
    FXCache.put(mapfile, ds_fx)
    return ds_fx
