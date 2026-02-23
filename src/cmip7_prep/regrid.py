# Example usage (in docstring):
# ds = xr.open_dataset("input.nc")
# zonal_p39 = zonal_mean_on_pressure_grid(ds, "ta", tables_path="cmip7-cmor-tables/tables")
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
from cmip7_prep.cache_tools import FXCache, RegridderCache
from cmip7_prep import vertical

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import dask.array as _da  # noqa: F401

    _HAS_DASK = True
except ModuleNotFoundError as e:
    _HAS_DASK = False

# Default weight maps; override via function args.
# optional bilinear map

INPUTDATA_DIR_noresm = Path("/datalake/NS9560K/diagnostics/land_xesmf_diag_data/")
DEFAULT_BILIN_MAP_NE30_noresm = Path(
    INPUTDATA_DIR_noresm / "map_ne30pg3_to_0.5x0.5_nomask_aave_da_c180515.nc"
)
DEFAULT_CONS_MAP_NE30_noresm = Path(
    INPUTDATA_DIR_noresm / "map_ne30pg3_to_0.5x0.5_nomask_aave_da_c180515.nc"
)
DEFAULT_BILIN_MAP_NE16_noresm = Path(
    INPUTDATA_DIR_noresm / "map_ne16pg3_to_1.9x2.5_nomask_scripgrids_c250425.nc"
)
DEFAULT_CONS_MAP_NE16_noresm = Path(
    INPUTDATA_DIR_noresm / "map_ne16pg3_to_1.9x2.5_nomask_scripgrids_c250425.nc"
)

INPUTDATA_DIR_cesm = Path("/glade/campaign/cesm/cesmdata/inputdata/")
DEFAULT_CONS_MAP_NE30_cesm = Path(
    INPUTDATA_DIR_cesm / "cpl/gridmaps/ne30pg3/map_ne30pg3_to_1x1d_aave.nc"
)
DEFAULT_BILIN_MAP_NE30_cesm = Path(
    INPUTDATA_DIR_cesm / "cpl/gridmaps/ne30pg3/map_ne30pg3_to_1x1d_bilin.nc"
)
DEFAULT_CONS_MAP_T232 = Path(
    INPUTDATA_DIR_cesm / "cpl/gridmaps/tx2_3v2/map_t232_TO_1x1d_aave.251023.nc"
)
DEFAULT_BILIN_MAP_T232 = Path(
    INPUTDATA_DIR_cesm / "cpl/gridmaps/tx2_3v2/map_t232_TO_1x1d_blin.251023.nc"
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
    lev_dim, lon_dim : str
        Names for vertical and longitude dimensions.
    ps_name, hyam_name, hybm_name, p0_name : str
        Names for surface pressure, hybrid A/B (midpoint) coefficients, and reference pressure.
    keep_attrs : bool
        Whether to keep variable attributes in the output.

    Returns
    -------
    xr.DataArray
        Zonal mean of the variable, interpolated to the target
        pressure levels, dims: (plev, lat, [time]).

    Example
    -------
    >>> import numpy as np
    >>> import xarray as xr
    >>> from cmip7_prep.regrid import zonal_mean_on_pressure_grid
    >>> ds = xr.Dataset({
    ...     'ta': (('time', 'lev', 'lat', 'lon'), np.random.rand(1, 2, 3, 4)),
    ...     'PS': (('time', 'lat', 'lon'), np.random.rand(1, 3, 4)),
    ...     'hyam': ('lev', [0.5, 0.3]),
    ...     'hybm': ('lev', [0.5, 0.7]),
    ...     'P0': 100000.0,
    ...     'lat': ('lat', [10, 20, 30]),
    ...     'lon': ('lon', [0, 90, 180, 270]),
    ... })
    >>> # This will fail unless vertical.to_plev is properly mocked or available
    >>> # zonal_mean_on_pressure_grid(ds, 'ta',
    >>> #      tables_path='cmip7-cmor-tables/tables', target='plev39')
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


def _pick_maps(
    varname: str,
    *,
    resolution: Optional[str] = "ne30",
    model: Optional[str] = "cesm",
    conservative_map: Optional[Path] = None,
    bilinear_map: Optional[Path] = None,
    force_method: Optional[str] = None,
) -> MapSpec:
    """Choose which precomputed map file to use for a variable."""
    cons = None
    bilin = None
    if model == "cesm":
        if resolution == "ne30":
            cons = (
                Path(conservative_map)
                if conservative_map
                else DEFAULT_CONS_MAP_NE30_cesm
            )
            bilin = Path(bilinear_map) if bilinear_map else DEFAULT_BILIN_MAP_NE30_cesm
        else:
            cons = Path(conservative_map) if conservative_map else DEFAULT_CONS_MAP_T232
            bilin = Path(bilinear_map) if bilinear_map else DEFAULT_BILIN_MAP_T232
    elif model == "noresm":
        if resolution == "ne30":
            cons = (
                Path(conservative_map)
                if conservative_map
                else DEFAULT_CONS_MAP_NE30_noresm
            )
            bilin = (
                Path(bilinear_map) if bilinear_map else DEFAULT_BILIN_MAP_NE30_noresm
            )
        elif resolution == "ne16":
            cons = (
                Path(conservative_map)
                if conservative_map
                else DEFAULT_CONS_MAP_NE16_noresm
            )
            bilin = (
                Path(bilinear_map) if bilinear_map else DEFAULT_BILIN_MAP_NE16_noresm
            )

    logger.info("Conservative_map is %s", str(cons))

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


def regrid_to_latlon_ds(
    ds_in: xr.Dataset,
    varnames: str | list[str],
    resolution: str,
    model: str,
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

    # Regrid each output var separately
    out_vars: dict[str, xr.DataArray] = {}
    names = [varnames] if isinstance(varnames, str) else list(varnames)
    for name in names:
        logger.info("Regridding var %s", name)
        out_vars[name] = regrid_to_latlon(
            ds_in,
            name,
            resolution,
            model,
            method=method,
            conservative_map=conservative_map,
            bilinear_map=bilinear_map,
            keep_attrs=keep_attrs,
            dtype=dtype,
            output_time_chunk=output_time_chunk,
        )
        logger.info("Finished regridding var %s", name)

    logger.info("conservative map is %s", conservative_map)
    logger.info("bilinear map is %s", bilinear_map)

    # Create an xarray dataset with the output vars
    ds_out = xr.Dataset(out_vars)

    # Attach time (and bounds) from the original dataset if requested
    if time_from is not None:
        ds_out = _attach_time_and_bounds(ds_out, time_from)

    # Pick the mapfile you used for conservative/bilinear selection
    spec = _pick_maps(
        varnames[0] if isinstance(varnames, list) else varnames,
        resolution=resolution,
        model=model,
        conservative_map=conservative_map,
        bilinear_map=bilinear_map,
        force_method="conservative",
    )  # fx always conservative

    # Regrid the fx data
    logger.info("using fx map: %s", spec.path)
    ds_fx = _regrid_fx_once(spec.path, ds_in, sftlf_path)  # ← uses cache
    if ds_fx is not None and len(ds_fx.data_vars) > 0:
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


def regrid_to_latlon(
    ds_in: xr.Dataset,
    varname: str,
    resolution: str,
    model: str,
    *,
    method: Optional[str] = None,
    conservative_map: Optional[Path] = None,
    bilinear_map: Optional[Path] = None,
    keep_attrs: bool = True,
    dtype: str | None = "float32",
    output_time_chunk: int | None = 12,
) -> xr.Dataset:
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

    var_da = ds_in[varname]  # always a DataArray
    if "ncol" not in var_da.dims and "lndgrid" not in var_da.dims:
        logger.info("Variable has no 'ncol' or 'lndgrid' dim; assuming ocn variable.")
        hdim = "tripolar"
        da2 = var_da  # Use the DataArray, not the whole Dataset
        non_spatial = [d for d in da2.dims if d not in ("yh", "xh")]
        # realm = "ocn"
        method = method or "conservative"  # force conservative for ocn
        # --- OCEAN: Normalize by sftof (sea fraction) if present ---
        sftof = _sftof_from_native(ds_in)
        if sftof is not None:
            logger.info("Normalizing ocean field by source sftof (sea fraction)")
            frac = sftof / 100.0
            da2 = da2.fillna(0) * frac
    else:
        # Move 'ncol' to the last position; return (da, non_spatial_dims)
        da2, non_spatial = _ensure_ncol_last(var_da)
        hdim = None
        # Determine hdim for atmos or land realms
        if "ncol" in var_da.dims:
            hdim = "ncol"
        elif "lndgrid" in var_da.dims:
            hdim = "lndgrid"

        # Normalize the land fields by land fraction
        if hdim == "lndgrid":
            da2 = _normalize_land_field(da2, ds_in)

    # Cast to save memory
    if dtype is not None and str(da2.dtype) != dtype:
        da2 = da2.astype(dtype)

    # Keep dask lazy and chunk along time if present
    if "time" in da2.dims and output_time_chunk and _HAS_DASK:
        da2 = da2.chunk({"time": output_time_chunk})

    # Select the xesmf map
    spec = _pick_maps(
        varname,
        resolution=resolution,
        model=model,
        conservative_map=conservative_map,
        bilinear_map=bilinear_map,
        force_method=method,
    )
    logger.info(
        "Regridding %s using %s map: %s ", varname, spec.method_label, spec.path
    )
    regridder = RegridderCache.get(spec.path, spec.method_label)
    logger.info("Regridder ready to use")
    # Tell xESMF to produce chunked output
    kwargs = {}
    if "time" in da2.dims and output_time_chunk:
        kwargs["output_chunks"] = {"time": output_time_chunk}

    # Transpose the input data  so that it can be regridded
    if hdim in ("ncol", "lndgrid"):
        da2_2d = (
            da2.rename({hdim: "lon"})
            .expand_dims({"lat": 1})  # add a dummy 'lat' of length 1
            .transpose(
                *non_spatial, "lat", "lon"
            )  # ensure last two dims are ('lat','lon')
        )
    else:
        logger.info("Creating da2_2d for ocean grid")
        da2_2d = da2.rename({"xh": "lon", "yh": "lat"})
        da2_2d = da2_2d.transpose(
            ..., "lat", "lon"
        )  # ensure last two dims are ('lat','lon')
        da2_2d = da2_2d.assign_coords(lon=((da2_2d.lon % 360)))

    logger.info(
        "da2d dims %s da2_2d range: %f to %f lat, %f to %f lon",
        da2_2d.dims,
        da2_2d["lat"].min().item(),
        da2_2d["lat"].max().item(),
        da2_2d["lon"].min().item(),
        da2_2d["lon"].max().item(),
    )
    logger.info("Invoking esmf regridder, da2_2d dims %s", da2_2d.dims)
    # Regrid the data
    out_norm = regridder(da2_2d, skipna=True, na_thres=1.0, **kwargs)
    logger.info("Regridding complete. out_norms dims: %s", out_norm.dims)

    # Denormalize the data if appropriate
    if hdim == "lndgrid":
        out = _denormalize_land_field(out_norm, ds_in, spec.path)
    else:
        out = out_norm

    lat = out["lat"].values
    lon = out["lon"].values
    ny, nx = len(lat), len(lon)

    # print(out.coords)

    # lon_bounds = out.lon_b.values  # 1D array of longitude bounds
    # lat_bounds = out.lat_b.values  # 1D array of latitude bounds
    # ny_b, nx_b = len(lat_bounds), len(lon_bounds)

    # print(f"latitudes  are {lat}")
    # print(f"longitudes are {lon}")
    # print(f"number of longitudes are {nx}")
    # print(f"number of latitudes  are {ny}")
    # weight_file = DEFAULT_CONS_MAP_NE16_noresm

    weight_file = spec.path
    weights = xr.open_dataset(weight_file)
    out_shape = weights.dst_grid_dims.load().data.tolist()[::-1]

    # print(f"computing lon_b_out")
    lon_b_out = np.zeros((out_shape[1], 2))
    lon_b_out[0:, 0] = weights.xv_b.data[0 : out_shape[1], 0]
    lon_b_out[:, 1] = weights.xv_b.data[1 : out_shape[1] + 1, 0]
    lon_b_out[-1, 1] = 360.0
    # print(f" lon_b is {lon_b_out}")

    # print(f"computing lat_b_out")
    lat_b_out = np.zeros((out_shape[0], 2))
    lat_b_out[0:, 0] = weights.yv_b.data[np.arange(out_shape[0]) * out_shape[1], 0]
    lat_b_out[:-1, 1] = lat_b_out[1:, 0]
    lat_b_out[-1, 1] = 90.0
    # print(f" lat_b is {lat_b_out}")

    out["lon"] = lon
    out["lat"] = lat
    out["areacella"] = _calculate_area_from_bounds(lon_b_out, lat_b_out)

    # Decide mapping by comparing lengths to (ny, nx)
    if "ncol" in var_da.dims:
        hdim = "ncol"
    elif "lndgrid" in var_da.dims:
        hdim = "lndgrid"

    if hdim not in ("ncol", "lndgrid"):
        # find the last two dims that came from xESMF
        spatial_dims = [d for d in out.dims if d not in non_spatial]
        if len(spatial_dims) < 2:
            raise ValueError(
                f"Unexpected output dims {out.dims}; need two spatial dims."
            )
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
        # Ocean realm, heuristic fallback: pick the dim whose size matches 180 as lat
        # choose the one closer to 180 as lat
        choose_lat = da if abs(na - 180) <= abs(nb - 180) else db
        choose_lon = db if choose_lat == da else da
        out = out.rename({choose_lat: "lat", choose_lon: "lon"})
        logger.debug("Final output dims: %s", out.dims)
        # lat1d, lon1d are coordinate variables written to the CMOR output file
        # out = out.assign_coords(lat=("lat", lat1d), lon=("lon", lon1d))

    # assign canonical 1-D coords
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
    if not pieces:
        return xr.Dataset()
    ds_fx = xr.Dataset(pieces)
    # normalize horizontal dim name to what your regrid code expects
    #    ds_fx = ds_fx.rename({"lndgrid": "ncol"})
    logger.info("sftlf in ds_fx: %s", "sftlf" in ds_fx)
    return ds_fx


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

    # Determine regridder
    regridder = RegridderCache.get(mapfile, "conservative")

    # Regrid sftlf from source if present
    if "sftlf" not in out_vars and "sftlf" in ds_fx_native:
        logger.info("Computing regridded sftlf")
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
        out = regridder(da2, skipna=True, na_thres=1.0)  # Regrid
        spatial = [d for d in out.dims if d in ("lat", "lon")]
        out = out.transpose(*spatial)
        out.name = "sftlf"
        out.attrs.update(da.attrs)
        out_vars["sftlf"] = out

    # For regridded grid, set sftof = 1 - sftlf
    if "sftlf" in out_vars:
        logger.info("Obtaining sftlf")
        sftlf = out_vars["sftlf"]
        logger.info("Computing regridded sftof as 1 - sftlf")
        sftof = (1.0 - sftlf / 100.0) * 100.0
        sftof = sftof.clip(min=0.0, max=100.0)
        sftof.name = "sftof"
        sftof.attrs["units"] = "%"
        sftof.attrs.setdefault("standard_name", "sea_area_fraction")
        sftof.attrs.setdefault("long_name", "Percentage of sea area")
        out_vars["sftof"] = sftof

    ds_fx = xr.Dataset(out_vars)
    FXCache.put(mapfile, ds_fx)
    return ds_fx


def _calculate_area_from_bounds(lon_b, lat_b):
    """
    Calculate 2D grid cell areas in m^2 from 1D bounds arrays.

    Parameters
    ----------
    lon_b : array-like, shape (nlon, 2)
        Longitude bounds for each cell.
    lat_b : array-like, shape (nlat, 2)
        Latitude bounds for each cell.

    Returns
    -------
    area : xarray.DataArray, shape (nlat, nlon)
        Grid cell areas with coordinates and metadata.
    """

    R = 6371000  # Earth radius in meters

    lon0 = lon_b[:, 0]
    lon1 = lon_b[:, 1]
    lat0 = lat_b[:, 0]
    lat1 = lat_b[:, 1]

    # Convert to radians
    lon0_rad = np.deg2rad(lon0)
    lon1_rad = np.deg2rad(lon1)
    lat0_rad = np.deg2rad(lat0)
    lat1_rad = np.deg2rad(lat1)

    # nlat = lat_b.shape[0]
    # nlon = lon_b.shape[0]

    # Broadcast to 2D
    dlon = lon1_rad - lon0_rad  # (nlon,)
    dlat_sin = np.sin(lat1_rad) - np.sin(lat0_rad)  # (nlat,)
    area = (
        R**2 * np.abs(dlon[np.newaxis, :]) * np.abs(dlat_sin[:, np.newaxis])
    )  # (nlat, nlon)

    # Build coordinates for centers
    lat_center = 0.5 * (lat0 + lat1)
    lon_center = 0.5 * (lon0 + lon1)

    area_da = xr.DataArray(
        area,
        dims=("lat", "lon"),
        coords={
            "lat": lat_center,
            "lon": lon_center,
        },
        name="areacella",
        attrs={
            "standard_name": "cell_area",
            "long_name": "Grid-Cell Area",
            "units": "m2",
        },
    )
    return area_da
