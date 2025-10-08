# cmip7_prep/regrid.py
"""Regridding utilities for CESM -> 1° lat/lon using precomputed ESMF weights."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple
import numpy as np
import xarray as xr
import xesmf as xe
#try:
#    import xesmf as xe
#except ModuleNotFoundError as e:
#    _HAS_XESMF = False
try:
    import dask.array as _da  # noqa: F401
    _HAS_DASK = True
except ModuleNotFoundError as e:
    _HAS_DASK = False

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


def _dst_latlon_1d_from_map(mapfile: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return canonical 1-D (lat, lon) for the destination grid.

    Robust to map files whose stored dst_grid_dims or reshape order is swapped.
    We derive 1-D axes by taking UNIQUE values from the 2-D center fields.
    """
    with _open_nc(mapfile) as m:
        lat2d = _read_array(m, "yc_b", "lat_b", "dst_grid_center_lat", "yc", "lat")
        lon2d = _read_array(m, "xc_b", "lon_b", "dst_grid_center_lon", "xc", "lon")

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
        lat1d = _read_array(m, "lat", "yc")
        lon1d = _read_array(m, "lon", "xc")
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


def _bounds_from_centers_1d(
    centers: np.ndarray, *, periodic: bool = False, period: float = 360.0
) -> np.ndarray:
    """Compute simple 1D bounds from 1D cell centers.
    For regular spacing this sets bounds[i] = [c[i]-dx/2, c[i]+dx/2].
    For periodic=True, wrap the last bound to period (e.g., lon 0..360)."""

    centers = np.asarray(centers, dtype="f8").ravel()
    n = centers.size
    b = np.empty((n, 2), dtype="f8")
    if n == 1:
        # Any small cell is fine for dummy grid; choose +/- 0.5 around center
        dx = 1.0
        b[0, 0] = centers[0] - dx / 2.0
        b[0, 1] = centers[0] + dx / 2.0
    else:
        # Estimate dx from adjacent centers (assume uniform spacing)
        dx = np.diff(centers).mean()
        b[:, 0] = centers - dx / 2.0
        b[:, 1] = centers + dx / 2.0
    if periodic:
        # Keep within [0, period]
        b = np.mod(b, period)
        # Ensure right bound of last cell connects to period exactly
        # (helps some ESMF periodic checks for 0..360)
        b[-1, 1] = (
            period if abs(b[-1, 1] - period) < 1e-6 or b[-1, 1] < 1e-6 else b[-1, 1]
        )
    return b


def _make_dummy_grids(mapfile: Path) -> tuple[xr.Dataset, xr.Dataset]:
    """Construct minimal ds_in/ds_out satisfying xESMF when reusing weights.
    Adds CF-style bounds for both lat and lon so conservative methods don’t
    trigger cf-xarray’s bounds inference on size-1 dimensions."""
    with _open_nc(mapfile) as m:
        nlat_in, nlon_in = _get_src_shape(m)
        lat_out_1d, lon_out_1d = _get_dst_latlon_1d(m)

    # --- Dummy INPUT grid (unstructured → represent as 2D with length-1 lat) ---
    lat_in = np.arange(nlat_in, dtype="f8")  # e.g., [0], length can be 1
    lon_in = np.arange(nlon_in, dtype="f8")
    ds_in = xr.Dataset(
        data_vars={
            "lat_bnds": (
                ("lat", "nbnds"),
                _bounds_from_centers_1d(lat_in, periodic=False),
            ),
            "lon_bnds": (
                ("lon", "nbnds"),
                _bounds_from_centers_1d(lon_in, periodic=False),
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
    lat_out_bnds = _bounds_from_centers_1d(lat_out_1d, periodic=False)
    lon_out_bnds = _bounds_from_centers_1d(lon_out_1d, periodic=True, period=360.0)

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
# Selection & utilities
# -------------------------


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

def _ensure_ncol_last(da: xr.DataArray):
    if "ncol" in da.dims:
        hdim = "ncol"
    elif "lndgrid" in da.dims:
        hdim = "lndgrid"
    else:
        raise ValueError(f"Expected 'ncol' or 'lndgrid' in dims; got {da.dims}")
    non_spatial = tuple(d for d in da.dims if d != hdim)
    return da.transpose(*non_spatial, hdim), non_spatial, hdim

# -------------------------
# Public API
# -------------------------
# regrid.py


def regrid_to_1deg_ds(
    ds_in: xr.Dataset,
    varname: str | list[str],
    *,
    time_from: xr.Dataset | None = None,
    method: Optional[str] = None,
    conservative_map: Optional[Path] = None,
    bilinear_map: Optional[Path] = None,
    keep_attrs: bool = True,
    dtype: str | None = "float32",
    output_time_chunk: int | None = 12,
) -> xr.Dataset:
    """Regrid var(s) and return a Dataset. If `carry` is provided, copy those
    names through unchanged when they are 1-D/non-spatial (no ncol/lat/lon)."""

    names = [varname] if isinstance(varname, str) else list(varname)
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

    # --- NEW: carry hybrid metadata (or any requested 1-D fields) unchanged ---
    ds_out = _attach_vertical_metadata(ds_out, ds_in)
    return ds_out


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

    var_da = ds_in[varname]  # always a DataArray

    da2, non_spatial, hdim = _ensure_ncol_last(var_da)

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
    )
    regridder = _RegridderCache.get(spec.path, spec.method_label)

    # tell xESMF to produce chunked output
    kwargs = {}
    if "time" in da2.dims and output_time_chunk:
        kwargs["output_chunks"] = {"time": output_time_chunk}

    da2_2d = (
        da2.rename({hdim: "lon"})
        .expand_dims({"lat": 1})  # add a dummy 'lat' of length 1
        .transpose(*non_spatial, "lat", "lon")  # ensure last two dims are ('lat','lon')
    )
    if "time" in da2_2d.dims and output_time_chunk:
        kwargs["output_chunks"] = {"time": output_time_chunk}

    out = regridder(da2_2d, **kwargs)  # current call that returns (*non_spatial, ?, ?)

    # --- NEW: robust lat/lon assignment based on destination grid lengths ---
    lat1d, lon1d = _dst_latlon_1d_from_map(spec.path)
    ny, nx = len(lat1d), len(lon1d)

    # find the last two dims that came from xESMF
    spatial_dims = [d for d in out.dims if d not in non_spatial]
    if len(spatial_dims) < 2:
        raise ValueError(f"Unexpected output dims {out.dims}; need two spatial dims.")

    da, db = spatial_dims[-2], spatial_dims[-1]
    na, nb = out.sizes[da], out.sizes[db]

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
