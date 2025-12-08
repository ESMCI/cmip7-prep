"""Utility functions for CMOR processing."""

from pathlib import Path
import warnings
import re
import datetime as dt
from importlib.resources import files, as_file

import cmor
import cftime
import numpy as np
import xarray as xr


_FILL_DEFAULT = 1.0e20
_HANDLE_RE = re.compile(r"^hdl:21\.14100/[0-9a-f\-]{36}$", re.IGNORECASE)
_UUID_RE = re.compile(r"^[0-9a-f\-]{36}$", re.IGNORECASE)


def packaged_dataset_json(filename: str = "cmor_dataset.json"):
    """Context manager yielding a real filesystem path to the packaged mapping file."""
    res = files("cmip7_prep").joinpath(f"data/{filename}")
    return as_file(res)


def filled_for_cmor(
    da: xr.DataArray, fill: float | None = None
) -> tuple[xr.DataArray, float]:
    """
    Replace NaNs with CMOR missing value (default 1e20) and return (data, fill).
    >>> import xarray as xr
    >>> arr = xr.DataArray([1.0, np.nan, 2.0])
    >>> filled_for_cmor(arr, fill=-999.0)
    (<xarray.DataArray (dim_0: 3)> Size: 12B
    array([   1., -999.,    2.], dtype=float32)
    Dimensions without coordinates: dim_0
    Attributes:
        _FillValue:     -999.0
        missing_value:  -999.0, -999.0)

    """
    if fill is None:
        # choose fill based on dtype
        f = np.array(
            _FILL_DEFAULT,
            dtype=da.dtype if np.issubdtype(da.dtype, np.floating) else "f8",
        ).item()
    else:
        f = np.array(
            fill, dtype=da.dtype if np.issubdtype(da.dtype, np.floating) else "f8"
        ).item()
    # only act on floating types
    if not np.issubdtype(da.dtype, np.floating):
        return da, f
    # replace NaNs with fill
    da2 = da.where(np.isfinite(da), other=f)
    # keep attrs helpful for downstream
    da2.attrs["_FillValue"] = f
    da2.attrs["missing_value"] = f
    if da2.dtype != np.float32:
        da2 = da2.astype(np.float32)
    return da2, f


# ---------------------------------------------------------------------
# Time encoding
# ---------------------------------------------------------------------


def encode_time_to_num(obj, units: str, calendar: str) -> np.ndarray:
    """
    Return numeric CF time for:
      - xarray.DataArray of cftime or datetime64
      - numpy.ndarray (any shape) of cftime / datetime64 / python datetime
      - scalar cftime / python datetime

    Always returns float64 ndarray of the same shape as input.
    >>> import cftime
    >>> arr = [cftime.DatetimeNoLeap(2000, 1, 1), cftime.DatetimeNoLeap(2000, 1, 2)]
    >>> encode_time_to_num(arr, units="days since 2000-01-01", calendar="noleap")
    array([0., 1.])
    """
    # Normalize to ndarray
    arr = obj.values if hasattr(obj, "values") else np.asarray(obj)

    # Already numeric → just cast
    if np.issubdtype(arr.dtype, np.number):
        return arr.astype("f8", copy=False)

    # datetime64 → list[datetime]
    if np.issubdtype(arr.dtype, "datetime64"):
        # convert to python datetime (UTC)
        ns = arr.astype("datetime64[ns]").astype("int64")
        out = []
        epoch = dt.datetime(1970, 1, 1)
        for n in ns.ravel():
            out.append(epoch + dt.timedelta(microseconds=n / 1000))
        seq = out

    # object dtype: cftime or python datetime already
    elif arr.dtype == object:
        seq = list(arr.ravel())

    else:
        raise TypeError(f"Unsupported time dtype {arr.dtype!r} for CF encoding")

    nums = np.asarray(cftime.date2num(seq, units=units, calendar=calendar), dtype="f8")

    return nums.reshape(arr.shape)


def bounds_from_centers_1d(vals: np.ndarray, kind: str) -> np.ndarray:
    """Compute [n,2] cell bounds from 1-D centers for 'lat' or 'lon'.

    - For 'lat': clamps to [-90, 90]
    - For 'lon': treats as periodic [0, 360)
    - Works with non-uniform spacing (uses midpoints between neighbors)
    """
    v = np.asarray(vals, dtype="f8").reshape(-1)
    n = v.size
    if n == 1:
        # Special case: single value, make a cell of width 1 centered on v[0]
        bounds = np.array([[v[0] - 0.5, v[0] + 0.5]], dtype="f8")
    elif n < 2:
        raise ValueError("Need at least 1 point to compute bounds")
    else:
        # neighbor midpoints
        mid = 0.5 * (v[1:] + v[:-1])  # length n-1
        bounds = np.empty((n, 2), dtype="f8")
        bounds[1:, 0] = mid
        bounds[:-1, 1] = mid

        # end caps: extrapolate by half-step at ends
        first_step = v[1] - v[0]
        last_step = v[-1] - v[-2]
        bounds[0, 0] = v[0] - 0.5 * first_step
        bounds[-1, 1] = v[-1] + 0.5 * last_step

    if kind == "lat":
        # clamp to physical limits
        bounds[:, 0] = np.maximum(bounds[:, 0], -90.0)
        bounds[:, 1] = np.minimum(bounds[:, 1], 90.0)
    elif kind == "lon":
        # wrap to [0, 360)
        bounds = bounds % 360.0
        # ensure continuity: each cell's upper bound matches next cell's lower bound
        for i in range(bounds.shape[0] - 1):
            bounds[i, 1] = bounds[i + 1, 0]
        # For the last cell, ensure wrap to 360 if needed
        if bounds[-1, 1] < bounds[-1, 0]:
            bounds[-1, 1] = 360.0
        # Postprocessing: round bounds to 8 decimals for continuity
        bounds = np.round(bounds, 8)

    return bounds


def roll_for_monotonic_with_bounds(lon, lon_bnds):
    """Roll lon and lon_bnds together so both are strictly increasing and aligned."""
    d = np.diff(lon)
    k = np.where(d < 0)[0]
    if k.size:
        shift = k[0] + 1
        lon = np.roll(lon, -shift)
        lon_bnds = np.roll(lon_bnds, -shift, axis=0)
        return lon, lon_bnds, -shift
    return lon, lon_bnds, 0


# --- CMOR attribute compat layer (handles both API variants) ---
def set_cmor_attr(name: str, value) -> None:
    """Set CMOR dataset attribute, handling both old and new API."""
    try:
        cmor.set_cur_dataset_attribute(name, value)  # new API
    except AttributeError:  # fallback for older CMOR
        cmor.setGblAttr(name, value)  # type: ignore[attr-defined]


def get_cmor_attr(name: str):
    """Get CMOR dataset attribute, handling both old and new API."""
    try:
        return cmor.get_cur_dataset_attribute(name)  # new API
    except AttributeError:  # fallback for older CMOR
        return cmor.getGblAttr(name)  # type: ignore[attr-defined]


def make_strictly_monotonic(x, direction="increasing"):
    """
    Return a float copy of x with strictly monotonic values by minimally nudging
    entries when needed.

    Parameters
    ----------
    x : array_like
        1-D array.
    direction : {"increasing", "decreasing"}
        Desired strict monotonic direction.

    Notes
    -----
    - Uses np.nextafter(prev, ±inf) to bump the *smallest possible* amount.
    - NaNs split the series into independent segments (left unchanged).
    - If an infinite value appears where a further increase/decrease is required,
      a ValueError is raised because it can't be nudged.
    >>> import numpy as np
    >>> arr = np.array([1, 2, 2, 3])
    >>> make_strictly_monotonic(arr)
    array([1., 2., 2., 3.])
    >>> arr = np.array([3, 2, 2, 1])
    >>> make_strictly_monotonic(arr, direction="decreasing")
    array([3., 2., 2., 1.])
    """
    y = np.asarray(x, dtype=float).copy()
    if y.ndim != 1:
        raise ValueError(f"x must be 1-D {y.ndim}")

    inc = direction.lower().startswith("inc")
    n = y.size
    i = 0
    while i < n:
        # skip NaNs; treat each finite segment independently
        if not np.isfinite(y[i]):
            i += 1
            continue
        # find contiguous finite segment [i:j)
        j = i + 1
        while j < n and np.isfinite(y[j]):
            j += 1

        smallest_normal_float = 1.0e-12
        # enforce strict monotonicity within y[i:j]
        for k in range(i + 1, j):
            prev = y[k - 1]
            curr = y[k]
            if inc:
                if prev == np.inf:
                    if not curr > prev:
                        raise ValueError(
                            "Cannot make sequence strictly increasing past +inf."
                        )
                if not curr > prev:
                    y[k] = max(curr, prev + smallest_normal_float)
            else:
                if prev == -np.inf:
                    if not curr < prev:
                        raise ValueError(
                            "Cannot make sequence strictly decreasing past -inf."
                        )
                if not curr < prev:
                    y[k] = min(curr, prev - smallest_normal_float)
        i = j  # move to next segment

    return y


def is_strictly_monotonic(arr):
    """
    simple test to see if 1d array is strictly monotonic
    >>> import numpy as np
    >>> arr = np.array([1, 2, 3])
    >>> bool(is_strictly_monotonic(arr))
    True
    >>> arr = np.array([3, 2, 1])
    >>> bool(is_strictly_monotonic(arr))
    True
    >>> arr = np.array([1, 2, 2, 3])
    >>> bool(is_strictly_monotonic(arr))
    False
    >>> arr = np.array([3., 2., 2., 1.])
    >>> bool(is_strictly_monotonic(arr))
    False
    """
    # Check for non-decreasing
    is_increasing = np.all(arr[:-1] < arr[1:])
    # Check for non-increasing
    is_decreasing = np.all(arr[:-1] > arr[1:])
    return is_increasing or is_decreasing


def sigma_mid_and_bounds(ds: xr.Dataset, levels: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (mid_sigma, bounds_sigma) in [0,1] for standard_hybrid_sigma.
    >>> import numpy as np
    >>> import xarray as xr
    >>> ds = xr.Dataset({
    ...     "hybm": ("mid", [0.2, 0.5, 0.8]),
    ...     "hybi": ("edge", [0.0, 0.4, 0.6, 1.0])
    ... })
    >>> levels = {"hybm": "hybm", "hybi": "hybi"}
    >>> sigma_mid, sigma_bnds = sigma_mid_and_bounds(ds, levels)
    >>> sigma_mid
    array([0.2, 0.5, 0.8])
    >>> sigma_bnds
    array([[0. , 0.4],
           [0.4, 0.6],
           [0.6, 1. ]])
    """
    lev_name = levels.get("src_axis_name", "lev")
    hybm_name = levels.get("hybm", "hybm")  # B mid
    # hyai_name = levels.get("hyai", "hyai")  # A interfaces (optional)
    hybi_name = levels.get("hybi", "hybi")  # B interfaces (preferred)
    ilev_name = levels.get("src_axis_bnds", "ilev")

    # 1) midpoints: prefer B mid (dimensionless 0..1); fallback to lev if already 0..1
    if hybm_name in ds:
        mid = np.asarray(ds[hybm_name].values, dtype="f8")
        mid = make_strictly_monotonic(mid)
    elif lev_name in ds:
        mid_candidate = np.asarray(ds[lev_name].values, dtype="f8")
        if np.nanmin(mid_candidate) >= 0.0 and np.nanmax(mid_candidate) <= 1.0:
            mid = mid_candidate
        else:
            raise ValueError(f"{lev_name} is not sigma (0..1);")
    else:
        raise KeyError("No sigma mid-levels found (need hybm or lev in [0,1]).")

    # 2) bounds: prefer B interfaces; else use ilev if 0..1; else synthesize
    if hybi_name in ds:
        edges = np.asarray(ds[hybi_name].values, dtype="f8")
        edges = make_strictly_monotonic(edges)
        if edges.ndim == 1 and edges.size == mid.size + 1:
            bnds = np.column_stack((edges[:-1], edges[1:]))
        else:
            raise ValueError(f"{hybi_name} has unexpected shape.")
    elif ilev_name in ds:
        ilev = np.asarray(ds[ilev_name].values, dtype="f8")
        if (
            np.nanmin(ilev) >= 0.0
            and np.nanmax(ilev) <= 1.0
            and ilev.size == mid.size + 1
        ):
            bnds = np.column_stack((ilev[:-1], ilev[1:]))
        else:
            # synthesize from mid if ilev isn't sigma
            edges = np.empty(mid.size + 1, dtype="f8")
            edges[1:-1] = 0.5 * (mid[:-1] + mid[1:])
            edges[0] = 0.0
            edges[-1] = 1.0
            bnds = np.column_stack((edges[:-1], edges[1:]))
    else:
        edges = np.empty(mid.size + 1, dtype="f8")
        edges[1:-1] = 0.5 * (mid[:-1] + mid[1:])
        edges[0] = 0.0
        edges[-1] = 1.0
        bnds = np.column_stack((edges[:-1], edges[1:]))

    # sanity: must be in [0,1]
    if (
        np.nanmin(mid) < 0.0
        or np.nanmax(mid) > 1.0
        or np.nanmin(bnds) < 0.0
        or np.nanmax(bnds) > 1.0
    ):
        raise ValueError("sigma mid/bounds not in [0,1].")

    if not is_strictly_monotonic(mid):
        raise ValueError(f"sigma mid {mid} not monotonic.")
    if not is_strictly_monotonic(bnds):
        raise ValueError(f"sigma bounds {bnds} not monotonic.")

    return mid, bnds


def resolve_table_filename(tables_path: Path, key: str) -> str:
    """Return CMOR table filename for a given key by searching common patterns."""
    # key like "Amon" or "coordinate"
    candidates = [
        tables_path / f"{key}.json",
        tables_path / f"CMIP7_{key}.json",
        tables_path / f"CMIP6_{key}.json",
        tables_path / f"{key.capitalize()}.json",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return f"{key}.json"


def _fx_glob_pattern(name: str) -> str:
    # CMOR filenames vary; this finds most fx files for this var
    # e.g., *_sftlf_fx_*.nc  or sftlf_fx_*.nc
    return f"**/*_{name}_fx_*.nc"


def open_existing_fx(outdir: Path, name: str) -> xr.DataArray | None:
    """Search recursively for an existing fx file for this variable name."""
    # Search recursively for an existing fx file for this var
    for p in outdir.rglob(_fx_glob_pattern(name)):
        try:
            ds = xr.open_dataset(p, engine="netcdf4")
            if name in ds:
                return ds[name]
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as e:
            # OSError: unreadable/corrupt file, low-level I/O; ValueError: engine/decoding issues
            warnings.warn(f"[fx] failed to open {p} with netcdf4: {e}", RuntimeWarning)
        except (ImportError, ModuleNotFoundError) as e:
            # netCDF4 backend not installed
            warnings.warn(f"[fx] netcdf4 backend unavailable: {e}", RuntimeWarning)

    return None
