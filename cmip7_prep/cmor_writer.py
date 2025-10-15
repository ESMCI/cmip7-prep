# pylint: disable=too-many-lines
"""Thin CMOR wrapper used by cmip7_prep.

This module centralizes CMOR session setup and writing so that the rest of the
pipeline can stay xarray-first. It supports either a dataset JSON file (preferred)
or directly injected global attributes, and creates axes based on the coordinates
present in the provided dataset. It also supports a packaged default
`cmor_dataset.json` living under `cmip7_prep/data/`.
"""

from contextlib import AbstractContextManager
from pathlib import Path
import json
import tempfile
import re
import types
import warnings
from importlib.resources import files, as_file
from typing import Any, Sequence, Optional, Union
import datetime as dt

import logging
import cftime
import cmor

import numpy as np
import xarray as xr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_FILL_DEFAULT = 1.0e20
_HANDLE_RE = re.compile(r"^hdl:21\.14100/[0-9a-f\-]{36}$", re.IGNORECASE)
_UUID_RE = re.compile(r"^[0-9a-f\-]{36}$", re.IGNORECASE)


def packaged_dataset_json(filename: str = "cmor_dataset.json"):
    """Context manager yielding a real filesystem path to the packaged mapping file."""
    res = files("cmip7_prep").joinpath(f"data/{filename}")
    return as_file(res)


def _filled_for_cmor(
    da: xr.DataArray, fill: float | None = None
) -> tuple[xr.DataArray, float]:
    """
    Replace NaNs with CMOR missing value (default 1e20) and return (data, fill).
    >>> import xarray as xr
    >>> arr = xr.DataArray([1.0, np.nan, 2.0])
    >>> _filled_for_cmor(arr, fill=-999.0)
    (<xarray.DataArray (dim_0: 3)> Size: 24B
    array([   1., -999.,    2.])
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
    return da2, f


# ---------------------------------------------------------------------
# Time encoding
# ---------------------------------------------------------------------


def _encode_time_to_num(obj, units: str, calendar: str) -> np.ndarray:
    """
    Return numeric CF time for:
      - xarray.DataArray of cftime or datetime64
      - numpy.ndarray (any shape) of cftime / datetime64 / python datetime
      - scalar cftime / python datetime

    Always returns float64 ndarray of the same shape as input.
    >>> import cftime
    >>> arr = [cftime.DatetimeNoLeap(2000, 1, 1), cftime.DatetimeNoLeap(2000, 1, 2)]
    >>> _encode_time_to_num(arr, units="days since 2000-01-01", calendar="noleap")
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


def _bounds_from_centers_1d(vals: np.ndarray, kind: str) -> np.ndarray:
    """Compute [n,2] cell bounds from 1-D centers for 'lat' or 'lon'.

    - For 'lat': clamps to [-90, 90]
    - For 'lon': treats as periodic [0, 360)
    - Works with non-uniform spacing (uses midpoints between neighbors)
    """
    v = np.asarray(vals, dtype="f8").reshape(-1)
    n = v.size
    if n < 2:
        raise ValueError("Need at least 2 points to compute bounds")

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
        # ensure each row is increasing in modulo arithmetic
        wrap = bounds[:, 1] < bounds[:, 0]
        if np.any(wrap):
            bounds[wrap, 1] += 360.0
    else:
        raise ValueError("kind must be 'lat' or 'lon'")

    return bounds


def _encode_time_bounds_to_num(tb, units: str, calendar: str) -> np.ndarray:
    """
    Encode bounds array of shape (..., 2) to numeric CF time.
    Returns float64 array with same shape.
    >>> import cftime
    >>> tb = [
    ...     [cftime.DatetimeNoLeap(2000, 1, 1), cftime.DatetimeNoLeap(2000, 1, 2)],
    ...     [cftime.DatetimeNoLeap(2000, 1, 2), cftime.DatetimeNoLeap(2000, 1, 3)]
    ... ]
    >>> _encode_time_bounds_to_num(tb, units="days since 2000-01-01", calendar="noleap")
    array([[0., 1.],
           [1., 2.]])
    """
    tba = tb.values if hasattr(tb, "values") else np.asarray(tb)
    if tba.ndim < 1 or tba.shape[-1] != 2:
        raise ValueError(f"time bounds must have last dim == 2, got {tba.shape}")
    left = _encode_time_to_num(tba[..., 0], units, calendar)
    right = _encode_time_to_num(tba[..., 1], units, calendar)
    return np.stack([left, right], axis=-1)


def _is_radians(vals: np.ndarray, units: str | None) -> bool:
    """
    Determine if values are in radians based on units.

    >>> import numpy as np
    >>> arr = np.array([0, np.pi/2, np.pi])
    >>> bool(_is_radians(arr, units="radian"))
    True
    >>> bool(_is_radians(arr, units="degrees"))
    False
    >>> bool(_is_radians(arr, units=None))
    True
    >>> arr = np.array([0, 90, 180])
    >>> bool(_is_radians(arr, units=None))
    False
    """
    # ...existing code...
    u = (units or "").strip().lower()
    if u in {"radian", "radians"}:
        return True
    if u:
        return False
    v = np.asarray(vals, dtype="f8")
    # Heuristic: lat in radians typically ≤ ~π/2 in magnitude; lon ≤ ~2π
    # If max |lat| ≤ π and some values are between ~-π and π, assume radians.
    return np.nanmax(np.abs(v)) <= (np.pi + 1e-6)


# --- CMOR attribute compat layer (handles both API variants) ---
def _set_attr(name: str, value) -> None:
    try:
        cmor.set_cur_dataset_attribute(name, value)  # new API
    except AttributeError:  # fallback for older CMOR
        cmor.setGblAttr(name, value)  # type: ignore[attr-defined]


def _get_attr(name: str):
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


def _sigma_mid_and_bounds(
    ds: xr.Dataset, levels: dict
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (mid_sigma, bounds_sigma) in [0,1] for standard_hybrid_sigma.
    >>> import numpy as np
    >>> import xarray as xr
    >>> ds = xr.Dataset({
    ...     "hybm": ("mid", [0.2, 0.5, 0.8]),
    ...     "hybi": ("edge", [0.0, 0.4, 0.6, 1.0])
    ... })
    >>> levels = {"hybm": "hybm", "hybi": "hybi"}
    >>> sigma_mid, sigma_bnds = _sigma_mid_and_bounds(ds, levels)
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


def _resolve_table_filename(tables_path: Path, key: str) -> str:
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


DatasetJsonLike = Union[str, Path, AbstractContextManager]


def _fx_glob_pattern(name: str) -> str:
    # CMOR filenames vary; this finds most fx files for this var
    # e.g., *_sftlf_fx_*.nc  or sftlf_fx_*.nc
    return f"**/*_{name}_fx_*.nc"


def _open_existing_fx(outdir: Path, name: str) -> xr.DataArray | None:
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


# ---------------------------------------------------------------------
# CMOR session
# ---------------------------------------------------------------------
class CmorSession(
    AbstractContextManager
):  # pylint: disable=too-many-instance-attributes
    """Context manager for CMOR sessions.

    Parameters
    ----------
    tables_path : str or Path
        Directory containing CMOR table JSONs (e.g., CMIP6_*.json or CMIP7_*.json).
    dataset_json : str or Path, optional
        Path to a cmor_dataset.json with the experiment/source metadata.
    dataset_attrs : dict, optional
        Alternative to `dataset_json`, allowing direct attribute injection.
    """

    def __init__(
        self,
        *,
        tables_path: Path | str,
        dataset_attrs: dict[str, str] | None = None,
        dataset_json: Optional[DatasetJsonLike] = None,
        tracking_prefix: str | None = None,
        # NEW: one log per run (session)
        log_dir: Path | str | None = None,
        log_name: str | None = None,
        outdir: Path | str | None = None,
    ) -> None:
        self.tables_path = Path(tables_path)
        self.dataset_attrs = dict(dataset_attrs or {})
        self.dataset_json = dataset_json
        self._dataset_json_cm = None
        self.tracking_prefix = tracking_prefix
        # logging config
        self._log_dir = Path(log_dir) if log_dir is not None else None
        self._log_name = log_name
        self._log_path: Path | None = None
        self._pending_ps = None
        self._outdir = Path(outdir or "./CMIP7").resolve()
        self._outdir.mkdir(parents=True, exist_ok=True)
        self._fx_written: set[str] = (
            set()
        )  # remembers which fx vars were written this run
        self._fx_cache: dict[str, xr.DataArray] = (
            {}
        )  # regridded fx fields cached in-memory

    def __enter__(self) -> "CmorSession":
        # Resolve logfile path if requested
        if self._log_dir is not None:
            ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            fname = self._log_name or f"cmor_{ts}.log"
            self._log_dir.mkdir(parents=True, exist_ok=True)
            self._log_path = (self._log_dir / fname).resolve()

        # Setup CMOR; pass logfile if CMOR supports it, else fall back
        try:
            if self._log_path is not None:
                logger.info("CMOR logfile: %s", self._log_path)
                cmor.setup(
                    inpath=str(self.tables_path),
                    netcdf_file_action=getattr(
                        cmor, "CMOR_REPLACE_3", getattr(cmor, "CMOR_REPLACE", 3)
                    ),
                    set_verbosity=cmor.CMOR_NORMAL,
                    logfile=str(self._log_path),  # supported by newer CMOR builds
                )
            else:
                cmor.setup(
                    inpath=str(self.tables_path),
                    netcdf_file_action=getattr(
                        cmor, "CMOR_REPLACE_3", getattr(cmor, "CMOR_REPLACE", 3)
                    ),
                    set_verbosity=cmor.CMOR_NORMAL,
                )
        except TypeError:
            # Older CMOR with no 'logfile' kw
            cmor.setup(
                inpath=str(self.tables_path),
                netcdf_file_action=getattr(
                    cmor, "CMOR_REPLACE_3", getattr(cmor, "CMOR_REPLACE", 3)
                ),
                set_verbosity=cmor.CMOR_NORMAL,
            )
            # best-effort fallback setter
            for name in ("set_logfile", "setLogFile", "set_log_file", "setLogfile"):
                if self._log_path is not None and hasattr(cmor, name):
                    getattr(cmor, name)(str(self._log_path))
                    break

        # Resolve dataset_json to a real filesystem path
        dj = self.dataset_json
        if dj is None:
            # packaged file → returns a context manager
            cm = packaged_dataset_json("cmor_dataset.json")
            self._dataset_json_cm = cm
            p = cm.__enter__()  # ← ENTER the CM, get a Path
        elif isinstance(dj, (str, Path)):
            p = Path(dj)
        else:
            # caller passed a context manager directly
            self._dataset_json_cm = dj
            p = dj.__enter__()  # ← ENTER the CM, get a Path

        with open(p, encoding="utf-8") as f:
            cfg = json.load(f)
        cfg["outpath"] = str(self._outdir)
        tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(cfg, tmp)
        tmp.close()
        cmor.dataset_json(str(tmp.name))

        try:
            prod = cmor.get_cur_dataset_attribute("product")  # type: ignore[attr-defined]
        except Exception:  # pylint: disable=broad-except
            prod = None
        if prod != "model-output":
            cmor.set_cur_dataset_attribute("product", "model-output")

        # long paragraph; split to keep lines < 100
        inst = _get_attr("institution_id") or "NCAR"
        license_text = (
            f"CMIP6 model data produced by {inst} is licensed under a Creative Commons "
            "Attribution 4.0 International License "
            "(https://creativecommons.org/licenses/by/4.0/). "
            "Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use "
            "governing CMIP6 output, including citation requirements and proper "
            "acknowledgment. Further information about this data, including some "
            "limitations, can be found via the further_info_url (recorded as a "
            "global attribute in this file). The data producers and data providers "
            "make no warranty, either express or implied, including, but not "
            "limited to, warranties of merchantability and fitness for a "
            "particular purpose. All liabilities arising from the supply of the "
            "information (including any liability arising in negligence) are "
            "excluded to the fullest extent permitted by law."
        )

        cmor.set_cur_dataset_attribute("license", license_text)

        # Tell CMOR how to build tracking_id; let CMOR generate it
        prefix = _get_attr("tracking_prefix")
        if not (isinstance(prefix, str) and prefix.startswith("hdl:")):
            _set_attr("tracking_prefix", "hdl:21.14100/")

        # If a non-handle tracking_id snuck in (e.g., bare UUID from JSON), clear it
        tid = _get_attr("tracking_id")
        if isinstance(tid, str) and not tid.startswith("hdl:21.14100/"):
            _set_attr(
                "tracking_id", ""
            )  # empty lets CMOR regenerate from tracking_prefix

        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            cmor.close()  # or your per-var close logic
        finally:
            if self._dataset_json_cm is not None:
                # Close the context manager we entered in __enter__
                self._dataset_json_cm.__exit__(exc_type, exc, tb)
                self._dataset_json_cm = None

    # -------------------------
    # internal helpers
    # -------------------------
    def _define_axes(self, ds: xr.Dataset, vdef: any) -> list[int]:
        """Define CMOR axis IDs for the variable in ds according to the CMOR tables.

        Rules:
        - time: numeric coord + optional bounds
        - horizontal: table_entry='latitude'/'longitude', with bounds (synth if missing)
        - hybrid sigma: table_entry='alev' defined by length only (+ z-factors a, b, p0, ps)
        - pressure levels: table_entry='plev' with coord_vals (+ optional bounds)
        - axis_ids order MUST match the order of ds[vdef.name].dims
        """

        # ---- helpers ----
        def _get_time_and_bounds(dsi: xr.Dataset):
            time_da = (
                dsi.coords["time"]
                if "time" in dsi.coords
                else (dsi["time"] if "time" in dsi else None)
            )
            if time_da is None:
                return None, None, None
            units = time_da.attrs.get("units", "days since 1850-01-01")
            cal = time_da.attrs.get(
                "calendar", time_da.encoding.get("calendar", "noleap")
            )
            tvals = _encode_time_to_num(time_da, units, cal)

            bname = time_da.attrs.get("bounds")
            tb = (
                dsi[bname]
                if isinstance(bname, str) and bname in dsi
                else (
                    dsi["time_bounds"]
                    if "time_bounds" in dsi
                    else (dsi["time_bnds"] if "time_bnds" in dsi else None)
                )
            )
            tbnum = _encode_time_to_num(tb, units, cal) if tb is not None else None
            return tvals, tbnum, str(units)

        def _get_1d_with_bounds(dsi: xr.Dataset, name: str, units_default: str):
            da = dsi.coords[name] if name in dsi.coords else dsi[name]
            vals = np.asarray(da.values, dtype="f8")
            units = da.attrs.get("units", units_default)

            # try common names; else synthesize conservative bounds
            cand = (f"{name}_bnds", f"{name}_bounds")
            b = None
            for cn in cand:
                if cn in dsi:
                    b = np.asarray(dsi[cn].values, dtype="f8")
                    break
            if b is None:
                dv = np.diff(vals)
                if vals.size >= 2:
                    lower = np.concatenate(
                        ([vals[0] - dv[0] / 2.0], vals[:-1] + dv / 2.0)
                    )
                    upper = np.concatenate(
                        (vals[1:] - dv / 2.0, [vals[-1] + dv[-1] / 2.0])
                    )
                    b = np.stack([lower, upper], axis=-1)
                else:
                    b = np.array([[vals[0] - 0.5, vals[0] + 0.5]], dtype="f8")
            return vals, b, units

        var_name = getattr(vdef, "name", None)
        if var_name is None or var_name not in ds:
            raise KeyError(f"Variable to write not found in dataset: {var_name!r}")
        var_da = ds[var_name]
        var_dims = list(var_da.dims)
        alev_id = None
        plev_id = None
        # ---- horizontal axes (use CMOR names) ----
        if "lat" not in ds or "lon" not in ds:
            raise KeyError(
                "Expected 'lat' and 'lon' in dataset for CMOR horizontal axes."
            )
        lat_vals, lat_bnds, _ = _get_1d_with_bounds(ds, "lat", "degrees_north")
        lon_vals, lon_bnds, _ = _get_1d_with_bounds(ds, "lon", "degrees_east")

        lat_id = cmor.axis(
            table_entry="latitude",
            units="degrees_north",
            coord_vals=lat_vals,
            cell_bounds=lat_bnds,
        )
        lon_id = cmor.axis(
            table_entry="longitude",
            units="degrees_east",
            coord_vals=lon_vals,
            cell_bounds=lon_bnds,
        )

        # ---- time axis ----
        time_id = None
        tvals, tbnds, t_units = _get_time_and_bounds(ds)
        if tvals is not None:
            time_id = cmor.axis(
                table_entry="time",
                units=t_units,
                coord_vals=tvals,
                cell_bounds=tbnds if tbnds is not None else None,
            )
        # --- vertical: standard_hybrid_sigma ---
        levels = getattr(vdef, "levels", {}) or {}

        if (levels.get("name") or "").lower() in {
            "standard_hybrid_sigma",
            "alevel",
            "alev",
        } or "lev" in var_dims:
            # names in the native ds
            hyam_name = levels.get("hyam", "hyam")  # A mid (dimensionless)
            hybm_name = levels.get("hybm", "hybm")  # B mid (dimensionless)
            hyai_name = levels.get("hyai", "hyai")  # A interfaces (optional)
            hybi_name = levels.get("hybi", "hybi")  # B interfaces (preferred)
            ps_name = levels.get("ps", "PS")

            # 0) sigma mid and bounds (0..1)
            sigma_mid, sigma_bnds = _sigma_mid_and_bounds(ds, levels)

            # 1) define axis using sigma
            alev_id = cmor.axis(
                table_entry="standard_hybrid_sigma",
                units="1",
                coord_vals=sigma_mid,
                cell_bounds=sigma_bnds,
            )

            # 2) z-factors: a(lev), b(lev), p0(scalar), ps(time,lat,lon)

            cmor.zfactor(
                zaxis_id=alev_id,
                zfactor_name="a",
                units="1",
                axis_ids=[alev_id],
                zfactor_values=np.asarray(ds[hyam_name].values),
                zfactor_bounds=np.asarray(ds[hyai_name].values),
            )
            cmor.zfactor(
                zaxis_id=alev_id,
                zfactor_name="b",
                units="1",
                axis_ids=[alev_id],
                zfactor_values=np.asarray(ds[hybm_name].values),
                zfactor_bounds=np.asarray(ds[hybi_name].values),
            )

            # p0 scalar
            cmor.zfactor(
                zaxis_id=alev_id, zfactor_name="p0", units="Pa", zfactor_values=1.0e5
            )
            # ps(time,lat,lon) zfactor must be DEFINED before the main variable
            ps_da = ds[ps_name]  # ensure units are Pa
            ps_zvar_id = cmor.zfactor(
                zaxis_id=alev_id,
                zfactor_name="ps",
                units="Pa",
                axis_ids=[
                    time_id,
                    lat_id,
                    lon_id,
                ],  # order must match var’s non-vertical axes
            )
            # stash to write before main variable
            self._pending_ps = (ps_zvar_id, ps_da)

        elif "plev" in var_dims:
            # Pressure levels expected in data
            plev = ds["plev"]
            pvals = np.asarray(plev.values, dtype="f8")
            punits = str(plev.attrs.get("units", "")).strip().lower()

            # Optional bounds
            pb = None
            for nm in ("plev_bnds", "plev_bounds"):
                if nm in ds:
                    pb = np.asarray(ds[nm].values, dtype="f8")
                    break

            # --- NEW: normalize to Pa ---
            # Convert hPa/mb/etc → Pa;
            # if units missing but magnitudes look like hPa (<= ~2000), assume hPa
            if punits in {"hpa", "mb", "mbar", "millibar"} or (
                punits == "" and (np.nanmax(pvals) if pvals.size else 0.0) <= 2000.0
            ):
                pvals = pvals * 100.0
                if pb is not None:
                    pb = pb * 100.0
            # (if punits already "pa" or values are already in Pa, do nothing)

            # Debug (optional)
            if pvals.size == 19:
                table_entry = "plev19"
            else:
                table_entry = "plev"

            levels = getattr(vdef, "levels", None) or {}
            name = levels.get("name")
            if isinstance(name, str):
                key = name.strip()
                # Accept entries like plev19, plev7h, plev27, etc.
                if key.lower().startswith("plev"):
                    table_entry = key

            plev_id = cmor.axis(
                table_entry=table_entry,
                units="Pa",
                coord_vals=pvals,
                cell_bounds=pb if pb is not None else None,
            )

        axes_ids = []
        if "time" in var_dims:
            axes_ids.append(time_id)
        if alev_id is not None:
            axes_ids.append(alev_id)
        if plev_id is not None:
            axes_ids.append(plev_id)

        axes_ids.extend([lat_id, lon_id])
        return axes_ids

    def _write_fx_2d(self, ds: xr.Dataset, name: str, units: str) -> None:
        if name not in ds:
            return
        table_filename = _resolve_table_filename(self.tables_path, "fx")
        cmor.load_table(table_filename)

        lat = ds["lat"].values
        lon = ds["lon"].values
        lat_b = ds.get("lat_bnds")
        lon_b = ds.get("lon_bnds")
        lat_b = (
            lat_b.values
            if isinstance(lat_b, xr.DataArray)
            else _bounds_from_centers_1d(lat, "lat")
        )
        lon_b = (
            lon_b.values
            if isinstance(lon_b, xr.DataArray)
            else _bounds_from_centers_1d(lon, "lon")
        )

        lat_id = cmor.axis(
            "latitude", "degrees_north", coord_vals=lat, cell_bounds=lat_b
        )
        lon_id = cmor.axis(
            "longitude", "degrees_east", coord_vals=lon, cell_bounds=lon_b
        )
        data_filled, fillv = _filled_for_cmor(ds[name])

        var_id = cmor.variable(name, units, [lat_id, lon_id], missing_value=fillv)

        cmor.write(
            var_id,
            np.asarray(data_filled),
        )
        cmor.close(var_id)

    def ensure_fx_written_and_cached(self, ds_regr: xr.Dataset) -> xr.Dataset:
        """Ensure sftlf and areacella exist in ds_regr and are written once as fx.
        If not present in ds_regr, try to read from existing CMOR fx files in outdir.
        If present in ds_regr but not yet written this run, write and cache them.
        Returns ds_regr augmented with any missing fx fields.
        """
        need = [("sftlf", "%"), ("areacella", "m2")]
        out = ds_regr

        for name, units in need:
            # 1) Already cached this run?
            if name in self._fx_cache:
                if name not in out:
                    out = out.assign({name: self._fx_cache[name]})
                continue

            # 2) Present in regridded dataset? (best case)
            if name in out:
                self._fx_cache[name] = out[name]
                if name not in self._fx_written:
                    # Convert landfrac to % if needed
                    if name == "sftlf":
                        v = out[name]
                        if (np.nanmax(v.values) <= 1.0) and v.attrs.get(
                            "units", ""
                        ) not in ("%", "percent"):
                            out = out.assign(
                                {
                                    name: (v * 100.0).assign_attrs(
                                        v.attrs | {"units": "%"}
                                    )
                                }
                            )
                            self._fx_cache[name] = out[name]
                    self._write_fx_2d(out, name, units)
                    self._fx_written.add(name)
                    continue

            # 3) Not present in ds_regr → try reading existing CMOR fx output
            if self._outdir:
                fx_da = _open_existing_fx(self._outdir, name)
                if fx_da is not None:
                    # Verify grid match (simple equality on lat/lon values)
                    if (
                        "lat" in out
                        and "lon" in out
                        and np.array_equal(out["lat"].values, fx_da["lat"].values)
                        and np.array_equal(out["lon"].values, fx_da["lon"].values)
                    ):
                        out = out.assign({name: fx_da})
                        self._fx_cache[name] = out[name]
                        self._fx_written.add(name)  # already exists on disk
                        continue
                # If grid mismatch, you could regrid fx_da here; for now, skip.
                # 4) Last resort: leave missing; caller may compute it later
        return out

    # public API
    # -------------------------
    def write_variable(
        self,
        ds: xr.Dataset,
        varname: str,
        vdef: Any,
    ) -> None:
        """Write one variable from ds to a CMOR-compliant NetCDF file."""
        # Pick CMOR table: prefer vdef.table, else vdef.realm (default Amon)
        table_key = (
            getattr(vdef, "table", None) or getattr(vdef, "realm", None) or "Amon"
        )

        table_filename = _resolve_table_filename(self.tables_path, table_key)
        cmor.load_table(table_filename)

        data = ds[vdef.name]
        data_filled, fillv = _filled_for_cmor(data)
        axes_ids = self._define_axes(ds, vdef)
        units = getattr(vdef, "units", "") or ""
        var_id = cmor.variable(
            getattr(vdef, "name", varname),
            units,
            axes_ids,
            positive=getattr(vdef, "positive", None),
            missing_value=fillv,
        )
        data = ds[varname]

        # ---- Prepare time info for this write (local, not cached) ----
        time_da = ds.coords.get("time")
        if time_da is None:
            time_da = ds.get("time")
        nt = 0

        self.ensure_fx_written_and_cached(ds)

        # ---- Main variable write ----

        cmor.write(
            var_id,
            np.asarray(data_filled),
            ntimes_passed=nt,
        )
        # ---- Hybrid ps streaming (if present) ----
        if self._pending_ps is not None:
            ps_id, ps_da = self._pending_ps
            if "time" in ps_da.dims:
                ps_da = ps_da.transpose("time", "lat", "lon")
                nt_ps = int(ps_da.sizes["time"])
            else:
                nt_ps = 0
            ps_filled, _ = _filled_for_cmor(ps_da)
            if nt_ps > 0:
                cmor.write(
                    ps_id,
                    np.asarray(ps_filled),
                    ntimes_passed=nt_ps,
                    store_with=var_id,
                )
            else:
                cmor.write(ps_id, np.asarray(ps_filled), store_with=var_id)
            self._pending_ps = None

        cmor.close(var_id)

    def write_variables(
        self,
        ds: xr.Dataset,
        cmip_vars: Sequence[str],
        mapping: "Mapping",
    ) -> None:
        """Write multiple CMIP variables from one dataset."""
        for v in cmip_vars:
            cfg = mapping.get_cfg(v) or {}
            table = cfg.get("table", "Amon")
            units = cfg.get("units", "")
            positive = cfg.get("positive") or None
            vdef = types.SimpleNamespace(
                name=v,
                table=table,
                realm=table,
                units=units,
                positive=positive,
            )
            # pylint: disable=broad-exception-caught
            try:
                self.write_variable(ds, v, vdef)
            except Exception as e:
                warnings.warn(f"[cmor] skipping {v} due to error: {e}", RuntimeWarning)
                # continue to next variable
