"""Thin CMOR wrapper used by cmip7_prep.

This module centralizes CMOR session setup and writing so that the rest of the
pipeline can stay xarray-first. It supports either a dataset JSON file (preferred)
or directly injected global attributes, and creates axes based on the coordinates
present in the provided dataset. It also supports a packaged default
`cmor_dataset.json` living under `cmip7_prep/data/`.
"""

from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
import os
import re
import types
import json
from importlib.resources import files, as_file
from typing import Any, Sequence
import datetime as dt
import cftime

try:
    import cmor
except Exception:  # pylint: disable=broad-except
    cmor = None  # pylint: disable=invalid-name

import numpy as np
import xarray as xr

_FILL_DEFAULT = 1.0e20
_HANDLE_RE = re.compile(r"^hdl:21\.14100/[0-9a-f\-]{36}$", re.IGNORECASE)
_UUID_RE = re.compile(r"^[0-9a-f\-]{36}$", re.IGNORECASE)


@contextmanager
def _as_path_cm(obj):
    """
    Yield a Path whether `obj` is already a path-like or a context manager
    (e.g., importlib.resources.as_file(...)).
    """
    # already a path-like → no-op context
    if isinstance(obj, (str, Path)):
        yield Path(obj)
        return

    # context manager that yields a path-like
    if hasattr(obj, "__enter__") and hasattr(obj, "__exit__"):
        with obj as p:
            yield Path(p)
        return

    raise TypeError(f"Unsupported dataset_json object type: {type(obj)!r}")


def _filled_for_cmor(
    da: xr.DataArray, fill: float | None = None
) -> tuple[xr.DataArray, float]:
    """Replace NaNs with CMOR missing value (default 1e20) and return (data, fill)."""
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


def _encode_time_bounds_to_num(tb, units: str, calendar: str) -> np.ndarray:
    """
    Encode bounds array of shape (..., 2) to numeric CF time.
    Returns float64 array with same shape.
    """
    tba = tb.values if hasattr(tb, "values") else np.asarray(tb)
    if tba.ndim < 1 or tba.shape[-1] != 2:
        raise ValueError(f"time bounds must have last dim == 2, got {tba.shape}")
    left = _encode_time_to_num(tba[..., 0], units, calendar)
    right = _encode_time_to_num(tba[..., 1], units, calendar)
    return np.stack([left, right], axis=-1)


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


def _bounds_from_centers(
    centers: np.ndarray, delta: float, *, clip=None, wrap=None
) -> np.ndarray:
    centers = np.asarray(centers, dtype="f8")
    half = 0.5 * delta
    left = centers - half
    right = centers + half
    if clip is not None:
        left = np.clip(left, clip[0], clip[1])
        right = np.clip(right, clip[0], clip[1])
    if wrap is not None:
        wmin, wmax = wrap
        width = wmax - wmin
        left = (left - wmin) % width + wmin
        right = (right - wmin) % width + wmin
    return np.stack([left, right], axis=1)


def _is_radians(vals: np.ndarray, units: str | None) -> bool:
    u = (units or "").strip().lower()
    if u in {"radian", "radians"}:
        return True
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
        dataset_json: Path | None = None,
        tracking_prefix: str | None = None,
        # NEW: one log per run (session)
        log_dir: Path | str | None = None,
        log_name: str | None = None,
    ) -> None:
        self.tables_path = Path(tables_path)
        self.dataset_attrs = dataset_attrs or {}
        self.dataset_json = Path(dataset_json) if dataset_json else None
        self.tracking_prefix = tracking_prefix
        # logging config
        self._log_dir = Path(log_dir) if log_dir is not None else None
        self._log_name = log_name
        self._log_path: Path | None = None

        # internals you already had…
        self._last_time_vals = None
        self._last_time_bnds = None
        self._last_ps_zvar_id = None
        self._last_ps_name = None

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

        if self.dataset_json is not None:
            cmor.dataset_json(str(self.dataset_json))
        else:
            try:
                res = files("cmip7_prep").joinpath("data/cmor_dataset.json")
                # as_file makes sure we have a real filesystem path even if the package is in a zip
                with as_file(res) as _p:
                    cmor.dataset_json(
                        str(_p)
                    )  # <-- this sets fake.dataset_json_path in tests
            except (
                FileNotFoundError,
                OSError,
                json.JSONDecodeError,
                ValueError,
            ):
                # Final fallback: still call dataset_json with a dummy string so fakes record it
                # (real CMOR would error if it tried to open it, but our tests use FakeCMOR)
                try:
                    cmor.dataset_json("cmor_dataset.json")
                except (
                    FileNotFoundError,
                    OSError,
                    json.JSONDecodeError,
                    ValueError,
                ):
                    pass

            # Load coordinates first (defines latitude/longitude/alev/plev, etc.)
            try:
                table = _resolve_table_filename(self.tables_path, "CV")
                cmor.set_cur_dataset_attribute(
                    "_controlled_vocabulary_file", os.path.basename(table)
                )
                table = _resolve_table_filename(self.tables_path, "coordinate")
                cmor.set_cur_dataset_attribute(
                    "_AXIS_ENTRY_FILE", os.path.basename(table)
                )
                table = _resolve_table_filename(self.tables_path, "formula_terms")
                cmor.set_cur_dataset_attribute(
                    "_FORMULA_VAR_FILE", os.path.basename(table)
                )
            except Exception:  # pylint: disable=broad-except
                pass
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

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401
        """Finalize CMOR, closing any open handles."""
        cmor.close()

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

        var_name = getattr(vdef, "name", None)
        if var_name is None or var_name not in ds:
            raise KeyError(f"Variable to write not found in dataset: {var_name!r}")
        var_da = ds[var_name]
        var_dims = list(var_da.dims)

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

        def _get_scalar_p0(dsi: xr.Dataset, p0_name: str) -> float:
            if p0_name in dsi:
                arr = dsi[p0_name]
                v = arr.values[()] if arr.shape == () else np.asarray(arr).ravel()[0]
                try:
                    return float(v)
                except (TypeError, ValueError, OverflowError):
                    pass
            return 1.0e5

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
        # keep for cmor.write()
        self._last_time_vals = tvals
        self._last_time_bnds = tbnds

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

        # ---- vertical axis (hybrid sigma or pressure) ----

        alev_id = None
        plev_id = None
        ps_zvar_id = None
        self._last_ps_zvar_id = None
        self._last_ps_name = None

        levels = getattr(vdef, "levels", None) or {}
        lev_kind = (levels.get("name") or "").lower()

        if lev_kind in {"standard_hybrid_sigma"} or "lev" in var_dims:
            # Names from mapping (with sensible defaults for CESM/CAM)
            lev_name = levels.get("src_axis_name", "lev")
            lev_bnds = levels.get("src_axis_bnds", "ilev")
            hyam_name = levels.get("hyam", "hyam")
            hybm_name = levels.get("hybm", "hybm")
            hyai_name = levels.get("hyai", "hyai")
            hybi_name = levels.get("hybi", "hybi")
            ps_name = levels.get("ps", "PS")
            p0_name = levels.get("p0", "P0")

            # 0) Required arrays present?
            for req in (lev_name, lev_bnds, hyam_name, hybm_name, ps_name, p0_name):
                if req not in ds:
                    raise KeyError(f"Hybrid sigma requires '{req}' in dataset.")
            # bounds for a/b are optional; axis bounds are mandatory
            if hyai_name not in ds or hybi_name not in ds:
                pass  # not fatal for zfactors; but axis bounds still come from lev_bnds

            # 1) Define the hybrid sigma axis with coord values + bounds
            lev_vals = np.asarray(ds[lev_name].values, dtype="f8")
            ilev_vals = np.asarray(
                ds[lev_bnds].values, dtype="f8"
            )  # shape (nlev, 2) or (nlev+1) to be reshaped
            # Ensure bounds are (nlev, 2)
            if ilev_vals.ndim == 1 and ilev_vals.size == lev_vals.size + 1:
                ilev_vals = np.column_stack([ilev_vals[:-1], ilev_vals[1:]])
            elif ilev_vals.ndim != 2 or ilev_vals.shape[1] != 2:
                raise ValueError(
                    f"{lev_bnds} must be (nlev, 2) or length nlev+1; got {ilev_vals.shape}"
                )

            alev_id = cmor.axis(
                table_entry="standard_hybrid_sigma",
                units="1",
                coord_vals=lev_vals,
                cell_bounds=ilev_vals,
            )

            # 2) Z-factors: a (Pa), b (1), p0 (Pa), ps(time,lat,lon) (Pa)
            p0 = float(np.asarray(ds[p0_name]).reshape(()))  # scalar
            hyam = np.asarray(ds[hyam_name].values, dtype="f8")
            hybm = np.asarray(ds[hybm_name].values, dtype="f8")

            # CMOR expects 'a' in Pa (use CESM hyam * p0) and 'b' unitless
            cmor.zfactor(
                zaxis_id=alev_id,
                zfactor_name="a",
                units="Pa",
                axis_ids=[alev_id],
                values=hyam * p0,
            )
            cmor.zfactor(
                zaxis_id=alev_id,
                zfactor_name="b",
                units="1",
                axis_ids=[alev_id],
                values=hybm,
            )
            cmor.zfactor(zaxis_id=alev_id, zfactor_name="p0", units="Pa", values=p0)

            # ps will be written alongside the main variable (so CMOR can snapshot it)
            if ps_name not in ds:
                raise KeyError(
                    "Hybrid coordinate requires surface pressure 'PS' in dataset."
                )
            ps_axes = [ax for ax in (time_id, lat_id, lon_id) if ax is not None]
            ps_zvar_id = cmor.zfactor(
                zaxis_id=alev_id, zfactor_name="ps", units="Pa", axis_ids=ps_axes
            )
            self._last_ps_zvar_id = ps_zvar_id
            self._last_ps_name = ps_name

        elif "plev" in var_dims:
            # Pressure levels expected in data
            plev = ds["plev"]
            pvals = np.asarray(plev.values, dtype="f8")
            punits = plev.attrs.get("units", "Pa")
            pb = None
            for nm in ("plev_bnds", "plev_bounds"):
                if nm in ds:
                    pb = np.asarray(ds[nm].values, dtype="f8")
                    break
            plev_id = cmor.axis(
                table_entry="plev",
                units=str(punits),
                coord_vals=pvals,
                cell_bounds=pb if pb is not None else None,
            )

        axes_ids = []
        if "time" in ds.dims:
            axes_ids.append(time_id)
        if alev_id is not None and "lev" in getattr(vdef, "dims", []):
            axes_ids.append(alev_id)
        if "plev" in getattr(vdef, "dims", []):
            axes_ids.append(plev_id)
        axes_ids.extend([lat_id, lon_id])

        return axes_ids

    # public API
    # -------------------------
    def write_variable(
        self, ds: xr.Dataset, varname: str, vdef: Any, outdir: Path
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
        # Optional variable attributes (e.g., cell_methods, long_name, standard_name)
        if getattr(vdef, "cell_methods", None):
            cmor.set_variable_attribute(var_id, "cell_methods", vdef.cell_methods)
        if getattr(vdef, "long_name", None):
            cmor.set_variable_attribute(var_id, "long_name", vdef.long_name)
        if getattr(vdef, "standard_name", None):
            cmor.set_variable_attribute(var_id, "standard_name", vdef.standard_name)
        data = ds[varname]

        # ---- Prepare time info for this write (local, not cached) ----
        time_da = ds.coords.get("time")
        if time_da is None:
            time_da = ds.get("time")
        tvals = None
        tbnds = None
        nt = 0
        if isinstance(time_da, xr.DataArray):
            nt = int(time_da.sizes.get("time", 0))
            # numeric time passes through; cftime → encode
            try:
                tvals = _encode_time_to_num(
                    time_da,  # works for numeric or cftime
                    units=time_da.attrs.get("units", "days since 1850-01-01"),
                    calendar=time_da.attrs.get(
                        "calendar", time_da.encoding.get("calendar", "noleap")
                    ),
                )
            except ValueError:
                # If numeric, just pass as numpy
                tvals = np.asarray(time_da, dtype="f8")

            # bounds (via CF 'bounds' attr or common names)
            bname = time_da.attrs.get("bounds")
            if isinstance(bname, str) and bname in ds:
                tb = ds[bname]
            elif "time_bounds" in ds:
                tb = ds["time_bounds"]
            elif "time_bnds" in ds:
                tb = ds["time_bnds"]
            else:
                tb = None
            if isinstance(tb, xr.DataArray):
                try:
                    tbnds = _encode_time_to_num(
                        tb,
                        units=time_da.attrs.get("units", "days since 1850-01-01"),
                        calendar=time_da.attrs.get(
                            "calendar", time_da.encoding.get("calendar", "noleap")
                        ),
                    )
                except ValueError:
                    tbnds = np.asarray(tb, dtype="f8")

        # ---- Hybrid ps streaming (if present) ----
        store_with = None
        if getattr(self, "_last_ps_zvar_id", None) is not None:
            store_with = self._last_ps_zvar_id
            ps_name = getattr(self, "_last_ps_name", None)
            if ps_name and ps_name in ds:
                ps_filled, _ = _filled_for_cmor(ds[ps_name])
                cmor.write(
                    store_with,
                    np.asarray(ps_filled),
                    time_vals=tvals,
                    time_bnds=tbnds,
                    ntimes_passed=nt,
                )

        # ---- Main variable write ----
        cmor.write(
            var_id,
            np.asarray(data_filled),
            ntimes_passed=nt,
            time_vals=tvals,
            time_bnds=tbnds,
            store_with=store_with,  # safe if None
        )

        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        outfile = outdir / f"{getattr(vdef, 'name', varname)}.nc"
        cmor.close(var_id, file_name=str(outfile))

    def write_variables(
        self,
        ds: xr.Dataset,
        cmip_vars: Sequence[str],
        mapping: "Mapping",
        *,
        outdir: Path,
    ) -> None:
        """Write multiple CMIP variables from one dataset."""
        for v in cmip_vars:
            cfg = mapping.get_cfg(v) or {}
            table = cfg.get("table", "Amon")
            units = cfg.get("units", "")
            vdef = types.SimpleNamespace(name=v, table=table, realm=table, units=units)
            self.write_variable(ds, v, vdef, outdir=outdir)
