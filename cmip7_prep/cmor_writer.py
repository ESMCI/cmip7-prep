"""Thin CMOR wrapper used by cmip7_prep.

This module centralizes CMOR session setup and writing so that the rest of the
pipeline can stay xarray-first. It supports either a dataset JSON file (preferred)
or directly injected global attributes, and creates axes based on the coordinates
present in the provided dataset. It also supports a packaged default
`cmor_dataset.json` living under `cmip7_prep/data/`.
"""

from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
import re
from importlib.resources import files as ir_files, as_file
from typing import Any, Optional, Dict, Union
import datetime as dt
import cftime
import cmor
import numpy as np
import xarray as xr


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


# ---------------------------------------------------------------------
# Packaged resource helpers
# ---------------------------------------------------------------------
def packaged_dataset_json(filename: str = "cmor_dataset.json") -> Any:
    """Return a context manager yielding a real FS path to a packaged dataset JSON.

    Looks under cmip7_prep/data/.
    Usage:
        with packaged_dataset_json() as p:
            cmor.dataset_json(str(p))
    """
    res = ir_files("cmip7_prep.data").joinpath(filename)
    p = as_file(res)
    print(f" p is {p}")
    return p


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


def _resolve_table_filename(tables_path: Path, table_key: str) -> str:
    """Return basename like 'CMIP6_Amon.json' or 'CMIP7_Amon.json' based on tables_path."""
    # If caller passed an explicit filename, keep it.
    if table_key.endswith(".json"):
        return table_key
    pstr = str(tables_path)
    if "CMIP6" in pstr:
        return f"CMIP6_{table_key}.json"
    if "CMIP7" in pstr:
        return f"CMIP7_{table_key}.json"
    return f"{table_key}.json"


# ---------------------------------------------------------------------
# CMOR session
# ---------------------------------------------------------------------
class CmorSession(AbstractContextManager):
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
        tables_path: Union[str, Path],
        dataset_json: Union[str, Path, None] = None,
        dataset_attrs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.tables_path = str(tables_path)
        self.dataset_json_path = str(dataset_json) if dataset_json else None
        self.dataset_attrs = dataset_attrs or {}

    def __enter__(self) -> "CmorSession":
        """Initialize CMOR and register dataset metadata."""
        # Setup CMOR with the Tables directory
        replace_flag = getattr(cmor, "CMOR_REPLACE_3", getattr(cmor, "CMOR_REPLACE", 0))
        verbosity = getattr(cmor, "CMOR_NORMAL", getattr(cmor, "CMOR_VERBOSE", 0))
        cmor.setup(
            inpath=str(self.tables_path),
            netcdf_file_action=replace_flag,
            set_verbosity=verbosity,
        )

        # ALWAYS seed CMOR’s internal dataset state from a JSON file,
        # then apply dataset_attrs as overrides.
        p = (
            self.dataset_json_path
            if self.dataset_json_path is not None
            else packaged_dataset_json()
        )
        cmor.dataset_json(str(p))
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
    def _define_axes(self, ds: xr.Dataset, vdef: Any) -> list[int]:
        """Create CMOR axis IDs based on dataset coordinates and optional vdef levels."""
        axes: list[int] = []

        # ---- time axis ----
        if "time" in ds.coords or "time" in ds:
            time = ds["time"]
            t_units = time.attrs.get("units", "days since 1850-01-01")
            cal = time.attrs.get("calendar", time.encoding.get("calendar", "noleap"))

            tvals = _encode_time_to_num(time, t_units, cal)

            # Optional bounds: try common names or CF 'bounds' attribute
            tb = ds.get("time_bnds") or ds.get("time_bounds")
            if tb is None and isinstance(time.attrs.get("bounds"), str):
                bname = time.attrs["bounds"]
                tb = ds.get(bname)

            t_bnds = None
            if tb is not None:
                try:
                    t_bnds = _encode_time_to_num(tb, t_units, cal)
                except ValueError:
                    t_bnds = None

            axes.append(
                cmor.axis(
                    table_entry="time",
                    units=str(t_units),
                    coord_vals=tvals,
                    cell_bounds=t_bnds if t_bnds is not None else None,
                )
            )

        # ---- vertical axis (plev or lev) ----
        levels_info = getattr(vdef, "levels", None)
        if "plev" in ds.coords:
            p = ds["plev"]
            p_units = p.attrs.get("units", "Pa")
            # If vdef specifies an axis entry (e.g., plev19), use that; else infer
            table_entry = None
            if isinstance(levels_info, dict):
                table_entry = levels_info.get("axis_entry") or levels_info.get("name")
            if table_entry is None:
                table_entry = "plev19" if p.size == 19 else "plev"

            p_bnds = ds.get("plev_bnds") or ds.get("plev_bounds")
            axes.append(
                cmor.axis(
                    table_entry=table_entry,
                    units=str(p_units),
                    coord_vals=p.values,
                    cell_bounds=p_bnds.values if p_bnds is not None else None,
                )
            )
        elif "lev" in ds.coords:
            lev = ds["lev"]
            table_entry = "alev"
            if isinstance(levels_info, dict):
                table_entry = levels_info.get("axis_entry", table_entry)
            axes.append(
                cmor.axis(table_entry=table_entry, units="1", coord_vals=lev.values)
            )

        # ---- horizontal axes ----
        if "lat" not in ds.coords or "lon" not in ds.coords:
            raise ValueError("Dataset missing required coordinates 'lat' and 'lon'.")

        lat = ds["lat"]
        lon = ds["lon"]
        lat_b = ds.get("lat_bnds") or ds.get("lat_bounds")
        lon_b = ds.get("lon_bnds") or ds.get("lon_bounds")

        lat_vals = np.asarray(lat.values, dtype="f8").reshape(-1)
        lon_vals = np.asarray(lon.values, dtype="f8").reshape(-1)

        # synthesize bounds if missing
        lat_b_vals = (
            np.asarray(lat_b.values, dtype="f8")
            if lat_b is not None
            else _bounds_from_centers_1d(lat_vals, "lat")
        )
        lon_b_vals = (
            np.asarray(lon_b.values, dtype="f8")
            if lon_b is not None
            else _bounds_from_centers_1d(lon_vals, "lon")
        )

        axes.append(
            cmor.axis(
                table_entry="latitude",
                units="degrees_north",
                coord_vals=lat_vals,
                cell_bounds=lat_b_vals,
            )
        )
        axes.append(
            cmor.axis(
                table_entry="longitude",
                units="degrees_east",
                coord_vals=lon_vals,
                cell_bounds=lon_b_vals,
            )
        )

        return axes

    # public API
    # -------------------------
    def write_variable(
        self, ds: xr.Dataset, varname: str, vdef: Any, outdir: Path
    ) -> None:
        """Write one variable from `ds` to a CMOR-compliant NetCDF file."""
        # Pick CMOR table: prefer vdef.table, else vdef.realm (default Amon)
        table_key = (
            getattr(vdef, "table", None) or getattr(vdef, "realm", None) or "Amon"
        )
        table_filename = _resolve_table_filename(self.tables_path, table_key)
        cmor.load_table(table_filename)

        axes_ids = self._define_axes(ds, vdef)

        units = getattr(vdef, "units", "") or ""
        var_id = cmor.variable(
            getattr(vdef, "name", varname),
            units,
            axes_ids,
            positive=getattr(vdef, "positive", None),
        )

        # Optional variable attributes (e.g., cell_methods, long_name, standard_name)
        if getattr(vdef, "cell_methods", None):
            cmor.set_variable_attribute(var_id, "cell_methods", vdef.cell_methods)
        if getattr(vdef, "long_name", None):
            cmor.set_variable_attribute(var_id, "long_name", vdef.long_name)
        if getattr(vdef, "standard_name", None):
            cmor.set_variable_attribute(var_id, "standard_name", vdef.standard_name)

        data = ds[varname]
        if "time" in data.dims:
            data = data.transpose("time", ...)

        data_np = np.asarray(data)
        # CMOR expects a NumPy array; this will materialize data as needed.

        try:
            # type: ignore[attr-defined]
            print("DEBUG tracking_id:", cmor.get_cur_dataset_attribute("tracking_id"))
        except Exception:  # pylint: disable=broad-except
            pass

        cmor.write(
            var_id, data_np, ntimes_passed=data_np.shape[0] if "time" in ds.dims else 1
        )

        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        outfile = outdir / f"{getattr(vdef, 'name', varname)}.nc"
        cmor.close(var_id, file_name=str(outfile))
