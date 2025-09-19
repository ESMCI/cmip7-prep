"""Thin CMOR wrapper used by cmip7_prep.

This module centralizes CMOR session setup and writing so that the rest of the
pipeline can stay xarray-first. It supports either a dataset JSON file (preferred)
or directly injected global attributes, and creates axes based on the coordinates
present in the provided dataset. It also supports a packaged default
`cmor_dataset.json` living under `cmip7_prep/data/`.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import xarray as xr
from xarray.coding.times import encode_cf_datetime

import cmor  # type: ignore


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
    res = files("cmip7_prep").joinpath(f"data/{filename}")
    return as_file(res)


# ---------------------------------------------------------------------
# Time encoding
# ---------------------------------------------------------------------
def _encode_time_to_num(time_da: xr.DataArray, units: str, calendar: str) -> np.ndarray:
    """Return numeric CF time values (float64) acceptable to CMOR.

    Tries xarray's encoder first; if that fails and cftime is available,
    falls back to cftime.date2num. Raises a ValueError with details otherwise.
    """
    # 1) xarray encoder (handles numpy datetime64 and cftime objects if cftime present)
    try:
        out = encode_cf_datetime(time_da.values, units=units, calendar=calendar)
        return np.asarray(out, dtype="f8")
    except (ValueError, TypeError) as exc_xr:
        last_err = exc_xr

    # 2) Optional cftime path (lazy import to keep lint/typecheck happy)
    try:
        import cftime  # type: ignore # pylint: disable=import-outside-toplevel

        seq = time_da.values.tolist()  # handles object arrays / cftime arrays
        out = cftime.date2num(seq, units=units, calendar=calendar)
        return np.asarray(out, dtype="f8")
    except Exception as exc_cf:  # noqa: BLE001 - we surface both causes together
        raise ValueError(
            f"Could not encode time to numeric CF values with units={units!r}, "
            f"calendar={calendar!r}. xarray error: {last_err}; cftime error: {exc_cf}"
        ) from exc_cf


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
        tables_path: str | Path,
        dataset_json: str | Path | None = None,
        dataset_attrs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.tables_path = str(tables_path)
        self.dataset_json_path = str(dataset_json) if dataset_json else None
        self.dataset_attrs = dataset_attrs or {}

    def __enter__(self) -> "CmorSession":
        """Initialize CMOR and register dataset metadata."""
        cmor.setup(inpath=self.tables_path, netcdf_file_action=cmor.CMOR_REPLACE_3)

        if self.dataset_json_path:
            cmor.dataset_json(self.dataset_json_path)
        elif self.dataset_attrs:
            for key, value in self.dataset_attrs.items():
                cmor.setGblAttr(key, value)
        else:
            # Fallback to packaged cmor_dataset.json if available
            with packaged_dataset_json() as p:
                cmor.dataset_json(str(p))

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
        if "lat" in ds.coords and "lon" in ds.coords:
            lat = ds["lat"]
            lon = ds["lon"]
            lat_b = ds.get("lat_bnds") or ds.get("lat_bounds")
            lon_b = ds.get("lon_bnds") or ds.get("lon_bounds")

            axes.append(
                cmor.axis(
                    table_entry="lat",
                    units="degrees_north",
                    coord_vals=lat.values,
                    cell_bounds=lat_b.values if lat_b is not None else None,
                )
            )
            axes.append(
                cmor.axis(
                    table_entry="lon",
                    units="degrees_east",
                    coord_vals=lon.values,
                    cell_bounds=lon_b.values if lon_b is not None else None,
                )
            )

        return axes

    # -------------------------
    # public API
    # -------------------------
    def write_variable(
        self, ds: xr.Dataset, varname: str, vdef: Any, outdir: Path
    ) -> None:
        """Write one variable from `ds` to a CMOR-compliant NetCDF file."""
        # Pick CMOR table: prefer vdef.table, else vdef.realm (default Amon)
        table_key = getattr(vdef, "table", None) or getattr(vdef, "realm", "Amon")
        table_key = str(table_key)
        candidate7 = Path(self.tables_path) / f"CMIP7_{table_key}.json"
        candidate6 = Path(self.tables_path) / f"CMIP6_{table_key}.json"

        if candidate7.exists():
            cmor.load_table(str(candidate7))
        elif candidate6.exists():
            cmor.load_table(str(candidate6))
        else:
            # Let CMOR search inpath; will raise if not found
            cmor.load_table(f"CMIP7_{table_key}.json")

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

        # CMOR expects a NumPy array; this will materialize data as needed.
        cmor.write(var_id, np.asarray(data), ntimes_passed=data.sizes.get("time", 1))

        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        outfile = outdir / f"{getattr(vdef, 'name', varname)}.nc"
        cmor.close(var_id, file_name=str(outfile))
