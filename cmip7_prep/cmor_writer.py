"""Thin CMOR wrapper used by cmip7_prep.

This module centralizes CMOR session setup and writing so that the rest of the
pipeline can stay xarray-first. It supports either a dataset JSON file (preferred)
or directly injected global attributes, and creates axes based on the coordinates
present in the provided dataset.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Dict, Optional

import cmor  # type: ignore
import numpy as np
import xarray as xr


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
            raise ValueError(
                "CmorSession requires either dataset_json path or dataset_attrs."
            )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Finalize CMOR, closing any open handles."""
        cmor.close()

    # -------------------------
    # internal helpers
    # -------------------------

    def _define_axes(self, ds: xr.Dataset, vdef: Any) -> list[int]:
        """Create CMOR axis IDs based on dataset coordinates and optional vdef levels.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset containing coordinates like time, lat, lon, and optionally plev/lev.
        vdef : Any
            Variable definition object. If it has a ``levels`` mapping (e.g.,
            ``{'axis_entry': 'plev19', 'name': 'plev', 'units': 'Pa'}``), that will
            be used to define the vertical axis.

        Returns
        -------
        list[int]
            List of CMOR axis IDs in the order they should be used for the variable.
        """
        axes: list[int] = []

        # Time axis
        if "time" in ds.coords or "time" in ds:
            time = ds["time"]
            t_units = time.attrs.get("units", "days since 1850-01-01")
            cal = time.attrs.get("calendar", "standard")
            if np.issubdtype(time.dtype, np.datetime64):
                # convert to numeric time using CF units if needed
                tvals = xr.conventions.times.encode_cf_datetime(
                    time.values, t_units, calendar=cal
                )
            else:
                tvals = time.values
            axes.append(
                cmor.axis(table_entry="time", units=str(t_units), coord_vals=tvals)
            )

        # Vertical axis (pressure or model levels)
        levels_info = getattr(vdef, "levels", None)
        # Prefer an explicit plev axis in the data
        if "plev" in ds.coords:
            p = ds["plev"]
            p_units = p.attrs.get("units", "Pa")
            # If vdef specifies an axis entry (e.g., plev19), use that; else let CMOR infer
            table_entry = None
            if isinstance(levels_info, dict):
                table_entry = levels_info.get("axis_entry") or levels_info.get("name")
            if table_entry is None:
                # Heuristic: choose plev19 if there are 19 levels, else generic plev
                table_entry = "plev19" if p.size == 19 else "plev"
            p_bnds = ds.get("plev_bnds")
            axes.append(
                cmor.axis(
                    table_entry=table_entry,
                    units=str(p_units),
                    coord_vals=p.values,
                    cell_bounds=p_bnds.values if p_bnds is not None else None,
                )
            )
        elif "lev" in ds.coords:
            # Generic hybrid "lev" axis; rely on table entry provided via vdef or default to "alev"
            lev = ds["lev"]
            table_entry = "alev"
            if isinstance(levels_info, dict):
                table_entry = levels_info.get("axis_entry", table_entry)
            axes.append(
                cmor.axis(table_entry=table_entry, units="1", coord_vals=lev.values)
            )

        # Latitude / Longitude
        if "lat" in ds.coords and "lon" in ds.coords:
            lat = ds["lat"]
            lon = ds["lon"]
            lat_b = ds.get("lat_bnds")
            lon_b = ds.get("lon_bnds")
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
        """Write one variable from `ds` to a CMOR-compliant NetCDF file.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset containing the variable and its coordinates.
        varname : str
            Name of the variable in `ds` to CMORize.
        vdef : Any
            An object with fields: ``name``, ``realm``,
            optional ``units``, ``positive``, and optional
            ``levels`` dict (see :meth:`_define_axes`).
            This is typically a light-weight holder.
        outdir : Path
            Output directory for the CMORized NetCDF file.
        """
        # Load the appropriate CMOR table without relying on exceptions.
        realm = getattr(vdef, "realm", "Amon")
        candidate7 = Path(self.tables_path) / f"CMIP7_{realm}.json"
        candidate6 = Path(self.tables_path) / f"CMIP6_{realm}.json"
        if candidate7.exists():
            cmor.load_table(str(candidate7))
        elif candidate6.exists():
            cmor.load_table(str(candidate6))
        else:
            # Fall back to passing the bare table name; CMOR will search its inpath.
            # This branch avoids broad exception handling while still being flexible.
            cmor.load_table(f"CMIP7_{realm}.json")

        axes_ids = self._define_axes(ds, vdef)
        units = getattr(vdef, "units", "")
        var_id = cmor.variable(
            getattr(vdef, "name", varname),
            units,
            axes_ids,
            positive=getattr(vdef, "positive", None),
        )

        # Optional variable attributes (e.g., cell_methods)
        if getattr(vdef, "cell_methods", None):
            cmor.set_variable_attribute(var_id, "cell_methods", vdef.cell_methods)

        # Ensure time is the leading dimension if present
        data = ds[varname]
        if "time" in data.dims:
            data = data.transpose("time", ...)

        cmor.write(var_id, data.values, ntimes_passed=data.sizes.get("time", 1))

        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        outfile = outdir / f"{getattr(vdef, 'name', varname)}.nc"
        cmor.close(var_id, file_name=str(outfile))
