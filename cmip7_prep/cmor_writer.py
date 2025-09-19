
# cmip7_prep/cmor_writer.py
from __future__ import annotations
import cmor
import numpy as np
import xarray as xr
from contextlib import AbstractContextManager
from pathlib import Path

class CmorSession(AbstractContextManager):
    """Thin wrapper around CMOR that supports a dataset_json path."""
    def __init__(self, tables_path: str | Path, dataset_json: str | Path | None = None, dataset_attrs: dict | None = None):
        self.tables_path = str(tables_path)
        self.dataset_json_path = str(dataset_json) if dataset_json else None
        self.dataset_attrs = dataset_attrs or {}

    def __enter__(self):
        cmor.setup(inpath=self.tables_path, netcdf_file_action=cmor.CMOR_REPLACE_3)
        if self.dataset_json_path:
            cmor.dataset_json(self.dataset_json_path)
        elif self.dataset_attrs:
            # Fallback: allow direct attribute injection if a JSON is not provided
            for k, v in self.dataset_attrs.items():
                cmor.setGblAttr(k, v)
        else:
            # Be explicit so users notice misconfiguration
            raise ValueError("CmorSession requires either dataset_json path or dataset_attrs.")
        return self

    def __exit__(self, exc_type, exc, tb):
        cmor.close()

    def _define_axes(self, ds: xr.Dataset, vdef):
        axes = []
        # time axis (expects proper CF units in ds['time'].attrs['units'], else supply default)
        t_units = ds["time"].attrs.get("units", "days since 1850-01-01")
        tvals = xr.conventions.times.encode_cf_datetime(ds["time"].values, t_units, calendar=ds["time"].attrs.get("calendar", "standard")) if np.issubdtype(ds["time"].dtype, np.datetime64) else ds["time"].values
        axes.append(cmor.axis(table_entry="time", units=str(t_units), coord_vals=tvals))

        # latitude / longitude (regular grid)
        if "lat" in ds.coords and "lon" in ds.coords:
            lat = ds["lat"].values
            lon = ds["lon"].values
            lat_b = ds["lat_bnds"].values if "lat_bnds" in ds else None
            lon_b = ds["lon_bnds"].values if "lon_bnds" in ds else None
            axes.append(cmor.axis(table_entry="lat", units="degrees_north", coord_vals=lat, cell_bounds=lat_b))
            axes.append(cmor.axis(table_entry="lon", units="degrees_east", coord_vals=lon, cell_bounds=lon_b))

        # pressure or model level axes would be added here if needed based on vdef
        return axes

    def write_variable(self, ds: xr.Dataset, varname: str, vdef, outdir: Path):
        # load proper table per realm (Amon, Lmon, etc.)
        tbl = f"CMIP7_{vdef.realm}.json"
        try:
            cmor.load_table(tbl)
        except Exception:
            # Fallback to CMIP6-style table names if CMIP7 tables aren't available yet
            cmor.load_table(f"CMIP6_{vdef.realm}.json")

        axes_ids = self._define_axes(ds, vdef)
        var_id = cmor.variable(vdef.name, getattr(vdef, "units", ""), axes_ids,
                               positive=getattr(vdef, "positive", None))

        # cell_methods as variable attribute
        if getattr(vdef, "cell_methods", None):
            cmor.set_variable_attribute(var_id, "cell_methods", vdef.cell_methods)

        data = ds[varname].values
        # make sure time is leading dimension if present
        if "time" in ds[varname].dims:
            data = ds[varname].transpose("time", ...).values
        cmor.write(var_id, data, ntimes_passed=data.shape[0] if data.ndim >= 1 else 1)

        # Close and write file
        outfile = Path(outdir) / f"{vdef.name}.nc"
        cmor.close(var_id, file_name=str(outfile))
