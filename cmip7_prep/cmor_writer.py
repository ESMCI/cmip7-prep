# cmip7_prep/cmor_writer.py
import cmor, numpy as np, xarray as xr
from contextlib import AbstractContextManager
from pathlib import Path

class CmorSession(AbstractContextManager):
    def __init__(self, tables_path, dataset_attrs):
        self.tables_path = str(tables_path)
        self.dataset_attrs = dataset_attrs

    def __enter__(self):
        cmor.setup(inpath=self.tables_path, netcdf_file_action=cmor.CMOR_REPLACE_3)
        cmor.dataset_json("cmor_dataset.json")  # or pass dict via cmor.setGblAttr
        cmor.load_table("CMIP7_Amon.json")      # swap per realm
        return self

    def __exit__(self, exc_type, exc, tb):
        cmor.close()

    def _define_axes(self, ds: xr.Dataset, vdef):
        axes = []
        # time
        axes.append(cmor.axis(table_entry="time", units=str(ds["time"].attrs.get("units","days since 1850-01-01")),
                              coord_vals=ds["time"].values.astype("float64")))
        # lat/lon (1° target) OR curvilinear (for OCN)
        if "lat" in ds.dims and "lon" in ds.dims:
            axes.append(cmor.axis(table_entry="lat", units="degrees_north",
                                  coord_vals=ds["lat"].values,
                                  cell_bounds=ds["lat_bnds"].values if "lat_bnds" in ds else None))
            axes.append(cmor.axis(table_entry="lon", units="degrees_east",
                                  coord_vals=ds["lon"].values,
                                  cell_bounds=ds["lon_bnds"].values if "lon_bnds" in ds else None))
        # vertical axes as needed (plev, lev)
        # ...
        return axes

    def write_variable(self, ds: xr.Dataset, varname: str, vdef, outdir: Path):
        # choose correct table per realm before defining axes
        cmor.load_table(f"CMIP7_{vdef.realm}.json")  # or CMIP6 fallback

        axes_ids = self._define_axes(ds, vdef)
        var_id = cmor.variable(vdef.name, vdef.units, axes_ids,
                               positive=vdef.positive if vdef.positive else None)
        # cell_methods if provided
        if vdef.cell_methods:
            cmor.set_variable_attribute(var_id, "cell_methods", vdef.cell_methods)

        data = ds[varname].transpose(...).values  # time-major
        cmor.write(var_id, data, ntimes_passed=data.shape[0])

        # write fx on target grid if applicable (once per run is enough)
        if vdef.realm in ("Amon","Lmon"):
            self._write_fx(ds)

        cmor.close(var_id, file_name=str(outdir / f"{vdef.name}.nc"))

    def _write_fx(self, ds):
        # Example: areacella, sftlf on the 1° grid
        if {"lat","lon"}.issubset(ds.dims):
            cmor.load_table("CMIP7_fx.json")
            lat = ds["lat"].values; lon = ds["lon"].values
            ax_t = cmor.axis(table_entry="time", units="days since 1850-01-01", coord_vals=np.array([0.0]))
            ax_la = cmor.axis(table_entry="lat", units="degrees_north", coord_vals=lat)
            ax_lo = cmor.axis(table_entry="lon", units="degrees_east", coord_vals=lon)
            # area
            if "cell_area" in ds:
                v = cmor.variable("areacella", "m2", [ax_la, ax_lo])
                cmor.write(v, ds["cell_area"].values[...])
                cmor.close(v)
            # land fraction
            if "sftlf" in ds:
                v = cmor.variable("sftlf", "%", [ax_la, ax_lo])
                cmor.write(v, ds["sftlf"].values[...])
                cmor.close(v)

