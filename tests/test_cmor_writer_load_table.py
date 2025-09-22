"""tests/test_cmor_writer_load_table.py"""

import types
import numpy as np
import xarray as xr

from cmip7_prep import cmor_writer as cw


def test_write_variable_loads_basename_table_and_defines_axes(fake_cmor, tmp_path):
    """test load basename table and define axes"""
    fake, tables_path = fake_cmor

    # Build minimal numeric-time dataset to avoid cftime dependence
    lat = xr.DataArray(np.array([-89.5, 89.5]), dims=("lat",))
    lon = xr.DataArray(np.array([0.5, 1.5, 2.5]), dims=("lon",))
    time = xr.DataArray(np.array([0.5], dtype="f8"), dims=("time",))
    tb = xr.DataArray(np.array([[0.0, 1.0]], dtype="f8"), dims=("time", "nbnd"))

    tas = xr.DataArray(
        np.ones((1, 2, 3), dtype="f4"),
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lat, "lon": lon},
        name="tas",
    )
    ds = xr.Dataset({"tas": tas, "time_bounds": tb})
    ds["time"].attrs["bounds"] = "time_bounds"

    outdir = tmp_path / "out"
    outdir.mkdir()

    # minimal vdef with table name
    vdef = types.SimpleNamespace(name="tas", table="Amon", units="K")

    # pylint: disable=using-constant-test
    with (
        cw.CmurSession
        if False
        else cw.CmorSession(
            tables_path=tables_path, dataset_attrs={"institution_id": "NCAR"}
        )
    ) as cm:  # noqa: E701
        cm.write_variable(ds, "tas", vdef, outdir=outdir)

    # Table basename should be used (resolved by inpath)
    assert fake.last_table == "CMIP6_Amon.json"

    # Axis calls: expect time, latitude, longitude in some order; verify entries exist
    entries = [a[0] for a in fake.axis_calls]
    assert "time" in entries
    assert "latitude" in entries
    assert "longitude" in entries

    # Variable and write were called
    assert fake.variable_calls and fake.write_calls
