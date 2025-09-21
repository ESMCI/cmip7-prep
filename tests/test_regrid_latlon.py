"""Unit tests for regridding: lat/lon naming/order and time-bounds propagation."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import xarray as xr
import pytest  # type: ignore

from cmip7_prep import regrid


class _FakeRegridder:
    """Return data with ('time','y','x') or ('time','x','y') dims to simulate xESMF."""

    # pylint: disable=too-few-public-methods
    def __init__(self, order: str = "yx"):
        assert order in ("yx", "xy")
        self.order = order

    def __call__(self, da: xr.DataArray, **kwargs) -> xr.DataArray:
        time_len = int(da.sizes.get("time", 1))
        if self.order == "yx":
            data = np.zeros((time_len, 180, 360), dtype=np.float32)
            return xr.DataArray(data, dims=("time", "y", "x"))
        data = np.zeros((time_len, 360, 180), dtype=np.float32)  # "xy"
        return xr.DataArray(data, dims=("time", "x", "y"))


@pytest.mark.parametrize("order", ["yx", "xy"])
def test_lat_lon_named_and_sized_correctly(monkeypatch, order):
    """regrid_to_1deg returns lat(180) in [-89.5,89.5] and lon(360) in [0.5,359.5]."""

    # Fake cache → our regridder
    # pylint: disable=protected-access
    monkeypatch.setattr(
        regrid._RegridderCache,
        "get",
        staticmethod(lambda path, method: _FakeRegridder(order)),
    )

    # Force canonical 1° lat/lon from the map
    lat1d = np.linspace(-89.5, 89.5, 180, dtype="f8")
    lon1d = np.arange(360, dtype="f8") + 0.5
    monkeypatch.setattr(regrid, "_dst_latlon_1d_from_map", lambda path: (lat1d, lon1d))

    # Minimal unstructured input dataset: (time, ncol)
    time_len, ncol = 2, 10
    ds_in = xr.Dataset(
        {"tas": (("time", "ncol"), np.ones((time_len, ncol), dtype=np.float32))},
        coords={"time": np.arange(time_len)},
    )

    out = regrid.regrid_to_1deg(
        ds_in,
        "tas",
        method="conservative",
        conservative_map=Path("dummy.nc"),  # not opened due to monkeypatch
        output_time_chunk=1,
        dtype="float32",
    )

    # Dims & coordinate ranges
    assert list(out.dims)[-2:] == ["lat", "lon"]
    assert out.sizes["lat"] == 180 and out.sizes["lon"] == 360
    assert np.isclose(float(out["lat"].min()), -89.5) and np.isclose(
        float(out["lat"].max()), 89.5
    )
    assert np.isclose(float(out["lon"].min()), 0.5) and np.isclose(
        float(out["lon"].max()), 359.5
    )


def test_regrid_to_1deg_ds_carries_time_bounds(monkeypatch):
    """regrid_to_1deg_ds propagates time and existing bounds from the source dataset."""
    # pylint: disable=protected-access
    monkeypatch.setattr(
        regrid._RegridderCache,
        "get",
        staticmethod(lambda path, method: _FakeRegridder("yx")),
    )
    lat1d = np.linspace(-89.5, 89.5, 180, dtype="f8")
    lon1d = np.arange(360, dtype="f8") + 0.5
    monkeypatch.setattr(regrid, "_dst_latlon_1d_from_map", lambda path: (lat1d, lon1d))

    # Source dataset with numeric time & bounds present
    time_len, ncol = 3, 5
    time_vals = np.arange(time_len, dtype="f8")
    tb = np.column_stack([time_vals, time_vals + 1.0]).astype("f8")  # (time, nbnd)
    ds_src = xr.Dataset(
        {
            "tas": (("time", "ncol"), np.ones((time_len, ncol), dtype=np.float32)),
            "time_bounds": (("time", "nbnd"), tb),
        },
        coords={"time": time_vals},
    )
    ds_src["time"].attrs.update(
        units="days since 2000-01-01", calendar="noleap", bounds="time_bounds"
    )

    ds_tmp = xr.Dataset({"tas": ds_src["tas"]})

    ds_out = regrid.regrid_to_1deg_ds(
        ds_tmp,
        "tas",
        time_from=ds_src,
        method="conservative",
        conservative_map=Path("dummy.nc"),
        output_time_chunk=1,
        dtype="float32",
    )

    # Assert the regridded dataset carries time & bounds
    assert "time" in ds_out.coords and "time_bounds" in ds_out
    assert ds_out["time"].attrs.get("bounds") == "time_bounds"
    assert ds_out["time_bounds"].dims == ("time", "nbnd")
    np.testing.assert_allclose(ds_out["time"].values, time_vals)
    np.testing.assert_allclose(ds_out["time_bounds"].values, tb)


def test_attrs_propagated(monkeypatch):
    """Attributes on the input DataArray are copied to the regridded output."""
    # fake regridder + lat/lon
    # pylint: disable=protected-access
    monkeypatch.setattr(
        regrid._RegridderCache, "get", staticmethod(lambda p, m: _FakeRegridder("yx"))
    )
    lat1d = np.linspace(-89.5, 89.5, 180)
    lon1d = np.arange(360, dtype="f8") + 0.5
    monkeypatch.setattr(regrid, "_dst_latlon_1d_from_map", lambda p: (lat1d, lon1d))

    ds_in = xr.Dataset(
        {"tas": (("time", "ncol"), np.ones((1, 3), np.float32))}, coords={"time": [0]}
    )
    ds_in["tas"].attrs.update(units="K", long_name="Near-surface air temperature")

    out = regrid.regrid_to_1deg(
        ds_in, "tas", conservative_map=Path("dummy.nc"), output_time_chunk=1
    )
    assert out.attrs.get("units") == "K"
    assert out.attrs.get("long_name") == "Near-surface air temperature"
