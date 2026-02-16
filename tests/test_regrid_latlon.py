"""Unit tests for lat/lon regridding and map selection."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import xarray as xr
import pytest  # type: ignore

from cmip7_prep import regrid


class _FakeRegridder:
    """Return data with lat/lon dims and coords to simulate xESMF output."""

    # pylint: disable=too-few-public-methods
    def __init__(self, nlat: int = 4, nlon: int = 8):
        self.nlat = nlat
        self.nlon = nlon

    def __call__(self, da: xr.DataArray, **kwargs) -> xr.DataArray:
        non_spatial = [d for d in da.dims if d not in ("lat", "lon")]
        shape = tuple(da.sizes[d] for d in non_spatial) + (self.nlat, self.nlon)
        data = np.zeros(shape, dtype=np.float32)
        lat = np.linspace(-90.0 + 90.0 / self.nlat, 90.0 - 90.0 / self.nlat, self.nlat)
        lon = np.linspace(0.0, 360.0 - 360.0 / self.nlon, self.nlon)
        coords = {"lat": lat, "lon": lon}
        for d in non_spatial:
            coords[d] = da.coords[d] if d in da.coords else np.arange(da.sizes[d])
        return xr.DataArray(
            data, dims=tuple(non_spatial) + ("lat", "lon"), coords=coords
        )


def _fake_weights(nlat: int = 4, nlon: int = 8) -> xr.Dataset:
    """Create a minimal map weights dataset used by area/bounds code."""
    lon_edges = np.linspace(0.0, 360.0, nlon + 1, dtype="f8")
    lat_edges = np.linspace(-90.0, 90.0, nlat + 1, dtype="f8")
    xv_b = np.zeros((nlon + 1, 1), dtype="f8")
    xv_b[:, 0] = lon_edges
    yv_b = np.zeros((nlat * nlon, 1), dtype="f8")
    yv_b[np.arange(nlat) * nlon, 0] = lat_edges[:-1]
    return xr.Dataset(
        {
            "dst_grid_dims": xr.DataArray(
                np.array([nlon, nlat], dtype=np.int32), dims=("grid_rank",)
            ),
            "xv_b": xr.DataArray(xv_b, dims=("xv_size", "nv")),
            "yv_b": xr.DataArray(yv_b, dims=("yv_size", "nv")),
        }
    )


@pytest.mark.parametrize(
    ("model", "resolution"),
    [("cesm", "ne30"), ("noresm", "ne16")],
)
def test_lat_lon_named_and_sized_correctly(monkeypatch, model, resolution):
    """regrid_to_latlon returns expected lat/lon dimensions and coordinates."""

    monkeypatch.setattr(
        regrid.RegridderCache,
        "get",
        staticmethod(lambda path, method: _FakeRegridder(4, 8)),
    )
    monkeypatch.setattr(regrid.xr, "open_dataset", lambda _: _fake_weights(4, 8))

    # Minimal unstructured input dataset: (time, ncol)
    time_len, ncol = 2, 10
    ds_in = xr.Dataset(
        {"tas": (("time", "ncol"), np.ones((time_len, ncol), dtype=np.float32))},
        coords={"time": np.arange(time_len)},
    )

    out = regrid.regrid_to_latlon(
        ds_in,
        varname="tas",
        resolution=resolution,
        model=model,
        method="conservative",
        conservative_map=Path("dummy.nc"),
        output_time_chunk=1,
        dtype="float32",
    )

    # Dims & coordinate ranges
    assert list(out.dims)[-2:] == ["lat", "lon"]
    assert out.sizes["lat"] == 4 and out.sizes["lon"] == 8
    assert np.isclose(float(out["lat"].min()), -67.5) and np.isclose(
        float(out["lat"].max()), 67.5
    )
    assert np.isclose(float(out["lon"].min()), 0.0) and np.isclose(
        float(out["lon"].max()), 315.0
    )
    assert "areacella" in out.coords


def test_regrid_to_latlon_ds_carries_time_bounds(monkeypatch):
    """regrid_to_latlon_ds propagates time and existing bounds from source."""
    monkeypatch.setattr(
        regrid.RegridderCache,
        "get",
        staticmethod(lambda path, method: _FakeRegridder(4, 8)),
    )
    monkeypatch.setattr(regrid.xr, "open_dataset", lambda _: _fake_weights(4, 8))
    monkeypatch.setattr(regrid, "_regrid_fx_once", lambda *args, **kwargs: None)

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

    ds_out = regrid.regrid_to_latlon_ds(
        ds_tmp,
        "tas",
        resolution="ne30",
        model="cesm",
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
    monkeypatch.setattr(
        regrid.RegridderCache,
        "get",
        staticmethod(lambda path, method: _FakeRegridder(4, 8)),
    )
    monkeypatch.setattr(regrid.xr, "open_dataset", lambda _: _fake_weights(4, 8))

    ds_in = xr.Dataset(
        {"tas": (("time", "ncol"), np.ones((1, 3), np.float32))}, coords={"time": [0]}
    )
    ds_in["tas"].attrs.update(units="K", long_name="Near-surface air temperature")

    out = regrid.regrid_to_latlon(
        ds_in,
        "tas",
        resolution="ne30",
        model="cesm",
        conservative_map=Path("dummy.nc"),
        output_time_chunk=1,
    )
    assert out.attrs.get("units") == "K"
    assert out.attrs.get("long_name") == "Near-surface air temperature"


def test_pick_maps_noresm_ne16_defaults():
    """noresm/ne16 map defaults and method preference work."""
    cons = regrid._pick_maps(  # pylint: disable=protected-access
        "pr", resolution="ne16", model="noresm", force_method="conservative"
    )
    bilin = regrid._pick_maps(  # pylint: disable=protected-access
        "tas", resolution="ne16", model="noresm"
    )
    assert cons.method_label == "conservative"
    assert cons.path == regrid.DEFAULT_CONS_MAP_NE16_noresm
    assert bilin.method_label == "bilinear"
    assert bilin.path == regrid.DEFAULT_BILIN_MAP_NE16_noresm
