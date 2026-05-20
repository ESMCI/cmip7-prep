"""Basic tests for cmor_writer.py using FakeCMOR."""

import json
from pathlib import Path
import numpy as np
import xarray as xr

from cmip7_prep.cmor_writer import CmorSession


def test_cmor_session_basic(tmp_path):
    """Test CmorSession context manager and basic variable writing using real CMOR."""
    tables_root = Path(__file__).parent.parent / "cmip7-cmor-tables"
    tables_root.mkdir(parents=True, exist_ok=True)
    # Create a simple dataset with lat/lon/time
    lat = np.linspace(-90, 90, 4)
    lon = np.linspace(0, 360, 8, endpoint=False)
    # Two months: January and February 2000
    time = np.array([15, 45])  # Midpoints (days since 2000-01-01)
    time_bnds = np.array([[0, 30], [30, 60]])  # Bounds for each month
    data = np.random.rand(2, 4, 8)
    ds = xr.Dataset(
        {
            "tas_tmin-h2m-hxy-u": (("time", "lat", "lon"), data),
            "lat": ("lat", lat),
            "lon": ("lon", lon),
            "time": ("time", time),
            "time_bnds": (("time", "bnds"), time_bnds),
        }
    )
    ds["lat"].attrs["units"] = "degrees_north"
    ds["lon"].attrs["units"] = "degrees_east"
    ds["time"].attrs["units"] = "days since 2000-01-01"
    ds["time"].attrs["calendar"] = "noleap"
    ds["time"].attrs["bounds"] = "time_bnds"
    ds.attrs["branded_variable"] = "tas_tmin-h2m-hxy-u"

    # pylint: disable=too-few-public-methods
    class VDef:
        """Minimal variable definition for testing."""

        name = "tas"
        branded_variable_name = "tas_tmin-h2m-hxy-u"
        units = "K"
        table = "atmos"
        levels = {}

    # pylint: disable=too-few-public-methods
    class CMIPVar:
        """Minimal CMIP variable wrapper for testing."""

        class BrandedName:
            """Branded variable name wrapper for testing."""

            name = "tas_tmin-h2m-hxy-u"

        branded_variable_name = BrandedName()

    dataset_json_path = Path(__file__).parent.parent / "data" / "cmor_dataset.json"
    log_dir = tmp_path
    log_name = "cmor_test.log"
    with CmorSession(
        tables_root=tables_root,
        dataset_json=dataset_json_path,
        dataset_attrs={
            "institution_id": "NCAR",
            "GLOBAL_IS_CMIP7": True,
            "branded_variable": {
                "variable_id": "tas",
                "table_id": "atmos",
                "plev": 50000,
                "description": "Vertical velocity at 500 hPa",
            },
        },
        log_dir=log_dir,
        log_name=log_name,
    ) as session:
        session.write_variable(ds, CMIPVar(), VDef())


def test_cmor_session_multiple_timeseries(tmp_path):
    """Test CmorSession with multiple timeseries files (datasets)."""
    tables_root = Path(__file__).parent.parent / "cmip7-cmor-tables"
    tables_root.mkdir(parents=True, exist_ok=True)
    # Create two timeseries datasets for two months
    lat = np.linspace(-90, 90, 4)
    lon = np.linspace(0, 360, 8, endpoint=False)
    time1 = np.array([15])  # January
    time2 = np.array([45])  # February
    time_bnds1 = np.array([[0, 30]])
    time_bnds2 = np.array([[30, 60]])
    data1 = np.random.rand(1, 4, 8)
    data2 = np.random.rand(1, 4, 8)
    ds1 = xr.Dataset(
        {
            "tas_tmin-h2m-hxy-u": (("time", "lat", "lon"), data1),
            "lat": ("lat", lat),
            "lon": ("lon", lon),
            "time": ("time", time1),
            "time_bnds": (("time", "bnds"), time_bnds1),
        }
    )
    ds2 = xr.Dataset(
        {
            "tas_tmin-h2m-hxy-u": (("time", "lat", "lon"), data2),
            "lat": ("lat", lat),
            "lon": ("lon", lon),
            "time": ("time", time2),
            "time_bnds": (("time", "bnds"), time_bnds2),
        }
    )
    # Concatenate along time
    ds = xr.concat([ds1, ds2], dim="time")
    ds["lat"].attrs["units"] = "degrees_north"
    ds["lon"].attrs["units"] = "degrees_east"
    ds["time"].attrs["units"] = "days since 2000-01-01"
    ds["time"].attrs["calendar"] = "noleap"
    ds["time"].attrs["bounds"] = "time_bnds"
    ds.attrs["branded_variable"] = "tas_tmin-h2m-hxy-u"

    # pylint: disable=too-few-public-methods
    class VDef:
        """Minimal variable definition for testing multiple timeseries."""

        name = "tas"
        branded_variable_name = "tas_tmin-h2m-hxy-u"
        units = "K"
        table = "atmos"
        levels = {}

    # pylint: disable=too-few-public-methods
    class CMIPVar:
        """Minimal CMIP variable wrapper for testing multiple timeseries."""

        class BrandedName:
            """Branded variable name wrapper for testing multiple timeseries."""

            name = "tas_tmin-h2m-hxy-u"

        branded_variable_name = BrandedName()

    dataset_json_path = Path(__file__).parent.parent / "data" / "cmor_dataset.json"
    log_dir = tmp_path
    log_name = "cmor_test_multi.log"
    with CmorSession(
        tables_root=tables_root,
        dataset_json=dataset_json_path,
        dataset_attrs={
            "institution_id": "NCAR",
            "GLOBAL_IS_CMIP7": True,
            "branded_variable": {
                "variable_id": "tas",
                "table_id": "atmos",
                "plev": 50000,
                "description": "Vertical velocity at 500 hPa",
            },
        },
        log_dir=log_dir,
        log_name=log_name,
    ) as session:
        session.write_variable(ds, CMIPVar(), VDef())


def test_cmor_session_zonal_mean_plev39(tmp_path):
    """Test CmorSession with a zonal-mean variable (time, plev, lat) — no lon dimension."""
    tables_root = Path(__file__).parent.parent / "cmip7-cmor-tables"
    tables_root.mkdir(parents=True, exist_ok=True)

    coord_json = tables_root / "tables" / "CMIP7_coordinate.json"
    with open(coord_json, encoding="utf-8") as f:
        coord_data = json.load(f)
    plev39_vals = np.array(coord_data["axis_entry"]["plev39"]["requested"], dtype="f8")

    lat = np.linspace(-90, 90, 8)
    time = np.array([15, 45])
    time_bnds = np.array([[0, 30], [30, 60]])
    data = np.random.rand(2, 39, 8)

    ds = xr.Dataset(
        {
            "ta_tavg-p39-hy-air": (("time", "plev", "lat"), data),
            "lat": ("lat", lat),
            "plev": ("plev", plev39_vals),
            "time": ("time", time),
            "time_bnds": (("time", "bnds"), time_bnds),
        }
    )
    ds["lat"].attrs["units"] = "degrees_north"
    ds["plev"].attrs.update(
        {"units": "Pa", "standard_name": "air_pressure", "positive": "down"}
    )
    ds["time"].attrs["units"] = "days since 2000-01-01"
    ds["time"].attrs["calendar"] = "noleap"
    ds["time"].attrs["bounds"] = "time_bnds"

    # pylint: disable=too-few-public-methods
    class VDef:
        """Minimal variable definition for zonal-mean plev39 testing."""

        name = "ta"
        branded_variable_name = "ta_tavg-p39-hy-air"
        units = "K"
        table = "atmos"
        levels = {"name": "plev39", "units": "Pa"}

    # pylint: disable=too-few-public-methods
    class CMIPVar:
        """Minimal CMIP variable wrapper for zonal-mean plev39 testing."""

        # pylint: disable=too-few-public-methods
        class BrandedName:
            """Branded variable name wrapper."""

            name = "ta_tavg-p39-hy-air"

        branded_variable_name = BrandedName()

    dataset_json_path = Path(__file__).parent.parent / "data" / "cmor_dataset.json"
    with CmorSession(
        tables_root=tables_root,
        dataset_json=dataset_json_path,
        dataset_attrs={
            "institution_id": "NCAR",
            "GLOBAL_IS_CMIP7": True,
            "branded_variable": {
                "variable_id": "ta",
                "table_id": "atmos",
                "description": "Air Temperature",
            },
        },
        log_dir=tmp_path,
        log_name="cmor_test_zonal_mean.log",
    ) as session:
        session.write_variable(ds, CMIPVar(), VDef())
