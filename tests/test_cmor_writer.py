"""Basic tests for cmor_writer.py using FakeCMOR."""

from pathlib import Path
import numpy as np
import xarray as xr

from cmip7_prep.cmor_writer import CmorSession


def test_cmor_session_basic(fake_cmor):
    """Test CmorSession context manager and basic variable writing."""
    fake, tables = fake_cmor
    # Create a simple dataset with lat/lon/time
    lat = np.linspace(-90, 90, 4)
    lon = np.linspace(0, 360, 8, endpoint=False)
    time = np.arange(2)
    data = np.random.rand(2, 4, 8)
    ds = xr.Dataset(
        {
            "tas": (("time", "lat", "lon"), data),
            "lat": ("lat", lat),
            "lon": ("lon", lon),
            "time": ("time", time),
        }
    )
    ds["lat"].attrs["units"] = "degrees_north"
    ds["lon"].attrs["units"] = "degrees_east"
    ds["time"].attrs["units"] = "days since 2000-01-01"
    ds["time"].attrs["calendar"] = "noleap"

    # pylint: disable=too-few-public-methods
    class VDef:
        """Minimal variable definition for testing."""

        name = "tas"
        units = "K"
        table = "atmos"
        levels = {}

    # pylint: disable=too-few-public-methods
    class CMIPVar:
        """Minimal CMIP variable wrapper for testing."""

        branded_variable_name = "tas"

    dataset_json_path = Path(__file__).parent.parent / "data" / "cmor_dataset.json"
    with CmorSession(tables_root=tables, dataset_json=dataset_json_path) as session:
        session.write_variable(ds, CMIPVar(), VDef())
    # Check FakeCMOR calls
    assert fake.variable_calls, "No variable calls recorded"
    assert fake.write_calls, "No write calls recorded"
