"""Basic tests for cmor_writer.py using FakeCMOR."""

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
        log_dir=log_dir,
        log_name=log_name,
    ) as session:
        session.write_variable(ds, CMIPVar(), VDef())
