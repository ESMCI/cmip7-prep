"""Basic tests for cmor_writer.py using FakeCMOR."""

import json
from pathlib import Path
import numpy as np
import pytest
import xarray as xr

from cmip7_prep.cmor_writer import CmorSession
from cmip7_prep.cmor_utils import build_dataset_cfg
from cmip7_prep.mapping_compat import _safe_eval


@pytest.mark.parametrize(
    ("model", "resolution"),
    [("cesm", "ne30"), ("noresm", "NorESM3-LM"), ("noresm", "NorESM3-MM")],
)
def test_cmor_session_basic(tmp_path, model, resolution):
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

    dataset_cfg = build_dataset_cfg(
        model=model,
        resolution=resolution,
        experiment="piControl",
        frequency="mon",
        ripf="r1i1p1f1",
        tables_root=tables_root,
        data_version="v20200101",
    )
    log_dir = tmp_path
    log_name = "cmor_test.log"
    with CmorSession(
        tables_root=tables_root,
        dataset_attrs=dataset_cfg,
        log_dir=log_dir,
        log_name=log_name,
        outdir=tmp_path / "cmor",
    ) as session:
        written_path = session.write_variable(ds, CMIPVar(), VDef())

    with xr.open_dataset(written_path) as written:
        with open(
            tables_root
            / "tables-cvs"
            / "split-view"
            / "required_global_attributes.json",
            encoding="utf-8",
        ) as handle:
            required_attrs = json.load(handle)
        assert set(required_attrs).issubset(written.attrs)
        assert {
            "branch_method",
            "grid",
            "member_id",
            "source_type",
            "sub_experiment_id",
            "sub_experiment",
        }.isdisjoint(written.attrs)
        assert written.attrs["variant_label"] == dataset_cfg["variant_label"]
        assert written.attrs["grid_label"] == dataset_cfg["grid_label"]
        assert written.attrs["source_id"] == dataset_cfg["source_id"]
        if model == "noresm":
            assert written.attrs["contact"] == dataset_cfg["contact"]
            assert written.attrs["EMD"] == dataset_cfg["EMD"]
        else:
            assert {"contact", "EMD"}.isdisjoint(written.attrs)
        np.testing.assert_allclose(
            written["tas"].squeeze(drop=True).values, data, rtol=1e-6
        )


@pytest.mark.parametrize(
    ("source_dim", "source_size", "target_dim", "output_dim", "variable_name"),
    [
        ("fates_levpft", 15, "vegtype", "type", "gppVgt"),
        ("fates_levlanduse", 5, "landuse", "landuse", "gppLut"),
    ],
)
def test_cmor_writes_mapped_vegetation_types(
    tmp_path, source_dim, source_size, target_dim, output_dim, variable_name
):
    """CMOR preserves mapped category labels, grouped values and missing urban."""
    source = xr.DataArray(
        np.broadcast_to(
            np.arange(1.0, source_size + 1)[None, :, None, None],
            (2, source_size, 4, 8),
        ),
        dims=("time", source_dim, "lat", "lon"),
        coords={
            "time": [15, 45],
            source_dim: np.arange(source_size),
            "lat": np.linspace(-90, 90, 4),
            "lon": np.linspace(0, 360, 8, endpoint=False),
        },
    )
    grouped = _safe_eval(
        f"map_multi_dimension(FATES_GPP_PF, {source_dim})",
        {"FATES_GPP_PF": source},
    )
    branded = f"{variable_name}_tavg-u-hxy-multi"
    ds = grouped.to_dataset(name=branded)
    ds["time_bnds"] = (("time", "bnds"), [[0, 30], [30, 60]])
    ds["time"].attrs.update(
        units="days since 2000-01-01", calendar="noleap", bounds="time_bnds"
    )
    ds["lat"].attrs["units"] = "degrees_north"
    ds["lon"].attrs["units"] = "degrees_east"

    class VDef:  # pylint: disable=too-few-public-methods
        """Land GPP on mapped categories."""

        name = variable_name
        branded_variable_name = branded
        units = "kg m-2 s-1"
        table = "land"
        levels = {}

    class CMIPVar:  # pylint: disable=too-few-public-methods
        """Branded variable identifier."""

        class BrandedName:  # pylint: disable=too-few-public-methods
            """Name wrapper used by the writer."""

            name = branded

        branded_variable_name = BrandedName()

    tables_root = Path(__file__).parent.parent / "cmip7-cmor-tables"
    cfg = build_dataset_cfg(
        model="noresm",
        resolution="NorESM3-LM",
        experiment="piControl",
        frequency="mon",
        ripf="r1i1p1f1",
        tables_root=tables_root,
        data_version="v20261103",
        realm="land",
    )
    with CmorSession(
        tables_root=tables_root,
        dataset_attrs=cfg,
        log_dir=tmp_path,
        log_name=f"cmor_{variable_name}.log",
        outdir=tmp_path / "cmor",
        realm="land",
    ) as session:
        written_path = session.write_variable(ds, CMIPVar(), VDef())

    with xr.open_dataset(written_path) as written:
        with open(
            tables_root / "tables" / "CMIP7_coordinate.json", encoding="utf-8"
        ) as handle:
            requested = json.load(handle)["axis_entry"][target_dim]["requested"]
        labels = [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in written["sector"].values
        ]
        assert written["sector"].dims == (output_dim,)
        assert labels == requested == grouped[target_dim].values.tolist()
        assert source_dim not in written.dims
        np.testing.assert_allclose(
            written[variable_name].transpose("time", output_dim, "lat", "lon").values,
            grouped.values,
            equal_nan=True,
        )


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

    dataset_cfg = build_dataset_cfg(
        model="cesm",
        resolution="ne30",
        experiment="piControl",
        frequency="mon",
        ripf="r1i1p1f1",
        tables_root=tables_root,
        data_version="v20200101",
    )
    log_dir = tmp_path
    log_name = "cmor_test_multi.log"
    with CmorSession(
        tables_root=tables_root,
        dataset_attrs=dataset_cfg,
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

    dataset_cfg = build_dataset_cfg(
        model="cesm",
        resolution="ne30",
        experiment="piControl",
        frequency="mon",
        ripf="r1i1p1f1",
        tables_root=tables_root,
        data_version="v20200101",
    )
    with CmorSession(
        tables_root=tables_root,
        dataset_attrs=dataset_cfg,
        log_dir=tmp_path,
        log_name="cmor_test_zonal_mean.log",
    ) as session:
        session.write_variable(ds, CMIPVar(), VDef())


def _dataset_without_time_bounds():
    """Return a dataset whose time coordinate carries no bounds."""
    data = np.random.rand(2, 4, 8)
    ds = xr.Dataset(
        {
            "tas_tmin-h2m-hxy-u": (("time", "lat", "lon"), data),
            "lat": ("lat", np.linspace(-90, 90, 4)),
            "lon": ("lon", np.linspace(0, 360, 8, endpoint=False)),
            "time": ("time", np.array([15, 45])),
        }
    )
    ds["lat"].attrs["units"] = "degrees_north"
    ds["lon"].attrs["units"] = "degrees_east"
    ds["time"].attrs["units"] = "days since 2000-01-01"
    ds["time"].attrs["calendar"] = "noleap"
    ds.attrs["branded_variable"] = "tas_tmin-h2m-hxy-u"
    return ds


class _VDef:  # pylint: disable=too-few-public-methods
    """Minimal variable definition."""

    name = "tas"
    branded_variable_name = "tas_tmin-h2m-hxy-u"
    units = "K"
    table = "atmos"
    levels: dict = {}


class _CMIPVar:  # pylint: disable=too-few-public-methods
    """Minimal CMIP variable wrapper."""

    class BrandedName:  # pylint: disable=too-few-public-methods
        """Branded variable name wrapper."""

        name = "tas_tmin-h2m-hxy-u"

    branded_variable_name = BrandedName()


def _session(tmp_path, realm):
    """Return a CmorSession for one realm, writing into tmp_path."""
    dataset_cfg = build_dataset_cfg(
        model="cesm",
        resolution="ne30",
        experiment="piControl",
        frequency="month",
        ripf="r1i1p1f1",
        tables_root=Path(__file__).parent.parent / "cmip7-cmor-tables",
        data_version="v20200101",
        realm=realm,
    )
    return CmorSession(
        tables_root=Path(__file__).parent.parent / "cmip7-cmor-tables",
        dataset_attrs=dataset_cfg,
        log_dir=tmp_path,
        log_name=f"cmor_{realm}.log",
        realm=realm,
    )


def test_missing_time_bounds_are_an_error_outside_land_ice(tmp_path):
    """A realm expected to write time bounds fails loudly without them.

    Only CISM land-ice output lacks bounds, so their absence anywhere else
    means something went wrong before CMORization and should be reported, not
    replaced with a plausible guess.
    """
    with _session(tmp_path, "atmos") as session:
        with pytest.raises(ValueError, match="expected to write them"):
            session.write_variable(_dataset_without_time_bounds(), _CMIPVar(), _VDef())


def test_missing_time_bounds_are_synthesized_for_land_ice(tmp_path):
    """Land ice is the exception: its bounds are built from the time values."""
    with _session(tmp_path, "landIce") as session:
        try:
            session.write_variable(_dataset_without_time_bounds(), _CMIPVar(), _VDef())
        except Exception as exc:  # pylint: disable=broad-except
            # Getting as far as CMOR's own checks means the bounds were built.
            # What CMOR then makes of an atmosphere variable in the land-ice
            # table is not what this test is about.
            assert "expected to write them" not in str(exc)
