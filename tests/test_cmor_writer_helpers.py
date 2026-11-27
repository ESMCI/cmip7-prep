"""tests/test_cmor_writer_helpers.py"""

import json

import netCDF4
import numpy as np
import pytest
import xarray as xr

from cmip7_prep.cmor_utils import encode_time_to_num
from cmip7_prep.cmor_writer import (
    normalize_written_cf_metadata,
    pressure_values_in_pa,
)

cftime = pytest.importorskip("cftime")


def test_encode_time_to_num_with_cftime_no_leap():
    """test  0001-01-16 12:00, 0001-02-15 12:00 in noleap"""
    t0 = cftime.DatetimeNoLeap(1, 1, 16, 12, 0, 0)
    t1 = cftime.DatetimeNoLeap(1, 2, 15, 12, 0, 0)
    arr = np.array([t0, t1], dtype=object)

    out = encode_time_to_num(arr, units="days since 0001-01-01", calendar="noleap")

    assert out.shape == (2,)
    assert out.dtype == np.float64
    assert np.all(np.isfinite(out))
    assert np.all(np.diff(out) > 0)  # strictly increasing


@pytest.mark.parametrize(
    ("units", "factor"),
    [("Pa", 1.0), ("hPa", 100.0), ("mbar", 100.0), ("kPa", 1000.0)],
)
def test_pressure_values_in_pa_converts_declared_units(units, factor):
    """Pressure formula terms are normalized to Pa without guessing units."""
    pressure = xr.DataArray([1.0, 2.0], attrs={"units": units})

    np.testing.assert_allclose(
        pressure_values_in_pa(pressure, "PS"),
        np.array([1.0, 2.0]) * factor,
    )


def test_pressure_values_in_pa_rejects_missing_units():
    """Formula pressure terms must declare their source units."""
    with pytest.raises(ValueError, match="must declare pressure units"):
        pressure_values_in_pa(xr.DataArray(100000.0), "PS")


def test_normalize_written_cf_metadata_repairs_hybrid_output(tmp_path):
    """Written hybrid metadata is explicit and bounds do not repeat units."""
    path = tmp_path / "hybrid.nc"
    table_path = tmp_path / "CMIP7_atmos.json"
    branded_name = "zg_tavg-al-hxy-u"
    table_path.write_text(
        json.dumps(
            {
                "variable_entry": {
                    branded_name: {
                        "modeling_realm": "atmos",
                        "dimensions": ["longitude", "latitude", "alevel", "time"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with netCDF4.Dataset(path, "w") as output:
        output.createDimension("lev", 2)
        output.createDimension("lat", 3)
        output.createDimension("lon", 4)
        output.createDimension("bnds", 2)
        output.createVariable("lat", "f8", ("lat",))
        output.createVariable("lon", "f8", ("lon",))
        lev = output.createVariable("lev", "f8", ("lev",))
        lev.bounds = "lev_bnds"
        lev.standard_name = "atmosphere_hybrid_sigma_pressure_coordinate"
        lev.formula_terms = "p0: p0 a: a b: b ps: ps"
        lev.units = "1"
        lev_bnds = output.createVariable("lev_bnds", "f8", ("lev", "bnds"))
        lev_bnds.standard_name = lev.standard_name
        lev_bnds.units = "1"
        lev_bnds.formula_terms = "p0: p0 a: a_bnds b: b_bnds ps: ps"
        output.createVariable("a", "f8", ("lev",)).original_units = "1"
        output.createVariable("b", "f8", ("lev",)).original_units = "1"
        output.createVariable("a_bnds", "f8", ("lev", "bnds")).original_units = "1"
        output.createVariable("b_bnds", "f8", ("lev", "bnds")).original_units = "1"
        output.createVariable("zg", "f4", ("lev", "lat", "lon"))

    normalize_written_cf_metadata(
        path,
        variable_id="zg",
        table_path=table_path,
        branded_name=branded_name,
    )

    with netCDF4.Dataset(path) as output:
        for name in ("a", "b", "a_bnds", "b_bnds"):
            assert output[name].units == "1"
        assert "units" not in output["lev_bnds"].ncattrs()
        assert "standard_name" not in output["lev_bnds"].ncattrs()
        assert output["lev_bnds"].formula_terms == "p0: p0 a: a_bnds b: b_bnds ps: ps"
        assert output["zg"].cell_measures == "area: areacella_ti-u-hxy-u"
        assert output.external_variables == "areacella_ti-u-hxy-u"
