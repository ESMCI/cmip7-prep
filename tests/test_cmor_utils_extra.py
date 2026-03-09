"""Additional unit tests for cmor_utils.py (functions not covered by doctests)."""

import numpy as np
import pytest
import xarray as xr

from cmip7_prep.cmor_utils import (
    bounds_from_centers_1d,
    filled_for_cmor,
    open_existing_fx,
    resolve_table_filename,
    roll_for_monotonic_with_bounds,
    sigma_mid_and_bounds,
)


# ---------------------------------------------------------------------------
# bounds_from_centers_1d
# ---------------------------------------------------------------------------


class TestBoundsFromCenters1d:
    """Tests for bounds_from_centers_1d covering lat, lon, plev, and edge cases."""

    def test_lat_shape(self):
        """Output has shape (n, 2) for n latitude points."""
        lats = np.linspace(-88.5, 88.5, 4)
        b = bounds_from_centers_1d(lats, "lat")
        assert b.shape == (4, 2)

    def test_lat_clamped_to_90(self):
        """Latitude bounds are clamped to [-90, 90]."""
        lats = np.array([-89.5, 0.0, 89.5])
        b = bounds_from_centers_1d(lats, "lat")
        assert b[0, 0] == pytest.approx(-90.0)
        assert b[-1, 1] == pytest.approx(90.0)

    def test_lat_internal_bounds_are_midpoints(self):
        """Internal bounds are midpoints between adjacent centers."""
        lats = np.array([10.0, 20.0, 30.0])
        b = bounds_from_centers_1d(lats, "lat")
        assert b[0, 1] == pytest.approx(15.0)
        assert b[1, 0] == pytest.approx(15.0)
        assert b[1, 1] == pytest.approx(25.0)

    def test_lon_shape(self):
        """Output has shape (n, 2) for n longitude points."""
        lons = np.linspace(1.25, 358.75, 4)
        b = bounds_from_centers_1d(lons, "lon")
        assert b.shape == (4, 2)

    def test_lon_continuity(self):
        """Adjacent longitude bounds share a common edge (no gaps)."""
        lons = np.linspace(1.25, 358.75, 144)
        b = bounds_from_centers_1d(lons, "lon")
        np.testing.assert_array_almost_equal(b[:-1, 1], b[1:, 0])

    def test_single_point_lat(self):
        """Single latitude point produces a width-1 cell centered on the value."""
        b = bounds_from_centers_1d(np.array([45.0]), "lat")
        assert b.shape == (1, 2)
        assert b[0, 0] == pytest.approx(44.5)
        assert b[0, 1] == pytest.approx(45.5)

    def test_plev_shape(self):
        """Pressure level input produces correct output shape."""
        plevs = np.array([100000.0, 50000.0, 10000.0])
        b = bounds_from_centers_1d(plevs, "plev")
        assert b.shape == (3, 2)


# ---------------------------------------------------------------------------
# roll_for_monotonic_with_bounds
# ---------------------------------------------------------------------------


class TestRollForMonotonicWithBounds:
    """Tests for roll_for_monotonic_with_bounds."""

    def test_already_monotonic_no_roll(self):
        """Returns unchanged arrays with shift=0 when input is already monotonic."""
        lon = np.array([0.0, 90.0, 180.0, 270.0])
        bnds = np.column_stack([lon - 45, lon + 45])
        out_lon, out_bnds, shift = roll_for_monotonic_with_bounds(lon, bnds)
        assert shift == 0
        np.testing.assert_array_equal(out_lon, lon)
        np.testing.assert_array_equal(out_bnds, bnds)

    def test_rolls_at_wraparound(self):
        """After rolling a non-monotonic sequence the result is strictly increasing."""
        lon = np.array([270.0, 0.0, 90.0, 180.0])
        bnds = np.column_stack([lon - 45, lon + 45])
        out_lon, _out_bnds, shift = roll_for_monotonic_with_bounds(lon, bnds)
        assert shift != 0
        assert np.all(np.diff(out_lon) > 0)


# ---------------------------------------------------------------------------
# sigma_mid_and_bounds
# ---------------------------------------------------------------------------


class TestSigmaMidAndBounds:
    """Tests for sigma_mid_and_bounds covering normal and synthesized-bounds cases."""

    def _make_ds(self):
        """Return a minimal dataset with hybm and hybi."""
        return xr.Dataset(
            {
                "hybm": ("mid", [0.1, 0.3, 0.6, 0.9]),
                "hybi": ("edge", [0.0, 0.2, 0.4, 0.7, 1.0]),
            }
        )

    def test_output_shapes(self):
        """mid has shape (n,) and bounds has shape (n, 2)."""
        ds = self._make_ds()
        mid, bnds = sigma_mid_and_bounds(ds, {"hybm": "hybm", "hybi": "hybi"})
        assert mid.shape == (4,)
        assert bnds.shape == (4, 2)

    def test_bounds_within_zero_one(self):
        """All sigma bounds are within [0, 1]."""
        ds = self._make_ds()
        _mid, bnds = sigma_mid_and_bounds(ds, {"hybm": "hybm", "hybi": "hybi"})
        assert np.all(bnds >= 0.0)
        assert np.all(bnds <= 1.0)

    def test_synthesizes_bounds_without_hybi(self):
        """When hybi is absent, bounds are synthesized with 0 and 1 as endpoints."""
        ds = xr.Dataset({"hybm": ("mid", [0.1, 0.3, 0.6, 0.9])})
        _mid, bnds = sigma_mid_and_bounds(ds, {"hybm": "hybm"})
        assert bnds.shape == (4, 2)
        assert bnds[0, 0] == pytest.approx(0.0)
        assert bnds[-1, 1] == pytest.approx(1.0)

    def test_raises_on_out_of_range(self):
        """Values outside [0, 1] raise ValueError."""
        ds = xr.Dataset({"hybm": ("mid", [0.1, 0.3, 1.5])})
        with pytest.raises(ValueError, match="sigma"):
            sigma_mid_and_bounds(ds, {"hybm": "hybm"})

    def test_raises_on_nonmonotonic(self):
        """Non-monotonic hybm values raise ValueError."""
        ds = xr.Dataset({"hybm": ("mid", [0.1, 0.3, 0.2, 0.9])})
        with pytest.raises(ValueError):
            sigma_mid_and_bounds(ds, {"hybm": "hybm"})


# ---------------------------------------------------------------------------
# resolve_table_filename
# ---------------------------------------------------------------------------


class TestResolveTableFilename:
    """Tests for resolve_table_filename."""

    def test_finds_cmip7_file(self, tmp_path):
        """Returns the full path when a CMIP7_<key>.json file exists."""
        (tmp_path / "CMIP7_Amon.json").touch()
        result = resolve_table_filename(tmp_path, "Amon")
        assert "CMIP7_Amon.json" in result

    def test_finds_capitalized_file(self, tmp_path):
        """Falls back to <Capitalized>.json when the CMIP7 variant is absent."""
        (tmp_path / "Coordinate.json").touch()
        result = resolve_table_filename(tmp_path, "coordinate")
        assert "Coordinate.json" in result

    def test_fallback_when_missing(self, tmp_path):
        """Returns '<key>.json' when no file is found."""
        result = resolve_table_filename(tmp_path, "Amon")
        assert result == "Amon.json"


# ---------------------------------------------------------------------------
# open_existing_fx
# ---------------------------------------------------------------------------


class TestOpenExistingFx:
    """Tests for open_existing_fx file discovery."""

    def test_returns_none_when_empty_dir(self, tmp_path):
        """Returns None when the output directory contains no fx files."""
        assert open_existing_fx(tmp_path, "sftlf") is None

    def test_no_match_for_nonconforming_name(self, tmp_path):
        """File without the required underscore prefix token returns None."""
        sub = tmp_path / "subdir"
        sub.mkdir()
        nc_path = sub / "sftlf_fx_v1.nc"  # lacks a leading '_sftlf_' glob match
        ds = xr.Dataset({"sftlf": xr.DataArray(np.ones((4, 8)), dims=["lat", "lon"])})
        ds.to_netcdf(nc_path)
        assert open_existing_fx(tmp_path, "sftlf") is None

    def test_finds_fx_with_correct_glob_pattern(self, tmp_path):
        """File matching **/*_sftlf_fx_*.nc is returned as a DataArray."""
        nc_path = tmp_path / "r1i1p1_sftlf_fx_test.nc"
        da = xr.DataArray(np.ones((4, 8)), dims=["lat", "lon"])
        xr.Dataset({"sftlf": da}).to_netcdf(nc_path)
        result = open_existing_fx(tmp_path, "sftlf")
        assert result is not None
        assert result.dims == ("lat", "lon")


# ---------------------------------------------------------------------------
# filled_for_cmor
# ---------------------------------------------------------------------------


class TestFilledForCmor:
    """Tests for filled_for_cmor NaN-replacement logic."""

    def test_replaces_nan_with_default_fill(self):
        """NaNs are replaced with 1e20 by default."""
        da = xr.DataArray([1.0, float("nan"), 3.0])
        filled, f = filled_for_cmor(da)
        assert np.isfinite(filled.values).all()
        assert f == pytest.approx(1.0e20, rel=1e-4)

    def test_replaces_nan_with_custom_fill(self):
        """NaNs are replaced with the caller-supplied fill value."""
        da = xr.DataArray([1.0, float("nan"), 3.0])
        filled, _f = filled_for_cmor(da, fill=-999.0)
        assert filled.values[1] == pytest.approx(-999.0)

    def test_non_float_array_unchanged(self):
        """Integer arrays pass through without modification."""
        da = xr.DataArray(np.array([1, 2, 3], dtype=np.int32))
        filled, _ = filled_for_cmor(da)
        np.testing.assert_array_equal(filled.values, [1, 2, 3])

    def test_returns_float32(self):
        """Output is always cast to float32."""
        da = xr.DataArray(np.array([1.0, 2.0], dtype=np.float64))
        filled, _ = filled_for_cmor(da)
        assert filled.dtype == np.float32
