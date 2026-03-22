"""Tests for mom6_static.py."""

import numpy as np
import pytest
import xarray as xr

from cmip7_prep.mom6_static import compute_cell_bounds_from_corners, ocean_fx_fields

# ---------------------------------------------------------------------------
# compute_cell_bounds_from_corners
# ---------------------------------------------------------------------------


class TestComputeCellBoundsFromCorners:
    """Tests for compute_cell_bounds_from_corners output shape and values."""

    def test_shape_1x1_grid(self):
        """A (2,2) corner array yields a (1,2) bounds array."""
        corners = np.array([[0.0, 1.0], [2.0, 3.0]])  # (2, 2) → 1x1 grid
        bounds = compute_cell_bounds_from_corners(corners)
        assert bounds.shape == (1, 2)

    def test_shape_2x3_grid(self):
        """A (3,4) corner array yields a (6,2) bounds array."""
        corners = np.arange(12, dtype=float).reshape(3, 4)  # (3, 4) → 2x3 grid
        bounds = compute_cell_bounds_from_corners(corners)
        assert bounds.shape == (6, 2)

    def test_min_max_within_corners(self):
        """Bounds are the min and max of the four cell corners."""
        corners = np.array([[0.0, 10.0], [5.0, 8.0]])  # 1×1 grid
        bounds = compute_cell_bounds_from_corners(corners)
        assert bounds[0, 0] == pytest.approx(0.0)
        assert bounds[0, 1] == pytest.approx(10.0)

    def test_with_nan_corner(self):
        """NaN corners are handled gracefully by nanmin/nanmax."""
        corners = np.array([[0.0, float("nan")], [5.0, 8.0]])
        # Should not raise; exact behaviour (NaN propagation) is acceptable
        compute_cell_bounds_from_corners(corners)

    def test_lower_bound_leq_upper_bound(self):
        """Lower bound is always <= upper bound for all cells."""
        corners = np.random.default_rng(42).uniform(-90, 90, (5, 5))
        bounds = compute_cell_bounds_from_corners(corners)
        assert np.all(bounds[:, 0] <= bounds[:, 1])


# ---------------------------------------------------------------------------
# ocean_fx_fields
# ---------------------------------------------------------------------------


class TestOceanFxFields:
    """Tests for ocean_fx_fields field extraction from a MOM6 static file."""

    def test_extracts_deptho_and_areacello(self, tmp_path):
        """deptho and areacello are returned when present in the static file."""
        nc_path = tmp_path / "ocean_static.nc"
        ds = xr.Dataset(
            {
                "deptho": xr.DataArray(np.ones((4, 5)), dims=["y", "x"]),
                "areacello": xr.DataArray(np.full((4, 5), 1e10), dims=["y", "x"]),
            }
        )
        ds.to_netcdf(nc_path)

        fx = ocean_fx_fields(nc_path)
        assert "deptho" in fx
        assert "areacello" in fx

    def test_missing_vars_not_returned(self, tmp_path):
        """Variables not in the recognised list are not included in the result."""
        nc_path = tmp_path / "ocean_static.nc"
        ds = xr.Dataset({"some_other_var": xr.DataArray([1.0, 2.0])})
        ds.to_netcdf(nc_path)

        fx = ocean_fx_fields(nc_path)
        assert "deptho" not in fx
        assert "areacello" not in fx
        assert len(fx) == 0
