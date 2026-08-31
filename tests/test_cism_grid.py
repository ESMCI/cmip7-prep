"""Tests for native CISM (land-ice) grid georeferencing."""

import numpy as np
import pytest

from cmip7_prep.cism_grid import (
    ICE_SHEET_CENTER,
    _median_distance_to,
    _warn_degenerate_cells,
    proj_for_ice_sheet,
    project_xy_to_latlon,
)

# Representative gris4 extent: 421 x 721 cells at 4 km in EPSG:3413.
GRIS4_X = -678650.0 + 4000.0 * np.arange(421)
GRIS4_Y = -3371350.0 + 4000.0 * np.arange(721)

# Representative ais8 extent: 761 x 761 cells at 8 km in EPSG:3031, centred on
# the South Pole -- so it contains both the 0/360 seam and the pole itself.
AIS8_X = -3040000.0 + 8000.0 * np.arange(761)
AIS8_Y = -3040000.0 + 8000.0 * np.arange(761)

# The 20 km SeaRISE Greenland grid uses a different projection (standard
# parallel 71, central meridian -39, large false easting/northing), so its
# native x/y are offset positives.
GRIS20_X = 20000.0 * np.arange(76)
GRIS20_Y = 20000.0 * np.arange(141)


def test_proj_for_ice_sheet_known_and_unknown():
    """Known ice sheets map to their EPSG code; anything else is rejected."""
    assert proj_for_ice_sheet("gris") == "EPSG:3413"
    assert proj_for_ice_sheet("ais") == "EPSG:3031"
    with pytest.raises(ValueError, match="No projection registered"):
        proj_for_ice_sheet("nope")


def test_greenland_lands_on_greenland():
    """A plausible gris4 extent projects onto Greenland, not somewhere else."""
    grid = project_xy_to_latlon(GRIS4_X, GRIS4_Y, "gris")
    assert grid.nx == 421 and grid.ny == 721
    assert grid.lon.shape == (721, 421)
    assert grid.lon_vertices.shape == (721, 421, 4)
    lat0, lon0, max_dist = ICE_SHEET_CENTER["gris"]
    assert _median_distance_to(grid.lat, grid.lon, lat0, lon0) < max_dist


def test_wrong_projection_is_rejected():
    """The 20 km SeaRISE grid projected as EPSG:3413 lands in Siberia."""
    with pytest.raises(ValueError, match="probably wrong for this grid"):
        project_xy_to_latlon(GRIS20_X, GRIS20_Y, "gris")


def test_antarctic_seam_cells_stay_contiguous():
    """Cells on the prime meridian must not appear to span the globe."""
    grid = project_xy_to_latlon(AIS8_X, AIS8_Y, "ais")
    spread = grid.lon_vertices.max(axis=-1) - grid.lon_vertices.min(axis=-1)
    # Away from the pole, no cell may span anything like a hemisphere.
    off_pole = np.abs(grid.lat) < 89.0
    assert spread[off_pole].max() < 10.0, (
        f"max off-pole corner spread {spread[off_pole].max():.1f} deg; "
        "the 0/360 branch correction is not working"
    )
    # Each corner stays within half a turn of its own center.
    assert np.abs(grid.lon_vertices - grid.lon[..., None]).max() < 180.0


def test_pole_cells_are_flagged(caplog):
    """The Antarctic domain contains the South Pole, so degenerate cells are logged."""
    grid = project_xy_to_latlon(AIS8_X, AIS8_Y, "ais")
    with caplog.at_level("WARNING"):
        n_bad = _warn_degenerate_cells(grid)
    assert n_bad > 0, "the ais8 domain contains the South Pole"
    assert "longitude is undefined" in caplog.text


def test_grid_too_small_to_derive_spacing():
    """A single-point axis gives no cell spacing to work from."""
    with pytest.raises(ValueError, match="too small to derive cell spacing"):
        project_xy_to_latlon(np.array([0.0]), np.array([0.0, 1.0]), "gris")
