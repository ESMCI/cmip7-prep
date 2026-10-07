"""Tests for putting CICE variables on the grid point they belong to.

CICE staggers its grid: the thermodynamic fields sit at the cell centre
(TLAT/TLON) and the velocities at the B-grid velocity point (ULAT/ULON).  Each
variable names its own coordinates, so they are read from the variable rather
than assumed, which is what issue #115 reports going wrong -- siu and siv were
written with centre coordinates.
"""

import numpy as np
import xarray as xr

from cmip7_prep.grids import CICE_BOUNDS_BY_COORD, CICE_GRID_VARS
from cmip7_prep.cmor_writer import (
    horizontal_coords_of,
    is_latitude,
    is_longitude,
    vertex_bounds_of,
)

NJ, NI, NVERT = 2, 3, 4


def _cice_dataset(with_u_bounds=True):
    """Return a dataset shaped like CICE output, with T and U grid points."""
    shape = (NJ, NI)
    ds = xr.Dataset(
        {
            "siconc": (
                ("nj", "ni"),
                np.zeros(shape),
                {"coordinates": "TLON TLAT time"},
            ),
            "siu": (
                ("nj", "ni"),
                np.zeros(shape),
                {"coordinates": "ULON ULAT time"},
            ),
            "no_coords": (("nj", "ni"), np.zeros(shape), {}),
        },
        coords={
            "TLAT": (
                ("nj", "ni"),
                np.full(shape, 10.0),
                {"units": "degrees_north", "bounds": "latt_bounds"},
            ),
            "TLON": (
                ("nj", "ni"),
                np.full(shape, 20.0),
                {"units": "degrees_east", "bounds": "lont_bounds"},
            ),
            # The velocity point deliberately carries no 'bounds' attribute, so
            # the name map is what has to find them.
            "ULAT": (("nj", "ni"), np.full(shape, 11.0), {"units": "degrees_north"}),
            "ULON": (("nj", "ni"), np.full(shape, 21.0), {"units": "degrees_east"}),
        },
    )
    ds["latt_bounds"] = (("nj", "ni", "nvertices"), np.zeros((NJ, NI, NVERT)))
    ds["lont_bounds"] = (("nj", "ni", "nvertices"), np.zeros((NJ, NI, NVERT)))
    if with_u_bounds:
        ds["latu_bounds"] = (("nj", "ni", "nvertices"), np.ones((NJ, NI, NVERT)))
        ds["lonu_bounds"] = (("nj", "ni", "nvertices"), np.ones((NJ, NI, NVERT)))
    return ds


class TestIdentifyingCoordinates:
    """Latitude and longitude are recognised by their CF attributes."""

    def test_latitude_by_units(self):
        """degrees_north marks a latitude whatever the variable is called."""
        assert is_latitude(xr.DataArray([0.0], attrs={"units": "degrees_north"}))

    def test_latitude_by_standard_name(self):
        """A standard_name of latitude is enough on its own."""
        assert is_latitude(xr.DataArray([0.0], attrs={"standard_name": "latitude"}))

    def test_longitude_by_units(self):
        """degrees_east marks a longitude."""
        assert is_longitude(xr.DataArray([0.0], attrs={"units": "degrees_east"}))

    def test_a_latitude_is_not_a_longitude(self):
        """The two are told apart rather than both matching."""
        lat = xr.DataArray([0.0], attrs={"units": "degrees_north"})
        assert not is_longitude(lat)


class TestChoosingTheGridPoint:
    """Each variable gets the coordinates its own attribute names."""

    def test_velocity_gets_the_velocity_point(self):
        """siu is on ULAT/ULON, which is what issue #115 reports."""
        ds = _cice_dataset()
        lat, lon = horizontal_coords_of(ds, ds["siu"], ds["TLAT"], ds["TLON"])
        assert lat.name == "ULAT"
        assert lon.name == "ULON"

    def test_thermodynamic_field_gets_the_centre(self):
        """siconc stays on TLAT/TLON."""
        ds = _cice_dataset()
        lat, lon = horizontal_coords_of(ds, ds["siconc"], ds["TLAT"], ds["TLON"])
        assert lat.name == "TLAT"
        assert lon.name == "TLON"

    def test_the_two_points_are_different_values(self):
        """The choice changes the numbers, not only the name.

        If it did not, the bug in the issue would have had no effect.
        """
        ds = _cice_dataset()
        u_lat, _ = horizontal_coords_of(ds, ds["siu"], ds["TLAT"], ds["TLON"])
        t_lat, _ = horizontal_coords_of(ds, ds["siconc"], ds["TLAT"], ds["TLON"])
        assert float(u_lat[0, 0]) != float(t_lat[0, 0])

    def test_the_attribute_decides_when_both_pairs_are_attached(self):
        """Attached coordinates cannot distinguish the grid points.

        TLAT and ULAT have the same dimensions, so xarray gives every variable
        on (nj, ni) all four of them.  Only the variable's own 'coordinates'
        attribute says which pair it is on, which is why that is read first.
        """
        ds = _cice_dataset()
        attached = {str(name) for name in ds["siu"].coords}
        assert {"TLAT", "ULAT"} <= attached, "both pairs should be attached"
        lat, lon = horizontal_coords_of(ds, ds["siu"], ds["TLAT"], ds["TLON"])
        assert (lat.name, lon.name) == ("ULAT", "ULON")

    def test_the_attribute_is_read_from_encoding_too(self):
        """Decoding coordinates moves the attribute out of attrs.

        xarray puts it in encoding instead, so a dataset read from a file keeps
        the information there rather than in attrs.
        """
        ds = _cice_dataset()
        del ds["siu"].attrs["coordinates"]
        ds["siu"].encoding["coordinates"] = "ULON ULAT time"
        lat, lon = horizontal_coords_of(ds, ds["siu"], ds["TLAT"], ds["TLON"])
        assert (lat.name, lon.name) == ("ULAT", "ULON")

    def test_a_variable_without_the_attribute_falls_back(self):
        """Output that names no coordinates uses the centre, as before."""
        ds = _cice_dataset()
        lat, lon = horizontal_coords_of(ds, ds["no_coords"], ds["TLAT"], ds["TLON"])
        assert lat.name == "TLAT"
        assert lon.name == "TLON"

    def test_an_attribute_naming_absent_variables_falls_back(self):
        """Coordinates named but not present do not leave us empty-handed."""
        ds = _cice_dataset()
        ds["siu"].attrs["coordinates"] = "NOWHERE_LON NOWHERE_LAT"
        lat, _ = horizontal_coords_of(ds, ds["siu"], ds["TLAT"], ds["TLON"])
        assert lat.name == "TLAT"


class TestFindingVertexBounds:
    """Bounds come from the same grid point as the coordinate."""

    def test_the_bounds_attribute_is_followed(self):
        """TLAT says where its bounds are, so that is used."""
        ds = _cice_dataset()
        bounds = vertex_bounds_of(ds, ds["TLAT"], "TLAT")
        assert bounds.name == "latt_bounds"

    def test_the_name_map_covers_a_missing_attribute(self):
        """ULAT carries no bounds attribute, so the map finds latu_bounds."""
        ds = _cice_dataset()
        assert vertex_bounds_of(ds, ds["ULAT"], "ULAT").name == "latu_bounds"
        assert vertex_bounds_of(ds, ds["ULON"], "ULON").name == "lonu_bounds"

    def test_the_velocity_bounds_are_not_the_centre_bounds(self):
        """Substituting centre bounds would misplace every cell corner."""
        ds = _cice_dataset()
        u_bounds = vertex_bounds_of(ds, ds["ULAT"], "ULAT")
        t_bounds = vertex_bounds_of(ds, ds["TLAT"], "TLAT")
        assert u_bounds.name != t_bounds.name
        assert float(u_bounds[0, 0, 0]) != float(t_bounds[0, 0, 0])

    def test_absent_bounds_return_nothing(self):
        """With no bounds for this point, None is returned rather than another
        point's bounds, leaving the caller to report it."""
        ds = _cice_dataset(with_u_bounds=False)
        assert vertex_bounds_of(ds, ds["ULAT"], "ULAT") is None

    def test_every_cice_grid_point_is_mapped(self):
        """The four grid points CICE writes all have an entry."""
        for point in ("T", "U", "N", "E"):
            assert f"{point}LAT" in CICE_BOUNDS_BY_COORD
            assert f"{point}LON" in CICE_BOUNDS_BY_COORD

    def test_the_carried_list_covers_coordinates_and_bounds(self):
        """What the driver copies over is every name in the map, both halves."""
        assert set(CICE_GRID_VARS) == set(CICE_BOUNDS_BY_COORD) | set(
            CICE_BOUNDS_BY_COORD.values()
        )
        assert "ULAT" in CICE_GRID_VARS
        assert "latu_bounds" in CICE_GRID_VARS
