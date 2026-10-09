"""The grid each realm's data is on.

Model components run on different grids, so the grid to regrid from depends on
which realm is being processed:

    atmos, atmosChem, aerosol, land   the atmosphere/land grid the case was
                                      run at: ne30, ne16, or 'regular' when
                                      the output already carries lat/lon
    ocean, ocnBgchem, seaIce          the model's tripolar grid: tnx1v4 for
                                      NorESM, tx2_3v2 for CESM
    landIce                           its own projected grid; CISM output is
                                      georeferenced from the x/y coordinates
                                      in the files, not regridded

:func:`resolution_for` returns it, so a caller gives only the atmosphere/land
resolution -- and only when processing a realm that uses it.
"""

from __future__ import annotations

# Grids the atmosphere and land may be run on.  'regular' means the output
# already carries lat/lon and is not regridded.
MODEL_RESOLUTIONS = ("ne30", "ne16", "regular")

# Realms whose input grid is the atmosphere/land grid the case was run on.
ATM_GRID_REALMS = frozenset({"atmos", "atmosChem", "aerosol", "land"})

# The ocean/sea-ice grid each model writes on.  These realms never share the
# atmosphere's grid, so passing one an atmosphere resolution would regrid
# through the wrong weights.
OCEAN_GRID = {"noresm": "tnx1v4", "cesm": "tx2_3v2"}

OCEAN_GRID_REALMS = frozenset({"ocean", "ocnBgchem", "seaIce"})

# Realms written on their native grid, with no ESMF regridding.  CISM land-ice
# output carries projected x/y coordinates, which cmor_writer georeferences with
# the ice sheet's own map projection (see cism_grid.project_xy_to_latlon), so no
# weight files are involved.  The value below is a pass-through: it names the
# entry that builds a regridder and discards it, which is what the unregridded
# path expects.
NATIVE_GRID_REALMS = frozenset({"landIce"})
NATIVE_GRID = "regular"


def resolution_for(model: str, realm: str, model_res: str | None = None) -> str:
    """Return the input grid name for one realm.

    ``model_res`` is the grid the atmosphere and land were run on.  It is
    required only for the realms on that grid, and ignored for the rest, so a
    sea-ice run need not supply one.
    """
    if realm in ATM_GRID_REALMS:
        if model_res is None:
            raise ValueError(
                f"Realm {realm!r} is on the atmosphere/land grid, so its "
                "resolution must be given"
            )
        if model_res not in MODEL_RESOLUTIONS:
            raise ValueError(
                f"Unknown atmosphere resolution {model_res!r}; "
                f"choose from {list(MODEL_RESOLUTIONS)}"
            )
        return model_res
    if realm in OCEAN_GRID_REALMS:
        try:
            return OCEAN_GRID[model]
        except KeyError:
            raise ValueError(
                f"No ocean grid known for model={model!r}; "
                f"known models: {sorted(OCEAN_GRID)}"
            ) from None
    if realm in NATIVE_GRID_REALMS:
        return NATIVE_GRID
    raise ValueError(f"No input grid known for realm {realm!r}")


# CICE staggers its grid: the thermodynamic fields sit at the cell centre (T),
# the velocities at the B-grid velocity point (U), and CICE6 adds the N and E
# points of the C grid.  Each point has its own latitude, longitude and vertex
# bounds, and a variable says which it is on through its 'coordinates'
# attribute.
CICE_BOUNDS_BY_COORD = {
    "TLAT": "latt_bounds",
    "TLON": "lont_bounds",
    "ULAT": "latu_bounds",
    "ULON": "lonu_bounds",
    "NLAT": "latn_bounds",
    "NLON": "lonn_bounds",
    "ELAT": "late_bounds",
    "ELON": "lone_bounds",
}

# Every coordinate and bounds variable of that grid.  Realizing a variable
# builds a new dataset around it, which leaves the bounds behind, so these are
# copied over -- all of them, so a velocity variable keeps ULAT/ULON and not
# just the centre (issue #115).
CICE_GRID_VARS = tuple(CICE_BOUNDS_BY_COORD) + tuple(CICE_BOUNDS_BY_COORD.values())
