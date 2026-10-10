"""The grid each realm's data is on.

Model components run on different grids, so the grid to regrid from depends on
which realm is being processed.  Which grid that is is stated per realm, per
resolution, in ``data/<model>_grids.yaml`` under ``grid_names_per_realm``
as each row's ``input_grid``, rather than being hardcoded here, so that adding
a model or changing a component's grid is a data change.

:func:`resolution_for` reads it.  ``'custom'`` is the way to run a case the
table does not otherwise describe: give it a ``custom`` block naming whatever
grids it ran, adding entries to ``resolutions`` for any that are new.  Those
grids may be regular lat/lon, in which case nothing is regridded, but they need
not be -- that is one possibility, not what ``custom`` means.  With no block at
all, nothing is regridded, since that is all that can be assumed.
"""

from __future__ import annotations

from .regrid_maps import get_realm_row, load_regrid_maps

# Resolutions the case may have been run at, as accepted on the command line.
MODEL_RESOLUTIONS = ("ne30", "NorESM3-LM", "NorESM3-MM", "custom")

# The extension point: a case the table does not otherwise describe, whose
# grids are declared by filling in the commented-out block at the end of
# 'grid_names_per_realm'.  Any grid may be named there, new ones included.
# With no block, nothing is regridded.
CUSTOM = "custom"


def resolution_for(model: str, realm: str, model_res: str | None = None) -> str:
    """Return the input grid name for one realm.

    ``model_res`` is the resolution the case was run at, which the table is
    keyed by along with the realm.
    """
    if model_res is None:
        raise ValueError(
            f"The resolution the case was run at must be given to find "
            f"realm {realm!r}'s input grid"
        )
    if model_res not in MODEL_RESOLUTIONS:
        raise ValueError(
            f"Unknown atmosphere resolution {model_res!r}; "
            f"choose from {list(MODEL_RESOLUTIONS)}"
        )
    declared = load_regrid_maps(model).get("grid_names_per_realm") or {}
    if model_res == CUSTOM and model_res not in declared:
        # A custom case whose grids have not been declared: nothing is
        # regridded, which is all that can be assumed about input the table
        # says nothing about.
        return CUSTOM
    return get_realm_row(model, model_res, realm)["input_grid"]


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
