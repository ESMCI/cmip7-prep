"""Native CISM (land-ice) grid support for CMORization.

CISM ice-sheet history files are on a projected (polar-stereographic) grid and
carry only Cartesian ``x``/``y`` coordinates in metres -- no latitude/longitude
and no grid-mapping metadata.  To write output on the native grid
(``grid_label = gn``), CMOR needs geographic cell-center latitude/longitude and
their corner vertices.

Because the projected ``x``/``y`` coordinates *are* present in the history files
(``x0``/``y0`` velocity grid and ``x1``/``y1`` scalar grid), we georeference them
directly with the ice sheet's known map projection via ``pyproj``.  This works
uniformly for either grid, so no per-grid ESMF mesh is required.  (An ESMF mesh
reader is kept below only as an optional cross-check.)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

# Map projection for each CISM ice-sheet grid (polar stereographic, WGS84).
#   gris -> EPSG:3413  (NSIDC Sea Ice Polar Stereographic North: lat_ts=70,
#                       lon_0=-45, lat_0=90)
#   ais  -> EPSG:3031  (Antarctic Polar Stereographic:           lat_ts=-71,
#                       lon_0=0,   lat_0=-90)
# Keyed by the --ice-sheet value.  Confirm these match the projection CISM used.
ICE_SHEET_PROJ = {
    "gris": "EPSG:3413",
    "ais": "EPSG:3031",
}


def proj_for_ice_sheet(ice_sheet: str) -> str:
    """Return the CRS string (e.g. 'EPSG:3413') for an ice sheet ('gris'/'ais')."""
    try:
        return ICE_SHEET_PROJ[ice_sheet]
    except KeyError:
        raise ValueError(
            f"No projection registered for ice_sheet={ice_sheet!r}; "
            f"known: {sorted(ICE_SHEET_PROJ)}"
        ) from None


@dataclass
class CismGrid:
    """Geographic coordinates for a structured CISM grid.

    All arrays are on the logically-rectangular ``(ny, nx)`` grid.  Vertices
    carry a trailing corner dimension of size 4 (ordered counter-clockwise).
    """

    lon: np.ndarray  # (ny, nx)
    lat: np.ndarray  # (ny, nx)
    lon_vertices: np.ndarray  # (ny, nx, 4)
    lat_vertices: np.ndarray  # (ny, nx, 4)
    nx: int
    ny: int


def project_xy_to_latlon(x: np.ndarray, y: np.ndarray, ice_sheet: str) -> CismGrid:
    """Georeference a projected CISM grid to lat/lon with the ice-sheet projection.

    Parameters
    ----------
    x, y:
        1-D projected coordinates in metres (e.g. ``x1``/``y1`` scalar grid or
        ``x0``/``y0`` velocity grid), read straight from the CISM history file.
    ice_sheet:
        'gris' or 'ais' -- selects the map projection (see :data:`ICE_SHEET_PROJ`).

    Returns
    -------
    CismGrid
        Cell-center ``lon``/``lat`` ``(ny, nx)`` and corner ``lon_vertices``/
        ``lat_vertices`` ``(ny, nx, 4)``.  Longitudes are in [0, 360).

    Notes
    -----
    - ``x``/``y`` are assumed regularly spaced; corner vertices are the four
      cell edges at ``x +/- dx/2``, ``y +/- dy/2`` transformed to lat/lon.
    - Uses ``always_xy=True`` so the transformer takes (easting, northing) and
      returns (lon, lat).
    """
    # Local import: pyproj is only needed for the land-ice realm.
    import pyproj  # pylint: disable=import-outside-toplevel,import-error

    x = np.asarray(x, dtype="f8")
    y = np.asarray(y, dtype="f8")
    nx, ny = int(x.size), int(y.size)
    if nx < 2 or ny < 2:
        raise ValueError(
            f"projected grid too small to derive cell spacing: nx={nx}, ny={ny}"
        )

    transformer = pyproj.Transformer.from_crs(
        proj_for_ice_sheet(ice_sheet), "EPSG:4326", always_xy=True
    )

    # Cell centers: (ny, nx) meshgrid of the projected coordinates -> lon/lat.
    x2d, y2d = np.meshgrid(x, y)  # both (ny, nx)
    lon, lat = transformer.transform(x2d, y2d)

    # Corner vertices: offset each center by +/- half a cell in x and y, CCW.
    dx = float(np.diff(x).mean())
    dy = float(np.diff(y).mean())
    off_x = np.array([-0.5, 0.5, 0.5, -0.5]) * dx
    off_y = np.array([-0.5, -0.5, 0.5, 0.5]) * dy
    lon_v = np.empty((ny, nx, 4), dtype="f8")
    lat_v = np.empty((ny, nx, 4), dtype="f8")
    for k in range(4):
        lo, la = transformer.transform(x2d + off_x[k], y2d + off_y[k])
        lon_v[..., k] = lo
        lat_v[..., k] = la

    # Normalize longitudes to [0, 360).
    lon = np.mod(lon, 360.0)
    lon_v = np.mod(lon_v, 360.0)

    return CismGrid(
        lon=lon, lat=lat, lon_vertices=lon_v, lat_vertices=lat_v, nx=nx, ny=ny
    )


# ---------------------------------------------------------------------------
# Optional: read geographic coordinates from an ESMF mesh instead of projecting.
# Kept as a cross-check against project_xy_to_latlon; not used by the pipeline.
# ---------------------------------------------------------------------------
def read_esmf_mesh(mesh_path: str | Path) -> CismGrid:
    """Read center/corner lat/lon from an ESMF mesh describing a structured grid.

    Notes
    -----
    - ``coordDim`` order is (lon, lat): column 0 is longitude, column 1 latitude.
    - ``elementConn`` node indices are 1-based (ESMF convention); converted to
      0-based here.  Fill values (``-1``) are clamped to the element's first node
      so the vertex array stays rectangular; a regular quad grid has 4 nodes/cell.
    - The flat element list is reshaped to ``(ny, nx)`` from ``origGridDims``,
      x (first dim) varying fastest.  ``elementCount`` must equal ``nx * ny``.
    """
    with xr.open_dataset(mesh_path) as ds:
        orig_dims = np.asarray(ds["origGridDims"].values).astype(int).tolist()
        if len(orig_dims) != 2:
            raise ValueError(
                f"{mesh_path}: expected a 2-D origGridDims, got {orig_dims}"
            )
        nx, ny = int(orig_dims[0]), int(orig_dims[1])

        center = np.asarray(ds["centerCoords"].values, dtype="f8")  # (nElem, 2)
        node = np.asarray(ds["nodeCoords"].values, dtype="f8")  # (nNode, 2)
        conn = np.asarray(ds["elementConn"].values)  # (nElem, maxNode)

    n_elem = center.shape[0]
    if n_elem != nx * ny:
        raise ValueError(
            f"{mesh_path}: elementCount ({n_elem}) != nx*ny ({nx}*{ny}={nx * ny}); "
            "the mesh does not match a structured grid of that shape."
        )

    lon = center[:, 0].reshape(ny, nx)
    lat = center[:, 1].reshape(ny, nx)

    ncorners = conn.shape[1]
    conn0 = conn.copy()
    fill = conn0 < 0
    conn0 = conn0 - 1  # 1-based -> 0-based
    if fill.any():
        first_valid = conn0[:, 0][:, None]
        conn0 = np.where(fill, first_valid, conn0)

    lon_v = node[conn0, 0].reshape(ny, nx, ncorners)
    lat_v = node[conn0, 1].reshape(ny, nx, ncorners)

    return CismGrid(
        lon=lon, lat=lat, lon_vertices=lon_v, lat_vertices=lat_v, nx=nx, ny=ny
    )
