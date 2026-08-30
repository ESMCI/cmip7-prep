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

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

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

# Nominal centre of each ice sheet and the radius within which most of its grid
# cells must fall, used as a backstop against a wrong projection (see
# _check_plausible).  Deliberately generous -- this only needs to catch a grid
# landing on the wrong part of the planet.
#
# A distance test is used rather than a lat/lon box because longitudes converge
# at the pole: a perfectly correct Greenland grid reaches high enough latitudes
# that its corners span a wide longitude range, including the 0/360 seam, so a
# longitude interval produces false alarms.  Latitude alone is not sufficient
# either -- the 20 km SeaRISE grid (standard_parallel 71, central_meridian -39)
# projected with EPSG:3413 yields latitudes of 65-80N, entirely plausible, while
# sitting near 108E in Siberia.  Distance from the ice sheet catches both.
ICE_SHEET_CENTER = {
    # ice_sheet: (lat, lon_east, max_median_distance_m)
    "gris": (72.0, -40.0, 2.0e6),
    "ais": (-90.0, 0.0, 3.0e6),
}


def _median_distance_to(
    lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float
) -> float:
    """Median great-circle distance (m) from the cells to a reference point."""
    r = 6371000.0
    la, lo = np.radians(lat), np.radians(lon)
    la0, lo0 = np.radians(lat0), np.radians(lon0)
    sin_half = (
        np.sin((la - la0) / 2.0) ** 2
        + np.cos(la0) * np.cos(la) * np.sin((lo - lo0) / 2.0) ** 2
    )
    return float(np.median(2.0 * r * np.arcsin(np.sqrt(np.clip(sin_half, 0.0, 1.0)))))


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
        ``lat_vertices`` ``(ny, nx, 4)``.  Center longitudes are in [0, 360);
        corner longitudes are contiguous with their own center and may fall
        just outside that interval near the 0/360 seam.

    Raises
    ------
    ValueError
        If the projected coordinates fall outside the plausible envelope for
        this ice sheet (see :data:`ICE_SHEET_BOUNDS`).

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

    # Normalize cell-center longitudes to [0, 360), then bring each cell's four
    # corners into the same 360-degree branch as its own center.  Wrapping the
    # corners independently splits any cell straddling the 0/360 seam: an 8 km
    # Antarctic cell centred on longitude 0 comes back with corners at 359.94
    # and 0.06 and appears to span the whole globe.  Corners may therefore fall
    # slightly outside [0, 360) -- CF requires them contiguous with the center,
    # not confined to a particular interval.
    lon = np.mod(lon, 360.0)
    lon_v = lon_v - 360.0 * np.round((lon_v - lon[..., None]) / 360.0)

    grid = CismGrid(
        lon=lon, lat=lat, lon_vertices=lon_v, lat_vertices=lat_v, nx=nx, ny=ny
    )
    _warn_degenerate_cells(grid)
    _check_plausible(grid, ice_sheet)
    return grid


def _warn_degenerate_cells(grid: CismGrid, spread_deg: float = 90.0) -> int:
    """Log a warning for cells whose corners still span a huge longitude range.

    After the branch correction above, a large residual spread means the cell is
    genuinely degenerate rather than merely seam-crossing -- in practice the
    cells at or adjacent to the pole, where longitude is undefined.  The
    Antarctic domain contains the South Pole, so this is expected there; it is a
    red flag anywhere else.  Returns the number of such cells.
    """
    spread = grid.lon_vertices.max(axis=-1) - grid.lon_vertices.min(axis=-1)
    n_bad = int((spread > spread_deg).sum())
    if n_bad:
        logger.warning(
            "%d of %d cells have corner longitudes spanning more than %.0f deg; "
            "these are at or near the pole, where longitude is undefined. "
            "Their cell areas and bounds should not be trusted.",
            n_bad,
            grid.lon.size,
            spread_deg,
        )
    return n_bad


def _check_plausible(grid: CismGrid, ice_sheet: str) -> None:
    """Fail loudly if the projected coordinates do not land on the ice sheet.

    A wrong projection produces perfectly well-formed output in the wrong place,
    which is far more damaging than a crash.  Compares the median cell against
    the nominal ice-sheet centre in :data:`ICE_SHEET_CENTER`.
    """
    ref = ICE_SHEET_CENTER.get(ice_sheet)
    if ref is None:
        logger.warning(
            "No plausibility reference for ice_sheet=%r; "
            "skipping the projection sanity check.",
            ice_sheet,
        )
        return

    lat0, lon0, max_dist = ref
    dist = _median_distance_to(grid.lat, grid.lon, lat0, lon0)
    if dist > max_dist:
        raise ValueError(
            f"projected grid for ice_sheet={ice_sheet!r} has a median cell "
            f"{dist / 1000:.0f} km from the expected centre "
            f"({lat0:.1f}N, {lon0:.1f}E), exceeding the {max_dist / 1000:.0f} km "
            f"limit.  The projection ({proj_for_ice_sheet(ice_sheet)}) is "
            f"probably wrong for this grid -- e.g. the 20 km SeaRISE Greenland "
            f"grid projected as EPSG:3413 lands near 108E, in Siberia.  "
            f"Grid spans lat {grid.lat.min():.2f}..{grid.lat.max():.2f}, "
            f"lon {grid.lon.min():.2f}..{grid.lon.max():.2f}."
        )

    logger.info(
        "CISM %s grid projected with %s: lat %.2f..%.2f, lon %.2f..%.2f, "
        "median cell %.0f km from (%.1fN, %.1fE)",
        ice_sheet,
        proj_for_ice_sheet(ice_sheet),
        grid.lat.min(),
        grid.lat.max(),
        grid.lon.min(),
        grid.lon.max(),
        dist / 1000,
        lat0,
        lon0,
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
