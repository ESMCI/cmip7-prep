# cmip7_prep/regrid.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple

import xarray as xr
import xesmf as xe
import numpy as np

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

# Default map files (can be overridden per call or via env vars)
DEFAULT_CONS_MAP = Path(
    # CESM area-avg conservative map (your existing file)
    "/glade/campaign/cesm/cesmdata/inputdata/cpl/gridmaps/ne30pg3/map_ne30pg3_to_1x1d_aave.nc"
)
# Optional bilinear map for intensive vars (set if you have one)
DEFAULT_BILIN_MAP = Path(
    # e.g., "map_ne30pg3_to_1x1d_bilin.nc"
    "/glade/campaign/cesm/cesmdata/inputdata/cpl/gridmaps/ne30pg3/map_ne30pg3_to_1x1d_bilin.nc"
)

# Variables that we consider "intensive" and prefer bilinear.
# Everything else defaults to conservative_normed / area-avg.
INTENSIVE_VARS = {
    # 2D near-surface
    "tas", "tasmin", "tasmax", "psl", "ps", "huss", "uas", "vas", "sfcWind",
    "ts", "prsn", "clt",
    # 3D state on lev/plev
    "ta", "ua", "va", "zg", "hus", "thetao", "uo", "vo", "so",
}

# --------------------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class MapSpec:
    method_label: str            # "conservative" or "bilinear" (label only)
    path: Path

class _RegridderCache:
    """Cache xESMF Regridders built from precomputed weight files."""
    _cache: Dict[Path, xe.Regridder] = {}

    @classmethod
    def get(cls, mapfile: Path, method_label: str) -> xe.Regridder:
        mapfile = mapfile.expanduser().resolve()
        if mapfile not in cls._cache:
            if not mapfile.exists():
                raise FileNotFoundError(f"Regrid weights not found: {mapfile}")
            # With filename=..., xESMF ignores in/out grids and method; it reads from map.
            cls._cache[mapfile] = xe.Regridder(
                xr.Dataset(), xr.Dataset(),
                method=method_label,
                filename=str(mapfile),
                reuse_weights=True,
            )
        return cls._cache[mapfile]

def _pick_maps(
    varname: str,
    conservative_map: Optional[Path] = None,
    bilinear_map: Optional[Path] = None,
    force_method: Optional[str] = None,  # "conservative" | "bilinear"
) -> MapSpec:
    """Decide which weights file to use for this variable."""
    cons = Path(conservative_map) if conservative_map else DEFAULT_CONS_MAP
    bilin = Path(bilinear

