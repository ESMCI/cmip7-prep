"""ESMF regrid weight files, loaded from the packaged YAML tables.

Each supported model has a ``data/<model>_regrid_maps.yaml`` table mapping a
resolution to its weight files::

    inputdata_dir: /nird/datalake/NS9560K/diagnostics/land_xesmf_diag_data/

    resolutions:
      ne16:
        conservative: map_ne16pg3_to_2x2_aave_c260531.nc
        bilinear: map_ne16pg3_to_2x2_blin_c260531.nc

``inputdata_dir`` is per model because the roots live on different machines.

The resolution key is the ``--resolution`` value ('ne16'), not the grid name
the weight file is built on ('ne16pg3').

An unknown resolution raises rather than falling back to another grid: the
previous fallback silently regridded to the ocean map, which produces output
that looks fine and is on the wrong grid.

``data/intensive_vars.yaml`` lists the variables that take the bilinear map.
It is shared by every model, since whether a quantity is intensive is a
property of the variable rather than of the model that wrote it.
"""

from __future__ import annotations

import copy
import logging
from functools import lru_cache
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"


@lru_cache(maxsize=None)
def _load_cached(model: str) -> dict:
    """Read and cache one model's regrid-map table."""
    path = DATA_DIR / f"{model}_regrid_maps.yaml"
    if not path.is_file():
        raise ValueError(f"No regrid-map table for model={model!r}; expected {path}")
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_regrid_maps(model: str) -> dict:
    """Return the whole table for ``model``.

    A copy is returned, so callers may not corrupt the cached table.
    """
    return copy.deepcopy(_load_cached(model))


def get_map_paths(model: str, resolution: str) -> dict[str, Path]:
    """Return the weight files for one model and resolution.

    The result has a 'conservative' key and, if the table defines one, a
    'bilinear' key.  Both are absolute paths.
    """
    table = _load_cached(model)
    resolutions = table.get("resolutions") or {}
    if resolution not in resolutions:
        raise ValueError(
            f"No regrid maps defined for model={model}, resolution={resolution}; "
            f"available: {sorted(resolutions)}"
        )

    root = Path(table.get("inputdata_dir", ""))
    entry = resolutions[resolution] or {}
    if "conservative" not in entry:
        raise ValueError(
            f"No conservative map for model={model}, resolution={resolution}; "
            "every resolution needs one, since fx fields are always conservative"
        )
    return {method: root / name for method, name in entry.items()}


@lru_cache(maxsize=None)
def load_intensive_vars() -> frozenset[str]:
    """Return the variables regridded bilinearly rather than conservatively.

    Shared across models, unlike the per-model map tables.
    """
    path = DATA_DIR / "intensive_vars.yaml"
    if not path.is_file():
        raise ValueError(f"No intensive-variable table; expected {path}")
    with open(path, encoding="utf-8") as handle:
        table = yaml.safe_load(handle) or {}
    return frozenset(table.get("intensive") or ())
