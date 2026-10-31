# pylint: disable=too-many-lines
# cmip7_prep/mapping_compat.py
"""
Mapping loader and evaluator for CMIP7-style YAML variable mapping files.

Overview
--------
This module translates between native model variable names (e.g. CESM output)
and CMIP7-standard variable names by loading a YAML mapping file.  The main
entry points are:

* :class:`Mapping` – loads a mapping YAML and provides ``realize`` /
  ``realize_all`` methods that construct xarray DataArrays for a requested
  CMIP variable from a native model dataset.
* :class:`VarConfig` – immutable dataclass holding the normalized
  configuration for a single CMIP variable entry.
* :func:`_to_varconfig` – converts a raw YAML dict for one variable into a
  ``VarConfig``.

YAML format
-----------
The mapping file must have a top-level ``variables:`` dict.  Each entry
describes one CMIP variable::

    variables:
      tas:
        table: Amon
        units: K
        sources:
          - model_var: T2m
      pr:
        table: Amon
        units: kg m-2 s-1
        sources:
          - model_var: PRECC
          - model_var: PRECL
        formula: "PRECC + PRECL"

Optional per-source ``freq`` tags allow different source variables to be
selected at different output frequencies.  Optional ``variants:`` lists allow
one logical CMIP name to map to several distinct realizations (e.g. different
pressure levels).

All mapping access is via the :class:`VarConfig` / ``cfg`` dict structure;
the raw YAML dicts are not exposed publicly.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import as_file
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping as TMapping, Optional
import re
import warnings
import logging

import numpy as np
import xarray as xr
import yaml  # runtime dep

from cmip7_prep.schema import mapping_errors

logger = logging.getLogger(__name__)


# Static grid / cell-measure fields that live INSIDE the model output files
# (e.g. the CICE cell area 'tarea', or the T/U grid lat/lon), rather than being
# written as their own per-variable timeseries files.  When such a field appears
# as a formula source (e.g. siarea = sum(siconc * tarea)), it never has a
# matching '*.<name>.*.nc' file, so it must not be required by the standalone
# file-existence check -- it is read from whichever data file is opened.  This
# is the single home for that knowledge (see also the sftlf->landfrac /
# areacella->area special cases in _realize_core).
STATIC_MODEL_VARS = frozenset(
    {
        "tarea",
        "uarea",
        "narea",
        "earea",
        "TLAT",
        "TLON",
        "ULAT",
        "ULON",
        "NLAT",
        "NLON",
        "ELAT",
        "ELON",
        "area",
        "landfrac",
        "landmask",
        "tmask",
        "wet",
    }
)


def packaged_mapping_resource(filename: str):
    """Context manager yielding a real filesystem path to the packaged mapping file.
    Example:
        >>> with packaged_mapping_resource("noresm_to_cmip7_atmos.yaml") as p:
        ...     str(p).endswith("noresm_to_cmip7_atmos.yaml")
        True
    """

    res = Path(__file__).parent.parent.parent / "data" / filename
    return as_file(res)


def package_mapping_resource_from_full_path(path: str | Path):
    """Context manager yielding a real filesystem path to a mapping YAML file."""
    return as_file(Path(path))


# pylint: disable=too-many-instance-attributes
@dataclass(frozen=True)
class VarConfig:
    """Immutable, normalized configuration for a single CMIP variable mapping.

    Populated by :func:`_to_varconfig` from a raw YAML entry.  Exactly one of
    ``source`` (single direct variable) or ``raw_variables`` + optional
    ``formula`` (multi-variable or derived) will be set for any given entry.

    Attributes
    ----------
    name:
        CMIP variable name (e.g. ``"tas"``).
    table:
        MIP table short name (e.g. ``"Amon"``).  ``"CMIP7_"`` prefixes are
        stripped on load.
    units:
        Target CMIP units string (e.g. ``"K"``).
    raw_variables:
        List of native model variable names required when a formula is used or
        when more than one source variable contributes.  ``None`` when
        ``source`` is set instead.
    source:
        Single native variable name used for a direct (no-formula) mapping.
        ``None`` when ``raw_variables`` is set instead.
    formula:
        Python/xarray expression string evaluated by :func:`_safe_eval`.
        Variable names in the expression correspond to keys in
        ``source_aliases`` (or directly to ``raw_variables`` if no aliases
        are defined).
    unit_conversion:
        Either a string expression ``"x * factor"`` or a dict with optional
        ``scale`` and/or ``offset`` keys applied after variable realization.
    positive:
        CF ``positive`` attribute (``"up"`` or ``"down"``), if applicable.
    cell_methods:
        CF ``cell_methods`` string (e.g. ``"time: mean"``).
    levels:
        Vertical level specification dict (e.g. ``{name: plev19, units: Pa}``).
    regrid_method:
        Regridding method hint (e.g. ``"bilinear"``).
    long_name:
        Human-readable variable description.
    standard_name:
        CF standard name.
    grids:
        Whether the variable is regridded before it is written::

            gn   written on the grid the data arrives on
            gr   interpolated to the target latitude-longitude grid

        One entry per output, so ``["gn", "gr"]`` writes the variable twice,
        once each way.  Defaults to ``["gr"]``, which is what every variable
        without the key gets.

        A global sum takes ``gn``, because the formula has already collapsed
        the horizontal grid and there is nothing left to interpolate.

        This is not the published ``grid_label``.  That is the gNNN code in
        ``data/<model>_grids.yaml``, read per realm by
        :func:`cmip7_prep.cmor_utils.build_dataset_cfg`.
    source_aliases:
        Mapping of formula token → native model variable name.  Allows the
        formula to use short alias names that differ from the actual variable
        names in the dataset.
    region:
        Optional region selector string (e.g. ``"global"``).
    """

    name: str
    table: Optional[str] = None
    units: Optional[str] = None
    raw_variables: Optional[List[str]] = None
    source: Optional[str] = None
    formula: Optional[str] = None
    unit_conversion: Optional[Any] = None
    positive: Optional[str] = None
    cell_methods: Optional[str] = None
    levels: Optional[Dict[str, Any]] = None
    regrid_method: Optional[str] = None
    long_name: Optional[str] = None
    standard_name: Optional[str] = None
    grids: Optional[List[str]] = None
    source_aliases: Optional[Dict[str, str]] = None  # alias → model_var
    region: Optional[str] = None

    def as_cfg(self) -> Dict[str, Any]:
        """Return a plain dict view for convenience in other modules.
        Example:
            >>> vc = VarConfig(name="tas", table="Amon", units="K")
            >>> sorted(vc.as_cfg().items())
            [('name', 'tas'), ('table', 'Amon'), ('units', 'K')]
        """
        d = {
            "name": self.name,
            "table": self.table,
            "units": self.units,
            "raw_variables": self.raw_variables,
            "source": self.source,
            "formula": self.formula,
            "unit_conversion": self.unit_conversion,
            "positive": self.positive,
            "cell_methods": self.cell_methods,
            "levels": self.levels,
            "regrid_method": self.regrid_method,
            "long_name": self.long_name,
            "standard_name": self.standard_name,
            "grids": self.grids,
            "source_aliases": self.source_aliases,
            "region": self.region,
        }
        return {k: v for k, v in d.items() if v is not None}


# ── formula namespace ────────────────────────────────────────────────────────
# The functions below, plus ``np`` and ``xr``, are the complete set of names a
# mapping formula may call.  formula_names relies on FORMULA_NAMESPACE too:
# a name listed here is never taken for a native variable, so adding a
# function makes it callable and keeps it out of the file search in one edit.


def verticalsum(arr, capped_at=None, dim="levsoi"):
    """Sum a DataArray along a soil/vertical dimension, optionally capping it.

    Parameters
    ----------
    arr       : xr.DataArray to reduce; non-DataArray values are passed through
    capped_at : optional upper bound applied to the summed result
    dim       : name of the dimension to sum over
    """
    # arr can be a DataArray or an expression
    if isinstance(arr, xr.DataArray):
        summed = arr.sum(dim=dim, skipna=True)
    else:
        summed = arr  # fallback, should be DataArray
    if capped_at is not None:
        summed = xr.where(summed > capped_at, capped_at, summed)
    return summed


def sumover_index(arr: xr.DataArray, indexlist: list, dimname: str) -> xr.DataArray:
    """
    Sum a DataArray over a subset of indices along a named dimension.

    Indices are 1-based.  Passing a single index collapses the dimension,
    which is how a single level (e.g. surface or basal velocity) is picked
    out of a vertical coordinate.

    Parameters
    ----------
    arr       : xr.DataArray with the dimension to reduce over
    indexlist : list of 1-based integer indices to sum over
    dimname   : name of the dimension to select/squeeze
    """
    if not isinstance(arr, xr.DataArray):
        raise TypeError(f"Expected xr.DataArray, got {type(arr).__name__}")
    if dimname not in arr.dims:
        raise ValueError(
            f"Dimension '{dimname}' not found in array dimensions {list(arr.dims)}"
        )
    if not indexlist:
        raise ValueError("indexlist must not be empty")

    # Account for zero-based indexing
    indexlist = [x - 1 for x in indexlist]

    # Select only the specified indices before summing — this ensures
    # indices not in indexlist are excluded from the sum entirely.
    return arr.isel({dimname: indexlist}).sum(dim=dimname)


def verticalmean(arr: xr.DataArray, levelname: str = "level") -> xr.DataArray:
    """
    Sigma-weighted vertical mean over a non-uniformly spaced dimension.

    Reproduces the weighting CISM uses for its own ``uvel_mean`` (see
    glissade.F90): values sit on level interfaces, with sigma running 0 at
    the upper surface to 1 at the bed, and each interface is weighted by the
    fraction of the column it represents.  The interior weights are the gaps
    between layer midpoints; the two end levels get the outer half-layers.

    A plain ``arr.mean(dim=...)`` is NOT equivalent: CISM's sigma levels are
    refined towards the bed by default, so an unweighted mean over-weights
    the closely spaced basal levels.

    Parameters
    ----------
    arr       : xr.DataArray whose vertical coordinate holds sigma values
    levelname : name of the vertical dimension
    """
    if not isinstance(arr, xr.DataArray):
        raise TypeError(f"Expected xr.DataArray, got {type(arr).__name__}")
    if levelname not in arr.dims:
        raise ValueError(
            f"Dimension '{levelname}' not found in array dimensions "
            f"{list(arr.dims)}"
        )
    if levelname not in arr.coords:
        raise ValueError(
            f"Dimension '{levelname}' has no coordinate values; "
            f"verticalmean needs the sigma levels to weight by."
        )

    sigma = np.asarray(arr[levelname].values, dtype="f8")
    if sigma.size < 2:
        raise ValueError(f"verticalmean needs at least 2 levels, got {sigma.size}")

    stag = 0.5 * (sigma[:-1] + sigma[1:])  # layer midpoints
    w = np.empty(sigma.size, dtype="f8")
    w[0] = stag[0] - sigma[0]  # top half-layer
    w[1:-1] = stag[1:] - stag[:-1]
    w[-1] = sigma[-1] - stag[-1]  # bottom half-layer
    w /= w.sum()  # guard against a sigma range other than [0, 1]

    weights = xr.DataArray(w, dims=(levelname,), coords={levelname: arr[levelname]})
    return (arr * weights).sum(dim=levelname)


# The radius CAM uses for its own grid, so that a cell area given in
# steradians becomes the area in square metres that CAM itself would report.
EARTH_RADIUS_M = 6371220.0

# The cell-area units a model writes, as the factor that converts them to
# square metres.  CAM writes 'area' in steradians, CLM writes 'area' in km^2
# and CICE writes 'tarea' in m^2.
_AREA_TO_M2: Dict[str, float] = {
    "m2": 1.0,
    "m^2": 1.0,
    "km2": 1.0e6,
    "km^2": 1.0e6,
    "sr": EARTH_RADIUS_M**2,
    "steradian": EARTH_RADIUS_M**2,
    "steradians": EARTH_RADIUS_M**2,
}


def _area_in_m2(area: xr.DataArray, area_units: Optional[str], caller: str):
    """Return the cell area converted to square metres.

    The units come from `area_units` when it is given, and from the array's
    own `units` attribute otherwise.
    """
    if not isinstance(area, xr.DataArray):
        raise TypeError(
            f"{caller} expected xr.DataArray for area, got {type(area).__name__}"
        )
    units = area_units if area_units is not None else area.attrs.get("units")
    if units is None:
        raise ValueError(
            f"{caller}: the cell area {area.name!r} has no 'units' attribute. "
            f"Give its units as area_units, one of {sorted(_AREA_TO_M2)}."
        )
    key = str(units).strip().lower().replace(" ", "")
    if key not in _AREA_TO_M2:
        raise ValueError(
            f"{caller}: cell area units {units!r} are not recognised. "
            f"The units it accepts are {sorted(_AREA_TO_M2)}."
        )
    return area.astype("float64") * _AREA_TO_M2[key]


def global_2d_sum(
    arr: xr.DataArray, area: xr.DataArray, area_units: Optional[str] = None
) -> xr.DataArray:
    """Integrate a quantity given per unit area over the whole globe.

    The quantity is multiplied by the area of each cell in square metres and
    summed over the dimensions the cell area has.  A field in kg m-2 becomes
    a total in kg, which is how the CMIP7 tables define an ``-hm-`` variable
    such as ``co2mass_tavg-u-hm-u``.  Every other dimension is left alone, so
    the result keeps one value per time.

    Cells where the quantity is missing are left out of the sum.  Masking the
    quantity in the formula therefore restricts the integral, and
    ``global_2d_sum(imass.where(sftgif > 0), area)`` covers the ice sheet alone.

    The arithmetic is done in double precision.  Model output is single
    precision, and a total over tens of thousands of cells loses several
    digits if it is summed that way.

    Parameters
    ----------
    arr        : xr.DataArray holding a quantity per unit area
    area       : xr.DataArray holding the area of each cell
    area_units : the units of `area`, for an area that carries no `units`
                 attribute.  Scaling or masking the area inside the formula
                 drops that attribute, so an expression such as
                 ``area * landfrac`` has to name its units here.

    >>> import numpy as np
    >>> area = xr.DataArray(np.full(4, 0.25), dims=("ncol",),
    ...                     attrs={"units": "m2"})
    >>> field = xr.DataArray(np.ones((2, 4)), dims=("time", "ncol"))
    >>> global_2d_sum(field, area).values
    array([1., 1.])
    >>> masked = field.where(xr.DataArray([True, True, False, False],
    ...                                   dims=("ncol",)))
    >>> global_2d_sum(masked, area).values
    array([0.5, 0.5])
    """
    if not isinstance(arr, xr.DataArray):
        raise TypeError(
            f"global_2d_sum expected xr.DataArray, got {type(arr).__name__}"
        )
    weights = _area_in_m2(area, area_units, "global_2d_sum")

    missing = [dim for dim in area.dims if dim not in arr.dims]
    if missing:
        raise ValueError(
            f"global_2d_sum: the cell area has dimensions {list(area.dims)} and the "
            f"field has {list(arr.dims)}, so {missing} cannot be summed over."
        )

    return (arr.astype("float64") * weights).sum(dim=area.dims, skipna=True)


def global_2d_mean(
    arr: xr.DataArray, area: xr.DataArray, area_units: Optional[str] = None
) -> xr.DataArray:
    """Average a field over the globe, weighting each cell by its area.

    The weighted sum is divided by the area it was taken over, so the units
    are the field's own and do not change.  That is what the CMIP7 tables ask
    for from an ``-hm-`` variable whose cell method is ``area: mean``, such as
    ``cfc11_tavg-u-hm-air``.  Every other dimension is left alone.

    Cells where the field is missing are left out of both the sum and the
    area, so ``global_2d_mean(tas.where(sftgif > 0), area)`` is the mean over the
    ice sheet rather than the globe.

    The weighting is by area.  Where a table asks for a different weight, such
    as the moles of air in each sample, the two agree only for a field that
    holds the same value everywhere.

    Parameters
    ----------
    arr        : xr.DataArray holding the field to average
    area       : xr.DataArray holding the area of each cell
    area_units : the units of `area`, for an area that carries no `units`
                 attribute

    >>> import numpy as np
    >>> area = xr.DataArray(np.array([1.0, 3.0]), dims=("ncol",),
    ...                     attrs={"units": "m2"})
    >>> field = xr.DataArray(np.array([[0.0, 4.0]]), dims=("time", "ncol"))
    >>> global_2d_mean(field, area).values
    array([3.])
    >>> uniform = xr.DataArray(np.full((2, 2), 7.0), dims=("time", "ncol"))
    >>> global_2d_mean(uniform, area).values
    array([7., 7.])
    """
    if not isinstance(arr, xr.DataArray):
        raise TypeError(
            f"global_2d_mean expected xr.DataArray, got {type(arr).__name__}"
        )
    weights = _area_in_m2(area, area_units, "global_2d_mean")

    missing = [dim for dim in area.dims if dim not in arr.dims]
    if missing:
        raise ValueError(
            f"global_2d_mean: the cell area has dimensions {list(area.dims)} and the "
            f"field has {list(arr.dims)}, so {missing} cannot be averaged over."
        )

    values = arr.astype("float64")
    counted = weights.where(values.notnull())
    return (values * weights).sum(dim=area.dims, skipna=True) / counted.sum(
        dim=area.dims, skipna=True
    )


def global_3d_mean(
    arr: xr.DataArray,
    thickness: xr.DataArray,
    area: xr.DataArray,
    area_units: Optional[str] = None,
    levelname: str = "lev",
) -> xr.DataArray:
    """Average a field over the whole atmosphere, weighting by the air in each cell.

    The vertical and the horizontal are collapsed together, each cell counting
    in proportion to the air it holds.  That is what the CMIP7 tables ask for
    from an ``-hm-`` variable whose cell method reads ``height: area: time:
    mean (with all samples weighted by the number of moles of air in the
    sample)``, such as ``ch4_tavg-u-hm-air``.  Every other dimension is left
    alone, so the result keeps one value per time.

    The weight is ``thickness * area``.  For CAM that is ``PDELDRY``, the dry
    air pressure thickness of each layer, times the cell area.  The moles of
    air in a cell are that product divided by the gravitational acceleration
    and the molar mass of air, and both are constants that cancel between the
    numerator and the denominator.  Any weight proportional to the air mass
    therefore gives the same answer, and the units of `thickness` do not
    matter.

    Cells where the field is missing are left out of both sums, so the mean is
    taken over the cells that have a value.

    Parameters
    ----------
    arr        : xr.DataArray holding the field to average
    thickness  : xr.DataArray holding a per-cell quantity proportional to the
                 mass of air, such as CAM's `PDELDRY`
    area       : xr.DataArray holding the area of each cell
    area_units : the units of `area`, for an area that carries no `units`
                 attribute
    levelname  : name of the vertical dimension to collapse

    >>> import numpy as np
    >>> area = xr.DataArray(np.array([1.0]), dims=("ncol",),
    ...                     attrs={"units": "m2"})
    >>> thickness = xr.DataArray(np.array([[1.0], [3.0]]),
    ...                          dims=("lev", "ncol"))
    >>> field = xr.DataArray(np.array([[0.0], [4.0]]), dims=("lev", "ncol"))
    >>> float(global_3d_mean(field, thickness, area).values)
    3.0
    >>> uniform = xr.DataArray(np.full((2, 2, 1), 7.0),
    ...                        dims=("time", "lev", "ncol"))
    >>> thick2 = xr.DataArray(np.array([[1.0], [3.0]]), dims=("lev", "ncol"))
    >>> global_3d_mean(uniform, thick2, area).values
    array([7., 7.])
    """
    if not isinstance(arr, xr.DataArray):
        raise TypeError(
            f"global_3d_mean expected xr.DataArray, got {type(arr).__name__}"
        )
    if not isinstance(thickness, xr.DataArray):
        raise TypeError(
            f"global_3d_mean expected xr.DataArray for thickness, got "
            f"{type(thickness).__name__}"
        )
    weights = _area_in_m2(area, area_units, "global_3d_mean") * thickness.astype(
        "float64"
    )

    if levelname not in arr.dims:
        raise ValueError(
            f"global_3d_mean: the field has dimensions {list(arr.dims)}, which do not "
            f"include the vertical dimension {levelname!r}."
        )
    if levelname not in thickness.dims:
        raise ValueError(
            f"global_3d_mean: the layer thickness has dimensions {list(thickness.dims)}, "
            f"which do not include the vertical dimension {levelname!r}."
        )
    missing = [dim for dim in area.dims if dim not in arr.dims]
    if missing:
        raise ValueError(
            f"global_3d_mean: the cell area has dimensions {list(area.dims)} and the "
            f"field has {list(arr.dims)}, so {missing} cannot be averaged over."
        )

    collapse = (levelname,) + tuple(area.dims)
    values = arr.astype("float64")
    counted = weights.where(values.notnull())
    return (values * weights).sum(dim=collapse, skipna=True) / counted.sum(
        dim=collapse, skipna=True
    )


def layer_thickness(
    pint: xr.DataArray, levelname: str = "ilev", outname: str = "lev"
) -> xr.DataArray:
    """Return the pressure thickness of each layer from the interface pressures.

    CAM writes ``PINT``, the pressure on the layer interfaces, which has one
    more value along its vertical dimension than the layers themselves.  The
    thickness of a layer is the difference between the two interfaces that
    bound it, so the result is one shorter and sits on the layer dimension.

    The interface coordinate is dropped rather than renamed, because its
    values belong to the interfaces and not to the layers between them.  The
    result therefore broadcasts against a field on ``lev`` by dimension name.

    The difference is taken as a magnitude, so the answer does not depend on
    whether the interfaces run from the top of the atmosphere downwards or
    the other way.

    This is the total pressure, so the thickness it gives is the moist one.
    A field defined against dry air is weighted about 0.25% too heavily in
    the moist layers, which is the water vapour the dry thickness leaves out.

    Parameters
    ----------
    pint      : xr.DataArray holding the pressure on the layer interfaces
    levelname : name of the interface dimension
    outname   : name the layer dimension should take

    >>> import numpy as np
    >>> pint = xr.DataArray(np.array([0.0, 1.0, 3.0]), dims=("ilev",))
    >>> layer_thickness(pint).values
    array([1., 2.])
    >>> layer_thickness(pint).dims
    ('lev',)
    """
    if not isinstance(pint, xr.DataArray):
        raise TypeError(
            f"layer_thickness expected xr.DataArray, got {type(pint).__name__}"
        )
    if levelname not in pint.dims:
        raise ValueError(
            f"layer_thickness: the interface pressure has dimensions "
            f"{list(pint.dims)}, which do not include {levelname!r}."
        )
    if pint.sizes[levelname] < 2:
        raise ValueError(
            f"layer_thickness needs at least 2 interfaces, got "
            f"{pint.sizes[levelname]}."
        )

    thickness = abs(pint.astype("float64").diff(dim=levelname))
    thickness = thickness.rename({levelname: outname})
    return thickness.drop_vars(outname, errors="ignore")


FORMULA_NAMESPACE: Dict[str, Any] = {
    "np": np,
    "xr": xr,
    "verticalsum": verticalsum,
    "sumover_index": sumover_index,
    "verticalmean": verticalmean,
    "global_2d_sum": global_2d_sum,
    "global_2d_mean": global_2d_mean,
    "global_3d_mean": global_3d_mean,
    "layer_thickness": layer_thickness,
}


def _safe_eval(expr: str, local_names: Dict[str, Any]) -> Any:
    """Evaluate a formula string in a restricted namespace.

    Built-in Python names are blocked (``__builtins__`` is empty).  Everything
    in :data:`FORMULA_NAMESPACE` is available in addition to whatever is passed
    via ``local_names``; see the individual functions above for their
    signatures.

    Parameters
    ----------
    expr:
        Arithmetic or xarray expression string (e.g. ``"PRECC + PRECL"``).
    local_names:
        Dict mapping variable token names to DataArrays (or other values).

    Returns
    -------
    Any
        Result of evaluating the expression — typically an
        ``xr.DataArray``.

    Example:
        >>> _safe_eval("x + 2", {"x": 3})
        5
        >>> import numpy as np
        >>> float(_safe_eval("np.mean(x)", {"x": [1, 2, 3]}))
        2.0
    """
    safe_globals = {"__builtins__": {}}

    safe_locals = local_names.copy()
    safe_locals.update(FORMULA_NAMESPACE)
    # pylint: disable=eval-used
    return eval(expr, safe_globals, safe_locals)


# ── minimal mapping form ─────────────────────────────────────────────────────
# Mapping files (schema: data/schemas/mapping.schema.yaml) hold a formula and
# only what cannot be looked up elsewhere.  expand_entry spells such an entry
# out in the form the rest of this module works with; the CMOR table supplies
# table, units and levels (Mapping._add_table_fields).

INTENSIVE_VARS_YAML = (
    Path(__file__).parent.parent.parent / "data" / "intensive_vars.yaml"
)

_IDENTIFIER = re.compile(r"\b[A-Za-z_]\w*\b")

# Vertical axes in a CMOR table's 'dimensions', as the level names the pipeline
# and writer dispatch on.  plevNN keeps its own name.
_HYBRID_LEVELS = {
    "alevel": "standard_hybrid_sigma",
    "alevhalf": "standard_hybrid_sigma_half",
}


@lru_cache(maxsize=None)
def load_intensive_vars() -> frozenset:
    """Return the root names that should be regridded bilinearly."""
    if not INTENSIVE_VARS_YAML.is_file():
        return frozenset()
    with open(INTENSIVE_VARS_YAML, encoding="utf-8") as handle:
        return frozenset((yaml.safe_load(handle) or {}).get("intensive") or ())


def formula_names(formula: str) -> List[str]:
    """Return the native variables a formula reads, sorted.

    That is every name in the expression except the functions it calls and
    the names in :data:`FORMULA_NAMESPACE`.

    >>> formula_names("(PRECC + PRECL) * 1.e6")
    ['PRECC', 'PRECL']
    >>> formula_names("verticalsum(SOILICE, capped_at=5000)")
    ['SOILICE']
    >>> formula_names("(siconc * tarea).where(siconc.coords['TLAT'] > 0).sum(dim=['nj'])")
    ['siconc', 'tarea']
    """
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"formula is not a valid expression: {formula!r}") from exc
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    return sorted(names - called - set(FORMULA_NAMESPACE))


def expand_entry(name: str, entry: Any, grids: List[str]) -> Any:
    """Spell a mapping entry out with ``sources``, the form the loader works with.

    A formula written per frequency becomes frequency-tagged sources; where the
    frequencies share one formula, the other frequencies' variable names become
    aliases of the monthly (or first) frequency's.  Entries that already list
    ``sources`` (the previous file format) are returned unchanged.

    >>> expand_entry("tas_tavg-h2m-hxy-u", "TREFHT", ["gr"])
    {'grids': ['gr'], 'regrid_method': 'bilinear', 'sources': [{'model_var': 'TREFHT'}]}
    >>> expand_entry("siu", {"formula": {"day": "siu_d", "mon": "siu"}}, ["gn"])["sources"]
    [{'model_var': 'siu_d', 'freq': 'day'}, {'model_var': 'siu', 'freq': 'mon'}]
    >>> e = expand_entry("x", {"formula": {"day": "A_d * B", "mon": "A * B"}}, ["gr"])
    >>> e["formula"]
    'A * B'
    >>> e["sources"][0]
    {'model_var': 'A_d', 'freq': 'day', 'alias': 'A'}
    """
    if isinstance(entry, str):
        entry = {"formula": entry}
    if not isinstance(entry, dict) or "sources" in entry:
        return entry
    per_freq = (entry.get("variants") or [entry])[0]["formula"]
    if not isinstance(per_freq, dict):
        per_freq = {None: per_freq}
    main_freq = "mon" if "mon" in per_freq else next(iter(per_freq))
    main = per_freq[main_freq]

    # A plain rename at every frequency needs no formula.
    rename = not entry.get("variants") and all(
        _IDENTIFIER.fullmatch(text.strip()) for text in per_freq.values()
    )
    if rename:
        sources = [
            (
                {"model_var": text.strip(), "freq": freq}
                if freq
                else {"model_var": text.strip()}
            )
            for freq, text in per_freq.items()
        ]
    else:
        sources, aliased = [], set()
        for freq, text in per_freq.items():
            if freq == main_freq:
                continue
            if _IDENTIFIER.sub("_", text) != _IDENTIFIER.sub("_", main):
                raise ValueError(
                    f"{name}: per-frequency formulas may differ only in variable names"
                )
            for model_var, alias in zip(
                _IDENTIFIER.findall(text), _IDENTIFIER.findall(main)
            ):
                source = {"model_var": model_var, "freq": freq, "alias": alias}
                if model_var != alias and source not in sources:
                    sources.append(source)
                    aliased.add(alias)
        try:
            names = formula_names(main)
        except ValueError:
            # Not valid Python: keep the file loadable; evaluating it will fail.
            names = [
                n
                for n in dict.fromkeys(_IDENTIFIER.findall(main))
                if n not in FORMULA_NAMESPACE
            ]
        for model_var in names:
            source = {"model_var": model_var}
            if model_var in aliased:
                source["freq"] = main_freq
            sources.append(source)

    expanded = {k: v for k, v in entry.items() if k not in ("formula", "variants")}
    expanded.setdefault("grids", list(grids))
    expanded.setdefault(
        "regrid_method",
        "bilinear" if name.split("_")[0] in load_intensive_vars() else "conservative",
    )
    expanded["sources"] = sources
    if entry.get("variants"):
        expanded["variants"] = [
            {
                **variant,
                "formula": (
                    variant["formula"][main_freq]
                    if isinstance(variant["formula"], dict)
                    else variant["formula"]
                ),
            }
            for variant in entry["variants"]
        ]
    elif not rename:
        expanded["formula"] = main
    return expanded


@lru_cache(maxsize=None)
def _table_entries(tables_root: str, table: str) -> Dict[str, dict]:
    """Return the variable_entry of ``<tables_root>/tables/CMIP7_<table>.json``, or {}."""
    path = Path(tables_root) / "tables" / f"CMIP7_{table}.json"
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle).get("variable_entry") or {}


def _levels(table_entry: dict) -> Optional[Dict[str, str]]:
    """Return the levels the pipeline needs, from a table entry's dimensions.

    >>> _levels({"dimensions": "longitude latitude plev19 time"})
    {'name': 'plev19'}
    >>> _levels({"dimensions": ["longitude", "latitude", "alevel", "time"]})
    {'name': 'standard_hybrid_sigma'}
    >>> _levels({"dimensions": "longitude latitude time"}) is None
    True
    """
    dims = table_entry.get("dimensions") or []
    if isinstance(dims, str):
        dims = dims.split()
    for dim in dims:
        if dim.startswith("plev") or dim in _HYBRID_LEVELS:
            return {"name": _HYBRID_LEVELS.get(dim, dim)}
    return None


class Mapping:
    """Load and evaluate a CMIP7 variable mapping YAML file.

    The YAML file must have a top-level ``variables:`` dict (see module
    docstring for format details).  On load, each entry is parsed into a
    :class:`VarConfig`.  Entries with a ``variants:`` list are expanded into
    numbered internal keys (``<name>:0``, ``<name>:1``, …); the bare name
    resolves to the first variant for backward compatibility.

    Parameters
    ----------
    path:
        Path to the YAML mapping file.

    Attributes
    ----------
    default_freq:
        If set, used as the fallback frequency when ``freq`` is not supplied
        to :meth:`realize` / :meth:`realize_all` / :meth:`get_cfg`.

    Notes
    -----
    Table names are normalized by stripping the ``"CMIP7_"`` prefix (e.g.
    ``"CMIP7_Amon"`` → ``"Amon"``).

    Use :meth:`from_packaged_default` to load the bundled mapping file without
    specifying an explicit path.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.default_freq: Optional[str] = None
        # Set by cmor_driver.py, so configs get table, units and levels from
        # the CMOR table (see _add_table_fields).
        self.tables_root: Optional[Path] = None
        self.realm: Optional[str] = None
        self._vars, self._raw, self._variant_keys = self._load_yaml(self.path)

    @classmethod
    def from_packaged_default(cls, filename: str) -> "Mapping":
        """Construct a Mapping from a packaged realm-specific YAML file."""
        logger.info("mapping file is %s", filename)
        with packaged_mapping_resource(filename) as p:
            return cls(p)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Mapping":
        """Construct a Mapping from a YAML file path."""
        logger.info("mapping file is %s", path)
        with package_mapping_resource_from_full_path(path) as p:
            return cls(p)

    # -----------------
    # Loading
    # -----------------
    @staticmethod
    def _load_yaml(path: Path):
        """Load CMIP7 YAML and return (vars_dict, raw_dict, variant_keys)."""
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if (
            not isinstance(data, dict)
            or "variables" not in data
            or not isinstance(data["variables"], dict)
        ):
            raise TypeError(
                "Unsupported YAML structure: expected top-level 'variables:' dict."
            )

        # Warn rather than raise, so one malformed entry does not stop a run.
        for message in mapping_errors(data):
            logger.warning("%s does not match the mapping schema: %s", path, message)
        grids = data.get("grids") or ["gr"]
        data = {
            name: expand_entry(name, cfg, grids)
            for name, cfg in data["variables"].items()
        }
        result: Dict[str, VarConfig] = {}
        raw: Dict[str, Any] = {}
        variant_keys: Dict[str, List[str]] = {}
        for name, cfg in data.items():
            if not isinstance(cfg, dict):
                raise TypeError("Each entry in the yaml file must be a dictionary.")
            variants = cfg.get("variants")
            if variants:
                # Expand into N internal entries keyed as "<name>:0", "<name>:1", etc.
                base_cfg = {k: v for k, v in cfg.items() if k != "variants"}
                internal_keys = []
                for i, variant in enumerate(variants):
                    internal_key = f"{name}:{i}"
                    variant_cfg = {**base_cfg, **variant}
                    raw[internal_key] = variant_cfg
                    result[internal_key] = _to_varconfig(name, variant_cfg)
                    internal_keys.append(internal_key)
                variant_keys[name] = internal_keys
                # Point base name to first variant for backward compat
                result[name] = result[f"{name}:0"]
                raw[name] = raw[f"{name}:0"]
            else:
                raw[name] = cfg
                result[name] = _to_varconfig(name, cfg)
                variant_keys[name] = [name]
        return result, raw, variant_keys

    # -----------------
    # Public API
    # -----------------
    def get_cfg(self, cmip_name: str, freq: Optional[str] = None) -> Dict[str, Any]:
        """Return the normalized config dict for a CMIP variable name.

        Parameters
        ----------
        cmip_name:
            Key into the mapping (e.g. 'tas').
        freq:
            Output frequency token (e.g. 'mon', 'day').
        """
        if cmip_name not in self._vars:
            raise KeyError(f"No mapping for {str(cmip_name)} in {self.path}")
        effective_freq = freq if freq is not None else self.default_freq
        raw = self._raw.get(cmip_name)
        if effective_freq is not None and raw is not None:
            return self._add_table_fields(
                _to_varconfig(cmip_name, raw, freq=effective_freq).as_cfg()
            )
        return self._add_table_fields(self._vars[cmip_name].as_cfg())

    def _add_table_fields(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Fill table, units and levels from the CMOR table, unless cfg sets them.

        Needs tables_root and realm; without them cfg is returned unchanged.
        """
        if self.tables_root is None:
            return cfg
        entry = _table_entries(str(self.tables_root), self.realm).get(cfg["name"], {})
        cfg.setdefault("table", self.realm)
        if "units" in entry:
            cfg.setdefault("units", entry["units"])
        levels = _levels(entry)
        if levels:
            cfg.setdefault("levels", levels)
        return cfg

    def timeseries_source_vars(
        self, cmip_name: str, freq: Optional[str] = None
    ) -> List[str]:
        """Source model vars that must exist as their own timeseries files.

        Excludes static grid / cell-measure fields (see
        :data:`STATIC_MODEL_VARS`) that are read from inside the data files
        rather than written as standalone ``*.<name>.*.nc`` timeseries.  Used by
        the pipeline's source-file existence check so that a formula source such
        as ``tarea`` (in ``siarea = sum(siconc * tarea)``) does not cause the
        variable to be skipped for want of a nonexistent ``*.tarea.*.nc`` file.
        """
        try:
            cfg = self.get_cfg(cmip_name, freq) or {}
        except KeyError:
            return []
        if cfg.get("source"):
            src_vars = [cfg["source"]]
        else:
            src_vars = list(cfg.get("raw_variables") or [])
        return [v for v in src_vars if v not in STATIC_MODEL_VARS]

    def iter_variable_names(self, freq: Optional[str] = None) -> List[str]:
        """Return top-level mapping variable names in YAML order.

        When ``freq`` is provided, variables whose sources are only tagged for
        other frequencies are excluded. Untagged sources remain eligible for
        all frequencies.
        """
        effective_freq = freq if freq is not None else self.default_freq
        names = []
        for cmip_name, internal_keys in self._variant_keys.items():
            if any(
                _mapping_entry_supports_frequency(self._raw.get(key), effective_freq)
                for key in internal_keys
            ):
                names.append(cmip_name)
        return names

    def realize(
        self, ds: xr.Dataset, cmip_name: str, freq: Optional[str] = None
    ) -> xr.DataArray:
        """Construct a CMIP variable as an xarray.DataArray from a native dataset.
        Only supports CMIP7 YAML config.
        """
        if cmip_name not in self._vars:
            warnings.warn(
                f"[mapping] source variable {cmip_name} not found in dataset — skipping",
                RuntimeWarning,
                stacklevel=1,
            )
            return None
        effective_freq = freq if freq is not None else self.default_freq
        raw = self._raw.get(cmip_name)
        if effective_freq is not None and raw is not None:
            vc = _to_varconfig(cmip_name, raw, freq=effective_freq)
        else:
            vc = self._vars[cmip_name]
        da = _realize_core(ds, vc)
        if da is not None:
            if vc.units:
                da.attrs["units"] = vc.units
            if vc.long_name:
                da.attrs.setdefault("long_name", vc.long_name)
            if vc.standard_name:
                da.attrs.setdefault("standard_name", vc.standard_name)

        return da

    def realize_all(
        self, ds: xr.Dataset, cmip_name: str, freq: Optional[str] = None
    ) -> List[tuple]:
        """Realize all variants of a CMIP variable.

        For non-variant entries returns a single-element list.
        For entries with a ``variants:`` key in the YAML, returns one
        ``(DataArray, cfg_dict)`` tuple per variant.

        Parameters
        ----------
        ds:
            Native model dataset.
        cmip_name:
            Branded variable name (top-level YAML key).
        freq:
            Output frequency token (e.g. 'mon', 'day').
        """
        if cmip_name not in self._vars:
            # One channel for both lines: warnings.warn goes to stderr and is
            # deduplicated per call site, so it would print once however many
            # variables are unmapped, while the separator printed every time.
            logger.warning("=" * 60)
            logger.warning(
                "[mapping] no mapping found for CMIP variable %s - skipping",
                cmip_name,
            )
            return []
        effective_freq = freq if freq is not None else self.default_freq
        results = []
        for internal_key in self._variant_keys.get(cmip_name, [cmip_name]):
            raw_entry = self._raw.get(internal_key)
            if effective_freq is not None and raw_entry is not None:
                vc = _to_varconfig(cmip_name, raw_entry, freq=effective_freq)
            else:
                vc = self._vars.get(internal_key, self._vars[cmip_name])
            logger.info(
                "realize_all: realizing variant '%s' with config: %s",
                internal_key,
                vc.as_cfg(),
            )
            da = _realize_core(ds, vc)
            if da is not None:
                if vc.units:
                    da.attrs["units"] = vc.units
                if vc.long_name:
                    da.attrs.setdefault("long_name", vc.long_name)
                if vc.standard_name:
                    da.attrs.setdefault("standard_name", vc.standard_name)
                results.append((da, self._add_table_fields(vc.as_cfg())))
        return results

    # No public access to _vars or _raw outside this file.


# -----------------
# Private routines
# -----------------
def _filter_sources(sources: List[Any], freq: Optional[str]) -> List[Any]:
    """Return the subset of source entries that match *freq*.

    If no entry carries a ``freq`` tag, all sources are returned unchanged.
    When a freq is requested and entries match, matched entries are returned
    together with any untagged entries (which are always included).
    When a freq is requested but no entry matches, untagged entries are used
    as a fallback; if there are none either, the full list is returned.

    Example:
        >>> srcs = [{'model_var': 'siu_d', 'freq': 'day'}, {'model_var': 'siu', 'freq': 'mon'}]
        >>> [s['model_var'] for s in _filter_sources(srcs, 'mon')]
        ['siu']
        >>> [s['model_var'] for s in _filter_sources(srcs, 'day')]
        ['siu_d']
        >>> srcs2 = [{'model_var': 'T2m'}]
        >>> _filter_sources(srcs2, 'mon')
        [{'model_var': 'T2m'}]
        >>> srcs3 = [{'model_var': 'v_d', 'freq': 'day'}, {'model_var': 'v', 'freq': 'mon'},
        ...          {'model_var': 'area'}]
        >>> [s['model_var'] for s in _filter_sources(srcs3, 'day')]
        ['v_d', 'area']
    """
    if not sources:
        return sources
    has_tags = any(isinstance(s, dict) and "freq" in s for s in sources)
    if not has_tags or freq is None:
        return sources
    matched = [s for s in sources if isinstance(s, dict) and s.get("freq") == freq]
    untagged = [s for s in sources if isinstance(s, dict) and "freq" not in s]
    if matched:
        return matched + untagged
    return untagged if untagged else sources


def _mapping_entry_supports_frequency(
    cfg: Optional[TMapping[str, Any]], freq: Optional[str]
) -> bool:
    """Return whether a raw mapping entry should be considered for *freq*.

    Entries with untagged sources are always eligible. Entries with only
    tagged sources are eligible only when at least one source matches *freq*.
    """
    if cfg is None or freq is None:
        return True
    sources = cfg.get("sources")
    if not isinstance(sources, list) or not sources:
        return True

    has_tagged_sources = False
    for item in sources:
        if not isinstance(item, dict):
            return True
        if "freq" not in item:
            return True
        has_tagged_sources = True
        if item.get("freq") == freq:
            return True
    return not has_tagged_sources


def _to_varconfig(
    name: str, cfg: TMapping[str, Any], freq: Optional[str] = None
) -> VarConfig:
    """Normalize a raw CMIP7 YAML entry dict into a :class:`VarConfig`.

    The ``sources`` list is required.  When ``freq`` is provided, only source
    entries whose ``freq`` tag matches (plus any untagged entries) are kept;
    see :func:`_filter_sources` for the full selection rules.

    If exactly one source remains after filtering and no formula is present,
    the result uses ``source`` (direct variable) rather than
    ``raw_variables``.  Scale factors defined on individual source entries are
    promoted to ``unit_conversion``; explicit ``unit_conversion`` keys in
    ``cfg`` take precedence.

    Example:
        >>> vc2 = _to_varconfig(
        ...     "pr",
        ...     {
        ...         "sources": [
        ...             {"model_var": "PRECC"},
        ...             {"model_var": "PRECL"}
        ...         ],
        ...         "formula": "PRECC + PRECL"
        ...     }
        ... )
        >>> vc2.raw_variables
        ['PRECC', 'PRECL']
        >>> vc3 = _to_varconfig("tas", {"sources": [{"model_var": "T2m", "scale": 1.0}]})
        >>> vc3.source
        'T2m'
        >>> srcs = [{"model_var": "siu_d", "freq": "day"}, {"model_var": "siu", "freq": "mon"}]
        >>> _to_varconfig("siu", {"sources": srcs}, freq="day").source
        'siu_d'
        >>> _to_varconfig("siu", {"sources": srcs}, freq="mon").source
        'siu'
    """
    # Determine table, normalize for test expectations
    table = cfg.get("table")
    if table and table.startswith("CMIP7_"):
        table_norm = table.replace("CMIP7_", "")
    else:
        table_norm = table

    # Only support CMIP7 'sources' key
    raw_from_sources: Optional[List[str]] = None
    formula = cfg.get("formula")
    source = None

    if "sources" not in cfg:
        raise ValueError("CMIP7 mapping entry must have a 'sources' key.")
    active = _filter_sources(cfg["sources"], freq)
    raw_from_sources = []
    aliases: Dict[str, str] = {}
    for item in active:
        if isinstance(item, dict) and "model_var" in item:
            mv = str(item["model_var"])
            raw_from_sources.append(mv)
            alias = str(item.get("alias", mv))
            aliases[alias] = mv

        elif isinstance(item, str):
            raw_from_sources.append(item)
            aliases[item] = item
    if not raw_from_sources:
        raise ValueError(f"raw_from_sources does not contain any values for {name}")
    if raw_from_sources and len(raw_from_sources) == 1 and not formula:
        source = raw_from_sources[0]
        raw_from_sources = None
        aliases = {}

    unit_conversion = cfg.get("unit_conversion")

    vc = VarConfig(
        name=name,
        table=table_norm,
        units=cfg.get("units"),
        raw_variables=raw_from_sources,
        source=source,
        formula=formula,
        unit_conversion=unit_conversion,
        positive=cfg.get("positive"),
        cell_methods=cfg.get("cell_methods"),
        levels=cfg.get("levels"),
        regrid_method=cfg.get("regrid_method"),
        long_name=cfg.get("long_name"),
        standard_name=cfg.get("standard_name"),
        grids=cfg.get("grids"),
        source_aliases=aliases if aliases else None,
        region=cfg.get("region"),
    )
    return vc


def _require_vars(
    ds: xr.Dataset, names: List[str], context: str
) -> Dict[str, xr.DataArray]:
    """Return a dict of DataArrays for the requested names, raising KeyError if any are missing.

    >>> import xarray as xr
    >>> ds = xr.Dataset({"a": xr.DataArray([1]), "b": xr.DataArray([2])})
    >>> sorted(_require_vars(ds, ["a", "b"], "ctx").keys())
    ['a', 'b']
    >>> _require_vars(ds, ["c"], "ctx")
    Traceback (most recent call last):
        ...
    KeyError: "ctx: missing variables ['c']"
    """
    missing = [n for n in names if n not in ds]
    if missing:
        raise KeyError(f"{context}: missing variables {missing}")
    return {n: ds[n] for n in names}


def _realize_core(ds: xr.Dataset, vc: VarConfig) -> xr.DataArray:
    """Extract or compute a DataArray from *ds* according to *vc*.

    Resolution priority:

    1. **Direct source** (``vc.source`` set) – returns ``ds[vc.source]``
       unchanged.
    2. **Single raw variable, no formula** – equivalent to direct source but
       derived from ``vc.raw_variables``.
    3. **Formula** – evaluates ``vc.formula`` via :func:`_safe_eval` using
       either ``vc.source_aliases`` (alias → native var) or
       ``vc.raw_variables`` as the variable namespace.

    Special cases: ``sftlf`` always reads ``landfrac``; ``areacella`` always
    reads ``area``, regardless of what ``raw_variables`` contains.

    Unit conversion and attribute assignment are handled by the caller
    (:meth:`Mapping.realize` / :meth:`Mapping.realize_all`).
    """
    logger.debug("Realizing variable '%s' with config: %s", vc.name, vc.as_cfg())
    if vc.source:
        if vc.source not in ds:
            raise KeyError(f"source variable {vc.source!r} not found in dataset")
        return ds[vc.source]

    if vc.name == "sftlf":
        model_vars = ["landfrac"]
    elif vc.name == "areacella":
        model_vars = ["area"]
    else:
        model_vars = vc.raw_variables
    logger.info("Required model vars to realize are %s", model_vars)

    # Identity mapping from a single raw variable
    if model_vars and vc.formula in (None, "", "null") and len(model_vars) == 1:
        var = model_vars[0]
        if var not in ds:
            raise KeyError(f"raw variable {var!r} not found in dataset")
        return ds[var]

    # Identity mapping from a formula
    if vc.formula:
        logger.debug("formula is %s", vc.formula)
        logger.debug("source aliases are %s", vc.source_aliases.items())

        # Determine da_dict for formula
        if vc.source_aliases:
            # Build da_dict using alias → model_var mapping
            missing = [mv for mv in vc.source_aliases.values() if mv not in ds]
            if missing:
                raise KeyError(f"realize({vc.name}): missing variables {missing}")
            da_dict = {alias: ds[mv] for alias, mv in vc.source_aliases.items()}
        elif model_vars:
            # Obtain a dictionary of DataArrays for the requested names
            da_dict = _require_vars(ds, model_vars, f"realize({vc.name})")
        else:
            raise ValueError(f"formula given for {vc.name} but no raw_variables listed")

        # Apply formula
        try:
            logger.info(
                "Applying formula: %s for name %s with model_vars %s",
                vc.formula,
                vc.name,
                model_vars,
            )
            result = _safe_eval(vc.formula, da_dict)

        except Exception as exc:
            raise ValueError(f"Error evaluating formula for {vc.name}: {exc}") from exc
        if not isinstance(result, xr.DataArray):
            raise ValueError(f"Formula for {vc.name} did not produce a DataArray")
        logger.debug(
            "Successfully evaluated formula for %s, result dims: %s",
            vc.name,
            result.dims,
        )
        return result

    raise ValueError(
        f"Mapping for {vc.name} is incomplete: set 'source' or 'raw_variables' or 'formula'."
    )
