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

from dataclasses import dataclass
from importlib.resources import as_file
from pathlib import Path
from typing import Any, Dict, List, Mapping as TMapping, Optional
import warnings
import logging
import sys

import numpy as np
import xarray as xr
import yaml  # runtime dep

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def packaged_mapping_resource(filename: str = "cesm_to_cmip7.yaml"):
    """Context manager yielding a real filesystem path to the packaged mapping file.
    Example:
        >>> with packaged_mapping_resource("cesm_to_cmip7.yaml") as p:
        ...     str(p).endswith("cesm_to_cmip7.yaml")
        True
    """

    res = Path(__file__).parent.parent.parent / "data" / filename
    return as_file(res)


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
    dims:
        Expected output dimension list (e.g. ``["time", "lat", "lon"]``).
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
    dims: Optional[List[str]] = None
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
            "dims": self.dims,
            "source_aliases": self.source_aliases,
            "region": self.region,
        }
        return {k: v for k, v in d.items() if v is not None}


def _safe_eval(expr: str, local_names: Dict[str, Any]) -> Any:
    """Evaluate a formula string in a restricted namespace.

    Built-in Python names are blocked (``__builtins__`` is empty).  The
    following names are always available in addition to whatever is passed via
    ``local_names``:

    * ``np`` – NumPy
    * ``xr`` – xarray
    * ``verticalsum(arr, capped_at=None, dim="levsoi")`` – sum a DataArray
      along a soil/vertical dimension, optionally capping the result.
    * ``sumoverpft(arr, pftlist, dimname)`` – sum a DataArray over a subset
      of PFT indices along a named dimension.

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

    # Add custom formula functions here
    def verticalsum(arr, capped_at=None, dim="levsoi"):
        # arr can be a DataArray or an expression
        if isinstance(arr, xr.DataArray):
            summed = arr.sum(dim=dim, skipna=True)
        else:
            summed = arr  # fallback, should be DataArray
        if capped_at is not None:
            summed = xr.where(summed > capped_at, capped_at, summed)
        return summed


    def sumoverpft(arr: xr.DataArray, pftlist: list, dimname: str) -> xr.DataArray:
        """
        Sum a DataArray over a subset of PFT indices along a named dimension.

        Parameters
        ----------
        arr     : xr.DataArray with a PFT dimension
        pftlist : list of integer PFT indices to sum over
        dimname : name of the PFT dimension to select/squeeze
        """
        if not isinstance(arr, xr.DataArray):
            raise TypeError(f"Expected xr.DataArray, got {type(arr).__name__}")
        if dimname not in arr.dims:
            raise ValueError(f"Dimension '{dimname}' not found in array dimensions {list(arr.dims)}")
        if not pftlist:
            raise ValueError("pftlist must not be empty")

        logger.info(f" i am here")

        # Account for zero-based indexing
        pftlist = [x-1 for x in pftlist]

        # Select only the specified PFT indices before summing —
        # this ensures indices not in pftlist are excluded from the sum entirely.
        return arr.isel({dimname: pftlist}).sum(dim=dimname)

    safe_locals = local_names.copy()
    safe_locals.update({
        "np": np,
        "xr": xr,
        "verticalsum": verticalsum,
        "sumoverpft": sumoverpft,
    })
    # pylint: disable=eval-used
    return eval(expr, safe_globals, safe_locals)


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
        self._vars, self._raw, self._variant_keys = self._load_yaml(self.path)

    @classmethod
    def from_packaged_default(cls, filename: str = "cesm_to_cmip7.yaml") -> "Mapping":
        """Construct a Mapping using the packaged default YAML."""
        logger.info("mapping file is %s", filename)
        with packaged_mapping_resource(filename) as p:
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

        data = data["variables"]
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
            return _to_varconfig(cmip_name, raw, freq=effective_freq).as_cfg()
        return self._vars[cmip_name].as_cfg()

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
            warnings.warn(
                f"[mapping] no mapping found for CMIP variable {cmip_name} — skipping",
                RuntimeWarning,
                stacklevel=2,
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
                results.append((da, vc.as_cfg()))
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
    scale_from_sources = None
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
            if scale_from_sources is None and "scale" in item:
                scale_from_sources = item["scale"]
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
    if scale_from_sources is not None:
        unit_conversion = {"scale": scale_from_sources}

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
        dims=cfg.get("dims"),
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
        logger.debug("source aliases are %s",vc.source_aliases.items())

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


def _apply_unit_conversion(da: xr.DataArray, rule: Any) -> xr.DataArray:
    """Apply a unit conversion rule to a DataArray.

    >>> import xarray as xr
    >>> da = xr.DataArray([1.0, 2.0, 3.0])
    >>> _apply_unit_conversion(da, {"scale": 2.0}).values.tolist()
    [2.0, 4.0, 6.0]
    >>> _apply_unit_conversion(da, {"offset": 10.0}).values.tolist()
    [11.0, 12.0, 13.0]
    >>> _apply_unit_conversion(da, "x * 2").values.tolist()
    [2.0, 4.0, 6.0]
    """
    if isinstance(rule, str):
        try:
            out = _safe_eval(rule, {"x": da})
        except Exception as exc:
            raise ValueError(
                f"Error evaluating unit_conversion expression: {exc}"
            ) from exc
        if not isinstance(out, xr.DataArray):
            raise ValueError("unit_conversion expression did not return a DataArray")
        return out

    if isinstance(rule, dict):
        scale = rule.get("scale", 1.0)
        offset = rule.get("offset", 0.0)
        return da * float(scale) + float(offset)

    raise TypeError(
        "unit_conversion must be a string expression or a dict with 'scale'/'offset'"
    )
