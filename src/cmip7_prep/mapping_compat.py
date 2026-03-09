# cmip7_prep/mapping_compat.py
"""
Mapping loader/evaluator for CMIP7-style YAML files.
Supports only CMIP7 YAML format (top-level 'variables:' dict).
All mapping access is via config structure (cfg/VarConfig).
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import as_file
from pathlib import Path
from typing import Any, Dict, List, Mapping as TMapping, Optional
import warnings
import logging

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
    """Normalized mapping entry for a single CMIP variable."""

    name: str
    table: Optional[str] = None
    units: Optional[str] = None
    raw_variables: Optional[List[str]] = None
    source: Optional[str] = None
    formula: Optional[str] = None
    unit_conversion: Optional[Any] = None  # str expr or {scale, offset}
    positive: Optional[str] = None
    cell_methods: Optional[str] = None
    levels: Optional[Dict[str, Any]] = None
    regrid_method: Optional[str] = None
    long_name: Optional[str] = None
    standard_name: Optional[str] = None
    dims: Optional[List[str]] = None

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
        }
        return {k: v for k, v in d.items() if v is not None}


def _safe_eval(expr: str, local_names: Dict[str, Any]) -> Any:
    """Evaluate a small arithmetic/xarray expression in a restricted environment.
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

    locals_safe = {
        "np": np,
        "xr": xr,
        "verticalsum": verticalsum,
    }
    locals_safe.update(local_names)
    # pylint: disable=eval-used
    return eval(expr, safe_globals, locals_safe)


class Mapping:
    """Load and evaluate a CMIP mapping YAML file.

    Parameters
    ----------
    path : str or Path
        Path to the YAML mapping file.

    Notes
    -----
    The loader accepts only a dict-based YAML style. All table names
    are normalized to a short form (e.g., 'Amon').
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.default_freq: Optional[str] = None
        self._vars, self._raw = self._load_yaml(self.path)

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
        """Load CMIP7 YAML and return (vars_dict, raw_dict)."""
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
        for name, cfg in data.items():
            if not isinstance(cfg, dict):
                raise TypeError("Each entry in the yaml file must be a dictionary.")
            raw[name] = cfg
            result[name] = _to_varconfig(name, cfg)
        return result, raw

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
        logger.info("Realizing variable '%s' with config: %s", cmip_name, vc.as_cfg())
        da = _realize_core(ds, vc)
        logger.info("Realized variable '%s' with dims: %s", cmip_name, da.dims)
        if da is not None:
            if vc.units:
                da.attrs["units"] = vc.units
            if vc.long_name:
                da.attrs.setdefault("long_name", vc.long_name)
            if vc.standard_name:
                da.attrs.setdefault("standard_name", vc.standard_name)

        return da

    # No public access to _vars or _raw outside this file.


# -----------------
# Private routines
# -----------------
def _filter_sources(sources: List[Any], freq: Optional[str]) -> List[Any]:
    """Return the subset of source entries that match *freq*.

    If no entry carries a ``freq`` tag, all sources are returned unchanged.
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
    """
    if not sources:
        return sources
    has_tags = any(isinstance(s, dict) and "freq" in s for s in sources)
    if not has_tags or freq is None:
        return sources
    matched = [s for s in sources if isinstance(s, dict) and s.get("freq") == freq]
    if matched:
        return matched
    untagged = [s for s in sources if isinstance(s, dict) and "freq" not in s]
    return untagged if untagged else sources


def _to_varconfig(
    name: str, cfg: TMapping[str, Any], freq: Optional[str] = None
) -> VarConfig:
    """Normalize a CMIP7 YAML entry into a VarConfig.

    Only supports CMIP7 'sources' key.

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
    for item in active:
        if isinstance(item, dict) and "model_var" in item:
            raw_from_sources.append(str(item["model_var"]))
            if scale_from_sources is None and "scale" in item:
                scale_from_sources = item["scale"]
        elif isinstance(item, str):
            raw_from_sources.append(item)
    if not raw_from_sources:
        raise ValueError(f"raw_from_sources does not contain any values for {str}")
    if raw_from_sources and len(raw_from_sources) == 1 and not formula:
        source = raw_from_sources[0]
        raw_from_sources = None

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
    """Create a DataArray according to a VarConfig (without unit conversion)."""
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

    # Identity mapping from a single raw variable
    if model_vars and vc.formula in (None, "", "null") and len(model_vars) == 1:
        var = model_vars[0]
        if var not in ds:
            raise KeyError(f"raw variable {var!r} not found in dataset")
        return ds[var]

    # Identity mapping from a formula
    if vc.formula:
        if not model_vars:
            raise ValueError(f"formula given for {vc.name} but no raw_variables listed")
        da_dict = _require_vars(ds, model_vars, f"realize({vc.name})")
        try:
            result = _safe_eval(vc.formula, da_dict)
        except Exception as exc:
            raise ValueError(f"Error evaluating formula for {vc.name}: {exc}") from exc
        if not isinstance(result, xr.DataArray):
            raise ValueError(f"Formula for {vc.name} did not produce a DataArray")
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
