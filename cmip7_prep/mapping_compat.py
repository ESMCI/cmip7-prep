# cmip7_prep/mapping_compat.py
r"""Mapping loader/evaluator compatible with CMIP6-style lists, CMIP7-style dicts,
and a dict wrapped under a top-level key 'variables:'.

Also supports a 'sources:' list where each item may be a plain string or a dict
with 'cesm_var' (e.g., {'cesm_var': 'TS'}). If there's exactly one source and
no formula, it's treated as a 1:1 mapping; otherwise sources become raw_variables.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files, as_file
from pathlib import Path
from typing import Any, Dict, List, Mapping as TMapping, Optional
import warnings

import numpy as np
import xarray as xr
import yaml  # runtime dep


def packaged_mapping_resource(filename: str = "cesm_to_cmip7.yaml"):
    """Context manager yielding a real filesystem path to the packaged mapping file
    >>> with packaged_mapping_resource("cesm_to_cmip7.yaml") as p:
    ...     str(p).endswith("cesm_to_cmip7.yaml")
    True
    """
    res = files("cmip7_prep").joinpath(f"data/{filename}")
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

    def as_cfg(self) -> Dict[str, Any]:
        """Return a plain dict view for convenience in other modules.
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
        }
        return {k: v for k, v in d.items() if v is not None}


def _normalize_table_name(value: Optional[str]) -> Optional[str]:
    """Return a short CMOR table name like 'Amon' from common inputs.
    >>> _normalize_table_name("CMIP6_Amon.json")
    'Amon'
    >>> _normalize_table_name("Amon")
    'Amon'
    >>> print(_normalize_table_name(None))
    None
    """
    if not value:
        return None
    s = str(value)
    if s.lower().endswith(".json"):
        s = s[:-5]
    if "_" in s:
        head, tail = s.split("_", 1)
        if head.upper().startswith("CMIP"):
            s = tail
    return s


def _safe_eval(expr: str, local_names: Dict[str, Any]) -> Any:
    """Evaluate a small arithmetic/xarray expression in a restricted environment.
    >>> _safe_eval("x + 2", {"x": 3})
    5
    >>> import numpy as np
    >>> _safe_eval("np.mean(x)", {"x": [1, 2, 3]})
    2.0
    """
    safe_globals = {"__builtins__": {}}
    locals_safe = {"np": np, "xr": xr}
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
    The loader accepts both dict- and list-based YAML styles. All table names
    are normalized to a short form (e.g., 'Amon').
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._vars: Dict[str, VarConfig] = self._load_yaml(self.path)

    @classmethod
    def from_packaged_default(cls, filename: str = "cesm_to_cmip7.yaml") -> "Mapping":
        """Construct a Mapping using the packaged default YAML."""
        with packaged_mapping_resource(filename) as p:
            return cls(p)

    # -----------------
    # Loading
    # -----------------
    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, VarConfig]:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Accept top-level "variables:" wrapper
        if (
            isinstance(data, dict)
            and "variables" in data
            and isinstance(data["variables"], dict)
        ):
            data = data["variables"]

        result: Dict[str, VarConfig] = {}

        if isinstance(data, dict):
            # CMIP7-style (or wrapped): keys are CMIP names
            for name, cfg in data.items():
                if not isinstance(cfg, dict):
                    continue
                result[name] = _to_varconfig(name, cfg)
        elif isinstance(data, list):
            # CMIP6-style: list with 'name' field
            for item in data:
                if not isinstance(item, dict) or "name" not in item:
                    continue
                name = str(item["name"])
                result[name] = _to_varconfig(name, item)
        else:
            raise TypeError(
                "Unsupported YAML structure: expected dict or list at top level."
            )

        return result

    # -----------------
    # Public API
    # -----------------
    def list_variables(self, table: Optional[str] = None) -> List[str]:
        """Return all CMIP variable names, optionally filtered by table."""
        if table is None:
            return sorted(self._vars.keys())
        t_short = _normalize_table_name(table)
        out = [k for k, v in self._vars.items() if v.table == t_short]
        return sorted(out)

    def get_cfg(self, cmip_name: str) -> Dict[str, Any]:
        """Return the normalized config dict for a CMIP variable name."""
        if cmip_name not in self._vars:
            raise KeyError(f"No mapping for {cmip_name!r} in {self.path}")
        return self._vars[cmip_name].as_cfg()

    def realize(self, ds: xr.Dataset, cmip_name: str) -> xr.DataArray:
        """Construct a CMIP variable as an xarray.DataArray from a native dataset."""
        if cmip_name not in self._vars:
            warnings.warn(
                f"[mapping] source variable {cmip_name} not found in dataset — skipping",
                RuntimeWarning,
                stacklevel=1,
            )
            return None
        vc = self._vars[cmip_name]

        da = _realize_core(ds, vc)
        if da is not None:
            if vc.unit_conversion is not None:
                da = _apply_unit_conversion(da, vc.unit_conversion)
            if vc.units:
                da.attrs["units"] = vc.units
            if vc.long_name:
                da.attrs.setdefault("long_name", vc.long_name)
            if vc.standard_name:
                da.attrs.setdefault("standard_name", vc.standard_name)

        return da


def _to_varconfig(name: str, cfg: TMapping[str, Any]) -> VarConfig:
    """Normalize a raw YAML entry into a VarConfig."""
    table = _normalize_table_name(cfg.get("table") or cfg.get("CMOR_table"))

    # 1) Accept your 'sources:' schema
    raw_from_sources: Optional[List[str]] = None
    if "sources" in cfg and isinstance(cfg["sources"], list):
        raw_from_sources = []
        for item in cfg["sources"]:
            if isinstance(item, str):
                raw_from_sources.append(item)
            elif isinstance(item, dict) and "cesm_var" in item:
                raw_from_sources.append(str(item["cesm_var"]))
        if not raw_from_sources:
            raw_from_sources = None

    # 2) Also accept traditional fields
    raw_vars = cfg.get("raw_variables") or cfg.get("raw_vars") or raw_from_sources
    if isinstance(raw_vars, str):
        raw_vars = [raw_vars]

    # Decide source vs raw_variables
    source = cfg.get("source")
    formula = cfg.get("formula")
    if source is None and raw_vars and len(raw_vars) == 1 and not formula:
        source = raw_vars[0]  # 1:1 mapping
        raw_vars = None

    vc = VarConfig(
        name=name,
        table=table,
        units=cfg.get("units"),
        raw_variables=raw_vars,
        source=source,
        formula=formula,
        unit_conversion=cfg.get("unit_conversion"),
        positive=cfg.get("positive"),
        cell_methods=cfg.get("cell_methods"),
        levels=cfg.get("levels"),
        regrid_method=cfg.get("regrid_method"),
        long_name=cfg.get("long_name"),
        standard_name=cfg.get("standard_name"),
    )
    return vc


def _require_vars(
    ds: xr.Dataset, names: List[str], context: str
) -> Dict[str, xr.DataArray]:
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

    # 2) identity mapping from a single raw variable
    if (
        vc.raw_variables
        and vc.formula in (None, "", "null")
        and len(vc.raw_variables) == 1
    ):
        var = vc.raw_variables[0]
        if var not in ds:
            raise KeyError(f"raw variable {var!r} not found in dataset")
        return ds[var]

    if vc.formula:
        if not vc.raw_variables:
            raise ValueError(f"formula given for {vc.name} but no raw_variables listed")
        env = _require_vars(ds, vc.raw_variables, f"realize({vc.name})")
        try:
            result = _safe_eval(vc.formula, env)
        except Exception as exc:
            raise ValueError(f"Error evaluating formula for {vc.name}: {exc}") from exc
        if not isinstance(result, xr.DataArray):
            raise ValueError(f"Formula for {vc.name} did not produce a DataArray")
        return result

    raise ValueError(
        f"Mapping for {vc.name} is incomplete: set 'source', or 'raw_variables', or 'formula'."
    )


def _apply_unit_conversion(da: xr.DataArray, rule: Any) -> xr.DataArray:
    """Apply a unit conversion rule to a DataArray."""
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
