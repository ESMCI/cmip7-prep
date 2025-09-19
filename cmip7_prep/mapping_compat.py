
# cmip7_prep/mapping_compat.py
r"""Mapping loader/evaluator compatible with CMIP6-style lists and CMIP7-style dicts.

This module provides a `Mapping` class that:
- Loads a YAML mapping file where **keys are CMIP variable names** (preferred), or
  a CMIP6-style *list* of entries with a ``name`` field.
- Exposes light config access via :meth:`get_cfg`.
- Builds CMIP variables from a native CESM `xarray.Dataset` via :meth:`realize`,
  supporting raw variables, formulas, and simple unit conversions.

Security note
-------------
Formulas and conversion expressions are evaluated from a trusted, local mapping file
with a restricted environment. We centralize the use of ``eval`` in a helper that
is clearly marked and limited, and we disable the linter warning only at that line.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping as TMapping, Optional

import numpy as np
import xarray as xr
import yaml  # runtime dependency; ignored by pylint via .pylintrc


def _normalize_table_name(value: Optional[str]) -> Optional[str]:
    """Return a short CMOR table name like 'Amon' from common inputs.

    Accepts: 'Amon', 'CMIP6_Amon.json', 'CMIP7_Amon.json', etc.
    """
    if not value:
        return None
    s = str(value)
    if s.lower().endswith(".json"):
        s = s[:-5]
    # strip CMIPx_ prefix if present
    if "_" in s:
        parts = s.split("_", 1)
        if len(parts) == 2 and parts[0].upper().startswith("CMIP"):
            s = parts[1]
    return s


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

    def as_cfg(self) -> Dict[str, Any]:
        """Return a plain dict view for convenience in other modules."""
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
        }
        return {k: v for k, v in d.items() if v is not None}


def _safe_eval(expr: str, local_names: Dict[str, Any]) -> Any:
    """Evaluate a small arithmetic/xarray expression in a restricted environment.

    Only the names provided in `local_names` are available, plus numpy/xarray.

    Used for:
      - combining raw variables in `formula`
      - simple unit conversions (string form)

    Mapping files are assumed trusted; we still minimize surface area by stripping
    builtins and only exposing needed names.
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

    # -----------------
    # Loading
    # -----------------
    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, VarConfig]:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        result: Dict[str, VarConfig] = {}

        if isinstance(data, dict):
            # CMIP7-style: keys are CMIP names
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
            raise TypeError("Unsupported YAML structure: expected dict or list at top level.")

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
        """Construct a CMIP variable as an xarray.DataArray from a native dataset.

        The mapping entry may specify:
        - `source`: a single CESM variable name (shortcut to pick the variable)
        - `raw_variables`: list of CESM variable names (used by formula or identity)
        - `formula`: Python expression combining raw variables
        - `unit_conversion`: str expression (using `x`) or dict {scale, offset}
        """
        if cmip_name not in self._vars:
            raise KeyError(f"No mapping for {cmip_name!r} in {self.path}")
        vc = self._vars[cmip_name]

        da = _realize_core(ds, vc)

        # apply unit conversion if requested
        if vc.unit_conversion is not None:
            da = _apply_unit_conversion(da, vc.unit_conversion)

        # set target units if provided
        if vc.units:
            da.attrs["units"] = vc.units

        return da


def _to_varconfig(name: str, cfg: TMapping[str, Any]) -> VarConfig:
    """Normalize a raw YAML entry into a :class:`VarConfig`."""
    table = _normalize_table_name(cfg.get("table") or cfg.get("CMOR_table"))
    raw_vars = cfg.get("raw_variables") or cfg.get("raw_vars") or None
    if isinstance(raw_vars, str):
        raw_vars = [raw_vars]
    levels = cfg.get("levels") or None
    vc = VarConfig(
        name=name,
        table=table,
        units=cfg.get("units"),
        raw_variables=raw_vars,
        source=cfg.get("source"),
        formula=cfg.get("formula"),
        unit_conversion=cfg.get("unit_conversion"),
        positive=cfg.get("positive"),
        cell_methods=cfg.get("cell_methods"),
        levels=levels,
        regrid_method=cfg.get("regrid_method"),
    )
    return vc


def _require_vars(ds: xr.Dataset, names: List[str], context: str) -> Dict[str, xr.DataArray]:
    missing = [n for n in names if n not in ds]
    if missing:
        raise KeyError(f"{context}: missing variables {missing}")
    return {n: ds[n] for n in names}


def _realize_core(ds: xr.Dataset, vc: VarConfig) -> xr.DataArray:
    """Create a DataArray according to a VarConfig (without unit conversion)."""
    # 1) direct source override
    if vc.source:
        if vc.source not in ds:
            raise KeyError(f"source variable {vc.source!r} not found in dataset")
        return ds[vc.source]

    # 2) identity mapping from a single raw variable
    if vc.raw_variables and vc.formula in (None, "", "null") and len(vc.raw_variables) == 1:
        var = vc.raw_variables[0]
        if var not in ds:
            raise KeyError(f"raw variable {var!r} not found in dataset")
        return ds[var]

    # 3) formula combination
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
    """Apply a unit conversion rule to a DataArray.

    Parameters
    ----------
    da : DataArray
        Input data.
    rule : str or dict
        - If str, evaluates an expression using ``x`` (the data), with ``np`` and ``xr`` available.
        - If dict, supports keys: ``scale`` and optional ``offset``.
    """
    if isinstance(rule, str):
        try:
            out = _safe_eval(rule, {"x": da})
        except Exception as exc:
            raise ValueError(f"Error evaluating unit_conversion expression: {exc}") from exc
        if not isinstance(out, xr.DataArray):
            raise ValueError("unit_conversion expression did not return a DataArray")
        return out

    if isinstance(rule, dict):
        scale = rule.get("scale", 1.0)
        offset = rule.get("offset", 0.0)
        return da * float(scale) + float(offset)

    raise TypeError("unit_conversion must be a string expression or a dict with 'scale'/'offset'")
