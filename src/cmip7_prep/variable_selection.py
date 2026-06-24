"""Helpers for assembling CMIP variable lists from mappings and data-request results.

Purpose mainly to be able to make cmorised variables that are not yet in the data-request, 
but which are defined in the mapping YAML. cmor_driver.py can use this
when the run-all-from-yaml argument is true. This expands the usability of cmip7-prep
so it can be used to generate cmorised output for additional MIPs or project related runs
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Tuple

from cmip7_prep.mapping_compat import Mapping


@dataclass(frozen=True)
class NamedEntry:
    """Minimal name wrapper matching the data-request object surface used downstream."""

    name: str

    def __str__(self) -> str:
        return self.name


@dataclass
class SyntheticVariable:
    """Mapping-backed stand-in for a data-request variable."""

    branded_variable_name: NamedEntry
    physical_parameter: NamedEntry
    attributes: dict[str, Any] = field(default_factory=dict)
    is_synthetic: bool = True


def _build_data_request_variable(
    cmip_name: str,
    physical_parameter: str | None = None,
    *,
    data_request: object | None = None,
    template_var: object | None = None,
    freq: str | None = None,
    realm: str | None = None,
) -> object | None:
    """Build a real data-request Variable when a DR context is available."""
    dr_ref = getattr(template_var, "dr", None) or data_request
    if dr_ref is None:
        return None

    variable_cls = type(template_var) if template_var is not None else None
    if variable_cls is None:
        try:
            module = importlib.import_module("data_request_api.query.data_request")
            variable_cls = module.Variable
        except (ImportError, AttributeError):
            return None

    attrs: dict[str, Any] = {
        "title": cmip_name,
        "branded_variable_name": NamedEntry(cmip_name),
        "physical_parameter": NamedEntry(physical_parameter or cmip_name),
    }
    if freq is not None:
        attrs["cmip7_frequency"] = NamedEntry(freq)
    if realm is not None:
        attrs["modelling_realm"] = [NamedEntry(realm)]

    synthetic = variable_cls.from_input(
        dr=dr_ref,
        id=f"synthetic::{cmip_name}",
        **attrs,
    )
    setattr(synthetic, "is_synthetic", True)
    return synthetic


def build_synthetic_variable(
    cmip_name: str,
    physical_parameter: str | None = None,
    *,
    data_request: object | None = None,
    template_var: object | None = None,
    freq: str | None = None,
    realm: str | None = None,
) -> object:
    """Build a synthetic variable, preferring a real DataRequest Variable surface."""
    data_request_var = _build_data_request_variable(
        cmip_name,
        physical_parameter,
        data_request=data_request,
        template_var=template_var,
        freq=freq,
        realm=realm,
    )
    if data_request_var is not None:
        return data_request_var

    branded_name = NamedEntry(cmip_name)
    parameter_name = NamedEntry(physical_parameter or cmip_name)
    return SyntheticVariable(
        branded_variable_name=branded_name,
        physical_parameter=parameter_name,
        attributes={"branded_variable_name": branded_name},
    )


def assemble_yaml_defined_cmip_vars(
    mapping: Mapping,
    data_request_vars: Iterable[object],
    *,
    data_request: object | None = None,
    freq: str | None = None,
    realm: str | None = None,
) -> Tuple[List[object], List[str]]:
    """Assemble variables in mapping order, reusing data-request matches when present."""
    vars_by_name = {}
    template_var = None
    for var in data_request_vars:
        branded_name = getattr(getattr(var, "branded_variable_name", None), "name", None)
        if branded_name is not None and branded_name not in vars_by_name:
            vars_by_name[branded_name] = var
        if template_var is None and hasattr(var, "dr") and hasattr(var, "attributes"):
            template_var = var

    resolved = []
    synthesized = []
    for cmip_name in mapping.iter_variable_names(freq=freq):
        existing = vars_by_name.get(cmip_name)
        if existing is None:
            resolved.append(
                build_synthetic_variable(
                    cmip_name,
                    data_request=data_request,
                    template_var=template_var,
                    freq=freq,
                    realm=realm,
                )
            )
            synthesized.append(cmip_name)
            continue
        resolved.append(existing)
    return resolved, synthesized