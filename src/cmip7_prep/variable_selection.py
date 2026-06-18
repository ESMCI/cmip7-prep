"""Helpers for assembling CMIP variable lists from mappings and data-request results."""

from __future__ import annotations

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


def build_synthetic_variable(
    cmip_name: str, physical_parameter: str | None = None
) -> SyntheticVariable:
    """Build the minimum variable object needed by the CMOR driver."""
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
    freq: str | None = None,
) -> Tuple[List[object], List[str]]:
    """Assemble variables in mapping order, reusing data-request matches when present."""
    vars_by_name = {}
    for var in data_request_vars:
        branded_name = getattr(getattr(var, "branded_variable_name", None), "name", None)
        if branded_name is not None and branded_name not in vars_by_name:
            vars_by_name[branded_name] = var

    resolved = []
    synthesized = []
    for cmip_name in mapping.iter_variable_names(freq=freq):
        existing = vars_by_name.get(cmip_name)
        if existing is None:
            resolved.append(build_synthetic_variable(cmip_name))
            synthesized.append(cmip_name)
            continue
        resolved.append(existing)
    return resolved, synthesized