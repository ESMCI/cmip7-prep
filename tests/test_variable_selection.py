"""Unit tests for YAML-driven CMIP variable assembly helpers."""

from __future__ import annotations

from types import SimpleNamespace

from cmip7_prep.mapping_compat import Mapping
from cmip7_prep.variable_selection import (
    NamedEntry,
    assemble_yaml_defined_cmip_vars,
    build_synthetic_variable,
)


def test_build_synthetic_variable_exposes_driver_surface():
    """Synthetic variables provide the minimum fields consumed by the driver."""
    var = build_synthetic_variable("tas")

    assert var.branded_variable_name.name == "tas"
    assert var.physical_parameter.name == "tas"
    assert var.attributes["branded_variable_name"].name == "tas"
    assert var.is_synthetic is True


def test_assemble_yaml_defined_cmip_vars_preserves_mapping_order(tmp_path):
    """Assembly should follow mapping order and reuse real objects when present."""
    yaml_str = """
variables:
  tas:
    table: Amon
    sources:
      - model_var: T2M
  pr:
    table: Amon
    sources:
      - model_var: PRECT
"""
    mapping_path = tmp_path / "mapping.yaml"
    mapping_path.write_text(yaml_str)
    mapping = Mapping(mapping_path)

    real_var = SimpleNamespace(
        branded_variable_name=NamedEntry("pr"),
        physical_parameter=NamedEntry("precipitation_flux"),
        attributes={"branded_variable_name": NamedEntry("pr")},
    )

    resolved, synthesized = assemble_yaml_defined_cmip_vars(
        mapping,
        [real_var],
        freq="mon",
    )

    assert [var.branded_variable_name.name for var in resolved] == ["tas", "pr"]
    assert resolved[1] is real_var
    assert resolved[0].is_synthetic is True
    assert synthesized == ["tas"]


def test_assemble_yaml_defined_cmip_vars_skips_incompatible_frequency(tmp_path):
    """Assembly should only include mapping variables compatible with the runtime frequency."""
    yaml_str = """
variables:
  tas:
    table: Amon
    sources:
      - model_var: T2M
        freq: mon
  pr:
    table: Amon
    sources:
      - model_var: PRECT
        freq: day
"""
    mapping_path = tmp_path / "freq_mapping.yaml"
    mapping_path.write_text(yaml_str)
    mapping = Mapping(mapping_path)

    resolved, synthesized = assemble_yaml_defined_cmip_vars(mapping, [], freq="day")

    assert [var.branded_variable_name.name for var in resolved] == ["pr"]
    assert synthesized == ["pr"]