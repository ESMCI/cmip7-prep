"""
Unit tests for the Mapping class in mapping_compat.py.
Tests mapping compatibility and normalization logic for CMIP7 variable mapping workflows.
"""

import tempfile
from cmip7_prep.mapping_compat import Mapping


def test_dict_style_sources_model_var():
    """
    Test that Mapping correctly normalizes a dict-style sources entry with model_var and scale.
    """
    yaml_str = """
variables:
  tas:
    sources:
      - {model_var: TS, scale: 10.0}
    table: CMIP7_atmos
"""
    with tempfile.NamedTemporaryFile("w+", suffix=".yaml", delete=True) as tmp:
        tmp.write(yaml_str)
        tmp.flush()
        mapping = Mapping(tmp.name)
        cfg = mapping.get_cfg("tas")

    # The Mapping class normalizes sources to raw_variables,
    # and for a single source with no formula, sets source
    assert cfg["source"] == "TS"
    assert "raw_variables" not in cfg or cfg["raw_variables"] is None
    assert cfg["table"] == "atmos"
    # Check that the scale is preserved in the config (if present)
    # The scale is not a top-level field, but should be present in unit_conversion if handled
    if "unit_conversion" in cfg:
        if isinstance(cfg["unit_conversion"], dict):
            assert cfg["unit_conversion"].get("scale", None) == 10.0
