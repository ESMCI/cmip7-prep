"""
Unit tests for the Mapping class in mapping_compat.py.
Tests mapping compatibility and normalization logic for CMIP7 variable mapping workflows.
"""

import tempfile
import xarray as xr
import numpy as np
import pytest
from cmip7_prep.mapping_compat import Mapping, _filter_sources, _to_varconfig


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


# ---------------------------------------------------------------------------
# _filter_sources unit tests
# ---------------------------------------------------------------------------

TAGGED_SOURCES = [
    {"model_var": "siu_d", "freq": "day"},
    {"model_var": "siu", "freq": "mon"},
]


def test_filter_sources_mon():
    """Monthly tag selects the monthly source entry."""
    result = _filter_sources(TAGGED_SOURCES, "mon")
    assert len(result) == 1
    assert result[0]["model_var"] == "siu"


def test_filter_sources_day():
    """Daily tag selects the daily source entry."""
    result = _filter_sources(TAGGED_SOURCES, "day")
    assert len(result) == 1
    assert result[0]["model_var"] == "siu_d"


def test_filter_sources_no_freq_returns_all():
    """When freq=None all sources are returned regardless of tags."""
    result = _filter_sources(TAGGED_SOURCES, None)
    assert result == TAGGED_SOURCES


def test_filter_sources_untagged_passthrough():
    """Sources without any freq tags are returned unchanged."""
    untagged = [{"model_var": "T2m"}, {"model_var": "TS"}]
    assert _filter_sources(untagged, "mon") == untagged


def test_filter_sources_unmatched_falls_back_to_untagged():
    """If no entry matches the requested freq, untagged entries are used."""
    mixed = [
        {"model_var": "siu_d", "freq": "day"},
        {"model_var": "siu_fallback"},  # no freq tag
    ]
    result = _filter_sources(mixed, "mon")
    assert len(result) == 1
    assert result[0]["model_var"] == "siu_fallback"


def test_filter_sources_empty():
    """Empty source list returns empty list unchanged."""
    assert _filter_sources([], "mon") == []


# ---------------------------------------------------------------------------
# _to_varconfig with freq
# ---------------------------------------------------------------------------


def test_to_varconfig_freq_selects_correct_source():
    """freq parameter resolves the matching tagged source as the single source."""
    cfg = {"sources": TAGGED_SOURCES, "table": "seaIce", "units": "m s-1"}
    assert _to_varconfig("siu", cfg, freq="mon").source == "siu"
    assert _to_varconfig("siu", cfg, freq="day").source == "siu_d"


def test_to_varconfig_no_freq_picks_first_tagged():
    """Without freq, both tagged entries are collected; the first becomes source only
    if there is exactly one after filtering — otherwise raw_variables is set."""
    cfg = {"sources": TAGGED_SOURCES}
    vc = _to_varconfig("siu", cfg)
    # Two entries remain → raw_variables, not source
    assert vc.raw_variables is not None
    assert len(vc.raw_variables) == 2


# ---------------------------------------------------------------------------
# Mapping.get_cfg / Mapping.realize with freq
# ---------------------------------------------------------------------------

_CICE_YAML = """
variables:
  siu_tavg-u-hxy-si:
    table: seaIce
    units: "m s-1"
    dims: [time, nj, ni]
    regrid_method: conservative
    sources:
      - model_var: siu_d
        freq: day
      - model_var: siu
        freq: mon
"""


@pytest.fixture()
def cice_mapping(tmp_path):
    """mapping for cice variables"""
    p = tmp_path / "test.yaml"
    p.write_text(_CICE_YAML)
    return Mapping(p)


def test_get_cfg_mon(cice_mapping):  # pylint: disable=redefined-outer-name
    """get_cfg with freq='mon' resolves to the monthly model variable."""
    cfg = cice_mapping.get_cfg("siu_tavg-u-hxy-si", freq="mon")
    assert cfg["source"] == "siu"


def test_get_cfg_day(cice_mapping):  # pylint: disable=redefined-outer-name
    """get_cfg with freq='day' resolves to the daily model variable."""
    cfg = cice_mapping.get_cfg("siu_tavg-u-hxy-si", freq="day")
    assert cfg["source"] == "siu_d"


def test_get_cfg_no_freq_returns_both(
    cice_mapping,
):  # pylint: disable=redefined-outer-name
    """get_cfg without freq returns all tagged sources as raw_variables."""
    cfg = cice_mapping.get_cfg("siu_tavg-u-hxy-si")
    # Both sources present → raw_variables, not single source
    assert "raw_variables" in cfg
    assert set(cfg["raw_variables"]) == {"siu_d", "siu"}


def test_realize_mon_picks_correct_var(
    cice_mapping,
):  # pylint: disable=redefined-outer-name
    """realize with freq='mon' loads data from the monthly variable."""
    ds = xr.Dataset(
        {
            "siu": (("time", "nj", "ni"), np.ones((2, 3, 4), dtype="f4")),
            "siu_d": (("time", "nj", "ni"), np.full((2, 3, 4), 9.0, dtype="f4")),
        }
    )
    da = cice_mapping.realize(ds, "siu_tavg-u-hxy-si", freq="mon")
    assert float(da.mean()) == pytest.approx(1.0)


def test_realize_day_picks_correct_var(
    cice_mapping,
):  # pylint: disable=redefined-outer-name
    """realize with freq='day' loads data from the daily variable."""
    ds = xr.Dataset(
        {
            "siu": (("time", "nj", "ni"), np.ones((2, 3, 4), dtype="f4")),
            "siu_d": (("time", "nj", "ni"), np.full((2, 3, 4), 9.0, dtype="f4")),
        }
    )
    da = cice_mapping.realize(ds, "siu_tavg-u-hxy-si", freq="day")
    assert float(da.mean()) == pytest.approx(9.0)
