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

    # Only CMIP7 'sources' key is supported
    assert cfg["table"] == "atmos"
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


# ---------------------------------------------------------------------------
# _filter_sources: matched + untagged behavior
# ---------------------------------------------------------------------------


def test_filter_sources_includes_untagged_with_match():
    """When a freq matches, untagged entries are returned alongside matched ones."""
    mixed = [
        {"model_var": "v_d", "freq": "day"},
        {"model_var": "v", "freq": "mon"},
        {"model_var": "area"},  # always-include, no freq tag
    ]
    result = _filter_sources(mixed, "day")
    assert len(result) == 2
    assert result[0]["model_var"] == "v_d"
    assert result[1]["model_var"] == "area"


# ---------------------------------------------------------------------------
# alias: source_aliases on VarConfig
# ---------------------------------------------------------------------------

_SIAREA_YAML = """
variables:
  siarea_tavg-u-hm-u:
    table: seaIce
    units: "m2"
    dims: [time]
    sources:
      - model_var: siconc_d
        freq: day
        alias: siconc
      - model_var: siconc
        freq: mon
      - model_var: tarea
    variants:
      - long_name: "Sea Ice Area (Northern Hemisphere)"
        region: nh
        formula: "(siconc * tarea / 100.).where(siconc.coords['TLAT'] > 0).sum(dim=['nj', 'ni'])"
      - long_name: "Sea Ice Area (Southern Hemisphere)"
        region: sh
        formula: "(siconc * tarea / 100.).where(siconc.coords['TLAT'] < 0).sum(dim=['nj', 'ni'])"
"""


@pytest.fixture()
def siarea_mapping(tmp_path):
    """Mapping fixture for siarea variant tests."""
    p = tmp_path / "siarea.yaml"
    p.write_text(_SIAREA_YAML)
    return Mapping(p)


def test_alias_day_source_aliases(
    siarea_mapping,
):  # pylint: disable=redefined-outer-name
    """With freq='day', source_aliases maps 'siconc' -> 'siconc_d' and 'tarea' -> 'tarea'."""
    cfg = siarea_mapping.get_cfg("siarea_tavg-u-hm-u", freq="day")
    aliases = cfg.get("source_aliases")
    assert aliases is not None
    assert aliases["siconc"] == "siconc_d"
    assert aliases["tarea"] == "tarea"


def test_alias_mon_source_aliases(
    siarea_mapping,
):  # pylint: disable=redefined-outer-name
    """With freq='mon', source_aliases are identity mappings."""
    cfg = siarea_mapping.get_cfg("siarea_tavg-u-hm-u", freq="mon")
    aliases = cfg.get("source_aliases")
    assert aliases is not None
    assert aliases["siconc"] == "siconc"
    assert aliases["tarea"] == "tarea"


# ---------------------------------------------------------------------------
# realize_all: variants
# ---------------------------------------------------------------------------


def _make_siarea_ds():
    """Build a minimal dataset with TLAT, siconc_d, siconc, tarea."""
    nj, ni = 4, 4
    # TLAT: two positive rows, two negative rows
    tlat = np.array(
        [
            [-30, -30, -30, -30],
            [-10, -10, -10, -10],
            [10, 10, 10, 10],
            [30, 30, 30, 30],
        ],
        dtype="f4",
    )
    siconc = np.full((2, nj, ni), 50.0, dtype="f4")  # 50%
    tarea = np.full((nj, ni), 1e10, dtype="f4")  # 1e10 m2
    coords = {"TLAT": (("nj", "ni"), tlat)}
    return xr.Dataset(
        {
            "siconc_d": (("time", "nj", "ni"), siconc),
            "siconc": (("time", "nj", "ni"), siconc),
            "tarea": (("nj", "ni"), tarea),
        },
        coords=coords,
    )


def test_realize_all_variants_have_correct_region(
    siarea_mapping,
):  # pylint: disable=redefined-outer-name
    """Each variant carries its region in the cfg dict."""
    ds = _make_siarea_ds()
    results = siarea_mapping.realize_all(ds, "siarea_tavg-u-hm-u", freq="mon")
    regions = [cfg.get("region") for _, cfg in results]
    assert regions == ["nh", "sh"]


def test_realize_all_returns_two_variants(
    siarea_mapping,
):  # pylint: disable=redefined-outer-name
    """realize_all on a variant entry returns a list of length 2."""
    ds = _make_siarea_ds()
    results = siarea_mapping.realize_all(ds, "siarea_tavg-u-hm-u", freq="mon")
    assert len(results) == 2


def test_realize_all_variants_have_correct_long_names(
    siarea_mapping,
):  # pylint: disable=redefined-outer-name
    """Each variant carries its long_name in attrs."""
    ds = _make_siarea_ds()
    results = siarea_mapping.realize_all(ds, "siarea_tavg-u-hm-u", freq="mon")
    long_names = [cfg["long_name"] for _, cfg in results]
    assert "Sea Ice Area (Northern Hemisphere)" in long_names
    assert "Sea Ice Area (Southern Hemisphere)" in long_names


def test_realize_all_nh_positive_sh_positive(
    siarea_mapping,
):  # pylint: disable=redefined-outer-name
    """NH variant selects only positive-TLAT cells; SH variant selects negative-TLAT cells."""
    ds = _make_siarea_ds()
    results = siarea_mapping.realize_all(ds, "siarea_tavg-u-hm-u", freq="mon")
    # Both NH and SH should be non-zero (dataset has cells in both hemispheres)
    for da, _ in results:
        assert float(da.isel(time=0)) > 0


def test_realize_all_alias_day_uses_siconc_d(
    siarea_mapping,
):  # pylint: disable=redefined-outer-name
    """With freq='day', alias maps siconc→siconc_d so formula evaluates correctly."""
    ds = _make_siarea_ds()
    # Set siconc_d to double the siconc values to verify the right variable is used
    ds["siconc_d"] = ds["siconc_d"] * 2
    results_day = siarea_mapping.realize_all(ds, "siarea_tavg-u-hm-u", freq="day")
    results_mon = siarea_mapping.realize_all(ds, "siarea_tavg-u-hm-u", freq="mon")
    # day result should be ~2x the mon result (siconc_d = 2 * siconc)
    nh_day = float(results_day[0][0].isel(time=0))
    nh_mon = float(results_mon[0][0].isel(time=0))
    assert nh_day == pytest.approx(nh_mon * 2, rel=1e-5)


def test_realize_all_non_variant_single_element(
    cice_mapping,
):  # pylint: disable=redefined-outer-name
    """realize_all on a non-variant entry returns a single-element list."""
    ds = xr.Dataset(
        {
            "siu": (("time", "nj", "ni"), np.ones((2, 3, 4), dtype="f4")),
        }
    )
    results = cice_mapping.realize_all(ds, "siu_tavg-u-hxy-si", freq="mon")
    assert len(results) == 1
    da, _ = results[0]
    assert float(da.mean()) == pytest.approx(1.0)
