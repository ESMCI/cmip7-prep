"""
Unit tests for the Mapping class in mapping_compat.py.
Tests mapping compatibility and normalization logic for CMIP7 variable mapping workflows.
"""

import json
import tempfile
import xarray as xr
import numpy as np
import pytest
from cmip7_prep.mapping_compat import (
    Mapping,
    _filter_sources,
    _safe_eval,
    _to_varconfig,
    expand_entry,
    formula_names,
    map_multi_dimension,
)


@pytest.fixture(name="fates_pfts")
def fixture_fates_pfts():
    """Each of the 15 source positions carries its 1-based PFT number."""
    return xr.DataArray(
        np.broadcast_to(np.arange(1.0, 16.0)[None, :, None], (2, 15, 3)),
        dims=("time", "fates_levpft", "ncol"),
        coords={"time": [15, 45], "fates_levpft": np.arange(15), "ncol": [1, 2, 3]},
        name="FATES_GPP_PF",
        attrs={"units": "kg m-2 s-1"},
    )


@pytest.mark.parametrize(
    "dimension_arg",
    ["fates_levpft", "'fates_levpft'", "dimname=fates_levpft"],
)
def test_map_multi_dimension_formula(fates_pfts, dimension_arg):
    """Dimension tokens are not sources and groups sum in CMIP order."""
    formula = f"map_multi_dimension(FATES_GPP_PF, {dimension_arg})"
    assert formula_names(formula) == ["FATES_GPP_PF"]
    cfg = expand_entry("gppVgt_tavg-u-hxy-multi", formula, ["gr"])
    assert cfg["sources"] == [{"model_var": "FATES_GPP_PF"}]
    result = _safe_eval(formula, {"FATES_GPP_PF": fates_pfts})
    assert result.dims == ("time", "vegtype", "ncol")
    assert result["vegtype"].values.tolist() == [
        "broadleaf_deciduous_trees",
        "broadleaf_evergreen_trees",
        "needleleaf_deciduous_trees",
        "needleleaf_evergreen_tree",
        "natural_grasses",
        "crops",
        "shrubs",
    ]
    np.testing.assert_array_equal(
        result.values,
        np.broadcast_to(np.array([11, 5, 3, 2, 39, 15, 45])[None, :, None], (2, 7, 3)),
    )
    xr.testing.assert_equal(result.sum("vegtype"), fates_pfts.sum("fates_levpft"))
    xr.testing.assert_equal(result["time"], fates_pfts["time"])
    xr.testing.assert_equal(result["ncol"], fates_pfts["ncol"])
    assert result.attrs == fates_pfts.attrs
    assert "fates_levpft" not in result.coords


def test_map_multi_dimension_realization(fates_pfts):
    """Minimal mapping expansion works through the normal realization path."""
    formula = "map_multi_dimension(FATES_GPP_PF, fates_levpft)"
    with tempfile.NamedTemporaryFile("w+", suffix=".yaml", delete=True) as tmp:
        tmp.write(f"variables:\n  gppVgt_tavg-u-hxy-multi: {formula}\n")
        tmp.flush()
        mapping = Mapping(tmp.name)
        assert mapping.timeseries_source_vars("gppVgt_tavg-u-hxy-multi", "mon") == [
            "FATES_GPP_PF"
        ]
        result = mapping.realize(fates_pfts.to_dataset(), "gppVgt_tavg-u-hxy-multi")
    xr.testing.assert_equal(result, map_multi_dimension(fates_pfts, "fates_levpft"))


def test_map_multi_dimension_preserves_dimension_order(fates_pfts):
    """The target dimension occupies the position of the source dimension."""
    arr = fates_pfts.transpose("ncol", "time", "fates_levpft")
    result = map_multi_dimension(arr, "fates_levpft")
    assert result.dims == ("ncol", "time", "vegtype")


def test_map_multi_dimension_propagates_missing_values(fates_pfts):
    """A missing member invalidates its group, not the unrelated groups."""
    arr = fates_pfts.where(fates_pfts["fates_levpft"] != 4)
    result = map_multi_dimension(arr, "fates_levpft")
    assert result.sel(vegtype="broadleaf_deciduous_trees").isnull().all()
    assert result.sel(vegtype="crops").notnull().all()


def test_map_multi_dimension_stays_lazy(fates_pfts):
    """Grouping a chunked field does not materialize the source data."""
    pytest.importorskip("dask.array")
    result = map_multi_dimension(
        fates_pfts.chunk({"time": 1, "fates_levpft": 3}), "fates_levpft"
    )
    assert result.chunks is not None
    xr.testing.assert_equal(
        result.compute(), map_multi_dimension(fates_pfts, "fates_levpft")
    )


@pytest.mark.parametrize("size", [14, 16])
def test_map_multi_dimension_rejects_incorrect_pft_count(size):
    """Missing crops and extra unclassified PFTs both fail explicitly."""
    arr = xr.DataArray(np.ones(size), dims="fates_levpft")
    with pytest.raises(ValueError, match=f"requires 15 source entries, got {size}"):
        map_multi_dimension(arr, "fates_levpft")


def test_map_multi_dimension_rejects_invalid_input(fates_pfts):
    """Unknown mappings, absent dimensions and target collisions are errors."""
    with pytest.raises(TypeError, match="Expected xr.DataArray"):
        map_multi_dimension(np.ones(15), "fates_levpft")
    with pytest.raises(TypeError, match="dimension name string"):
        map_multi_dimension(fates_pfts, 1)
    with pytest.raises(ValueError, match="No multi-dimension mapping"):
        map_multi_dimension(fates_pfts, "unknown")
    with pytest.raises(ValueError, match="not found"):
        map_multi_dimension(fates_pfts.isel(fates_levpft=0), "fates_levpft")
    with pytest.raises(ValueError, match="already exists"):
        map_multi_dimension(
            fates_pfts.assign_coords(vegtype="existing"), "fates_levpft"
        )


@pytest.fixture(name="fates_landuse")
def fixture_fates_landuse():
    """Source labels deliberately differ from the 1-based mapping positions."""
    return xr.DataArray(
        np.broadcast_to(np.arange(1, 6)[None, :, None], (2, 5, 3)),
        dims=("time", "fates_levlanduse", "ncol"),
        coords={
            "time": [15, 45],
            "fates_levlanduse": [
                "primary",
                "secondary",
                "rangeland",
                "pasture",
                "crops",
            ],
            "ncol": [1, 2, 3],
        },
        name="FATES_GPP_LU",
        attrs={"units": "kg m-2 s-1"},
    )


@pytest.mark.parametrize(
    "dimension_arg",
    ["fates_levlanduse", "'fates_levlanduse'", "dimname=fates_levlanduse"],
)
def test_map_multi_dimension_landuse_formula(fates_landuse, dimension_arg):
    """Land use sums the five source positions and leaves urban missing."""
    formula = f"map_multi_dimension(FATES_GPP_LU, {dimension_arg})"
    assert formula_names(formula) == ["FATES_GPP_LU"]
    result = _safe_eval(formula, {"FATES_GPP_LU": fates_landuse})
    assert result.dims == ("time", "landuse", "ncol")
    assert result["landuse"].values.tolist() == [
        "primary_and_secondary_land",
        "pastures",
        "crops",
        "urban",
    ]
    np.testing.assert_allclose(
        result.values,
        np.broadcast_to(np.array([3, 7, 5, np.nan])[None, :, None], (2, 4, 3)),
        equal_nan=True,
    )
    xr.testing.assert_equal(result["time"], fates_landuse["time"])
    xr.testing.assert_equal(result["ncol"], fates_landuse["ncol"])
    assert result.attrs == fates_landuse.attrs
    assert "fates_levlanduse" not in result.coords


def test_map_multi_dimension_landuse_realization(fates_landuse):
    """The configured land-use mapping works through normal realization."""
    formula = "map_multi_dimension(FATES_GPP_LU, fates_levlanduse)"
    with tempfile.NamedTemporaryFile("w+", suffix=".yaml", delete=True) as tmp:
        tmp.write(f"variables:\n  gppLut_tavg-u-hxy-multi: {formula}\n")
        tmp.flush()
        mapping = Mapping(tmp.name)
        result = mapping.realize(fates_landuse.to_dataset(), "gppLut_tavg-u-hxy-multi")
    xr.testing.assert_equal(
        result, map_multi_dimension(fates_landuse, "fates_levlanduse")
    )


def test_map_multi_dimension_landuse_missing_and_dimension_order(fates_landuse):
    """Missing primary data does not affect pastures or the dimension order."""
    arr = fates_landuse.where(fates_landuse["fates_levlanduse"] != "primary").transpose(
        "ncol", "time", "fates_levlanduse"
    )
    result = map_multi_dimension(arr, "fates_levlanduse")
    assert result.dims == ("ncol", "time", "landuse")
    assert result.sel(landuse="primary_and_secondary_land").isnull().all()
    assert result.sel(landuse="urban").isnull().all()
    assert (result.sel(landuse="pastures") == 7).all()
    assert (result.sel(landuse="crops") == 5).all()


def test_map_multi_dimension_landuse_stays_lazy(fates_landuse):
    """The missing urban group remains lazy along with the sums."""
    pytest.importorskip("dask.array")
    result = map_multi_dimension(
        fates_landuse.chunk({"time": 1, "fates_levlanduse": 2}), "fates_levlanduse"
    )
    assert result.chunks is not None
    xr.testing.assert_equal(
        result.compute(), map_multi_dimension(fates_landuse, "fates_levlanduse")
    )


@pytest.mark.parametrize("size", [4, 6])
def test_map_multi_dimension_rejects_incorrect_landuse_count(size):
    """Land-use input must contain exactly the five FATES categories."""
    arr = xr.DataArray(np.ones(size), dims="fates_levlanduse")
    with pytest.raises(ValueError, match=f"requires 5 source entries, got {size}"):
        map_multi_dimension(arr, "fates_levlanduse")


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


def test_iter_variable_names_respects_frequency(
    cice_mapping,
):  # pylint: disable=redefined-outer-name
    """iter_variable_names keeps entries that support the requested frequency."""
    assert cice_mapping.iter_variable_names(freq="mon") == ["siu_tavg-u-hxy-si"]
    assert cice_mapping.iter_variable_names(freq="day") == ["siu_tavg-u-hxy-si"]


def test_iter_variable_names_skips_incompatible_frequency(tmp_path):
    """iter_variable_names excludes entries tagged only for other frequencies."""
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
  ps:
    table: Amon
    sources:
      - model_var: PS
"""
    path = tmp_path / "freq_filter.yaml"
    path.write_text(yaml_str)
    mapping = Mapping(path)

    assert mapping.iter_variable_names(freq="mon") == ["tas", "ps"]
    assert mapping.iter_variable_names(freq="day") == ["pr", "ps"]


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


def test_iter_variable_names_variant_entry_stays_visible(
    siarea_mapping,
):  # pylint: disable=redefined-outer-name
    """Variant-backed variables are returned once as their top-level YAML key."""
    assert siarea_mapping.iter_variable_names(freq="mon") == ["siarea_tavg-u-hm-u"]


# ---------------------------------------------------------------------------
# sumover_index: a single index collapses the dimension, selecting one level
# ---------------------------------------------------------------------------
@pytest.fixture(name="uvel")
def uvel_fixture():
    """A (time, level, y0, x0) field with distinguishable values per level."""
    return xr.DataArray(
        np.arange(2 * 11 * 4 * 5, dtype="f8").reshape(2, 11, 4, 5),
        dims=("time", "level", "y0", "x0"),
        coords={"level": np.linspace(0.0, 1.0, 11)},
        name="uvel",
    )


def test_sumover_index_single_is_isel(uvel):  # pylint: disable=redefined-outer-name
    """One index reproduces isel and drops the dimension.

    This is what the landIce surface and basal velocity formulas rely on:
    sumover_index(uvel, indexlist=[1]) is the top of the ice column and
    indexlist=[11] is the bed.
    """
    for index in (1, 11):
        out = _safe_eval(
            f"sumover_index(uvel, indexlist=[{index}], dimname='level')",
            {"uvel": uvel},
        )
        assert out.dims == ("time", "y0", "x0")
        assert "level" not in out.coords
        assert (out == uvel.isel(level=index - 1)).all()


def test_sumover_index_is_one_based(uvel):  # pylint: disable=redefined-outer-name
    """Index 1 is the first element, not the second."""
    out = _safe_eval(
        "sumover_index(uvel, indexlist=[1], dimname='level')", {"uvel": uvel}
    )
    assert (out == uvel.isel(level=0)).all()
    assert not (out == uvel.isel(level=1)).all()


def test_sumover_index_multiple_sums(uvel):  # pylint: disable=redefined-outer-name
    """Several indices sum, and indices outside the list are excluded."""
    out = _safe_eval(
        "sumover_index(uvel, indexlist=[1, 2, 3], dimname='level')", {"uvel": uvel}
    )
    assert (out == uvel.isel(level=[0, 1, 2]).sum(dim="level")).all()


def test_sumover_index_rejects_empty_list(uvel):  # pylint: disable=redefined-outer-name
    """An empty index list is a formula bug, not a no-op."""
    with pytest.raises(ValueError, match="indexlist must not be empty"):
        _safe_eval("sumover_index(uvel, indexlist=[], dimname='level')", {"uvel": uvel})


# ---------------------------------------------------------------------------
# verticalmean: sigma-weighted column average
# ---------------------------------------------------------------------------
# The 11 sigma levels a CISM Greenland run actually uses: 0 at the upper
# surface, 1 at the bed, refined towards the bed.
CISM_SIGMA = [
    0.0,
    0.231404958677686,
    0.407407407407407,
    0.544378698224852,
    0.653061224489796,
    0.740740740740741,
    0.8125,
    0.8719723183391,
    0.921810699588477,
    0.96398891966759,
    1.0,
]


@pytest.fixture(name="uvel_sigma")
def uvel_sigma_fixture():
    """A velocity profile falling linearly from 100 m/yr at the surface to 40 at the bed."""
    sigma = np.array(CISM_SIGMA)
    profile = 100.0 - 60.0 * sigma
    data = np.broadcast_to(profile[:, None, None], (sigma.size, 3, 2)).copy()
    return xr.DataArray(
        data, dims=("level", "y0", "x0"), coords={"level": sigma}, name="uvel"
    )


def test_verticalmean_matches_analytic(
    uvel_sigma,
):  # pylint: disable=redefined-outer-name
    """For a linear profile the column mean is exact: 100 - 60/2 = 70 m/yr."""
    out = _safe_eval("verticalmean(uvel, levelname='level')", {"uvel": uvel_sigma})
    assert out.dims == ("y0", "x0")
    np.testing.assert_allclose(out.values, 70.0, rtol=1e-12)


def test_verticalmean_beats_plain_mean(
    uvel_sigma,
):  # pylint: disable=redefined-outer-name
    """An unweighted mean is biased towards the bed, where levels bunch up."""
    weighted = _safe_eval("verticalmean(uvel, levelname='level')", {"uvel": uvel_sigma})
    plain = uvel_sigma.mean(dim="level")
    assert float(plain.values[0, 0]) < float(weighted.values[0, 0])
    assert abs(float(plain.values[0, 0]) - 70.0) > 5.0


def test_verticalmean_matches_cism(uvel_sigma):  # pylint: disable=redefined-outer-name
    """Weights must match glissade.F90's uvel_mean accumulation exactly."""
    sigma = np.array(CISM_SIGMA)
    stag = 0.5 * (sigma[:-1] + sigma[1:])
    expected = np.empty(sigma.size)
    expected[0] = stag[0]
    expected[1:-1] = stag[1:] - stag[:-1]
    expected[-1] = 1.0 - stag[-1]
    profile = 100.0 - 60.0 * sigma
    out = _safe_eval("verticalmean(uvel, levelname='level')", {"uvel": uvel_sigma})
    np.testing.assert_allclose(out.values[0, 0], float((profile * expected).sum()))


def test_verticalmean_requires_coordinate_values():
    """Without sigma values on the dimension there is nothing to weight by."""
    arr = xr.DataArray(np.ones((4, 2)), dims=("level", "x"), name="arr")
    with pytest.raises(ValueError, match="no coordinate values"):
        _safe_eval("verticalmean(arr)", {"arr": arr})


def test_verticalmean_rejects_bad_dim(
    uvel_sigma,
):  # pylint: disable=redefined-outer-name
    """A wrong dimension name is reported before anything else is attempted."""
    with pytest.raises(ValueError, match="Dimension 'lev' not found"):
        _safe_eval("verticalmean(uvel, levelname='lev')", {"uvel": uvel_sigma})


# ---------------------------------------------------------------------------
# global_2d_sum: a quantity per unit area integrated over the globe
# ---------------------------------------------------------------------------
# Four cells of pi steradians each cover the whole sphere, so a field of 1.0
# integrates to the area of the Earth.
EARTH_AREA_M2 = 4.0 * np.pi * 6371220.0**2


@pytest.fixture(name="tmco2")
def tmco2_fixture():
    """A (time, ncol) column burden of 1 kg m-2 on a four-cell grid."""
    return xr.DataArray(
        np.ones((3, 4), dtype="f4"),
        dims=("time", "ncol"),
        name="TMCO2",
    )


def _area(values, units, dims=("ncol",)):
    """A cell area carrying the units attribute a model writes."""
    return xr.DataArray(
        np.asarray(values, dtype="f8"), dims=dims, name="area", attrs={"units": units}
    )


def test_global_2d_sum_converts_steradians(
    tmco2,
):  # pylint: disable=redefined-outer-name
    """CAM writes 'area' in steradians, and the total comes out in kg."""
    area = _area(np.full(4, np.pi), "steradians")
    out = _safe_eval("global_2d_sum(TMCO2, area)", {"TMCO2": tmco2, "area": area})
    assert out.dims == ("time",)
    assert float(out[0]) == pytest.approx(EARTH_AREA_M2)


def test_global_2d_sum_converts_square_kilometres(
    tmco2,
):  # pylint: disable=redefined-outer-name
    """CLM writes 'area' in km^2, which is a million square metres."""
    area = _area(np.full(4, 2.0), "km^2")
    out = _safe_eval("global_2d_sum(TMCO2, area)", {"TMCO2": tmco2, "area": area})
    assert float(out[0]) == pytest.approx(8.0e6)


def test_global_2d_sum_leaves_square_metres_alone(
    tmco2,
):  # pylint: disable=redefined-outer-name
    """CICE writes 'tarea' in m^2, and those areas are summed as they are."""
    area = _area(np.full(4, 2.0), "m2")
    out = _safe_eval("global_2d_sum(TMCO2, tarea)", {"TMCO2": tmco2, "tarea": area})
    assert float(out[0]) == pytest.approx(8.0)


def test_global_2d_sum_sums_two_horizontal_dimensions():
    """A CICE field on (nj, ni) collapses to one value per time."""
    field = xr.DataArray(np.ones((2, 3, 5)), dims=("time", "nj", "ni"))
    area = _area(np.full((3, 5), 4.0), "m2", dims=("nj", "ni"))
    out = _safe_eval("global_2d_sum(field, tarea)", {"field": field, "tarea": area})
    assert out.dims == ("time",)
    assert float(out[0]) == pytest.approx(60.0)


def test_global_2d_sum_leaves_out_masked_cells(
    tmco2,
):  # pylint: disable=redefined-outer-name
    """Masking the field in the formula restricts the integral to what is left."""
    area = _area(np.full(4, 2.0), "m2")
    masked = tmco2.where(xr.DataArray([True, True, False, False], dims=("ncol",)))
    out = _safe_eval("global_2d_sum(masked, area)", {"masked": masked, "area": area})
    assert float(out[0]) == pytest.approx(4.0)


def test_global_2d_sum_keeps_one_value_per_time(
    tmco2,
):  # pylint: disable=redefined-outer-name
    """Each time is integrated on its own."""
    field = tmco2 * xr.DataArray([1.0, 2.0, 3.0], dims=("time",))
    area = _area(np.full(4, 0.25), "m2")
    out = _safe_eval("global_2d_sum(field, area)", {"field": field, "area": area})
    assert out.sizes["time"] == 3
    assert out.values == pytest.approx([1.0, 2.0, 3.0])


def test_global_2d_sum_works_in_double_precision(
    tmco2,
):  # pylint: disable=redefined-outer-name
    """Model output is single precision, and a total over many cells is not."""
    area = _area(np.full(4, 1.0), "m2")
    out = _safe_eval("global_2d_sum(TMCO2, area)", {"TMCO2": tmco2, "area": area})
    assert tmco2.dtype == np.dtype("f4")
    assert out.dtype == np.dtype("f8")


def test_global_2d_sum_needs_area_units(tmco2):  # pylint: disable=redefined-outer-name
    """Scaling the area in a formula drops its units, so they have to be named."""
    area = xr.DataArray(np.full(4, 1.0), dims=("ncol",), name="area")
    with pytest.raises(ValueError, match="no 'units' attribute"):
        _safe_eval("global_2d_sum(TMCO2, area)", {"TMCO2": tmco2, "area": area})
    out = _safe_eval(
        "global_2d_sum(TMCO2, area, area_units='m2')", {"TMCO2": tmco2, "area": area}
    )
    assert float(out[0]) == pytest.approx(4.0)


def test_global_2d_sum_rejects_unknown_area_units(
    tmco2,
):  # pylint: disable=redefined-outer-name
    """Units nobody has written a conversion for stop the variable."""
    area = _area(np.full(4, 1.0), "hectares")
    with pytest.raises(ValueError, match="not recognised"):
        _safe_eval("global_2d_sum(TMCO2, area)", {"TMCO2": tmco2, "area": area})


def test_global_2d_sum_rejects_a_mismatched_grid(
    tmco2,
):  # pylint: disable=redefined-outer-name
    """An area on a grid the field is not on is a mapping bug, not a broadcast."""
    area = _area(np.full((3, 5), 1.0), "m2", dims=("nj", "ni"))
    with pytest.raises(ValueError, match="cannot be summed over"):
        _safe_eval("global_2d_sum(TMCO2, area)", {"TMCO2": tmco2, "area": area})


# ---------------------------------------------------------------------------
# Minimal mapping form (data/schemas/mapping.schema.yaml)
# ---------------------------------------------------------------------------
_MINIMAL_YAML = """
grids: [gr]
variables:
  tas_tavg-h2m-hxy-u: TREFHT
  pr_tavg-u-hxy-u: (PRECC + PRECL) * 1000
  ta_tavg-p19-hxy-air: T
  cl_tavg-al-hxy-u: {formula: CLOUD, units: "1"}
  rsds_tavg-u-hxy-u: {formula: FSDS, positive: up}
  tos_tavg-u-hxy-sea: {formula: tos, grids: [gn, gr]}
  siu_tavg-u-hxy-si:
    formula: {day: siu_d, mon: siu}
  siarea_tavg-u-hm-u:
    variants:
      - region: nh
        formula:
          day: (siconc_d * tarea).where(siconc_d.coords['TLAT'] > 0).sum(dim=['nj', 'ni'])
          mon: (siconc * tarea).where(siconc.coords['TLAT'] > 0).sum(dim=['nj', 'ni'])
      - region: sh
        formula:
          day: (siconc_d * tarea).where(siconc_d.coords['TLAT'] < 0).sum(dim=['nj', 'ni'])
          mon: (siconc * tarea).where(siconc.coords['TLAT'] < 0).sum(dim=['nj', 'ni'])
"""

# A stand-in for cmip7-cmor-tables: only the fields the mapping reads.
_TABLE = {
    "tas_tavg-h2m-hxy-u": {
        "units": "K",
        "dimensions": "longitude latitude time height2m",
    },
    "ta_tavg-p19-hxy-air": {
        "units": "K",
        "dimensions": ["longitude", "latitude", "plev19", "time"],
    },
    "cl_tavg-al-hxy-u": {"units": "%", "dimensions": "longitude latitude alevel time"},
}


@pytest.fixture(name="minimal")
def minimal_fixture(tmp_path):
    """A Mapping over _MINIMAL_YAML that knows where the (fake) CMOR tables are."""
    (tmp_path / "tables").mkdir()
    (tmp_path / "tables" / "CMIP7_atmos.json").write_text(
        json.dumps({"variable_entry": _TABLE})
    )
    (tmp_path / "mapping.yaml").write_text(_MINIMAL_YAML)
    mapping = Mapping(tmp_path / "mapping.yaml")
    mapping.tables_root = tmp_path
    mapping.realm = "atmos"
    return mapping


def test_minimal_short_form(minimal):
    """A string entry is the formula; a bare name is a plain source."""
    assert minimal.get_cfg("tas_tavg-h2m-hxy-u")["source"] == "TREFHT"
    cfg = minimal.get_cfg("pr_tavg-u-hxy-u")
    assert cfg["formula"] == "(PRECC + PRECL) * 1000"
    assert cfg["raw_variables"] == ["PRECC", "PRECL"]


def test_minimal_grids_regrid_positive(minimal):
    """File grids default, entry override, regrid method and positive."""
    assert minimal.get_cfg("tas_tavg-h2m-hxy-u")["grids"] == ["gr"]
    assert minimal.get_cfg("tos_tavg-u-hxy-sea")["grids"] == ["gn", "gr"]
    assert minimal.get_cfg("tas_tavg-h2m-hxy-u")["regrid_method"] == "bilinear"
    assert minimal.get_cfg("pr_tavg-u-hxy-u")["regrid_method"] == "conservative"
    assert minimal.get_cfg("rsds_tavg-u-hxy-u")["positive"] == "up"


def test_minimal_table_fields(minimal):
    """Table, units and levels come from the CMOR table; declared units win."""
    cfg = minimal.get_cfg("ta_tavg-p19-hxy-air")
    assert (cfg["table"], cfg["units"], cfg["levels"]) == (
        "atmos",
        "K",
        {"name": "plev19"},
    )
    cl = minimal.get_cfg("cl_tavg-al-hxy-u")
    assert cl["units"] == "1"
    assert cl["levels"] == {"name": "standard_hybrid_sigma"}


def test_minimal_per_frequency(minimal):
    """A per-frequency formula picks the frequency's model variable."""
    assert minimal.get_cfg("siu_tavg-u-hxy-si", freq="mon")["source"] == "siu"
    assert minimal.get_cfg("siu_tavg-u-hxy-si", freq="day")["source"] == "siu_d"
    assert minimal.iter_variable_names(freq="3hr") == [
        name for name in minimal.iter_variable_names() if name != "siu_tavg-u-hxy-si"
    ]


def test_minimal_variants_per_frequency(minimal):
    """Variant formulas are realized per region, and per frequency."""
    ds = _make_siarea_ds()
    ds["siconc_d"] = ds["siconc_d"] * 2
    mon = minimal.realize_all(ds, "siarea_tavg-u-hm-u", freq="mon")
    day = minimal.realize_all(ds, "siarea_tavg-u-hm-u", freq="day")
    assert [cfg["region"] for _, cfg in mon] == ["nh", "sh"]
    assert float(day[0][0].isel(time=0)) == pytest.approx(
        2 * float(mon[0][0].isel(time=0))
    )
