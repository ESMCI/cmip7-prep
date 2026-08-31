"""Tests for the YAML-backed history-file include-pattern tables."""

import pytest
import yaml

from cmip7_prep import include_patterns as ip
from cmip7_prep.include_patterns import (
    DATA_DIR,
    all_include_patterns,
    get_include_patterns,
    load_include_patterns,
    patterns_for_variable,
    sampling_from_branded_name,
)

MODELS = ["cesm", "noresm"]
# Time signifiers that appear in CMIP7 branded variable names.  Only the ones a
# model writes to its own history tape need an include-pattern entry; the rest
# fall back to time-averaged files (see patterns_for_variable).
SAMPLINGS = {"tavg", "tpt", "ti", "tsum", "tmin", "tmax", "tminavg", "tmaxavg"}


@pytest.mark.parametrize("model", MODELS)
def test_table_is_well_formed(model):
    """Realm -> frequency -> sampling -> non-empty list of string patterns."""
    table = load_include_patterns(model)
    assert table, f"{model} table is empty"
    for realm, by_frequency in table.items():
        assert isinstance(by_frequency, dict), f"{model}/{realm} is not a mapping"
        for frequency, by_sampling in by_frequency.items():
            where = f"{model}/{realm}/{frequency}"
            assert isinstance(frequency, str), where
            assert isinstance(by_sampling, dict), f"{where} is not a mapping"
            assert by_sampling, f"{where} has no sampling entries"
            for sampling, patterns in by_sampling.items():
                assert sampling in SAMPLINGS, f"{where}: unknown sampling {sampling!r}"
                assert isinstance(patterns, list) and patterns
                assert all(isinstance(p, str) and p for p in patterns)


@pytest.mark.parametrize("model", MODELS)
def test_yaml_file_is_the_source_of_truth(model):
    """The loader returns exactly what the packaged YAML file contains."""
    with open(DATA_DIR / f"{model}_include_patterns.yaml", encoding="utf-8") as handle:
        assert load_include_patterns(model) == yaml.safe_load(handle)


def test_returned_table_is_a_copy():
    """Mutating the result must not corrupt the cached table."""
    first = load_include_patterns("noresm")
    first["atmos"]["mon"]["tavg"].append("bogus")
    assert "bogus" not in load_include_patterns("noresm")["atmos"]["mon"]["tavg"]


def test_known_lookups():
    """Patterns other parts of the pipeline depend on."""
    assert get_include_patterns("noresm", "atmos", "mon") == ["cam.h0a"]
    assert get_include_patterns("cesm", "ocean", "mon") == [
        "mom6.h.z",
        "mom6.h.native.",
    ]
    # NorESM writes 3hr atmosphere output to a different file than CESM.
    assert get_include_patterns("noresm", "atmos", "3hr") == ["cam.h4a"]
    assert get_include_patterns("cesm", "atmos", "3hr") == ["cam.h3a"]


def test_ice_sheet_placeholder_is_filled():
    """landIce patterns are per ice sheet."""
    assert get_include_patterns("noresm", "landIce", "yr", ice_sheet="gris") == [
        "cism.gris.h"
    ]
    assert get_include_patterns("noresm", "landIce", "yr", ice_sheet="ais") == [
        "cism.ais.h"
    ]


def test_ice_sheet_is_required_when_the_pattern_needs_one():
    """Forgetting --ice-sheet must fail, not emit a literal brace."""
    with pytest.raises(ValueError, match="requires --ice-sheet"):
        get_include_patterns("noresm", "landIce", "yr")


def test_unknown_model_realm_and_frequency_all_raise():
    """The error names what was asked for, so a typo is obvious."""
    with pytest.raises(ValueError, match="No include-pattern table"):
        get_include_patterns("nosuchmodel", "atmos", "mon")
    with pytest.raises(ValueError, match="No include_patterns defined"):
        get_include_patterns("noresm", "nosuchrealm", "mon")
    with pytest.raises(ValueError, match="No include_patterns defined"):
        get_include_patterns("cesm", "land", "6hr")


def test_missing_sampling_raises_rather_than_returning_averages():
    """Asking for instantaneous output that does not exist must not fall back.

    Silently returning the time-averaged files would produce a well-formed but
    wrong instantaneous variable.
    """
    # NorESM daily atmosphere output is time-averaged only.
    with pytest.raises(ValueError, match="sampling=tpt"):
        get_include_patterns("noresm", "atmos", "day", sampling="tpt")
    # seaIce has no instantaneous output at any frequency.
    with pytest.raises(ValueError, match="writes no tpt output"):
        all_include_patterns("noresm", "seaIce", sampling="tpt")


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize(
    "realm,averaged,instantaneous",
    [("atmos", "cam.h0a", "cam.h0i"), ("land", "clm2.h0a", "clm2.h0i")],
)
def test_instantaneous_output_is_declared(model, realm, averaged, instantaneous):
    """Both models write monthly instantaneous output to the 'i' tape."""
    assert get_include_patterns(model, realm, "mon", sampling="tpt") == [instantaneous]
    assert get_include_patterns(model, realm, "mon") == [averaged]
    assert all_include_patterns(model, realm, sampling="tpt") == [instantaneous]


def test_all_include_patterns_spans_frequencies_without_duplicates():
    """NorESM land shares clm2.h2a between 3hr and yr; it appears once."""
    patterns = all_include_patterns("noresm", "land")
    assert patterns == ["clm2.h0a", "clm2.h0i", "clm2.h1a", "clm2.h2a"]
    assert len(patterns) == len(set(patterns))


def test_all_include_patterns_needs_an_ice_sheet_too():
    """The placeholder is filled on the all-frequencies path as well."""
    assert all_include_patterns("noresm", "landIce", "ais") == ["cism.ais.h"]
    with pytest.raises(ValueError, match="requires --ice-sheet"):
        all_include_patterns("noresm", "landIce")


# ---------------------------------------------------------------------------
# Instantaneous output.  No shipped table declares 'tpt' yet, so these use a
# synthetic model table to exercise the code path ahead of the data.
# ---------------------------------------------------------------------------
@pytest.fixture(name="mixed_model")
def mixed_model_fixture(tmp_path, monkeypatch):
    """A model whose mon frequency has both samplings and whose day has only tavg."""
    # Written literally rather than dumped: yaml.safe_dump sorts keys, and the
    # returned pattern order follows file order.
    (tmp_path / "fake_include_patterns.yaml").write_text(
        "atmos:\n"
        "  mon:\n"
        "    tavg: [cam.h0a]\n"
        "    tpt: [cam.h0i]\n"
        "  day:\n"
        "    tavg: [cam.h1a]\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(ip, "DATA_DIR", tmp_path)
    ip._load_cached.cache_clear()  # pylint: disable=protected-access
    yield "fake"
    ip._load_cached.cache_clear()  # pylint: disable=protected-access


def test_sampling_selects_one_file_family(mixed_model):
    """tavg and tpt resolve to different history files at the same frequency."""
    assert get_include_patterns(mixed_model, "atmos", "mon", sampling="tavg") == [
        "cam.h0a"
    ]
    assert get_include_patterns(mixed_model, "atmos", "mon", sampling="tpt") == [
        "cam.h0i"
    ]


def test_sampling_defaults_to_time_average(mixed_model):
    """CMORizing without an explicit sampling gets averaged output."""
    assert get_include_patterns(mixed_model, "atmos", "mon") == ["cam.h0a"]


def test_bulk_collection_takes_both_samplings(mixed_model):
    """gen_timeseries collects everything, since it cannot know what is needed."""
    assert all_include_patterns(mixed_model, "atmos") == [
        "cam.h0a",
        "cam.h0i",
        "cam.h1a",
    ]


def test_bulk_collection_can_be_narrowed(mixed_model):
    """--sampling skips frequencies that lack that sampling entirely."""
    assert all_include_patterns(mixed_model, "atmos", sampling="tavg") == [
        "cam.h0a",
        "cam.h1a",
    ]
    # day has no instantaneous output, so only mon contributes.
    assert all_include_patterns(mixed_model, "atmos", sampling="tpt") == ["cam.h0i"]


# ---------------------------------------------------------------------------
# Routing a CMIP7 variable to the right history files
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "branded,expected",
    [
        ("tas_tavg-u-hxy-u", "tavg"),
        ("ps_tpt-u-hxy-u", "tpt"),
        ("acabf_tavg-u-hxy-is", "tavg"),
        ("areacella_ti-u-hxy-u", "ti"),
        ("tasmax_tmax-u-hxy-u", "tmax"),
    ],
)
def test_sampling_is_read_from_the_branded_name(branded, expected):
    """The signifier sits between the first underscore and the first dash."""
    assert sampling_from_branded_name(branded) == expected


def test_sampling_rejects_a_bare_variable_name():
    """A plain name has no compound part to read a signifier from."""
    with pytest.raises(ValueError, match="not a branded variable name"):
        sampling_from_branded_name("tas")


def test_instantaneous_variable_routes_to_the_i_tape():
    """The point of the exercise: _tpt must not be built from averaged files."""
    assert patterns_for_variable("noresm", "atmos", "mon", "ps_tpt-u-hxy-u") == [
        "cam.h0i"
    ]
    assert patterns_for_variable("noresm", "atmos", "mon", "tas_tavg-u-hxy-u") == [
        "cam.h0a"
    ]


def test_undeclared_signifier_falls_back_and_warns(caplog):
    """ti/tsum/tmin/tmax keep reading averaged files, but say so.

    Failing outright would regress variables that work today; falling back
    silently would hide a real question about which tape they belong in.
    """
    with caplog.at_level("WARNING"):
        patterns = patterns_for_variable(
            "noresm", "atmos", "mon", "areacella_ti-u-hxy-u"
        )
    assert patterns == ["cam.h0a"]
    assert "falling back to time-averaged files" in caplog.text


def test_missing_time_average_still_raises():
    """The fallback must not paper over a genuinely absent realm/frequency."""
    with pytest.raises(ValueError, match="No include_patterns defined"):
        patterns_for_variable("cesm", "land", "6hr", "mrsos_tavg-u-hxy-u")
