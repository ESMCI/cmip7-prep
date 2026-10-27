"""Tests for scripts/convert_request_csv.py."""

import csv
import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
# pylint: disable=wrong-import-position
from convert_request_csv import CONFIG, read_request_csv
from cmip7_prep.schema import mapping_errors

COLUMNS = [
    "Branded Variable Name",
    "Modelling Realm - Primary",
    "CESM Variable Name (T4S)",
    "Formula",
    "CMIP7 Frequency",
    "Region",
]


def _read(tmp_path, rows):
    """Write ``rows`` (tuples in COLUMNS order) to a CSV and read it back."""
    path = tmp_path / "request.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        writer.writerows(rows)
    return read_request_csv(path, CONFIG)


def test_rename_and_formula(tmp_path):
    """A bare source is a rename; a formula is kept, minus its trailing comment."""
    mappings, skipped = _read(
        tmp_path,
        [
            ("tas_tavg-h2m-hxy-u", "atmos", "TREFHT", "", "mon", "glb"),
            (
                "clt_tavg-u-hxy-u",
                "atmos",
                "CLDTOT",
                "CLDTOT*100 #cam.h0a",
                "mon",
                "glb",
            ),
        ],
    )
    assert mappings["atmos"] == {
        "tas_tavg-h2m-hxy-u": "TREFHT",
        "clt_tavg-u-hxy-u": "CLDTOT*100",
    }
    assert not skipped


def test_chunits_becomes_units(tmp_path):
    """chunits(expr, units=...) is the schema's ``units`` field."""
    mappings, _ = _read(
        tmp_path,
        [("c_tavg-u-hxy-lnd", "land", "X", "chunits(X, units='g m-2')", "mon", "glb")],
    )
    assert mappings["land"]["c_tavg-u-hxy-lnd"] == {"formula": "X", "units": "g m-2"}


def test_formula_per_frequency(tmp_path):
    """Rows that differ by frequency become a per-frequency formula."""
    mappings, _ = _read(
        tmp_path,
        [
            ("siage_tavg-u-hxy-si", "seaIce", "siage_d, siage", "", "day", "glb"),
            ("siage_tavg-u-hxy-si", "seaIce", "siage_d, siage", "", "mon", "glb"),
        ],
    )
    assert mappings["seaIce"]["siage_tavg-u-hxy-si"] == {
        "formula": {"day": "siage_d", "mon": "siage"}
    }


def test_same_formula_across_regions_is_one_entry(tmp_path):
    """A region that only repeats the global formula adds nothing."""
    mappings, _ = _read(
        tmp_path,
        [
            ("so2_tavg-al-hxy-u", "aerosol", "SO2", "", "mon", "glb"),
            ("so2_tavg-al-hxy-u", "aerosol", "SO2", "", "mon", "30S-90S"),
        ],
    )
    assert mappings["aerosol"] == {"so2_tavg-al-hxy-u": "SO2"}


def test_differing_regions_become_variants(tmp_path):
    """Different formulas per region are separate outputs."""
    mappings, _ = _read(
        tmp_path,
        [
            ("v_tavg-u-hm-u", "seaIce", "a", "a*2", "mon", "nh"),
            ("v_tavg-u-hm-u", "seaIce", "a", "a*3", "mon", "sh"),
        ],
    )
    assert mappings["seaIce"]["v_tavg-u-hm-u"] == {
        "variants": [
            {"region": "nh", "formula": "a*2"},
            {"region": "sh", "formula": "a*3"},
        ]
    }


@pytest.mark.parametrize(
    "source, formula, reason",
    [
        ("N/A", "", "no CESM source variable"),
        ("A, B", "", "several source variables"),
        ("X", "yeartomonth_data(X)", "unsupported function yeartomonth_data"),
        ("X", "X when above TROP_P", "not a valid expression"),
    ],
)
def test_unmappable_rows_are_reported(tmp_path, source, formula, reason):
    """Rows the pipeline cannot evaluate are skipped and say why."""
    mappings, skipped = _read(
        tmp_path, [("v_tavg-u-hxy-u", "atmos", source, formula, "mon", "glb")]
    )
    assert mappings["atmos"] == {}
    assert [row[:2] for row in skipped] == [("atmos", "v_tavg-u-hxy-u")]
    assert reason in skipped[0][2]


def test_other_realms_are_ignored(tmp_path):
    """Realms without a mapping file are not an error."""
    mappings, skipped = _read(tmp_path, [("v", "ocean2", "X", "", "mon", "glb")])
    assert "ocean2" not in mappings and not skipped


def test_checked_in_files_match_the_csv():
    """data/cesm_to_cmip7_*.yaml are what the converter makes of the request CSV."""
    data = os.path.join(os.path.dirname(__file__), "..", "data")
    csv_path = os.path.join(data, "cmip7.data_request.cesm.v1.0.airtv1225.csv")
    mappings, _ = read_request_csv(csv_path, CONFIG)
    for realm, variables in mappings.items():
        with open(
            os.path.join(data, f"cesm_to_cmip7_{realm}.yaml"), encoding="utf-8"
        ) as f:
            on_disk = yaml.safe_load(f)["variables"]
        assert on_disk == variables, realm
        assert not mapping_errors({"variables": on_disk})


def test_positional_slices_work_on_unstructured_arrays(tmp_path):
    """X[:,k,:,:] becomes X[:,k,...], valid with or without lat and lon."""
    mappings, _ = _read(
        tmp_path,
        [("f_tavg-u-hxy-u", "land", "P", "P[:,1,:,:]+P[:,2,:,:]", "mon", "glb")],
    )
    assert mappings["land"]["f_tavg-u-hxy-u"] == "P[:,1,...]+P[:,2,...]"


def test_tuple_formula_is_skipped(tmp_path):
    """A comma where a + was meant would make a tuple; it is reported instead."""
    mappings, skipped = _read(
        tmp_path, [("f_tavg-u-hxy-lnd", "land", "A, B, C", "(A+B,C)", "mon", "glb")]
    )
    assert mappings["land"] == {}
    assert "tuple" in skipped[0][2]
