"""Tests for scripts/validate_cmor_output.py."""

# pylint: disable=wrong-import-position

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import validate_cmor_output
from validate_cmor_output import (
    collect_log_records,
    parse_log_variable,
    resolve_cmip_root,
    scan_output_tree,
    select_produced_files,
    summarize_dimension_inventory,
)
from build_validation_html import build_site_data, collect_reports


def test_resolve_cmip_root_accepts_parent(tmp_path):
    """A parent directory containing CMIP7/CMIP resolves to CMIP7."""
    cmip_root = tmp_path / "CMIP7" / "CMIP"
    cmip_root.mkdir(parents=True)
    assert resolve_cmip_root(tmp_path) == tmp_path / "CMIP7"


def test_resolve_cmip_root_accepts_direct_cmip7(tmp_path):
    """A direct CMIP7 path resolves unchanged."""
    cmip_root = tmp_path / "CMIP7" / "CMIP"
    cmip_root.mkdir(parents=True)
    print(resolve_cmip_root(tmp_path))
    assert resolve_cmip_root(tmp_path) == tmp_path / "CMIP7"


def test_parse_args_accepts_only_source_ids_for_selected_model(monkeypatch, tmp_path):
    """Source IDs are constrained to the selected model's valid names."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_cmor_output.py",
            "--model",
            "noresm",
            "--source-id",
            "NorESM3-MM",
            "--realm",
            "atmos",
            "--frequency",
            "mon",
            "--root-output-path",
            str(tmp_path),
        ],
    )

    args = validate_cmor_output.parse_args()

    assert args.source_id == "NorESM3-MM"


def test_parse_args_rejects_source_id_from_another_model(monkeypatch, tmp_path):
    """A globally valid source ID cannot be paired with the wrong model."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_cmor_output.py",
            "--model",
            "noresm",
            "--source-id",
            "CESM3",
            "--realm",
            "atmos",
            "--frequency",
            "mon",
            "--root-output-path",
            str(tmp_path),
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        validate_cmor_output.parse_args()


def test_parse_log_variable_extracts_variable_name():
    """CMOR log names carry the branded variable name."""
    assert parse_log_variable(Path("cmor_tas_glb_mon_1-10.log")) == "tas"


def test_collect_log_records_filters_to_expected_variables(tmp_path):
    """Only expected variables should be kept when collecting logs."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "cmor_tas_glb_mon_1-10.log").write_text(
        "ERROR: something failed\n", encoding="utf-8"
    )
    (log_dir / "cmor_pr_glb_mon_1-10.log").write_text("all good\n", encoding="utf-8")

    records = collect_log_records(log_dir, {"tas"})
    assert sorted(records) == ["tas"]
    assert records["tas"][0].has_error is True


def test_collect_log_records_filters_to_the_frequency(tmp_path):
    """A variable written at two frequencies has a log for each, and the
    errors of one say nothing about the other."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "cmor_tas_glb_mon_1-10.log").write_text("all good\n", encoding="utf-8")
    (log_dir / "cmor_tas_glb_day_1-10.log").write_text(
        "ERROR: something failed\n", encoding="utf-8"
    )

    monthly = collect_log_records(log_dir, {"tas"}, "mon")
    assert monthly["tas"][0].has_error is False

    daily = collect_log_records(log_dir, {"tas"}, "day")
    assert daily["tas"][0].has_error is True


def test_scan_output_tree_inventories_dims(tmp_path):
    """Produced CMOR files should be discovered and their dims inventoried."""
    file_path = (
        tmp_path
        / "CMIP7"
        / "CMIP"
        / "NCC"
        / "NorESM3"
        / "piControl"
        / "r1i1p1f1"
        / "glb"
        / "mon"
        / "tas"
        / "hxy"
        / "gr"
        / "tas_Amon_noresm_piControl_r1i1p1f1_gr_185001-185012.nc"
    )
    file_path.parent.mkdir(parents=True)
    data = xr.Dataset(
        {
            "tas": xr.DataArray(
                np.ones((2, 3, 4), dtype=np.float32),
                dims=("time", "lat", "lon"),
                coords={
                    "time": np.array([0, 1]),
                    "lat": np.array([-10.0, 0.0, 10.0]),
                    "lon": np.array([0.0, 90.0, 180.0, 270.0]),
                },
            )
        }
    )
    data.to_netcdf(file_path)

    produced, inventory, errors = scan_output_tree(
        tmp_path / "CMIP7",
        model="noresm",
        source_id="NorESM3",
        experiment="piControl",
        frequency="mon",
        expected_variables={"tas_hxy"},
        ensemble_member=None,
    )
    print(produced)
    assert sorted(produced) == ["tas_hxy"]
    assert not errors
    summary = summarize_dimension_inventory(inventory)
    assert summary[0]["variable"] == "tas"
    assert summary[0]["dims"] == ["time, lat, lon"]
    assert summary[0]["grid_types"] == ["gr"]


def test_scan_output_tree_can_find_a_cross_realm_variable_by_full_brand(tmp_path):
    """Cross-realm brands are checked against the shared CMIP variable path."""
    cmip_root = tmp_path / "CMIP7"
    file_path = (
        cmip_root
        / "CMIP"
        / "NCC"
        / "NorESM3"
        / "piControl"
        / "r1i1p1f1"
        / "glb"
        / "mon"
        / "abs550bc"
        / "tavg-u-hxy-u"
        / "gr"
        / "abs550bc_Amon_noresm_piControl_r1i1p1f1_gr_185001-185012.nc"
    )
    file_path.parent.mkdir(parents=True)
    xr.Dataset({"abs550bc": (("time", "lat", "lon"), np.ones((1, 2, 2)))}).to_netcdf(
        file_path
    )

    produced, inventory, errors = scan_output_tree(
        cmip_root,
        model="noresm",
        source_id="NorESM3",
        experiment="piControl",
        frequency="mon",
        expected_variables={"abs550bc_tavg-u-hxy-u"},
    )

    assert list(produced) == ["abs550bc_tavg-u-hxy-u"]
    assert inventory[0].variable == "abs550bc"
    assert not errors


def test_missing_branded_variable_is_not_hidden_by_shared_root_name(tmp_path):
    """A produced hur brand must not make a different missing hur brand appear."""
    cmip_root = tmp_path / "CMIP7"
    produced_brand = "hur_tavg-p19-hxy-air"
    missing_brand = "hur_tavg-p19-hxy-u"
    file_path = (
        cmip_root
        / "CMIP"
        / "NCC"
        / "NorESM3"
        / "piControl"
        / "r1i1p1f1"
        / "glb"
        / "mon"
        / "hur"
        / "tavg-p19-hxy-air"
        / "gr"
        / "hur_Amon_noresm_piControl_r1i1p1f1_gr_000101-000112.nc"
    )
    file_path.parent.mkdir(parents=True)
    xr.Dataset(
        {
            "hur": (
                ("time", "plev", "lat", "lon"),
                np.ones((1, 1, 2, 2)),
            )
        }
    ).to_netcdf(file_path)
    ownership, expected_variables = validate_cmor_output.classify_validation_variables(
        {produced_brand, missing_brand},
        {produced_brand: ("atmos",), missing_brand: ("atmos",)},
        "atmos",
        {produced_brand: {}, missing_brand: {}},
    )

    scanned_files, inventory, errors = scan_output_tree(
        cmip_root,
        model="noresm",
        source_id="NorESM3",
        experiment="piControl",
        frequency="mon",
        expected_variables=set(expected_variables),
    )
    produced_files = select_produced_files(expected_variables, scanned_files)
    report = validate_cmor_output.build_report(
        args=SimpleNamespace(
            model="noresm",
            source_id="NorESM3",
            realm="atmos",
            experiment="piControl",
            frequency="mon",
            ensemble_member="r1i1p1f1",
            root_output_path=str(tmp_path),
        ),
        yaml_path=tmp_path / "mapping.yaml",
        expected_variables=expected_variables,
        log_records={},
        produced_files=produced_files,
        dimension_inventory=[],
        inspection_errors=errors,
        plot_outputs={"timeseries": {}, "maps": {}, "zonal": {}},
        variable_provenance={},
        mapped_elsewhere=[],
        unmapped_requested_variables=[],
    )

    assert ownership.owned_here == {produced_brand, missing_brand}
    assert list(scanned_files) == [produced_brand]
    assert inventory[0].variable == "hur"
    assert report["produced_variables"] == [produced_brand]
    assert report["expected_but_not_produced"] == [missing_brand]


def test_collect_reports_prefers_source_specific_over_legacy_reports(tmp_path):
    """Old pooled reports do not appear beside their source-specific replacement."""
    reports = tmp_path / "validation_reports"
    legacy = reports / "noresm_atmos_piControl_mon"
    current = reports / "noresm_NorESM3-LM_atmos_piControl_mon"
    legacy.mkdir(parents=True)
    current.mkdir(parents=True)
    base_scope = {
        "model": "noresm",
        "realm": "atmos",
        "experiment": "piControl",
        "frequency": "mon",
        "ensemble_member": None,
    }
    (legacy / "validation_summary.json").write_text(
        json.dumps({"scope": base_scope}), encoding="utf-8"
    )
    (current / "validation_summary.json").write_text(
        json.dumps({"scope": {**base_scope, "source_id": "NorESM3-LM"}}),
        encoding="utf-8",
    )

    collected = collect_reports(reports)

    assert [report["subset_id"] for report in collected] == [
        "noresm_NorESM3-LM_atmos_piControl_mon"
    ]


def test_scan_output_tree_uses_only_selected_source_id(tmp_path):
    """Files from other resolutions are excluded from validation and overlap checks."""
    cmip_root = tmp_path / "CMIP7"
    expected = "tas_hxy"
    paths = []
    for source_id in ("NorESM3-LM", "NorESM3-MM"):
        file_path = (
            cmip_root
            / "CMIP"
            / "NCC"
            / source_id
            / "piControl"
            / "r1i1p1f1"
            / "glb"
            / "mon"
            / "tas"
            / "hxy"
            / "gr"
            / f"tas_Amon_noresm_piControl_r1i1p1f1_gr_{source_id}.nc"
        )
        file_path.parent.mkdir(parents=True)
        xr.Dataset({"tas": (("time", "lat", "lon"), np.ones((1, 2, 2)))}).to_netcdf(
            file_path
        )
        paths.append(file_path)

    produced, inventory, errors = scan_output_tree(
        cmip_root,
        model="noresm",
        source_id="NorESM3-LM",
        experiment="piControl",
        frequency="mon",
        expected_variables={expected},
    )

    assert produced[expected] == [paths[0]]
    assert len(inventory) == 1
    assert not errors


def test_default_report_dir_separates_source_ids(tmp_path):
    """Resolution-specific validation reports do not overwrite each other."""
    base_args = {
        "root_output_path": str(tmp_path),
        "model": "noresm",
        "realm": "atmos",
        "experiment": "piControl",
        "frequency": "mon",
    }

    lm_dir = validate_cmor_output.default_report_dir(
        SimpleNamespace(**base_args, source_id="NorESM3-LM")
    )
    mm_dir = validate_cmor_output.default_report_dir(
        SimpleNamespace(**base_args, source_id="NorESM3-MM")
    )

    assert lm_dir != mm_dir
    assert "NorESM3-LM" in lm_dir.name


def test_validation_site_data_carries_realm_ownership_categories():
    """The HTML report receives cross-realm and mapping-error classifications."""
    data = build_site_data(
        [
            {
                "subset_id": "noresm_atmosChem_historical_mon",
                "mapped_elsewhere": [
                    {
                        "variable": "abs550bc_tavg-u-hxy-u",
                        "owner_realms": ["aerosol"],
                        "found_in_output": True,
                        "files": ["abs550bc.nc"],
                    }
                ],
                "unmapped_requested_variables": ["ccldncl_tavg-u-hxy-ccl"],
                "expected_variables": [],
                "produced_variables": [],
            }
        ]
    )

    subset = data["subsets"][0]
    assert subset["mapped_elsewhere"][0]["found_in_output"] is True
    assert subset["unmapped_requested_variables"] == ["ccldncl_tavg-u-hxy-ccl"]
    mapped_variable = next(
        variable
        for variable in subset["variables"]
        if variable["name"] == "abs550bc_tavg-u-hxy-u"
    )
    assert mapped_variable["status"] == "mapped_elsewhere"
    assert mapped_variable["mapping_owner"] == ["aerosol"]
    assert mapped_variable["found_in_output"] is True
    assert mapped_variable["output_files"] == ["abs550bc.nc"]
