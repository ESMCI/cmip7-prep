"""Tests for scripts/validate_cmor_output.py."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from validate_cmor_output import (  # pylint: disable=wrong-import-position
    collect_log_records,
    parse_log_variable,
    resolve_cmip_root,
    scan_output_tree,
    summarize_dimension_inventory,
)


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


def test_parse_log_variable_extracts_variable_name():
    """CMOR log names carry the branded variable name."""
    assert parse_log_variable(Path("cmor_20260101T010203Z_tas.log")) == "tas"


def test_collect_log_records_filters_to_expected_variables(tmp_path):
    """Only expected variables should be kept when collecting logs."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "cmor_20260101T010203Z_tas.log").write_text(
        "ERROR something failed\n", encoding="utf-8"
    )
    (log_dir / "cmor_20260101T010203Z_pr.log").write_text(
        "all good\n", encoding="utf-8"
    )

    records = collect_log_records(log_dir, {"tas"})
    assert sorted(records) == ["tas"]
    assert records["tas"][0].has_error is True


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
        experiment="piControl",
        frequency="mon",
        expected_variables={"tas_hxy"},
        ensemble_member=None,
        resolution=None,
    )

    assert sorted(produced) == ["tas_hxy"]
    assert not errors
    summary = summarize_dimension_inventory(inventory)
    assert summary[0]["variable"] == "tas"
    assert summary[0]["dims"] == ["time, lat, lon"]
    assert summary[0]["grid_types"] == ["gr"]
