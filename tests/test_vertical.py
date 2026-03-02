"""Tests for vertical.py: _read_requested_levels, _resolve_p0, and to_plev."""

import json
import os

import numpy as np
import pytest
import xarray as xr

from cmip7_prep.vertical import _read_requested_levels, _resolve_p0, to_plev


# ---------------------------------------------------------------------------
# _read_requested_levels
# ---------------------------------------------------------------------------


class TestReadRequestedLevels:
    """Tests for _read_requested_levels pressure-level JSON reader."""

    def _write_coord_json(self, dirpath, levels, filename="CMIP7_coordinate.json"):
        """Write a minimal coordinate JSON file and return the directory path."""
        data = {"axis_entry": {"plev19": {"requested": levels}}}
        path = os.path.join(str(dirpath), filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        return dirpath

    def test_reads_plev19_from_cmip7(self, tmp_path):
        """Reads requested levels from a CMIP7_coordinate.json file."""
        levels = [100000, 92500, 85000]
        self._write_coord_json(str(tmp_path), levels)
        result = _read_requested_levels(tmp_path, axis_name="plev19")
        np.testing.assert_array_equal(result, np.array(levels, dtype="f8"))

    def test_reads_from_cmip6_fallback(self, tmp_path):
        """Falls back to CMIP6_coordinate.json when the CMIP7 file is absent."""
        levels = [100000, 50000]
        data = {"axis_entry": {"plev19": {"requested": levels}}}
        (tmp_path / "CMIP6_coordinate.json").write_text(
            json.dumps(data), encoding="utf-8"
        )
        result = _read_requested_levels(tmp_path, axis_name="plev19")
        np.testing.assert_array_equal(result, np.array(levels, dtype="f8"))

    def test_raises_if_no_coord_json(self, tmp_path):
        """Raises FileNotFoundError when no coordinate JSON is found."""
        with pytest.raises(FileNotFoundError, match="coordinate table JSON"):
            _read_requested_levels(tmp_path, axis_name="plev19")

    def test_returns_float64(self, tmp_path):
        """Returned array has dtype float64."""
        self._write_coord_json(str(tmp_path), [100000, 50000])
        result = _read_requested_levels(tmp_path)
        assert result.dtype == np.float64


# ---------------------------------------------------------------------------
# _resolve_p0
# ---------------------------------------------------------------------------


class TestResolveP0:
    """Tests for _resolve_p0 reference-pressure resolver."""

    def test_scalar_variable(self):
        """Reads P0 from a scalar dataset variable."""
        ds = xr.Dataset({"P0": ((), 95000.0)})
        assert _resolve_p0(ds) == pytest.approx(95000.0)

    def test_global_attribute(self):
        """Reads P0 from a global dataset attribute when the variable is absent."""
        ds = xr.Dataset(attrs={"P0": 98000.0})
        assert _resolve_p0(ds) == pytest.approx(98000.0)

    def test_default_when_absent(self):
        """Returns 100000.0 Pa when P0 is not present in the dataset."""
        assert _resolve_p0(xr.Dataset()) == pytest.approx(100000.0)

    def test_size1_array(self):
        """Reads P0 from a size-1 array variable."""
        ds = xr.Dataset({"P0": ("lev", [97500.0])})
        assert _resolve_p0(ds) == pytest.approx(97500.0)


# ---------------------------------------------------------------------------
# to_plev
# ---------------------------------------------------------------------------


def _make_hybrid_ds(nlev: int = 5, ntim: int = 2, ncol: int = 4):
    """Return a minimal hybrid-level dataset for plev interpolation tests."""
    hyam = np.linspace(0.0, 0.3, nlev)
    hybm = np.linspace(0.0, 0.7, nlev)
    ps_data = np.full((ntim, ncol), 101325.0)
    ta_data = np.random.default_rng(0).uniform(200.0, 300.0, (ntim, nlev, ncol))
    return xr.Dataset(
        {
            "ta": xr.DataArray(ta_data, dims=["time", "lev", "ncol"]),
            "PS": xr.DataArray(ps_data, dims=["time", "ncol"]),
            "hyam": xr.DataArray(hyam, dims=["lev"]),
            "hybm": xr.DataArray(hybm, dims=["lev"]),
            "P0": ((), 100000.0),
        }
    )


def _write_coord_json(dirpath, levels, axis_name="plev19"):
    """Write a minimal CMIP7_coordinate.json to dirpath."""
    data = {"axis_entry": {axis_name: {"requested": levels}}}
    path = os.path.join(str(dirpath), "CMIP7_coordinate.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


class TestToPlev:
    """Integration tests for the to_plev hybrid-to-pressure interpolation."""

    def test_output_has_plev_dim(self, tmp_path):
        """Output variable has a 'plev' dimension."""
        _write_coord_json(tmp_path, [90000.0, 70000.0, 50000.0])
        ds = _make_hybrid_ds()
        result = to_plev(ds, "ta", tmp_path, target="plev19")
        assert "plev" in result["ta"].dims

    def test_plev_coordinate_values(self, tmp_path):
        """Output plev coordinate matches the requested levels."""
        requested = [90000.0, 70000.0, 50000.0]
        _write_coord_json(tmp_path, requested)
        ds = _make_hybrid_ds()
        result = to_plev(ds, "ta", tmp_path, target="plev19")
        np.testing.assert_array_equal(result["ta"].coords["plev"].values, requested)

    def test_drops_hybrid_coeffs(self, tmp_path):
        """hyam, hybm, and P0 are removed from the output dataset."""
        _write_coord_json(tmp_path, [90000.0, 50000.0])
        ds = _make_hybrid_ds()
        result = to_plev(ds, "ta", tmp_path, target="plev19")
        assert "hyam" not in result
        assert "hybm" not in result
        assert "P0" not in result

    def test_raises_on_missing_ps(self, tmp_path):
        """KeyError is raised when PS is absent from the dataset."""
        _write_coord_json(tmp_path, [90000.0])
        ds = _make_hybrid_ds().drop_vars("PS")
        with pytest.raises(KeyError, match="Missing required variables"):
            to_plev(ds, "ta", tmp_path, target="plev19")
