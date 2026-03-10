"""Tests for cache_tools.py: FXCache, RegridderCache, and open_nc."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from cmip7_prep.cache_tools import FXCache, RegridderCache, open_nc


@pytest.fixture(autouse=True)
def clear_caches():
    """Clear class-level caches before and after each test."""
    FXCache.clear()
    RegridderCache.clear()
    yield
    FXCache.clear()
    RegridderCache.clear()


class TestFXCache:
    """Unit tests for the FXCache class-level dictionary cache."""

    def test_get_missing_returns_none(self):
        """get() returns None for a key that was never put."""
        assert FXCache.get(Path("nonexistent.nc")) is None

    def test_put_then_get(self):
        """get() returns the exact object that was put."""
        key = Path("map.nc")
        ds = xr.Dataset({"sftlf": xr.DataArray(np.ones((3, 3)))})
        FXCache.put(key, ds)
        assert FXCache.get(key) is ds

    def test_clear_empties_cache(self):
        """After clear(), get() returns None for previously stored keys."""
        key = Path("map.nc")
        FXCache.put(key, xr.Dataset({"sftlf": xr.DataArray(np.ones((2, 2)))}))
        FXCache.clear()
        assert FXCache.get(key) is None

    def test_multiple_keys_stored_independently(self):
        """Multiple keys coexist without interfering with each other."""
        k1, k2 = Path("a.nc"), Path("b.nc")
        ds1 = xr.Dataset({"sftlf": xr.DataArray([1.0])})
        ds2 = xr.Dataset({"areacella": xr.DataArray([2.0])})
        FXCache.put(k1, ds1)
        FXCache.put(k2, ds2)
        assert FXCache.get(k1) is ds1
        assert FXCache.get(k2) is ds2

    def test_put_overwrites_existing_key(self):
        """Putting the same key twice replaces the earlier value."""
        key = Path("map.nc")
        ds1 = xr.Dataset({"a": xr.DataArray([1.0])})
        ds2 = xr.Dataset({"a": xr.DataArray([2.0])})
        FXCache.put(key, ds1)
        FXCache.put(key, ds2)
        assert FXCache.get(key) is ds2


class TestRegridderCache:
    """Unit tests for the RegridderCache class-level cache."""

    def test_clear_does_not_raise_on_empty(self):
        """clear() is a no-op when the cache is already empty."""
        RegridderCache.clear()  # should not raise

    def test_get_missing_file_raises(self, tmp_path):
        """get() raises FileNotFoundError when the weight file does not exist."""
        with pytest.raises(FileNotFoundError, match="Regrid weights not found"):
            RegridderCache.get(tmp_path / "nonexistent.nc", "conservative")


class TestOpenNc:
    """Unit tests for the open_nc helper function."""

    def test_missing_file_raises(self, tmp_path):
        """open_nc raises FileNotFoundError for a non-existent path."""
        with pytest.raises(FileNotFoundError, match="Weight file not found"):
            open_nc(tmp_path / "ghost.nc")

    def test_opens_valid_netcdf(self, tmp_path):
        """open_nc successfully opens a valid NetCDF file."""
        path = tmp_path / "test.nc"
        ds = xr.Dataset({"x": xr.DataArray([1.0, 2.0, 3.0])})
        ds.to_netcdf(path)
        result = open_nc(path)
        assert "x" in result
        result.close()
