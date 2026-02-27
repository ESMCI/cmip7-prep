"""CMIP7 preparation toolkit: regridding and CMOR writing for CESM outputs."""

import importlib.metadata as importlib_metadata


def _get_version() -> str:
    """Return the installed package version, or 'unknown' if not available."""
    try:
        return importlib_metadata.version("cmip7-prep")
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


__version__ = _get_version()
