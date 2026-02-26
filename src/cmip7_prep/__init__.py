"""CMIP7 preparation toolkit: regridding and CMOR writing for CESM outputs."""

import subprocess


def _get_git_version():
    try:
        version = (
            subprocess.check_output(
                ["git", "describe", "--tags", "--abbrev=0"], stderr=subprocess.STDOUT
            )
            .decode("utf-8")
            .strip()
        )
        return version
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


__version__ = _get_git_version()

__all__ = ["regrid", "cmor_writer"]
