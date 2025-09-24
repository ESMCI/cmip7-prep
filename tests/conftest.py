"""Shared test fixtures and a FakeCMOR stand-in for unit tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple
import uuid
import numpy as np
import pytest

# Import production module at top-level (avoids C0415)
from cmip7_prep import cmor_writer as cw


class FakeCMOR:  # pylint: disable=too-many-instance-attributes
    """Minimal CMOR stand-in that mimics constants and API surface used by cmor_writer."""

    # Common constants across CMOR builds
    CMOR_REPLACE = 0
    CMOR_REPLACE_3 = 0
    CMOR_APPEND = 1
    CMOR_APPEND_3 = 1
    CMOR_NORMAL = 1
    CMOR_VERBOSE = 2
    CMOR_QUIET = 0

    def __init__(self) -> None:
        """Create a fake CMOR session state."""
        self.attrs: dict[str, Any] = {}
        self.inpath: str | None = None
        self.dataset_json_path: str | None = None
        self.last_table: str | None = None
        self.axis_calls: list[tuple] = []
        self.variable_calls: list[tuple] = []
        self.write_calls: list[tuple] = []
        self.logfile: str | None = None
        self.closed = False
        self.closed_file = ""

    # --- core API used by CmorSession ---
    # pylint: disable=unused-argument
    def setup(
        self,
        inpath: str,
        netcdf_file_action: int | None = None,
        set_verbosity: int | None = None,
    ) -> None:
        """Record the tables path."""
        self.inpath = str(inpath)

    def dataset_json(self, rcfile: str) -> None:
        """Record the dataset JSON path and seed a few CV attributes."""
        self.dataset_json_path = str(rcfile)
        ip = Path(self.inpath or ".")
        self.attrs.setdefault("_controlled_vocabulary_file", str(ip / "CMIP6_CV.json"))
        self.attrs.setdefault("_AXIS_ENTRY_FILE", str(ip / "CMIP6_coordinate.json"))
        self.attrs.setdefault("_FORMULA_VAR_FILE", str(ip / "CMIP6_formula_terms.json"))

    # New API names
    def set_cur_dataset_attribute(self, key: str, value: Any) -> None:
        """Set a dataset/global attribute."""
        self.attrs[str(key)] = value

    def get_cur_dataset_attribute(self, key: str) -> Any:
        """Get a dataset/global attribute."""
        return self.attrs.get(str(key), "")

    # Legacy API names (keep camelCase to match CMOR)  # pylint: disable=invalid-name
    def setGblAttr(self, key: str, value: Any) -> None:  # noqa: N802
        """Legacy alias for set_cur_dataset_attribute."""
        self.set_cur_dataset_attribute(key, value)

    def getGblAttr(self, key: str) -> Any:  # noqa: N802
        """Legacy alias for get_cur_dataset_attribute."""
        return self.get_cur_dataset_attribute(key)

    # Tables & variable definitions
    def load_table(self, name: str) -> int:
        """Record last table loaded and return a fake handle."""
        self.last_table = str(name)
        return 0

    def axis(
        self, table_entry: str, units: str, coord_vals, cell_bounds=None, **_
    ) -> int:
        """Record axis definition and return a fake axis id."""
        self.axis_calls.append(
            (
                table_entry,
                units,
                np.asarray(coord_vals),
                None if cell_bounds is None else np.asarray(cell_bounds),
            )
        )
        return len(self.axis_calls)

    def variable(self, table_entry: str, units: str, axis_ids, **_) -> int:
        """Record variable definition and return a fake var id."""
        self.variable_calls.append((table_entry, units, tuple(axis_ids)))
        return 10

    def write(self, var_id: int, data, **_) -> None:  # pylint: disable=unused-argument
        """Record a write call; auto-generate tracking_id from tracking_prefix if needed."""
        tid = self.attrs.get("tracking_id", "")
        prefix = self.attrs.get("tracking_prefix", "")
        if (
            (tid == "" or tid is None)
            and isinstance(prefix, str)
            and prefix.startswith("hdl:")
        ):
            self.attrs["tracking_id"] = f"{prefix}{uuid.uuid4()}"
        self.write_calls.append((var_id, np.asarray(data)))

    # pylint: disable=unused-argument
    def close(self, var_id: int | None = None, file_name: str | None = None) -> None:
        """Close the current object/file."""
        self.closed = True
        self.closed_file = file_name or ""


@pytest.fixture()
def fake_cmor(monkeypatch, tmp_path) -> Tuple[FakeCMOR, Path]:
    """Patch cmor_writer.cmor with FakeCMOR and return (fake, tables_path)."""
    fake = FakeCMOR()
    monkeypatch.setattr(cw, "cmor", fake, raising=True)
    tables = tmp_path / "CMIP6_Tables"
    tables.mkdir()
    return fake, tables


# Back-compat for any older tests that reference `_FakeCMOR`
_FakeCMOR = FakeCMOR  # noqa: N816
