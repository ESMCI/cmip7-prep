r"""Utilities for reading and querying a CMIP Data Request export.

Supports Airtable-style JSON (with a top-level ``records`` list) and CSV exports.
Provides a small helper class :class:`DReq` to select variables by table (e.g., Amon)
and by group tags (e.g., "baseline_monthly").

Example
-------
>>> dr = DReq("data_request_v1.2.2.csv")
>>> vars_amon = dr.select(table="Amon", group_regexes=[r"baseline[_\s-]*monthly"])
>>> 'tas' in vars_amon
True
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional


def _compile_patterns(regexes: Optional[Iterable[str]]) -> List[re.Pattern[str]]:
    """Compile a list of regexes case-insensitively.

    Parameters
    ----------
    regexes : iterable of str or None
        Patterns to compile. If None, returns an empty list.

    Returns
    -------
    list[Pattern]
        Compiled regexes (case-insensitive).
    """
    if not regexes:
        return []
    return [re.compile(rx, re.IGNORECASE) for rx in regexes]


class DReq:
    """Reader/query helper for a CMIP Data Request export (CSV or JSON).

    Attributes
    ----------
    path : Path
        Path to the Data Request export.
    rows : list[dict]
        Parsed rows (Airtable fields dicts).
    table_keys : list[str]
        Column names that may hold the CMOR table value.
    group_keys : list[str]
        Column names that may hold grouping/bucket tags.
    short_name_keys : list[str]
        Column names that may hold the CMIP variable short name.
    group_patterns : list[Pattern]
        Regex patterns used to detect the desired group (optional).
    """

    # Heuristics for common column names across exports
    _DEFAULT_TABLE_KEYS = ["Table", "table", "CMOR table", "CMOR_table"]
    _DEFAULT_GROUP_KEYS = [
        "Group",
        "Groups",
        "Bucket",
        "MIP bucket",
        "Baseline group",
        "Notes",
        "Tags",
    ]
    _DEFAULT_SHORT_NAME_KEYS = [
        "Short name",
        "short_name",
        "Variable",
        "Name",
        "name",
        "CMIP Variable",
        "Compound name",
    ]

    def __init__(
        self,
        path: str | Path,
        *,
        group_regexes: Optional[Iterable[str]] = None,
        table_keys: Optional[List[str]] = None,
        group_keys: Optional[List[str]] = None,
        short_name_keys: Optional[List[str]] = None,
    ) -> None:
        self.path: Path = Path(path)
        self.rows: List[Dict[str, Any]] = self._read(self.path)
        self.table_keys: List[str] = table_keys or self._DEFAULT_TABLE_KEYS
        self.group_keys: List[str] = group_keys or self._DEFAULT_GROUP_KEYS
        self.short_name_keys: List[str] = (
            short_name_keys or self._DEFAULT_SHORT_NAME_KEYS
        )
        self.group_patterns = _compile_patterns(group_regexes)

    # ---------------------------
    # I/O
    # ---------------------------

    @staticmethod
    def _read(path: Path) -> List[Dict[str, Any]]:
        """Read a CSV or JSON export into a list of dictionaries.

        Supports Airtable JSON with ``{"records": [{"fields": {...}}, ...]}``.
        """
        suffix = path.suffix.lower()
        if suffix == ".json":
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "records" in data:
                return [rec.get("fields", {}) for rec in data["records"]]
            if isinstance(data, list):
                return [dict(row) for row in data]
            raise ValueError("Unrecognized JSON structure for data request.")
        # CSV fallback
        with path.open("r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    # ---------------------------
    # Row helpers
    # ---------------------------

    def _field(self, row: Dict[str, Any], keys: Iterable[str]) -> str:
        """Return first non-empty field value among candidate keys as string."""
        for k in keys:
            if k in row and row[k] not in (None, ""):
                return str(row[k])
        return ""

    def _table_name(self, row: Dict[str, Any]) -> str:
        """Return the normalized CMOR table name (e.g., 'Amon')."""
        table = self._field(row, self.table_keys)
        return table.replace(".json", "")

    def _short_name(self, row: Dict[str, Any]) -> Optional[str]:
        """Return the CMIP variable short name, if present.

        Also supports "Compound name" like "Amon.tas" by splitting on '.'.
        """
        for k in self.short_name_keys:
            v = row.get(k)
            if v:
                v = str(v).strip()
                if k == "Compound name" and "." in v:
                    return v.split(".", 1)[1]
                return v
        return None

    def _matches_group(self, row: Dict[str, Any]) -> bool:
        """Return True if any group pattern matches the row's group/bucket text.

        If no group_patterns were provided at construction, this returns False.
        """
        if not self.group_patterns:
            return False
        hay = " ".join(self._field(row, [k]) for k in self.group_keys)
        return any(p.search(hay) for p in self.group_patterns)

    # ---------------------------
    # Public selection
    # ---------------------------

    def select(
        self,
        *,
        table: str = "Amon",
        include_all_in_table: bool = False,
    ) -> List[str]:
        """Select variable short names by table and (optionally) group.

        Parameters
        ----------
        table : str, default 'Amon'
            CMOR table to filter on (e.g., 'Amon', 'Lmon').
        include_all_in_table : bool, default False
            If True, ignore group_patterns and return all variables in the table.

        Returns
        -------
        list[str]
            Ordered unique list of short names.
        """
        selected: List[str] = []
        for row in self.rows:
            if self._table_name(row) != table:
                continue
            if include_all_in_table or self._matches_group(row):
                name = self._short_name(row)
                if name:
                    selected.append(name)

        # de-duplicate preserving order
        seen: set[str] = set()
        ordered: List[str] = []
        for name in selected:
            if name not in seen:
                seen.add(name)
                ordered.append(name)
        return ordered

    def rows_for(self, short_name: str) -> List[Dict[str, Any]]:
        """Return all Data Request rows matching a given CMIP variable short name."""
        out: List[Dict[str, Any]] = []
        for row in self.rows:
            name = self._short_name(row)
            if name == short_name:
                out.append(row)
        return out
