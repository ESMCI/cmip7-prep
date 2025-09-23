# cmip7_prep/dreq_search.py
"""Search utilities for the CMIP data request v1.2.2 CSV.

This module supports:
- Dotted queries like "Amon.ta" or "Amon." (table + variable prefix).
- Subsetting by "CMIP7 Variable Groups" (e.g., baseline_monthly).
- Choosing which column to match on and which column to return.
By default both matching and returned values use the "Physical Parameter" column.
"""

from __future__ import annotations

import csv
import io
import re
from importlib.resources import as_file
from importlib.resources import files
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Union

# Column aliases (lowercased) tailored to the provided header
_NAME_FIELDS = (
    "branded variable name",  # fallback if physical parameter missing
    "cmip6 compound name",
    "cmip7 compound name",
)
_TABLE_FIELDS = (
    "cmip6 table (legacy)",
    "table section (cmip6)",
)
_GROUP_FIELDS = ("cmip7 variable groups",)
_PHYSICAL_FIELD = "physical parameter"  # primary short CMIP var name

_SPLIT = re.compile(r"[;,]\s*|\s*\|\s*")


def packaged_dreq_csv(filename: str = "datarequest_v1_2_2_variables.csv") -> Path:
    """Return a concrete filesystem Path to the packaged data request CSV."""
    res = files("cmip7_prep.data").joinpath(filename)
    with as_file(res) as pth:
        return Path(pth)


def _normalize_row(raw_row: Mapping[str, Any]) -> dict[str, str]:
    """Lowercase headers and coerce values to stripped strings (or '')."""
    return {
        str(k).strip().lower(): ("" if v is None else str(v).strip())
        for k, v in raw_row.items()
    }


def _first_nonempty(row: Mapping[str, str], keys: Sequence[str]) -> str:
    """Return the first non-empty cell among the given keys ('' if none)."""
    for key in keys:
        val = row.get(key, "")
        if val:
            return val
    return ""


def _split_groups(cell: str) -> list[str]:
    """Split a groups cell on commas/semicolons/pipes/spaces."""
    if not cell:
        return []
    toks = [t for t in _SPLIT.split(cell) if t]
    if len(toks) <= 1 and " " in cell:
        toks = [t for t in cell.split() if t]
    return toks


def _extract_table_from_compound(compound: str) -> str:
    """Extract the table (e.g., 'Amon') from 'Amon.tas' or 'Amon tas'."""
    if not compound:
        return ""
    if "." in compound:
        return compound.split(".", 1)[0].strip()
    return compound.split()[0].strip()


def _extract_name_from_compound(compound: str) -> str:
    """Extract the variable name (e.g., 'tas') from 'Amon.tas' or 'Amon tas'."""
    if not compound:
        return ""
    if "." in compound:
        return compound.split(".", 1)[1].strip()
    toks = compound.split()
    return toks[1].strip() if len(toks) > 1 else ""


def _detect_dialect(text: str) -> csv.Dialect:
    """Detect CSV dialect; fallback to a simple comma-delimited dialect."""
    try:
        return csv.Sniffer().sniff(text, delimiters=",;\t")
    except csv.Error:
        # pylint: disable=too-few-public-methods
        class _D(csv.excel):
            delimiter = ","

        return _D()


def find_variables_by_prefix(
    csv_path: Union[str, Path, None],
    prefix: str,
    *,
    include_groups: Iterable[str] | None = None,
    exclude_groups: Iterable[str] | None = None,
    where: Mapping[str, Union[str, Sequence[str]]] | None = None,
    exact: bool = False,
    match_field: Optional[str] = _PHYSICAL_FIELD,
    return_field: Optional[str] = _PHYSICAL_FIELD,
) -> List[str]:
    """Search the Data Request CSV for variables.

    Supports dotted queries:
      - "Amon."      → all variables in Amon
      - "Amon.ta"    → variables in Amon whose *match_field* starts with 'ta'
      - "AERmon.mmr" → variables in AERmon whose *match_field* starts with 'mmr'

    By default, both matching and returned values use the **Physical Parameter** column.

    Parameters
    ----------
    csv_path
        Path to the CSV; if None, uses the packaged file under cmip7_prep/data/.
    prefix
        Variable prefix or dotted query "<table>.<prefix>".
    include_groups / exclude_groups
        Filter rows by entries in "CMIP7 Variable Groups".
    where
        Extra case-insensitive equality filters, e.g. {"cmip7 frequency": "monthly"}.
        Special key "table" refers to "CMIP6 Table (legacy)".
    exact
        If True, require an exact match on *match_field* (instead of startswith).
    match_field
        Column to match against (default: "Physical Parameter").
    return_field
        Column to return (default: "Physical Parameter").

    Returns
    -------
    list[str]
        Sorted unique values from *return_field* for rows that match.
    """
    if csv_path is None:
        csv_path = packaged_dreq_csv()

    include_groups = {g.lower() for g in (include_groups or [])}
    exclude_groups = {g.lower() for g in (exclude_groups or [])}
    where = dict(where or {})

    # Dotted query parsing
    table_from_prefix: Optional[str] = None
    var_prefix = prefix
    if "." in prefix:
        left, right = prefix.split(".", 1)
        table_from_prefix = (left or "").strip()
        var_prefix = (right or "").strip()
    var_prefix_lc = var_prefix.lower()

    # Inject table filter from dotted query
    if table_from_prefix:
        where["table"] = table_from_prefix

    match_key = (match_field or _PHYSICAL_FIELD).strip().lower()
    return_key = (return_field or _PHYSICAL_FIELD).strip().lower()

    # Load CSV
    path = Path(csv_path)
    text = path.read_text(encoding="utf-8-sig")
    dialect = _detect_dialect(text)
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)

    out: set[str] = set()

    for rr in reader:
        row = _normalize_row(rr)

        # Resolve table
        table_val = _first_nonempty(row, _TABLE_FIELDS)
        if not table_val:
            compound = _first_nonempty(
                row, ("cmip6 compound name", "cmip7 compound name")
            )
            table_val = _extract_table_from_compound(compound)

        want_table = str(where.get("table", "")).strip().lower()
        if want_table and table_val.strip().lower() != want_table:
            continue

        # Value to MATCH
        match_val = row.get(match_key, "")
        if not match_val:
            branded = _first_nonempty(row, ("branded variable name",))
            compound = _first_nonempty(
                row, ("cmip6 compound name", "cmip7 compound name")
            )
            match_val = (branded or _extract_name_from_compound(compound) or "").strip()

        if not match_val:
            continue

        name_lc = match_val.lower()
        if exact:
            if var_prefix and name_lc != var_prefix_lc:
                continue
            if not var_prefix and not table_from_prefix:
                continue
        else:
            if var_prefix and not name_lc.startswith(var_prefix_lc):
                continue

        # Extra where filters (not 'table')
        ok = True
        for k_raw, val in where.items():
            if k_raw == "table":
                continue
            k = str(k_raw).strip().lower()
            cell_lc = row.get(k, "").lower()
            if isinstance(val, (list, tuple, set)):
                if cell_lc not in {str(v).strip().lower() for v in val}:
                    ok = False
                    break
            else:
                if cell_lc != str(val).strip().lower():
                    ok = False
                    break
        if not ok:
            continue

        # Group filters
        group_cell = _first_nonempty(row, _GROUP_FIELDS)
        groups = {g.lower() for g in _split_groups(group_cell)} if group_cell else set()
        if include_groups and not groups & include_groups:
            continue
        if exclude_groups and groups & exclude_groups:
            continue

        # Value to RETURN
        ret_val = row.get(return_key, "").strip()
        if not ret_val:
            ret_val = match_val

        out.add(ret_val)

    return sorted(out)
