"""Tests for cmip7_prep.dreq_search using the packaged datarequest CSV."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import csv
import io

import pytest

from cmip7_prep.dreq_search import (
    packaged_dreq_csv,
    find_variables_by_prefix,
)

# Lowercased header keys we’ll look for
H_TABLE = "cmip6 table (legacy)"
H_BRANDED = "branded variable name"
H_PHYS = "physical parameter"
H_GROUPS = "cmip7 variable groups"
H_COMP6 = "cmip6 compound name"
H_COMP7 = "cmip7 compound name"


def _read_packaged_rows():
    """Yield normalized (lowercased headers) rows from the packaged CSV."""
    path = packaged_dreq_csv()
    text = Path(path).read_text(encoding="utf-8-sig")
    # Dialect sniff with safe fallback
    try:
        dialect = csv.Sniffer().sniff(text, delimiters=",;\t")
    except csv.Error:

        class _D(csv.excel):  # pylint: disable=too-few-public-methods
            delimiter = ","

        dialect = _D()
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    for raw in reader:
        row = {
            (k.strip().lower() if k is not None else ""): (
                "" if v is None else str(v).strip()
            )
            for k, v in raw.items()
        }
        yield row


def _first_row_with(
    *,
    table_required: bool = True,
    need_group: Optional[str] = None,
    need_branded_diff: bool = False,
    allow_empty_table_if_compound_present: bool = False,
) -> Optional[dict]:
    """Find a row meeting certain conditions; None if not found."""
    for row in _read_packaged_rows():
        table = row.get(H_TABLE, "")
        phys = row.get(H_PHYS, "")
        branded = row.get(H_BRANDED, "")
        if not phys and not branded:
            continue

        if table_required and not table:
            if allow_empty_table_if_compound_present:
                comp = row.get(H_COMP6, "") or row.get(H_COMP7, "")
                if not comp:
                    continue
            else:
                continue

        if need_group:
            groups = (row.get(H_GROUPS, "") or "").lower()
            if need_group.lower() not in groups:
                continue

        if need_branded_diff and (not branded or branded == phys):
            continue

        return row
    return None


def _table_from_compound(row: dict) -> Optional[str]:
    comp = row.get(H_COMP6, "") or row.get(H_COMP7, "")
    if not comp:
        return None
    if "." in comp:
        return comp.split(".", 1)[0].strip()
    return comp.split()[0].strip() if comp else None


def test_packaged_csv_exists_and_nonempty():
    """File should be there"""
    p = packaged_dreq_csv()
    assert p.exists()
    txt = p.read_text(encoding="utf-8-sig")
    assert len(txt) > 100  # non-trivial file


def test_amon_all_returns_nonempty_unique_list():
    """A dotted-table query like 'Amon.' should return a non-empty, unique list."""
    got = find_variables_by_prefix(None, "Amon.")
    assert isinstance(got, list)
    assert len(got) == len(set(got))  # unique
    # All entries should be non-empty strings
    assert all(isinstance(v, str) and v for v in got)


def test_group_subset_is_subset_or_skipped_if_absent():
    """include_groups should yield a subset of the table list, or skip if group is absent."""
    all_amon = set(find_variables_by_prefix(None, "Amon."))
    # Detect whether 'baseline_monthly' appears in any row for Amon
    row = _first_row_with(need_group="baseline_monthly")
    if row is None:
        pytest.skip("No 'baseline_monthly' group in packaged CSV (skipping)")
    amon_baseline = set(
        find_variables_by_prefix(None, "Amon.", include_groups={"baseline_monthly"})
    )
    assert amon_baseline.issubset(all_amon)


def test_exact_and_prefix_for_some_table_variable():
    """Pick any row with table + physical parameter and verify exact and prefix queries."""
    row = _first_row_with(table_required=True)
    if row is None:
        pytest.skip("No suitable row with table+physical parameter found.")
    table = row.get(H_TABLE, "")
    phys = row.get(H_PHYS, "") or row.get(H_BRANDED, "")
    assert table and phys

    # Exact dotted query must contain the physical parameter
    got_exact = find_variables_by_prefix(None, f"{table}.{phys}", exact=True)
    assert phys in got_exact

    # Prefix query with first 2–3 chars should also contain phys
    prefix = phys[:3] if len(phys) >= 3 else phys
    got_prefix = find_variables_by_prefix(None, f"{table}.{prefix}")
    assert phys in got_prefix


def test_branded_match_returns_physical_if_different():
    """If a row has a different Branded vs Physical, matching on Branded returns Physical."""
    row = _first_row_with(need_branded_diff=True)
    if row is None:
        pytest.skip("No row with differing Branded/Physical names found.")
    table = row.get(H_TABLE, "") or _table_from_compound(row)
    branded = row.get(H_BRANDED, "")
    phys = row.get(H_PHYS, "") or branded
    assert table and branded and phys

    # Use a prefix of branded to query; return_field should be Physical Parameter
    q = f"{table}.{branded[:5]}"
    got = find_variables_by_prefix(
        None,
        q,
        match_field="Branded Variable Name",
        return_field="Physical Parameter",
    )
    assert phys in got


def test_table_fallback_from_compound_when_table_missing_or_empty():
    """Rows lacking table should still be found via compound name fallback."""
    row = _first_row_with(
        table_required=False, allow_empty_table_if_compound_present=True
    )
    if row is None:
        pytest.skip("No row with empty table but compound present found.")
    table = _table_from_compound(row)
    phys = row.get(H_PHYS, "") or row.get(H_BRANDED, "")
    assert table and phys

    got = find_variables_by_prefix(None, f"{table}.")
    assert phys in got
