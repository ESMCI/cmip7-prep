#!/usr/bin/env python3
"""query_missing_vars.py — Produce a merged CSV of existing + missing CMIP7 variables.

The output CSV contains two sections:

1. **Existing variables** — every variable already in the mapping YAML, serialised
   with the same logic as :mod:`yaml_to_csv`.  Running
   ``convert_csv_to_yaml.py --model cesm`` on this section alone reproduces the
   original YAML exactly.

2. **Missing variables** — stub rows for CMIP7-requested variables that have no
   entry in the YAML.  Metadata (long name, units, dimensions, cell methods, …)
   comes from the CMIP7 table JSON files; CESM-specific columns
   (``CESM Variable Name``, ``Formula``, ``Scale``, ``Freq``, ``Alias``) are
   left blank for the user to fill in.

Usage
-----
    python scripts/query_missing_vars.py \\
        [--yaml   data/cesm_to_cmip7.yaml] \\
        [--tables cmip7-cmor-tables/tables] \\
        [--realm  atmos land ocean seaIce] \\
        [--output cesm_full.csv]

Workflow
--------
    # 1.  Generate the merged CSV
    python scripts/query_missing_vars.py --output cesm_full.csv

    # 2.  Fill in CESM Variable Name / Formula / Scale / Freq / Alias for new entries

    # 3.  Regenerate the YAML (existing variables round-trip perfectly;
    #     filled-in new ones are appended)
    python scripts/convert_csv_to_yaml.py --model cesm \\
        --input cesm_full.csv --output data/cesm_to_cmip7.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import warnings
from pathlib import Path

import yaml

# Import the row-serialisation logic from yaml_to_csv so the existing-variable
# section of the output is byte-for-byte identical to what yaml_to_csv produces.
_SCRIPTS = Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS))
from yaml_to_csv import CESM_COLUMNS, variable_to_rows  # noqa: E402

# Extended column list for this script's output only.  The two extra columns
# (Priority, Experiments) are data-request metadata and are not part of the
# standard CESM_COLUMNS used by yaml_to_csv / convert_csv_to_yaml.
_OUTPUT_COLUMNS = CESM_COLUMNS + ["Priority", "Experiments"]

# ── constants ─────────────────────────────────────────────────────────────────

# Table JSON files that do not contain variable_entry blocks.
_SKIP_TABLE_IDS = {
    "CV",
    "coordinate",
    "formula_terms",
    "grids",
    "cell_measures",
    "long_name_overrides",
}

# Dimension names that imply a specific levels configuration.
_LEVELS_BY_DIM = {
    "plev19": {"name": "plev19", "units": "Pa"},
    "plev39": {"name": "plev39", "units": "Pa"},
    "alevel": {
        "name": "standard_hybrid_sigma",
        "units": "1",
        "src_axis_name": "lev",
        "src_axis_bnds": "ilev",
    },
}


# ── CMIP7 table helpers ───────────────────────────────────────────────────────


def load_cmip7_tables(tables_dir: Path) -> dict[str, dict]:
    """Return ``{branded_var_name: entry}`` from all CMIP7 table JSON files.

    Each entry is the raw ``variable_entry`` dict augmented with ``_table_id``.

    >>> import tempfile, json, os
    >>> tmp = tempfile.mkdtemp()
    >>> tbl = {"Header": {}, "variable_entry": {"tas_foo": {"long_name": "Temp", "units": "K", "dimensions": ["time", "lat"], "modeling_realm": "atmos", "standard_name": "", "cell_methods": "", "positive": ""}}}
    >>> with open(os.path.join(tmp, "CMIP7_atmos.json"), "w") as f: json.dump(tbl, f)
    >>> result = load_cmip7_tables(Path(tmp))
    >>> list(result.keys())
    ['tas_foo']
    >>> result['tas_foo']['_table_id']
    'atmos'
    """
    lookup: dict[str, dict] = {}
    for json_path in sorted(tables_dir.glob("CMIP7_*.json")):
        table_id = json_path.stem.replace("CMIP7_", "")
        if table_id in _SKIP_TABLE_IDS:
            continue
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        for bvn, entry in data.get("variable_entry", {}).items():
            lookup[bvn] = dict(entry, _table_id=table_id)
    return lookup


def _parse_dims(raw) -> list:
    """Normalise a CMIP7 table dimensions value to a list of strings.

    Most tables store dims as a list; the fx table stores them as a
    space-separated string.

    >>> _parse_dims(["longitude", "latitude", "time"])
    ['longitude', 'latitude', 'time']
    >>> _parse_dims("longitude latitude")
    ['longitude', 'latitude']
    >>> _parse_dims("")
    []
    """
    if isinstance(raw, list):
        return [str(d).strip() for d in raw if str(d).strip()]
    if isinstance(raw, str):
        return [d for d in re.split(r"[\s,]+", raw.strip()) if d]
    return []


def _infer_levels(dims: list) -> dict:
    """Return a levels dict inferred from dimension names, or ``{}`` if none found.

    >>> _infer_levels(["time", "plev19", "lat", "lon"])
    {'name': 'plev19', 'units': 'Pa'}
    >>> _infer_levels(["time", "alevel", "lat", "lon"])
    {'name': 'standard_hybrid_sigma', 'units': '1', 'src_axis_name': 'lev', 'src_axis_bnds': 'ilev'}
    >>> _infer_levels(["time", "lat", "lon"])
    {}
    """
    for dim in dims:
        if dim in _LEVELS_BY_DIM:
            return _LEVELS_BY_DIM[dim]
    return {}


def table_entry_to_stub_row(bvn: str, entry: dict) -> dict:
    """Convert a CMIP7 table entry to a stub CSV row.

    CESM-specific columns (``CESM Variable Name``, ``Formula``, ``Scale``,
    ``Freq``, ``Alias``, ``Regrid Method``) are left blank for the user to fill in.

    >>> entry = {"long_name": "Surface Temperature", "units": "K", "dimensions": ["longitude", "latitude", "time"], "modeling_realm": "atmos", "standard_name": "surface_temperature", "cell_methods": "time: mean", "positive": "", "_table_id": "atmos"}
    >>> row = table_entry_to_stub_row("ts_foo", entry)
    >>> row["CMIP Variable Name"]
    'ts_foo'
    >>> row["Table"]
    'atmos'
    >>> row["Units"]
    'K'
    >>> row["Dimensions"]
    '["longitude", "latitude", "time"]'
    >>> row["CESM Variable Name"]
    ''
    >>> row["Levels Name"]
    ''
    """
    dims = _parse_dims(entry.get("dimensions", []))
    levels = _infer_levels(dims)

    # modeling_realm may be multi-word (e.g. "ocnBgchem ocean"); use first token.
    realm_raw = entry.get("modeling_realm", entry.get("_table_id", ""))
    table = realm_raw.split()[0] if realm_raw.strip() else entry.get("_table_id", "")

    return {
        "CMIP Variable Name": bvn,
        "Table": table,
        "Long Name": entry.get("long_name", ""),
        "Standard Name": entry.get("standard_name", ""),
        "Units": entry.get("units", ""),
        "Dimensions": json.dumps(dims),
        "CESM Variable Name": "",
        "Formula": "",
        "Scale": "",
        "Freq": "",
        "Alias": "",
        "Cell Methods": entry.get("cell_methods", ""),
        "Regrid Method": "",
        "Region": "",
        "Positive": entry.get("positive", ""),
        "Priority": entry.get("_priority", ""),
        "Experiments": json.dumps(entry.get("_experiments", [])),
        "Levels Name": levels.get("name", ""),
        "Levels Units": levels.get("units", ""),
        "Levels Src Axis Name": levels.get("src_axis_name", ""),
        "Levels Src Axis Bnds": levels.get("src_axis_bnds", ""),
    }


# ── data request helper ───────────────────────────────────────────────────────


def get_requested_var_metadata(
    realms: list[str] | None = None,
) -> dict[str, dict]:
    """Query the CMIP7 data request API and return metadata for requested variables.

    Args:
        realms: Optional realm filter list (e.g. ``["atmos", "land"]``).
                ``None`` retrieves all realms.

    Returns:
        Mapping of branded variable name → ``{"priority": int, "experiments": [str, ...]}``.
    """
    from data_request_api.query import data_request as dr
    from data_request_api.content import dump_transformation as dt

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        content_dic = dt.get_transformed_content()
    DR = dr.DataRequest.from_separated_inputs(**content_dic)

    kwargs: dict = dict(skip_if_missing=False, operation="all")
    variables = (
        [
            v
            for realm in realms
            for v in DR.find_variables(modelling_realm=realm, **kwargs)
        ]
        if realms
        else list(DR.find_variables(**kwargs))
    )

    result: dict[str, dict] = {}
    for v in variables:
        bvn = str(v.attributes["branded_variable_name"])
        priority = DR.find_priority_per_variable(v)
        experiments = sorted(
            {
                str(e.attributes["id"])
                for opp in DR.find_opportunities_per_variable(v)
                for e in DR.find_experiments_per_opportunity(opp)
            }
        )
        result[bvn] = {"priority": priority, "experiments": experiments}
    return result


# ── core logic ────────────────────────────────────────────────────────────────


def build_merged_rows(
    yaml_path: Path,
    tables_dir: Path,
    realms: list[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Return ``(existing_rows, stub_rows)`` for the merged CSV.

    *existing_rows* — full-fidelity rows produced by :func:`variable_to_rows`
    for every variable already in *yaml_path*.  Round-tripping these through
    ``convert_csv_to_yaml.py --model cesm`` reproduces the original YAML.

    *stub_rows* — one blank-CESM-column row per CMIP7-requested variable that
    is absent from *yaml_path*, sorted by table then branded variable name.
    """
    with open(yaml_path, encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)
    existing_vars: dict = yaml_data.get("variables", {})

    # Rows for all variables already in the YAML (preserving original order).
    existing_rows: list[dict] = [
        row
        for name, var in existing_vars.items()
        for row in variable_to_rows(name, var)
    ]

    table_lookup = load_cmip7_tables(tables_dir)

    print(f"Querying CMIP7 data request (realms={realms or 'all'}) …", flush=True)
    var_metadata = get_requested_var_metadata(realms)
    requested = set(var_metadata)

    # Annotate existing rows with priority and experiments from the data request.
    for row in existing_rows:
        meta = var_metadata.get(row["CMIP Variable Name"])
        if meta:
            row["Priority"] = meta["priority"]
            row["Experiments"] = json.dumps(meta["experiments"])
    print(f"  {len(requested)} variables requested by CMIP7 data request")
    print(f"  {len(existing_vars)} variables already in {yaml_path.name}")

    missing_bvns = (requested & set(table_lookup)) - set(existing_vars)
    not_in_tables = requested - set(table_lookup) - set(existing_vars)
    if not_in_tables:
        print(
            f"  {len(not_in_tables)} requested variable(s) not found in CMIP7 tables "
            f"(skipped): {sorted(not_in_tables)[:5]}"
            f"{'…' if len(not_in_tables) > 5 else ''}"
        )

    stub_rows = [
        table_entry_to_stub_row(
            bvn,
            dict(
                table_lookup[bvn],
                _priority=var_metadata[bvn]["priority"],
                _experiments=var_metadata[bvn]["experiments"],
            ),
        )
        for bvn in missing_bvns
    ]
    stub_rows.sort(key=lambda r: (r["Table"], r["CMIP Variable Name"]))
    print(f"  {len(stub_rows)} new stub rows added")
    return existing_rows, stub_rows


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Write a merged CSV of existing YAML variables (full fidelity) "
            "and stub rows for CMIP7-requested variables not yet in the YAML."
        )
    )
    parser.add_argument(
        "--yaml",
        default="data/cesm_to_cmip7.yaml",
        help="Existing mapping YAML (default: data/cesm_to_cmip7.yaml)",
    )
    parser.add_argument(
        "--tables",
        default="cmip7-cmor-tables/tables",
        help="Directory of CMIP7_*.json table files (default: cmip7-cmor-tables/tables)",
    )
    parser.add_argument(
        "--realm",
        nargs="+",
        metavar="REALM",
        default=None,
        help=(
            "Realm filter for the data request query "
            "(e.g. atmos land ocean seaIce). Default: all realms."
        ),
    )
    parser.add_argument(
        "--output",
        default="cesm_full.csv",
        help="Output CSV file path (default: cesm_full.csv)",
    )
    args = parser.parse_args()

    yaml_path = Path(args.yaml)
    tables_dir = Path(args.tables)

    if not yaml_path.exists():
        sys.exit(f"error: YAML file not found: {yaml_path}")
    if not tables_dir.is_dir():
        sys.exit(f"error: tables directory not found: {tables_dir}")

    existing_rows, stub_rows = build_merged_rows(
        yaml_path, tables_dir, realms=args.realm
    )
    all_rows = existing_rows + stub_rows

    output_path = Path(args.output)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_OUTPUT_COLUMNS, restval="")
        writer.writeheader()
        writer.writerows(all_rows)

    print(
        f"Wrote {len(all_rows)} rows ({len(existing_rows)} existing + {len(stub_rows)} new) to {output_path}"
    )
    if stub_rows:
        print(
            "\nNext steps:\n"
            "  1. Fill in 'CESM Variable Name' / 'Formula' / 'Scale' / 'Freq' / 'Alias' for new rows.\n"
            f"  2. python scripts/convert_csv_to_yaml.py --model cesm \\\n"
            f"         --input {output_path} --output data/cesm_to_cmip7.yaml"
        )


if __name__ == "__main__":
    main()
