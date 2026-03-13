"""yaml_to_csv.py — Generate a CSV from a YAML variable-mapping file.

The CSV produced here uses the CESM column layout understood by convert_csv_to_yaml.py
(--model cesm).  The primary use-case is bootstrapping a CESM CSV from an upstream
YAML file such as
  https://github.com/ESMCI/cmip7-prep/blob/features/seaice2/data/cesm_to_cmip7.yaml

Usage
-----
    python yaml_to_csv.py --input cesm_to_cmip7.yaml --output cesm_data.csv

The script is intentionally simple: it does a best-effort translation and flags any
variables that could not be represented cleanly so the user can review them.
"""

import yaml
import csv
import argparse

# Column names expected by convert_csv_to_yaml.py when --model cesm is used.
CESM_COLUMNS = [
    "CMIP Variable Name",
    "Table",
    "Long Name",
    "Standard Name",
    "Units",
    "Dimensions",
    "CESM Variable Name",
    "Cell Methods",
    "Regrid Method",
]


def sources_to_expr(sources: list) -> str:
    """Build a human-readable expression from a list of source dicts.

    Each source dict may have:
        model_var : str  (required)
        scale     : float (optional) — multiplied against the variable
        freq      : str   (optional) — preferred sampling frequency, ignored here

    Rules
    -----
    * Single source, no scale  →  "VAR"
    * Single source, with scale →  "VAR * scale"
    * Multiple sources          →  "VAR1 + VAR2 + ..."
      (scale, if present on any source, is represented as "VAR * scale")

    >>> sources_to_expr([{"model_var": "TREFHT"}])
    'TREFHT'
    >>> sources_to_expr([{"model_var": "QFLX", "scale": -1.0}])
    'QFLX * -1.0'
    >>> sources_to_expr([{"model_var": "PRECC"}, {"model_var": "PRECL"}])
    'PRECC + PRECL'
    >>> sources_to_expr([{"model_var": "SOIL1C", "scale": 0.001}, {"model_var": "SOIL2C", "scale": 0.001}])
    'SOIL1C * 0.001 + SOIL2C * 0.001'
    >>> sources_to_expr([])
    ''
    """
    if not sources:
        return ""
    parts = []
    for src in sources:
        model_var = src.get("model_var", "")
        scale = src.get("scale")
        if scale is not None:
            parts.append(f"{model_var} * {scale}")
        else:
            parts.append(model_var)
    return " + ".join(parts)


def variable_to_row(name: str, var: dict) -> dict:
    """Convert one YAML variable entry to a CSV row dict.

    Variables with a top-level ``variants`` key (region-specific sub-formulas)
    cannot be fully represented in a flat CSV.  They are included using their
    first variant formula and flagged with a ``#VARIANTS`` suffix in the
    ``CESM Variable Name`` cell so the user knows to review them.

    >>> row = variable_to_row("tas", {"table": "atmos", "units": "K", "dims": ["time", "lat", "lon"], "sources": [{"model_var": "TREFHT"}]})
    >>> row["CMIP Variable Name"]
    'tas'
    >>> row["CESM Variable Name"]
    'TREFHT'
    >>> row["Dimensions"]
    'time, lat, lon'

    >>> row2 = variable_to_row("pr", {"table": "atmos", "units": "kg m-2 s-1", "dims": ["time", "lat", "lon"], "formula": "PRECC + PRECL", "sources": [{"model_var": "PRECC"}, {"model_var": "PRECL"}]})
    >>> row2["CESM Variable Name"]
    'PRECC + PRECL'
    """
    # ── CESM Variable Name ──────────────────────────────────────────────────
    formula = var.get("formula")
    sources = var.get("sources", [])
    variants = var.get("variants")

    if formula:
        cesm_var = formula
    elif sources:
        cesm_var = sources_to_expr(sources)
    elif variants:
        # Use the first variant formula and mark for review
        cesm_var = variants[0].get("formula", "") + "  #VARIANTS"
    else:
        cesm_var = ""

    if variants and not (formula or (not sources)):
        # sources present but variants also present — append flag
        cesm_var = cesm_var + "  #VARIANTS"

    # ── dims ────────────────────────────────────────────────────────────────
    dims = var.get("dims", [])
    dims_str = ", ".join(str(d) for d in dims)

    return {
        "CMIP Variable Name": name,
        "Table": var.get("table", ""),
        "Long Name": var.get("long_name", ""),
        "Standard Name": var.get("standard_name", ""),
        "Units": var.get("units", ""),
        "Dimensions": dims_str,
        "CESM Variable Name": cesm_var,
        "Cell Methods": var.get("cell_methods", ""),
        "Regrid Method": var.get("regrid_method", ""),
    }


def yaml_to_csv(yaml_path: str, csv_path: str) -> int:
    """Convert *yaml_path* to *csv_path* and return the number of rows written.

    >>> import tempfile, os, yaml
    >>> data = {"dataset_overrides": {"source_id": "CESM3"}, "variables": {"tas": {"table": "atmos", "units": "K", "dims": ["time", "lat", "lon"], "sources": [{"model_var": "TREFHT"}]}}}
    >>> with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
    ...     yaml.dump(data, f)
    ...     ypath = f.name
    >>> import csv as _csv, tempfile as _tmp
    >>> cpath = _tmp.mktemp(suffix=".csv")
    >>> yaml_to_csv(ypath, cpath)
    1
    >>> os.unlink(ypath); os.unlink(cpath)
    """
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    variables = data.get("variables", {})
    rows = [variable_to_row(name, var) for name, var in variables.items()]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CESM_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert a YAML variable-mapping file to a CSV suitable for "
            "convert_csv_to_yaml.py --model cesm."
        )
    )
    parser.add_argument(
        "--input",
        default="cesm_to_cmip7.yaml",
        help="Input YAML file (default: cesm_to_cmip7.yaml)",
    )
    parser.add_argument(
        "--output",
        default="cesm_data.csv",
        help="Output CSV file (default: cesm_data.csv)",
    )
    args = parser.parse_args()

    n = yaml_to_csv(args.input, args.output)
    print(f"wrote {n} entries to {args.output}")


if __name__ == "__main__":
    main()
