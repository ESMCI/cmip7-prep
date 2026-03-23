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

import json
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
    "CESM Variable Name",  # comma-separated source variable name(s); used by convert_csv_to_yaml.py as skip filter
    "Formula",  # the formula string, present only when original had one
    "Scale",  # comma-separated scale factors, positionally aligned with CESM Variable Name
    "Freq",  # comma-separated sampling frequencies, positionally aligned with CESM Variable Name
    "Alias",  # comma-separated source aliases, positionally aligned with CESM Variable Name
    "Cell Methods",
    "Regrid Method",
    "Region",
    "Positive",
    "Levels Name",
    "Levels Units",
    "Levels Src Axis Name",
    "Levels Src Axis Bnds",
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


def sources_to_names(sources: list) -> str:
    """Return a comma-separated list of model variable names from *sources*.

    Only the ``model_var`` field is included; scale factors and other
    sub-fields are intentionally omitted so the result is a plain list of
    CESM variable names suitable for the ``CESM Variable Name`` CSV column.

    >>> sources_to_names([{"model_var": "TREFHT"}])
    'TREFHT'
    >>> sources_to_names([{"model_var": "PRECC"}, {"model_var": "PRECL"}])
    'PRECC, PRECL'
    >>> sources_to_names([{"model_var": "QFLX", "scale": -1.0}])
    'QFLX'
    >>> sources_to_names([])
    ''
    """
    return ", ".join(
        src.get("model_var", "") for src in sources if src.get("model_var")
    )


def sources_to_scale_freq_alias(sources: list) -> tuple:
    """Return (scale_str, freq_str, alias_str) for the Scale/Freq/Alias CSV columns.

    Each returned string is a comma-separated list positionally aligned with
    the ``CESM Variable Name`` column.  If all sources lack a given attribute
    the corresponding string is empty.

    >>> sources_to_scale_freq_alias([{"model_var": "TREFHT"}])
    ('', '', '')
    >>> sources_to_scale_freq_alias([{"model_var": "QFLX", "scale": -1.0}])
    ('-1.0', '', '')
    >>> sources_to_scale_freq_alias([{"model_var": "siconc", "freq": "day"}, {"model_var": "tarea"}])
    ('', 'day, ', '')
    >>> sources_to_scale_freq_alias([{"model_var": "A", "scale": 0.001, "freq": "day", "alias": "a"}, {"model_var": "B", "scale": 0.001}])
    ('0.001, 0.001', 'day, ', 'a, ')
    >>> sources_to_scale_freq_alias([])
    ('', '', '')
    """
    if not sources:
        return ("", "", "")

    def _col(attr):
        vals = [str(src.get(attr, "")) for src in sources]
        return ", ".join(vals) if any(v for v in vals) else ""

    return (_col("scale"), _col("freq"), _col("alias"))


def variable_to_rows(name: str, var: dict) -> list:
    """Convert one YAML variable entry to a list of CSV row dicts.

    Variables without ``variants`` produce a single row.  Variables with
    ``variants`` produce one row per variant, each carrying the variant's
    ``formula``, ``long_name``, and ``region``.  Per-source attributes
    (``scale``, ``freq``, ``alias``) are written to the ``Scale``, ``Freq``,
    and ``Alias`` columns as comma-separated values positionally aligned with
    ``CESM Variable Name``.

    >>> rows = variable_to_rows("tas", {"table": "atmos", "units": "K", "dims": ["time", "lat", "lon"], "sources": [{"model_var": "TREFHT"}]})
    >>> len(rows)
    1
    >>> rows[0]["CMIP Variable Name"]
    'tas'
    >>> rows[0]["CESM Variable Name"]
    'TREFHT'
    >>> rows[0]["Formula"]
    ''
    >>> rows[0]["Scale"]
    ''
    >>> rows[0]["Dimensions"]
    '["time", "lat", "lon"]'
    >>> rows[0]["Region"]
    ''

    >>> rows2 = variable_to_rows("pr", {"table": "atmos", "units": "kg m-2 s-1", "dims": ["time", "lat", "lon"], "formula": "PRECC + PRECL", "sources": [{"model_var": "PRECC"}, {"model_var": "PRECL"}]})
    >>> rows2[0]["Formula"]
    'PRECC + PRECL'
    >>> rows2[0]["CESM Variable Name"]
    'PRECC, PRECL'

    >>> var_with_scale = {"table": "atmos", "units": "kg m-2 s-1", "dims": ["time", "lat", "lon"], "sources": [{"model_var": "QFLX", "scale": -1.0}]}
    >>> rows_scale = variable_to_rows("evspsbl", var_with_scale)
    >>> rows_scale[0]["Formula"]
    ''
    >>> rows_scale[0]["Scale"]
    '-1.0'

    >>> var_with_variants = {"table": "seaIce", "units": "m2", "dims": ["time"], "sources": [{"model_var": "siconc", "freq": "day"}, {"model_var": "tarea"}], "variants": [{"long_name": "NH", "region": "nh", "formula": "siconc.where(lat>0)"}, {"long_name": "SH", "region": "sh", "formula": "siconc.where(lat<0)"}]}
    >>> rows3 = variable_to_rows("siarea_tavg-u-hm-u", var_with_variants)
    >>> len(rows3)
    2
    >>> rows3[0]["Region"]
    'nh'
    >>> rows3[0]["Long Name"]
    'NH'
    >>> rows3[0]["Formula"]
    'siconc.where(lat>0)'
    >>> rows3[0]["CESM Variable Name"]
    'siconc, tarea'
    >>> rows3[1]["Region"]
    'sh'
    >>> rows3[0]["Freq"]
    'day, '
    """
    formula = var.get("formula")
    sources = var.get("sources", [])
    variants = var.get("variants")
    levels = var.get("levels", {})

    dims = var.get("dims", [])
    # Use JSON so that list-of-lists dims round-trip correctly.
    dims_str = json.dumps(dims)

    scale_str, freq_str, alias_str = sources_to_scale_freq_alias(sources)

    base = {
        "CMIP Variable Name": name,
        "Table": var.get("table", ""),
        "Standard Name": var.get("standard_name", ""),
        "Units": var.get("units", ""),
        "Dimensions": dims_str,
        "Cell Methods": var.get("cell_methods", ""),
        "Regrid Method": var.get("regrid_method", ""),
        "Positive": var.get("positive", ""),
        "Scale": scale_str,
        "Freq": freq_str,
        "Alias": alias_str,
        "Levels Name": levels.get("name", ""),
        "Levels Units": levels.get("units", ""),
        "Levels Src Axis Name": levels.get("src_axis_name", ""),
        "Levels Src Axis Bnds": levels.get("src_axis_bnds", ""),
    }

    if variants and isinstance(variants, list) and variants:
        rows = []
        cesm_var = sources_to_names(sources)
        for v in variants:
            row = dict(base)
            row["Long Name"] = v.get("long_name", var.get("long_name", ""))
            row["CESM Variable Name"] = cesm_var
            row["Formula"] = v.get("formula", "")
            row["Region"] = v.get("region", "")
            rows.append(row)
        return rows

    # No variants — single row
    cesm_var = sources_to_names(sources)

    row = dict(base)
    row["Long Name"] = var.get("long_name", "")
    row["CESM Variable Name"] = cesm_var
    row["Formula"] = formula or ""
    row["Region"] = ""
    return [row]


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
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    variables = data.get("variables", {})
    rows = [
        row for name, var in variables.items() for row in variable_to_rows(name, var)
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
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
