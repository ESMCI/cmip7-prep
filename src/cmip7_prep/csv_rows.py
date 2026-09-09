"""Turn a YAML variable entry into CSV rows.

The CSV layout is the CESM one that convert_csv_to_yaml.py reads back
(--model cesm).  query_missing_vars.py uses these helpers so its output
matches that layout exactly.
"""

# Column names expected by convert_csv_to_yaml.py when --model cesm is used.
CESM_COLUMNS = [
    "CMIP Variable Name",
    "Table",
    "Long Name",
    "Standard Name",
    "Units",
    # comma-separated source name(s); convert_csv_to_yaml uses it as a skip filter
    "CESM Variable Name",
    "Formula",  # the formula string, present only when original had one
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


def sources_to_names(sources: list[dict]) -> str:
    """Return a comma-separated list of model variable names from *sources*.

    Only the ``model_var`` field is included;
    sub-fields that are intentionally omitted so the result is a plain list of
    CESM variable names suitable for the ``CESM Variable Name`` CSV column.

    >>> sources_to_names([{"model_var": "TREFHT"}])
    'TREFHT'
    >>> sources_to_names([{"model_var": "PRECC"}, {"model_var": "PRECL"}])
    'PRECC, PRECL'
    >>> sources_to_names([{"model_var": "sialgc", "freq": "day"}])
    'sialgc'
    >>> sources_to_names([])
    ''
    """
    return ", ".join(
        src.get("model_var", "") for src in sources if src.get("model_var")
    )


def sources_to_freq_alias(sources: list[dict]) -> tuple[str, str]:
    """Return (freq_str, alias_str) for the Freq/Alias CSV columns.

    Each returned string is a comma-separated list positionally aligned with
    the ``CESM Variable Name`` column.  If all sources lack a given attribute
    the corresponding string is empty.

    >>> sources_to_freq_alias([{"model_var": "TREFHT"}])
    ('', '')
    >>> sources_to_freq_alias([{"model_var": "siconc", "freq": "day"}, {"model_var": "tarea"}])
    ('day, ', '')
    >>> sources_to_freq_alias([{"model_var": "A", "freq": "day", "alias": "a"}, {"model_var": "B"}])
    ('day, ', 'a, ')
    >>> sources_to_freq_alias([])
    ('', '')
    """
    if not sources:
        return ("", "")

    def _col(attr):
        vals = [str(src.get(attr, "")) for src in sources]
        return ", ".join(vals) if any(v for v in vals) else ""

    return (_col("freq"), _col("alias"))


def variable_to_rows(name: str, var: dict) -> list:
    """Convert one YAML variable entry to a list of CSV row dicts.

    Variables without ``variants`` produce a single row.  Variables with
    ``variants`` produce one row per variant, each carrying the variant's
    ``formula``, ``long_name``, and ``region``.  Per-source attributes
    (``freq``, ``alias``) are written to the ``Freq``,
    and ``Alias`` columns as comma-separated values positionally aligned with
    ``CESM Variable Name``.

    >>> tas = {"table": "atmos", "units": "K",
    ...        "sources": [{"model_var": "TREFHT"}]}
    >>> rows = variable_to_rows("tas", tas)
    >>> len(rows)
    1
    >>> rows[0]["CMIP Variable Name"]
    'tas'
    >>> rows[0]["CESM Variable Name"]
    'TREFHT'
    >>> rows[0]["Formula"]
    ''
    >>> rows[0]["Region"]
    ''

    >>> pr = {"table": "atmos", "units": "kg m-2 s-1",
    ...       "formula": "PRECC + PRECL",
    ...       "sources": [{"model_var": "PRECC"}, {"model_var": "PRECL"}]}
    >>> rows2 = variable_to_rows("pr", pr)
    >>> rows2[0]["Formula"]
    'PRECC + PRECL'
    >>> rows2[0]["CESM Variable Name"]
    'PRECC, PRECL'

    >>> var_with_variants = {
    ...     "table": "seaIce", "units": "m2",
    ...     "sources": [{"model_var": "siconc", "freq": "day"},
    ...                 {"model_var": "tarea"}],
    ...     "variants": [
    ...         {"long_name": "NH", "region": "nh", "formula": "siconc.where(lat>0)"},
    ...         {"long_name": "SH", "region": "sh", "formula": "siconc.where(lat<0)"},
    ...     ],
    ... }
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

    freq_str, alias_str = sources_to_freq_alias(sources)

    base = {
        "CMIP Variable Name": name,
        "Table": var.get("table", ""),
        "Standard Name": var.get("standard_name", ""),
        "Units": var.get("units", ""),
        "Cell Methods": var.get("cell_methods", ""),
        "Regrid Method": var.get("regrid_method", ""),
        "Positive": var.get("positive", ""),
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
