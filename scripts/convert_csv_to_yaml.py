import sys
import csv
import json
import yaml
import re
import argparse
from typing import Optional

# ── model configurations ─────────────────────────────────────────────────────
# Each config defines how to read a model-specific CSV and what metadata to write.
#
#   key_column          : CSV column used as the top-level YAML key
#   column_map          : CSV column → YAML key; "_source_expr" is special-cased
#                         (parsed as a variable name or math formula)
#   realm_column        : CSV column containing the realm/table value
#   keep_realms         : list of realms to keep; None means keep all
#   source_column       : CSV column containing the model variable expression
#   source_skip_phrases : rows whose source column contains any of these are dropped
#   dataset_overrides   : written verbatim as the top-level "dataset_overrides" block
#   default_input       : default CSV path when --input is omitted
#   default_output      : default YAML path when --output is omitted

MODEL_CONFIGS = {
    "noresm": {
        "default_input": "data.csv",
        "default_output": "data.yaml",
        "normalize_dim_names": True,
        "dataset_overrides": {
            "institution_id": "NCC",
            "source_id": "NorESM3",
            "nominal_resolution": "200 km",
            "yaml_coax_dummy": [0, 1, 2],
        },
        "key_column": "Branded Variable Name",
        "column_map": {
            "Modelling Realm - Primary": "table",
            "CMIP6 Compound Name": "long_name",
            "Description": "description",
            "Units (from Physical Parameter)": "units",
            "Dimensions": "dims",
            "NorESM3 name (dependency)": "_source_expr",
            "CMIP7 Freq.": "freq",
        },
        "realm_column": "Modelling Realm - Primary",
        "keep_realms": ["atmos", "land"],
        "source_column": "NorESM3 name (dependency)",
        "source_skip_phrases": [
            "?",
            "n/a",
            "derived",
            "IN SURF DATASET",
            "can be derived",
        ],
    },
    "cesm": {
        "default_input": "cesm_data.csv",
        "default_output": "cesm_data.yaml",
        "normalize_dim_names": False,
        "dataset_overrides": {
            "institution_id": "NCAR",
            "source_id": "CESM3",
            "nominal_resolution": "100 km",
        },
        "key_column": "CMIP Variable Name",
        "column_map": {
            "Table": "table",
            "Long Name": "long_name",
            "Standard Name": "standard_name",
            "Units": "units",
            "Dimensions": "dims",
            "CESM Variable Name": "_source_expr",
            # "Formula" overrides _source_expr when present (round-trip format).
            # "Scale", "Freq", "Alias" are merged into sources in post-processing.
            "Formula": "_formula",
            "Scale": "_scale",
            "Freq": "_freq",
            "Alias": "_alias",
            "Cell Methods": "cell_methods",
            "Regrid Method": "regrid_method",
            "Region": "region",
            "Positive": "positive",
            "Levels Name": "_levels_name",
            "Levels Units": "_levels_units",
            "Levels Src Axis Name": "_levels_src_axis_name",
            "Levels Src Axis Bnds": "_levels_src_axis_bnds",
        },
        "realm_column": "Table",
        "keep_realms": None,  # keep all realms
        "source_column": "CESM Variable Name",
        "source_skip_phrases": [],
    },
}


class InlineListDumper(yaml.Dumper):
    """YAML dumper that can render lists in inline (flow) style."""

    pass


def inline_list_representer(dumper, data):
    """Represent a Python list as an inline YAML sequence (flow style)."""
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


# Register the inline list representer for both the custom dumper and the
# default PyYAML dumper so that yaml.dump() picks it up even when no
# explicit Dumper is provided.
yaml.add_representer(list, inline_list_representer, Dumper=InlineListDumper)
yaml.add_representer(list, inline_list_representer, Dumper=yaml.Dumper)


def should_keep(row, config):
    """Return True if this row should be included in the output.

    Filters on realm and source-column content using the model config.

    >>> cfg = MODEL_CONFIGS["noresm"]
    >>> should_keep({"Modelling Realm - Primary": "atmos", "NorESM3 name (dependency)": "T2M"}, cfg)
    True
    >>> should_keep({"Modelling Realm - Primary": "ocean", "NorESM3 name (dependency)": "SST"}, cfg)
    False
    >>> should_keep({"Modelling Realm - Primary": "atmos", "NorESM3 name (dependency)": ""}, cfg)
    False
    >>> should_keep({"Modelling Realm - Primary": "atmos", "NorESM3 name (dependency)": "n/a"}, cfg)
    False
    """
    realm_col = config["realm_column"]
    source_col = config["source_column"]
    keep_realms = config.get("keep_realms")
    skip_phrases = config.get("source_skip_phrases", [])

    if keep_realms is not None and row.get(realm_col) not in keep_realms:
        return False

    source = row.get(source_col, "").strip()
    if not source:
        return False
    for phrase in skip_phrases:
        if phrase in source:
            return False
    return True


def is_math_expression(expr: str) -> bool:
    """
    Return True if expr is a mathematical expression,
    Return False if it is a single variable name.

    >>> is_math_expression("PRECC + PRECL")
    True
    >>> is_math_expression("PRECC * 0.001")
    True
    >>> is_math_expression("verticalsum(SOILWATER)")
    True
    >>> is_math_expression("PRECC")
    False
    >>> is_math_expression("T2M")
    False
    """
    expr = expr.strip()
    if re.search(r"[+\-*/^%]", expr):
        return True
    if re.search(r"\w+\s*\(", expr):
        return True
    # Commenting out, this is definitely classifying to much as expressions,
    # not sure if we are missing somethng, though...
    # if re.search(r'\d', expr):
    #     return True
    if re.search(r"\w+\s+\w+", expr):
        return True
    if re.search("FATES", expr):
        return True

    return False


def extract_variables(expr: str) -> list:
    """
    Extract variable names from a mathematical expression,
    ignoring operators, numbers, and known functions/modules.

    >>> extract_variables("PRECC + PRECL")
    [{'model_var': 'PRECC'}, {'model_var': 'PRECL'}]
    >>> extract_variables("PRECC * 0.001")
    [{'model_var': 'PRECC'}]
    >>> extract_variables("verticalsum(SOILWATER, capped_at=1000)")
    [{'model_var': 'SOILWATER'}]
    >>> extract_variables("np.sqrt(U**2 + V**2)")
    [{'model_var': 'U'}, {'model_var': 'V'}]
    """

    # known functions and modules to ignore
    ignore = {
        "np",
        "xr",
        "math",  # modules
        "sqrt",
        "abs",
        "sum",
        "min",
        "max",
        "mean",  # common functions
        "verticalsum",
        "where",
        "zeros",
        "ones",  # custom/xarray functions
        "True",
        "False",
        "None",  # python keywords
        "dim",
        "capped_at",
        "skipna",  # common keyword arguments
    }

    # find all words in the expression
    all_words = re.findall(r"[a-zA-Z_]\w*", expr)

    # filter out ignored words and pure numbers
    variables = []
    for word in all_words:
        word_dict = {}
        if word not in ignore:
            word_dict["model_var"] = word
            variables.append(word_dict)
    if "FATES" in expr and "FATES_FRAC" not in expr:
        variables.append({"model_var": "FATES_FRAC"})
    return variables


def analyse_expression(expr: str) -> dict:
    """
    Analyse an expression and return whether it is a math
    expression and what variables it contains.

    >>> analyse_expression("PRECC + PRECL")
    {'is_math': True, 'variables': [{'model_var': 'PRECC'}, {'model_var': 'PRECL'}]}
    >>> analyse_expression("T2M")
    {'is_math': False, 'variables': [{'model_var': 'T2M'}]}
    """
    expr = expr.strip()

    if not is_math_expression(expr):
        return {
            "is_math": False,
            "variables": [
                {"model_var": expr}
            ],  # single variable is the expression itself
        }
    return {"is_math": True, "variables": extract_variables(expr)}


def strip_quotes(obj):
    """Recursively strip quotes from all string values in a dict."""
    if isinstance(obj, dict):
        return {k: strip_quotes(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [strip_quotes(i) for i in obj]
    if isinstance(obj, str):
        return obj.strip("'\"")
    return obj


def fix_number_norwegian_format(value):
    """Normalize a value using Norwegian number formatting to a standard numeric string.

    >>> fix_number_norwegian_format("1,5")
    '1.5'
    >>> fix_number_norwegian_format("1.000,5")
    '1000.5'
    >>> fix_number_norwegian_format(1.5)
    1.5
    """
    if isinstance(value, str):
        value = value.replace(".", "").replace(",", ".")
    if isinstance(value, str):
        value = value.replace(
            "\u2212", "-"
        )  # replace unicode minus sign with regular hyphen
    return value


def clean_string(value, normalize_dim_names=False):
    """Clean a string by stripping whitespace and optionally normalising dimension names.

    Dimension name normalisation (longitude→lon, latitude→lat) is only applied
    when normalize_dim_names=True (NorESM); for CESM the original names are preserved.

    >>> clean_string("longitude", normalize_dim_names=True)
    'lon'
    >>> clean_string("latitude", normalize_dim_names=True)
    'lat'
    >>> clean_string("longitude", normalize_dim_names=False)
    'longitude'
    >>> clean_string("latitude", normalize_dim_names=False)
    'latitude'
    >>> clean_string("  lev  ")
    'lev'
    >>> clean_string("'time'")
    'time'
    """
    if isinstance(value, str):
        value = value.replace("'", "").replace('"', "")  # remove quotes
        value = value.strip()
    if normalize_dim_names:
        if value == "longitude":
            value = "lon"
        elif value == "latitude":
            value = "lat"
        elif value == "alevel":
            value = "lev"
    return value


def clean_strings(values, normalize_dim_names=False):
    """Clean a list of strings."""
    if isinstance(values, list):
        return [clean_string(v, normalize_dim_names) for v in values]
    elif isinstance(values, str):
        return clean_string(values, normalize_dim_names)
    return values


def include_fates_frac(expr):
    """If the expression contains "FATES" but not "FATES_FRAC", include "FATES_FRAC" as an additional source variable."""
    if "FATES" not in expr:
        return expr
    if "FATES_FRAC" in expr:
        return expr
    return f"({expr})*FATES_FRAC"


# ── read csv ──────────────────────────────────────────────────────────────────
def _parse_csv_identifiers(value: str) -> Optional[list[str]]:
    """Parse *value* as a comma-separated list of plain variable-name identifiers.

    Returns a list of identifier strings if every comma-separated token is a
    valid Python identifier (letters/digits/underscores, starting with a
    letter or underscore).  Returns ``None`` if any token is not a plain
    identifier, signalling that the value should be handled as a formula
    expression by ``analyse_expression`` instead.

    >>> _parse_csv_identifiers("TREFHT")
    ['TREFHT']
    >>> _parse_csv_identifiers("siconc, tarea")
    ['siconc', 'tarea']
    >>> _parse_csv_identifiers("PRECC, PRECL")
    ['PRECC', 'PRECL']
    >>> _parse_csv_identifiers("CLDTOT * 100") is None
    True
    >>> _parse_csv_identifiers("PRECC + PRECL") is None
    True
    """
    import re as _re

    parts = [p.strip() for p in value.split(",")]
    if all(_re.match(r"^[A-Za-z_]\w*$", p) for p in parts if p):
        return [p for p in parts if p]
    return None


def _split_positional(s: str, n: int) -> list:
    """Split comma-separated string *s* into exactly *n* stripped tokens.

    Pads with empty strings if *s* has fewer than *n* tokens; truncates if more.

    >>> _split_positional("day, mon, ", 3)
    ['day', 'mon', '']
    >>> _split_positional("-1.0", 1)
    ['-1.0']
    >>> _split_positional("", 2)
    ['', '']
    >>> _split_positional("a, b", 3)
    ['a', 'b', '']
    """
    parts = [p.strip() for p in s.split(",")]
    parts.extend([""] * (n - len(parts)))
    return parts[:n]


def _build_entry(row, config):
    """Build a single variable entry dict from one CSV row."""
    column_map = config["column_map"]
    entry = {}

    for csv_col, yaml_key in column_map.items():
        if csv_col not in row:
            continue  # skip columns absent from this CSV
        value = row[csv_col].strip()
        if yaml_key == "_formula":
            # An explicit (possibly empty) Formula column is authoritative: clear
            # any formula that _source_expr may have set from the human-readable
            # CESM Variable Name column (round-trip case).
            if value:
                entry["formula"] = value
            else:
                entry.pop("formula", None)
            continue
        if not value:
            continue  # skip blank cells

        if yaml_key == "dims":
            normalize = config.get("normalize_dim_names", False)
            if value.startswith("["):
                # JSON-encoded dims from yaml_to_csv (handles flat and nested lists).
                try:
                    dims = json.loads(value)
                except json.JSONDecodeError:
                    dims = clean_strings(value.split(","), normalize)
            else:
                dims = clean_strings(value.split(","), normalize)
            entry["dims"] = dims
        elif yaml_key == "units":
            entry["units"] = fix_number_norwegian_format(value)
        elif yaml_key == "_source_expr":
            names = _parse_csv_identifiers(value)
            if names is not None:
                # New format: comma-separated plain variable names.
                # Scale/Freq/Alias will be merged in post-processing.
                entry["sources"] = [{"model_var": n} for n in names]
            else:
                # Fallback: legacy formula expression in CESM Variable Name.
                result = analyse_expression(value)
                if result["is_math"]:
                    entry["formula"] = include_fates_frac(value)
                    entry["sources"] = result["variables"]
                else:
                    entry["sources"] = result["variables"]

        elif yaml_key in ("_scale", "_freq", "_alias"):
            entry[yaml_key] = value

        else:
            # Includes: table, long_name, standard_name, cell_methods,
            # regrid_method, region, positive, _levels_name, _levels_units,
            # _levels_src_axis_name, _levels_src_axis_bnds
            entry[yaml_key] = value

    # Build levels dict from explicit columns (if present) or fall back to
    # auto-generating from "lev" in dims (needed for NorESM which has no
    # dedicated levels columns).
    levels_name = entry.pop("_levels_name", None)
    levels_units = entry.pop("_levels_units", None)
    levels_src_axis_name = entry.pop("_levels_src_axis_name", None)
    levels_src_axis_bnds = entry.pop("_levels_src_axis_bnds", None)

    if levels_name:
        levels = {"name": levels_name}
        if levels_units:
            levels["units"] = levels_units
        if levels_src_axis_name:
            levels["src_axis_name"] = levels_src_axis_name
        if levels_src_axis_bnds:
            levels["src_axis_bnds"] = levels_src_axis_bnds
        entry["levels"] = levels
    elif "dims" in entry and "lev" in entry["dims"]:
        # Fallback for models without explicit levels columns (e.g., NorESM).
        entry["levels"] = {
            "name": "standard_hybrid_sigma",
            "units": "1",
            "src_axis_name": "lev",
            "src_axis_bnds": "ilev",
        }

    # Merge Scale/Freq/Alias columns into the per-source dicts.
    scale_str = entry.pop("_scale", None)
    freq_str = entry.pop("_freq", None)
    alias_str = entry.pop("_alias", None)
    if "sources" in entry and (scale_str or freq_str or alias_str):
        sources = entry["sources"]
        n = len(sources)
        scales = _split_positional(scale_str or "", n)
        freqs = _split_positional(freq_str or "", n)
        aliases = _split_positional(alias_str or "", n)
        for i, src in enumerate(sources):
            if scales[i]:
                try:
                    src["scale"] = float(scales[i])
                except ValueError:
                    pass
            if freqs[i]:
                src["freq"] = freqs[i]
            if aliases[i]:
                src["alias"] = aliases[i]

    return entry


# Fields that belong to each variant, not to the base variable entry.
_VARIANT_FIELDS = ("long_name", "formula", "region")


def read_csv(filepath, config):
    """Read CSV and return a data dict ready for YAML output.

    The structure is::

        {
            "dataset_overrides": {...},
            "variables": {
                "<cmip_name>": { ... },
                ...
            }
        }

    Variables with multiple CSV rows (same CMIP Variable Name, different Region)
    are reconstructed into a ``variants`` list.
    """
    key_col = config["key_column"]

    # First pass: collect all (name, entry) pairs, preserving order.
    all_entries = []
    with open(filepath, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:

            if not should_keep(row, config):
                continue
            name = row[key_col].strip()
            if not name:
                continue
            entry = _build_entry(row, config)

            all_entries.append((name, entry))

    # Second pass: group by name and reconstruct variants where multiple rows
    # share the same CMIP Variable Name.
    grouped = {}
    for name, entry in all_entries:
        grouped.setdefault(name, []).append(entry)

    data = {}
    for name, entries in grouped.items():
        if len(entries) == 1:
            data[name] = entries[0]
        else:
            # Multiple rows → variants variable.
            # Base fields come from the first row (they are identical across rows).
            base = {k: v for k, v in entries[0].items() if k not in _VARIANT_FIELDS}
            variants = []
            # print(base)
            for e in entries:
                variant = {k: e[k] for k in _VARIANT_FIELDS if k in e}
                variants.append(variant)
            # print(variants)
            base["variants"] = variants
            data[name] = base

    return {
        "dataset_overrides": config["dataset_overrides"],
        "variables": data,
    }


# ── write yaml ────────────────────────────────────────────────────────────────
def write_yaml(data, filepath):
    """Write dictionary to a YAML file with minor formatting improvements."""
    with open(filepath, "w") as f:
        yaml.dump(data, f, default_flow_style=None)
    with open(filepath, "r") as f:
        lines = f.readlines()
    modified_lines = []
    for line in lines:
        if "{model_var:" in line:
            # Only reformat single-key source dicts: {model_var: VAR} → model_var: VAR
            # Leave multi-key dicts (e.g. {model_var: VAR, scale: -1.0}) untouched.
            line = re.sub(r"\{model_var: (\w+)\}", r"model_var: \1", line)
        if "FATES_FRAC" in line:
            line = line.replace("FATES_FRAC", "FATES_FRACTION")
        if "yaml_coax_dummy" in line:
            continue
        time_signifiers = [
            "_tavg-",
            "_tpt-",
            "_tclmdc-",
            "_ti-",
            "_tmin-",
            "_tminavg-",
            "_tmax-",
            "_tmaxavg-",
        ]
        add_newline = (
            any(sig in line for sig in time_signifiers) or "variables:" in line
        )
        if add_newline:
            modified_lines.append("\n")
        modified_lines.append(line)
    with open(filepath, "w") as f:
        f.writelines(modified_lines)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Convert CSV variable metadata to YAML for CMIP7 regridding pipelines."
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_CONFIGS.keys()),
        default="noresm",
        help="Model whose CSV column layout to use (default: noresm)",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Input CSV file (default: model-specific)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output YAML file (default: model-specific)",
    )
    args = parser.parse_args()

    config = MODEL_CONFIGS[args.model]
    input_file = args.input or config["default_input"]
    output_file = args.output or config["default_output"]

    data = read_csv(input_file, config)
    write_yaml(data, output_file)
    print(f"wrote {len(data['variables'])} entries to {output_file}")


if __name__ == "__main__":
    InlineListDumper.add_representer(list, inline_list_representer)
    main()
