import csv
import yaml
import re
import argparse

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
        "dataset_overrides": {
            "institution_id": "NCC",
            "source_id": "NorESM3",
            "nominal_resolution": "200 km",
        },
        "key_column": "Branded Variable Name",
        "column_map": {
            "Modelling Realm-Primary": "table",
            "CMIP6 Compound Name": "long_name",
            "Description": "description",
            "Units (from Physical Parameter)": "units",
            "Dimensions": "dims",
            "NorESM3 name (dependency)": "_source_expr",
            "CMIP7 Freq.": "freq",
        },
        "realm_column": "Modelling Realm-Primary",
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
            "Cell Methods": "cell_methods",
            "Regrid Method": "regrid_method",
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
    >>> should_keep({"Modelling Realm-Primary": "atmos", "NorESM3 name (dependency)": "T2M"}, cfg)
    True
    >>> should_keep({"Modelling Realm-Primary": "ocean", "NorESM3 name (dependency)": "SST"}, cfg)
    False
    >>> should_keep({"Modelling Realm-Primary": "atmos", "NorESM3 name (dependency)": ""}, cfg)
    False
    >>> should_keep({"Modelling Realm-Primary": "atmos", "NorESM3 name (dependency)": "n/a"}, cfg)
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
    """Convert a string with Norwegian number format to a float.

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


def clean_string(value):
    """Clean a string by stripping whitespace and normalising dimension names.

    >>> clean_string("longitude")
    'lon'
    >>> clean_string("latitude")
    'lat'
    >>> clean_string("  lev  ")
    'lev'
    >>> clean_string("'time'")
    'time'
    """
    if isinstance(value, str):
        value = value.replace("'", "").replace('"', "")  # remove quotes
        value = value.strip()
    if value == "longitude":
        value = "lon"
    elif value == "latitude":
        value = "lat"
    return value


def clean_strings(values):
    """Clean a list of strings."""
    if isinstance(values, list):
        return [clean_string(v) for v in values]
    elif isinstance(values, str):
        return clean_string(values)
    return values


# ── read csv ──────────────────────────────────────────────────────────────────
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
    """
    data = {}
    key_col = config["key_column"]
    column_map = config["column_map"]

    with open(filepath, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:

            if not should_keep(row, config):
                continue

            entry = {}
            for csv_col, yaml_key in column_map.items():
                if csv_col not in row:
                    continue  # skip columns absent from this CSV
                value = row[csv_col].strip()
                if not value:
                    continue  # skip blank cells

                if yaml_key == "dims":
                    dims = clean_strings(value.split(","))
                    entry["dims"] = dims
                    if "lev" in dims:
                        entry["levels"] = {
                            "name": "standard_hybrid_sigma",
                            "units": "1",
                            "src_axis_name": "lev",
                            "src_axis_bnds": "ilev",
                        }
                elif yaml_key == "units":
                    entry["units"] = fix_number_norwegian_format(value)
                elif yaml_key == "_source_expr":
                    result = analyse_expression(value)
                    if result["is_math"]:
                        entry["formula"] = value
                        entry["sources"] = result["variables"]
                    else:
                        entry["sources"] = result["variables"]
                else:
                    entry[yaml_key] = value

            name = row[key_col].strip()
            if name:
                data[name] = entry

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
            line = line.replace("{model_var:", "model_var: ").replace("}", "")
        modified_lines.append(line)
        if "units:" in line or "source_id:" in line:
            modified_lines.append("\n")
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
