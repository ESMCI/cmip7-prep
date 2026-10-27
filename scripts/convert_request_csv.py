"""Generate the CESM3 mapping files from the CMIP7 data request CSV.

    python scripts/convert_request_csv.py --output-dir data --report skipped.csv

This is a separate script from ``convert_csv_to_yaml.py`` on purpose.  That
script reads one row per variable and writes one string per variable, and
``MODEL_CONFIGS`` there is tied to the NorESM column layout (and to an older CESM
one).  The CESM data request CSV is shaped differently:

* It has one row per compound name, so a branded variable appears once per
  frequency and region.  The rows have to be merged into one mapping entry,
  with a per-frequency formula where they differ.  The converter's short form
  can only write a single string.
* Its formulas are free text -- trailing ``#`` comments, ``chunits(...)``
  wrappers, functions the pipeline does not have -- so each one is parsed, and
  rows that cannot be evaluated are skipped and reported instead of being
  written as mappings that fail at run time.
* It writes the file-level ``grids:`` line the checked-in mappings start with.

Keeping this apart leaves ``convert_csv_to_yaml.py``, and so the NorESM
workflow and its tests, untouched.  Grid labels and per-variable grid overrides
are shared with it by import.  If the two scripts are merged later, the merging
of rows (``_entry_for``) and the formula parsing (``parse_request_formula``)
are the parts worth keeping.

Writes ``cesm_to_cmip7_<realm>.yaml``, checked against
``data/schemas/mapping.schema.yaml`` before anything is written.
"""

import argparse
import ast
import csv
import os
import re
import sys

import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), "../", "src"))
sys.path.append(os.path.dirname(__file__))

# pylint: disable=wrong-import-position
from cmip7_prep.mapping_compat import FORMULA_NAMESPACE
from cmip7_prep.schema import mapping_errors
from convert_csv_to_yaml import GRIDS_OVERRIDES, REALM_GRIDS

# The realm grids of convert_csv_to_yaml, plus ocean biogeochemistry, which
# stays on the model's native grid like the ocean.
GRIDS = {**REALM_GRIDS, "ocnBgchem": ["gn"]}

CONFIG = {
    "model": "cesm",
    "default_input": "data/cmip7.data_request.cesm.v1.0.airtv1225.csv",
    "key_column": "Branded Variable Name",
    "realm_column": "Modelling Realm - Primary",
    "source_column": "CESM Variable Name (T4S)",
    "formula_column": "Formula",
    "freq_column": "CMIP7 Frequency",
    "region_column": "Region",
    "realms": [
        "atmos",
        "atmosChem",
        "aerosol",
        "land",
        "landIce",
        "seaIce",
        "ocean",
        "ocnBgchem",
    ],
}

# Frequencies a per-frequency formula may be keyed by (the mapping schema).
FORMULA_FREQUENCIES = {"1hr", "3hr", "6hr", "day", "mon", "yr"}

_CHUNITS = re.compile(
    r"^chunits\((?P<inner>.+),\s*units=['\"](?P<units>[^'\"]+)['\"]\)$"
)
_SUM_OVER = re.compile(r"\bsum\((?P<arr>\w+),\s*['\"](?P<dim>\w+)['\"]\)")
# X[:, k, :, :] picks one level of a (time, level, lat, lon) array.  CLM and CICE
# write unstructured (time, level, ncol) arrays, where that fourth index is out
# of range, so the trailing full slices are written as an ellipsis instead.
_TRAILING_SLICES = re.compile(r"\[(?P<head>:,\s*\d+)(?:,\s*:)+\]")
_NOT_AVAILABLE = {"", "n/a", "?"}


def parse_request_formula(source: str, formula: str, freq: str = "") -> tuple:
    """Turn one row's source and formula cells into ``(formula, units, reason)``.

    Exactly one of ``formula`` and ``reason`` is set: ``reason`` says why the
    row cannot be expressed as a mapping entry.  ``units`` comes from a
    ``chunits(expr, units='...')`` wrapper, which the mapping spells as a
    ``units`` field instead.  ``freq`` picks between a daily variable and its
    monthly twin when the source cell lists both, as the sea-ice rows do.

    >>> parse_request_formula("TREFHT", "")
    ('TREFHT', None, None)
    >>> parse_request_formula("CLDTOT", "CLDTOT*100  #cam.h0a")
    ('CLDTOT*100', None, None)
    >>> parse_request_formula("CCN6", "chunits(CCN6, units='cm-3')")
    ('CCN6', 'cm-3', None)
    >>> parse_request_formula("TOT_CLD_VISTAU", 'sum(TOT_CLD_VISTAU, "lev")')
    ('TOT_CLD_VISTAU.sum(dim="lev")', None, None)
    >>> parse_request_formula("P", "P[:,1,:,:]+P[:,2,:,:]")
    ('P[:,1,...]+P[:,2,...]', None, None)
    >>> parse_request_formula("A, B, C", "(A+B,C)")[2]
    'formula is a tuple, not a single expression'
    >>> parse_request_formula("CLIMODIS [COSP]", "")
    ('CLIMODIS', None, None)
    >>> parse_request_formula("sisi_d, sisi", "", "day")
    ('sisi_d', None, None)
    >>> parse_request_formula("sisi_d, sisi", "", "mon")
    ('sisi', None, None)
    >>> parse_request_formula("A, B", "")[2]
    'several source variables but no formula'
    >>> parse_request_formula("X", "yeartomonth_data(X)")[2]
    'unsupported function yeartomonth_data'
    >>> parse_request_formula("N/A", "")[2]
    'no CESM source variable'
    """
    source = source.replace("[COSP]", "").strip()
    if source.lower() in _NOT_AVAILABLE:
        return None, None, "no CESM source variable"
    formula = formula.split("#", 1)[0].replace('\\"', '"').strip()
    if not formula:
        names = [n.strip() for n in source.split(",") if n.strip()]
        if not all(re.fullmatch(r"[A-Za-z_]\w*", n) for n in names):
            return None, None, "several source variables but no formula"
        if len(names) == 2 and names[0] == names[1] + "_d":
            return (names[0] if freq == "day" else names[1]), None, None
        if len(names) != 1:
            return None, None, "several source variables but no formula"
        return names[0], None, None

    units = None
    match = _CHUNITS.match(formula)
    if match:
        formula, units = match["inner"].strip(), match["units"]
    formula = _SUM_OVER.sub(r'\g<arr>.sum(dim="\g<dim>")', formula)
    formula = _TRAILING_SLICES.sub(r"[\g<head>,...]", formula)

    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError:
        return None, None, "formula is not a valid expression"
    if isinstance(tree.body, ast.Tuple):
        # A stray comma, as in (A+B,C) for A+B+C, would evaluate to a tuple.
        return None, None, "formula is a tuple, not a single expression"
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id not in FORMULA_NAMESPACE
        ):
            return None, None, f"unsupported function {node.func.id}"
    return formula, units, None


def _entry_for(rows: list) -> tuple:
    """Build one mapping entry from every parsed row of a branded variable.

    ``rows`` holds ``(frequency, region, formula, units)``.  Returns
    ``(entry, notes)``: a formula string or an entry dict, and anything dropped
    to fit the mapping schema.
    """
    notes = []
    by_region: dict = {}
    for freq, region, formula, units in rows:
        by_region.setdefault(region, {}).setdefault(freq, (formula, units))

    def per_frequency(freqs: dict):
        formulas = {f: v[0] for f, v in freqs.items()}
        if len(set(formulas.values())) == 1:
            return next(iter(formulas.values()))
        kept = {f: v for f, v in formulas.items() if f in FORMULA_FREQUENCIES}
        for freq in sorted(set(formulas) - set(kept)):
            notes.append(f"formula for frequency {freq} dropped")
        return kept or None

    units_seen = [r[3] for r in rows if r[3]]
    entry: dict = {}
    if units_seen:
        entry["units"] = units_seen[0]
        if len(set(units_seen)) > 1:
            notes.append(f"conflicting units, kept {units_seen[0]}")

    # A global row is the variable itself; the regional rows beside it repeat it
    # for a subregion and are not separate outputs.  Variables with no global
    # row (the hemispheric sea-ice totals) have one output per region.
    if "glb" in by_region:
        dropped = sorted(set(by_region) - {"glb"})
        if dropped:
            notes.append(f"regions {dropped} ignored")
        by_region = {"glb": by_region["glb"]}
    resolved = {region: per_frequency(freqs) for region, freqs in by_region.items()}
    resolved = {r: f for r, f in resolved.items() if f}
    if not resolved:
        return None, notes
    distinct = {repr(f) for f in resolved.values()}
    if len(distinct) == 1:
        formula = next(iter(resolved.values()))
    else:
        formula = None
        entry["variants"] = [
            {"region": region, "formula": f} for region, f in sorted(resolved.items())
        ]
    if formula is not None:
        entry["formula"] = formula
    if list(entry) == ["formula"] and isinstance(entry["formula"], str):
        return entry["formula"], notes
    return entry, notes


def read_request_csv(filepath, config):
    """Read the CMIP7 request CSV into ``({realm: {name: entry}}, skipped)``.

    ``skipped`` lists ``(realm, name, reason)`` for every variable that could
    not be mapped, so nothing is dropped silently.
    """
    key_col, realm_col = config["key_column"], config["realm_column"]
    grouped: dict = {}
    problems: dict = {}
    with open(filepath, "r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            realm = row[realm_col].strip()
            name = row[key_col].strip()
            if realm not in config["realms"] or not name:
                continue
            formula, units, reason = parse_request_formula(
                row[config["source_column"]],
                row[config["formula_column"]],
                row[config["freq_column"]].strip(),
            )
            key = (realm, name)
            if reason:
                problems.setdefault(key, set()).add(reason)
                continue
            grouped.setdefault(key, []).append(
                (
                    row[config["freq_column"]].strip(),
                    row[config["region_column"]].strip(),
                    formula,
                    units,
                )
            )

    mappings = {realm: {} for realm in config["realms"]}
    skipped = []
    for (realm, name), rows in grouped.items():
        entry, notes = _entry_for(rows)
        if entry is None:
            problems.setdefault((realm, name), set()).update(notes)
            continue
        override = GRIDS_OVERRIDES.get(name)
        if override and set(override) <= {"gn", "gr"}:
            if set(override) != set(GRIDS[realm]):
                entry = {"formula": entry} if isinstance(entry, str) else entry
                entry["grids"] = list(override)
        mappings[realm][name] = entry
    for (realm, name), reasons in sorted(problems.items()):
        if name not in mappings[realm]:
            skipped.append((realm, name, "; ".join(sorted(reasons))))
    return mappings, skipped


def write_request_yaml(realm, variables, filepath, source_csv):
    """Write one realm's mapping file: a header, the default grids, the entries."""
    body = yaml.safe_dump(
        {"variables": variables}, sort_keys=True, default_flow_style=False, width=10**6
    )
    lines = [f"# Generated by scripts/convert_csv_to_yaml.py from {source_csv}.\n"]
    lines.append("# Edit the CSV and regenerate rather than editing this file.\n")
    lines.append(f"grids: [{GRIDS[realm][0]}]\n")
    for line in body.splitlines(keepends=True):
        # A blank line before each entry, as the hand-written files have.
        if re.match(r"^  \S", line):
            lines.append("\n")
        lines.append(line)
    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(lines)


def convert(args, config):
    """Generate every realm's mapping file from a request-layout CSV."""
    input_file = args.input or config["default_input"]
    mappings, skipped = read_request_csv(input_file, config)
    total = 0
    for realm, variables in mappings.items():
        errors = mapping_errors({"variables": variables})
        if errors:
            raise ValueError(f"{realm}: generated mapping fails the schema: {errors}")
        path = os.path.join(args.output_dir, f"{config['model']}_to_cmip7_{realm}.yaml")
        write_request_yaml(realm, variables, path, os.path.basename(input_file))
        total += len(variables)
        print(f"wrote {len(variables)} entries to {path}")
    print(f"total: {total} entries; {len(skipped)} variables skipped")
    if args.report:
        with open(args.report, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["realm", "variable", "reason"])
            writer.writerows(skipped)
        print(f"skipped variables listed in {args.report}")


def main():
    """Parse the command line and generate the mapping files."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument(
        "--input",
        default=None,
        help=f"Request CSV (default: {CONFIG['default_input']})",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for the mapping files (default: current directory)",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Write the variables that could not be mapped to this CSV",
    )
    convert(parser.parse_args(), CONFIG)


if __name__ == "__main__":
    main()
