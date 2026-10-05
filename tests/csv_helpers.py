"""Shared helpers for the ``convert_csv_to_yaml`` test modules.

The tests for that script are split across three modules -- unit tests for the
parsing helpers, integration tests for ``read_csv``, and tests for the warning
machinery -- and all three need to build a CSV on disk.  The column lists live
here too, so a spreadsheet layout change is one edit rather than three.
"""

import csv

# The CESM spreadsheet layout, as convert_csv_to_yaml expects to receive it.
CESM_FIELDNAMES = [
    "CMIP Branded Variable Name",
    "Table",
    "Long Name",
    "Standard Name",
    "Units",
    "Dimensions",
    "CESM Variable Name",
    "Formula",
    "Scale",
    "Freq",
    "Alias",
    "Cell Methods",
    "Regrid Method",
]

# The NorESM spreadsheet layout.
NORESM_FIELDNAMES = [
    "Branded Variable Name",
    "Modelling Realm - Primary",
    "CMIP6 Compound Name",
    "Description",
    "Units (from Physical Parameter)",
    "Dimensions",
    "NorESM3 name (dependency)",
    "CMIP7 Freq.",
]


def write_temp_csv(tmp_path, fieldnames, rows):
    """Write *rows* to a CSV under *tmp_path* and return its path as a string."""
    path = tmp_path / "test.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return str(path)
