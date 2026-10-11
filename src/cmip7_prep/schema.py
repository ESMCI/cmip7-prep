"""Validate mapping files against data/schemas/mapping.schema.yaml.

The schema is the contract for the mapping YAML; this module only loads it and
hands it to jsonschema.  Edit the schema, not this module, to change what a
mapping file may contain.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import jsonschema
from jsonschema.exceptions import best_match
import yaml

SCHEMA_PATH = (
    Path(__file__).parent.parent.parent / "data" / "schemas" / "mapping.schema.yaml"
)


@lru_cache(maxsize=None)
def _validator() -> jsonschema.Draft202012Validator:
    """Load the schema once, check that it is itself valid, and build a validator."""
    with open(SCHEMA_PATH, encoding="utf-8") as handle:
        schema = yaml.safe_load(handle)
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


def mapping_errors(data: Any) -> list[str]:
    """Return one message per schema violation in a loaded mapping file.

    An empty list means the file is valid.  Where the schema offers
    alternatives (an entry needs 'formula' or 'variants'), the most specific
    reason is reported.

    >>> mapping_errors({"variables": {"tas": "TREFHT"}})
    []
    >>> bad = {"formula": "T", "grids": ["gm"], "scale": 2}
    >>> print("\\n".join(mapping_errors({"variables": {"tas": bad}})))
    variables/tas: Additional properties are not allowed ('scale' was unexpected)
    variables/tas/grids/0: 'gm' is not one of ['gn', 'gr']
    """
    errors = sorted(
        (best_match([error]) for error in _validator().iter_errors(data)),
        key=lambda e: [str(p) for p in e.path],
    )
    return [
        f"{'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}" for e in errors
    ]
