"""Reading the controlled vocabulary, to reject bad values before a run starts.

CMOR validates global attributes against the controlled vocabulary when it
writes, and refuses the write if one is not registered.  That check is the right
one, but it arrives late: by then the time series stage has already run, which
is the expensive part.  The helpers here read the same file CMOR reads, so a
misspelled experiment can be rejected in the first second instead of the last.

The vocabulary lives in the CMOR tables checkout, not in this package, so every
function takes the tables root.  It is read on demand and cached, since a run
asks about a handful of values at most.
"""

from __future__ import annotations

import difflib
import json
import os
from functools import lru_cache
from pathlib import Path

# Where the vocabulary sits relative to the tables checkout.
CV_RELATIVE_PATH = Path("tables-cvs") / "cmor-cvs.json"

# The one key the file wraps everything in.
CV_ROOT_KEY = "CV"


def cv_path(tables_root: os.PathLike | str) -> Path:
    """Return the path of the controlled vocabulary within a tables checkout."""
    return Path(tables_root) / CV_RELATIVE_PATH


@lru_cache(maxsize=4)
def _load(tables_root: str) -> dict:
    """Return the vocabulary's contents, cached per tables checkout."""
    path = cv_path(tables_root)
    if not path.is_file():
        raise FileNotFoundError(
            f"No controlled vocabulary at {path}. Point --tables-root at a "
            "cmip7-cmor-tables checkout, or update the one you have."
        )
    document = json.loads(path.read_text(encoding="utf-8"))
    return document.get(CV_ROOT_KEY, document)


def allowed_values(tables_root: os.PathLike | str, key: str) -> list[str]:
    """Return the registered values of one controlled attribute, sorted.

    ``key`` is a vocabulary key such as ``experiment_id`` or ``source_id``.
    Some attributes nest -- the licences live at ``license.license_id`` -- so a
    dotted path walks down to them.  An unknown key raises, listing what is
    controlled at that level, so a typo in the key is as loud as one in a value.
    """
    entry = _load(str(tables_root))
    walked = []
    for part in key.split("."):
        if not isinstance(entry, dict) or part not in entry:
            where = ".".join(walked) or "the vocabulary"
            raise KeyError(
                f"{key!r} is not a controlled attribute in "
                f"{cv_path(tables_root)}; {where} controls: "
                f"{sorted(entry) if isinstance(entry, dict) else entry}"
            )
        entry = entry[part]
        walked.append(part)
    if isinstance(entry, dict):
        return sorted(entry)
    if isinstance(entry, list):
        return sorted(str(value) for value in entry)
    return [str(entry)]


def validate(tables_root: os.PathLike | str, key: str, value: str) -> None:
    """Raise ValueError unless ``value`` is registered for ``key``.

    The message names the nearest registered values, since the usual cause is a
    small misspelling and 75 experiment ids are too many to print.
    """
    allowed = allowed_values(tables_root, key)
    if value in allowed:
        return
    suggestions = difflib.get_close_matches(value, allowed, n=5, cutoff=0.5)
    hint = (
        f" Did you mean: {', '.join(suggestions)}?"
        if suggestions
        else f" It controls {len(allowed)} value(s); see {cv_path(tables_root)}."
    )
    raise ValueError(f"{key} {value!r} is not in the controlled vocabulary.{hint}")
