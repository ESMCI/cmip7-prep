"""The vYYYYMMDD edition a CMORized dataset belongs to."""

import datetime as dt
import re

_DATA_VERSION_RE = re.compile(r"^v\d{8}$")


def checked_data_version(value: str) -> str:
    """Return the data version unchanged, or raise if it is not vYYYYMMDD.

    One dataset carries one version however many invocations wrote it, so the
    value is given on the command line rather than taken from the clock. A run
    split into ten-year chunks would otherwise put each chunk in the directory
    of the day it happened to run on.
    """
    text = str(value).strip()
    if not _DATA_VERSION_RE.match(text):
        raise ValueError(
            f"--data-version takes vYYYYMMDD, the date of the edition, not {value!r}"
        )
    try:
        dt.date(int(text[1:5]), int(text[5:7]), int(text[7:9]))
    except ValueError as exc:
        raise ValueError(f"--data-version: {value!r} is not a date") from exc
    return text
