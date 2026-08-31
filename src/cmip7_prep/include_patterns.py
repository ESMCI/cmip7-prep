"""History-file include patterns, loaded from the packaged YAML tables.

Each supported model has a ``data/<model>_include_patterns.yaml`` table mapping
realm, frequency and time sampling to the history-file name substrings that hold
that output::

    atmos:
      mon:
        tavg: [cam.h0a]
        tpt:  [cam.h0i]
      day:
        tavg: [cam.h1a]

The sampling level exists because CMIP7 frequency does not distinguish
time-averaged from instantaneous output -- both a ``tavg`` and a ``tpt``
variable can be monthly.  The branded variable name carries it, and CAM encodes
it in the file name as the tape's averaging flag ('a' or 'i').

``gen_timeseries.py`` globs for ``*<pattern>*`` when collecting raw history
files; ``cmor_driver.py`` uses the same patterns to find the time-series files
it CMORizes.  Keeping the tables in YAML lets a new realm or a changed history
file name be added without touching Python.

Patterns may contain a ``{ice_sheet}`` placeholder, used by the land-ice realm
so a single CISM domain (gris or ais) is selected per run.  The helpers here
fill it in; a caller that forgets to supply an ice sheet gets a ValueError
rather than a pattern with a literal brace in it.
"""

from __future__ import annotations

import copy
import logging
from functools import lru_cache
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"


@lru_cache(maxsize=None)
def _load_cached(model: str) -> dict[str, dict[str, list[str]]]:
    """Read and cache one model's include-pattern table."""
    path = DATA_DIR / f"{model}_include_patterns.yaml"
    if not path.is_file():
        raise ValueError(
            f"No include-pattern table for model={model!r}; expected {path}"
        )
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_include_patterns(model: str) -> dict[str, dict[str, list[str]]]:
    """Return the realm/frequency pattern table for ``model``.

    A copy is returned, so callers may not corrupt the cached table.
    """
    return copy.deepcopy(_load_cached(model))


def _substitute_ice_sheet(
    patterns: list[str], realm: str, ice_sheet: str | None
) -> list[str]:
    """Fill the '{ice_sheet}' placeholder, if the patterns use one."""
    if not any("{ice_sheet}" in pattern for pattern in patterns):
        return list(patterns)
    if ice_sheet is None:
        raise ValueError(f"realm {realm!r} requires --ice-sheet (gris or ais)")
    return [pattern.format(ice_sheet=ice_sheet) for pattern in patterns]


def _select_sampling(
    by_sampling: dict[str, list[str]],
    sampling: str | None,
    where: str,
) -> list[str]:
    """Pick one sampling out of a frequency entry, or all of them.

    ``sampling`` of None means every sampling, which is what bulk collection
    wants.  Naming one that the model does not write is an error rather than an
    empty result: reading time-averaged files for an instantaneous variable
    produces plausible-looking output that is silently wrong, and an empty
    pattern list makes the caller quietly do nothing.
    """
    if sampling is None:
        patterns: list[str] = []
        for entry in by_sampling.values():
            patterns.extend(entry)
        return patterns
    if sampling not in by_sampling:
        raise ValueError(
            f"No include_patterns defined for {where}, sampling={sampling}; "
            f"available: {sorted(by_sampling)}"
        )
    return list(by_sampling[sampling])


def get_include_patterns(
    model: str,
    realm: str,
    frequency: str,
    ice_sheet: str | None = None,
    sampling: str | None = "tavg",
) -> list[str]:
    """Return the include patterns for one model, realm, frequency and sampling.

    ``sampling`` defaults to 'tavg', since CMORizing a variable needs exactly one
    file family and most CMIP7 variables are time-averaged.  Pass the sampling
    parsed from the branded variable name to select instantaneous output.
    """
    try:
        by_sampling = _load_cached(model)[realm][frequency]
    except KeyError:
        raise ValueError(
            f"No include_patterns defined for model={model}, "
            f"realm={realm}, frequency={frequency}"
        ) from None
    where = f"model={model}, realm={realm}, frequency={frequency}"
    patterns = _select_sampling(by_sampling, sampling, where)
    patterns = _substitute_ice_sheet(patterns, realm, ice_sheet)
    logger.info("Looking for pattern: %s", patterns)
    return patterns


def all_include_patterns(
    model: str,
    realm: str,
    ice_sheet: str | None = None,
    sampling: str | None = None,
) -> list[str]:
    """Return the patterns for every frequency of one model and realm.

    Used when collecting raw history files, where all frequencies of a realm are
    gathered in one pass rather than one frequency at a time.  ``sampling`` of
    None collects time-averaged and instantaneous output together, which is what
    the time-series step wants: it cannot know which the later CMORization will
    ask for.  Naming one narrows the sweep.
    """
    try:
        by_frequency = _load_cached(model)[realm]
    except KeyError:
        raise ValueError(
            f"No include_patterns defined for model={model}, realm={realm}"
        ) from None

    # Frequencies can share a history file (NorESM land uses clm2.h2a for both
    # 3hr and yr), so de-duplicate while preserving order to avoid globbing the
    # same pattern twice.
    patterns: list[str] = []
    matched = False
    for frequency, by_sampling in by_frequency.items():
        if sampling is not None and sampling not in by_sampling:
            continue
        matched = True
        where = f"model={model}, realm={realm}, frequency={frequency}"
        for pattern in _substitute_ice_sheet(
            _select_sampling(by_sampling, sampling, where), realm, ice_sheet
        ):
            if pattern not in patterns:
                patterns.append(pattern)

    if sampling is not None and not matched:
        raise ValueError(
            f"No include_patterns defined for model={model}, realm={realm}, "
            f"sampling={sampling}; the model writes no {sampling} output for "
            f"this realm"
        )
    return patterns


def sampling_from_branded_name(branded_name: str) -> str:
    """Return the time signifier from a CMIP7 branded variable name.

    Branded names are ``<root>_<temporal>-<vertical>-<horizontal>-<area>``, so
    ``tas_tavg-u-hxy-u`` gives 'tavg' and ``ps_tpt-u-hxy-u`` gives 'tpt'.

    >>> sampling_from_branded_name("tas_tavg-u-hxy-u")
    'tavg'
    >>> sampling_from_branded_name("acabf_tpt-u-hxy-is")
    'tpt'
    """
    if "_" not in branded_name:
        raise ValueError(
            f"{branded_name!r} is not a branded variable name "
            f"(expected <root>_<temporal>-<vertical>-<horizontal>-<area>)"
        )
    _root, compound = branded_name.rsplit("_", 1)
    return compound.split("-")[0]


def patterns_for_variable(
    model: str,
    realm: str,
    frequency: str,
    branded_name: str,
    ice_sheet: str | None = None,
) -> list[str]:
    """Return the include patterns for one CMIP7 variable.

    The sampling comes from the variable's own branded name, because CMIP7
    frequency does not distinguish time-averaged from instantaneous output.

    Signifiers the table does not declare (``ti``, ``tsum``, ``tmin``, ``tmax``
    and friends) fall back to the time-averaged patterns, which is where they
    have always been read from, but log a warning: whether they need their own
    history tape is a per-model question that has not been answered here.
    """
    sampling = sampling_from_branded_name(branded_name)
    try:
        return get_include_patterns(
            model, realm, frequency, ice_sheet=ice_sheet, sampling=sampling
        )
    except ValueError:
        if sampling == "tavg":
            raise
        logger.warning(
            "No %r include patterns for model=%s realm=%s frequency=%s; "
            "falling back to time-averaged files for %s. If %s output lives in "
            "its own history tape, add it to the include-pattern table.",
            sampling,
            model,
            realm,
            frequency,
            branded_name,
            sampling,
        )
        return get_include_patterns(
            model, realm, frequency, ice_sheet=ice_sheet, sampling="tavg"
        )
