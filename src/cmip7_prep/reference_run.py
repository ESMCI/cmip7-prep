"""Plan a full-chain run over one archived case, for every realm.

This is the planning half of ``scripts/run_reference_case.py``: it works out
which commands a complete run needs, in which order, without running anything.
Keeping it here rather than in the script means the plan can be tested, and
printed for inspection before a long run is started.

The archive layout assumed is CIME's short-term archive:

    <case-dir>/<component>/hist/     history files, one component per realm
    <case-dir>/<component>/proc/...  where processed output conventionally goes

though the output root is given separately, so a run never has to write back
into an archive it may not own.
"""

from __future__ import annotations

import glob
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .grids import ATM_RESOLUTIONS, needs_atmos_res
from .include_patterns import all_include_patterns, load_include_patterns

# Which component directory holds each realm's history files.  Mirrors
# REALM_COMPONENT_MAP in scripts/cmor_driver.py.
REALM_COMPONENT = {
    "atmos": "atm",
    "aerosol": "atm",
    "atmosChem": "atm",
    "land": "lnd",
    "ocean": "ocn",
    "ocnBgchem": "ocn",
    "seaIce": "ice",
    "landIce": "glc",
}

# The ice sheet to process when a realm is per-ice-sheet and none was named.
DEFAULT_ICE_SHEET = "gris"

STAGES = ("timeseries", "cmor", "validate")

# Frequencies are processed coarsest first, so the cheapest and most used
# output exists soonest.  Alphabetical order would put 'day' before 'mon' and
# '1hr' first of all, leaving the monthly output until last.
FREQUENCY_ORDER = ("yr", "mon", "day", "6hr", "3hr", "1hr")

# The steps run with the interpreter running this, not whatever "python" means
# on PATH: that keeps every stage in the environment the run was started from,
# and works when the runner was invoked as python3 or by absolute path.
PYTHON = sys.executable or "python"

# CMORization is run with one worker: running it in parallel has not been
# tested.  Its --workers are Dask workers in any case, which is a different
# thing from the processes gen_timeseries.py spreads file reads over, so the
# two are not set from one number.
CMOR_WORKERS = 1


@dataclass
class Step:
    """One command the run will execute.

    ``key`` identifies the step in the final report; ``realm`` and ``frequency``
    are carried so a failure can be attributed without parsing the command.
    """

    key: str
    stage: str
    realm: str
    command: list[str]
    frequency: str | None = None
    log_name: str = ""

    def __post_init__(self) -> None:
        if not self.log_name:
            self.log_name = f"{self.key}.log"


@dataclass
class Plan:
    """A complete run: its steps, and the directories they use."""

    steps: list[Step] = field(default_factory=list)
    timeseries_root: Path | None = None
    cmor_root: Path | None = None
    skipped: list[str] = field(default_factory=list)

    def for_stage(self, stage: str) -> list[Step]:
        """Return the steps belonging to one stage."""
        return [step for step in self.steps if step.stage == stage]


def realms_for(model: str) -> list[str]:
    """Return the realms a model's include-pattern table declares."""
    return list(load_include_patterns(model))


def frequencies_for(model: str, realm: str) -> list[str]:
    """Return the frequencies declared for one model and realm, coarsest first.

    The order is the order a run will process them in, so it is chosen rather
    than alphabetical: see FREQUENCY_ORDER.
    """
    declared = load_include_patterns(model)[realm]
    known = [f for f in FREQUENCY_ORDER if f in declared]
    rest = sorted(f for f in declared if f not in FREQUENCY_ORDER)
    return known + rest


def written_frequencies(
    case_dir: os.PathLike | str,
    model: str,
    realm: str,
    frequencies: Sequence[str],
    ice_sheet: str | None = None,
) -> list[str]:
    """Return those frequencies the case actually wrote history files for.

    A table declares every frequency a realm *can* produce, but a given run
    writes only the streams it was configured for.  Planning a CMORization step
    for a frequency with no history files behind it guarantees a failure that
    says nothing about the code, so those frequencies are dropped from the plan
    instead.
    """
    inputdir = history_dir(case_dir, realm)
    written = []
    for frequency in frequencies:
        patterns = all_include_patterns(model, realm, ice_sheet, None, [frequency])
        if any(
            glob.glob(os.path.join(inputdir, f"*{pattern}*")) for pattern in patterns
        ):
            written.append(frequency)
    return written


def history_dir(case_dir: os.PathLike | str, realm: str) -> Path:
    """Return the history directory holding one realm's output."""
    component = REALM_COMPONENT[realm]
    return Path(case_dir) / component / "hist"


def build_plan(
    case_dir: os.PathLike | str,
    outdir: os.PathLike | str,
    *,
    model: str,
    realms: Sequence[str] | None = None,
    frequencies: Sequence[str] | None = None,
    years: str | None = None,
    stages: Sequence[str] = STAGES,
    atmos_res: str | None = None,
    experiment: str,
    workers: int = 4,
    ice_sheet: str | None = None,
    overwrite_timeseries: bool = False,
    variant_label: str | None = None,
    tables_root: os.PathLike | str | None = None,
    scripts_dir: os.PathLike | str = "scripts",
) -> Plan:
    """Return the full plan for one reference-case run.

    ``realms`` of None takes every realm the model declares; a realm whose
    history directory is absent is recorded in ``Plan.skipped`` rather than
    failing the run, since an archived case need not hold every component.

    ``atmos_res`` is the grid the atmosphere and land were run on; every
    other realm's grid is derived from it or from the model.

    ``workers`` applies to time series generation only; CMORization always runs
    with one worker (see CMOR_WORKERS).

    ``variant_label`` and ``tables_root`` are passed to CMORization unchanged,
    so the same command can produce output for submission rather than only
    exercising the chain.

    ``years`` is passed through to gen_timeseries.py as ``--years-spec`` and so
    uses its format, ``first:last:increment``.
    """
    unknown = [stage for stage in stages if stage not in STAGES]
    if unknown:
        raise ValueError(f"Unknown stage(s) {unknown}; choose from {list(STAGES)}")
    wanted_realms = list(realms) if realms else realms_for(model)
    on_atmos_grid = [realm for realm in wanted_realms if needs_atmos_res(realm)]
    if on_atmos_grid and atmos_res is None:
        raise ValueError(
            "An atmosphere resolution is needed for "
            f"{on_atmos_grid}; the other realms derive their own grid"
        )
    if atmos_res is not None and atmos_res not in ATM_RESOLUTIONS:
        raise ValueError(
            f"Unknown atmosphere resolution {atmos_res!r}; "
            f"choose from {list(ATM_RESOLUTIONS)}"
        )

    case_dir = Path(case_dir)
    outdir = Path(outdir)
    scripts_dir = Path(scripts_dir)
    timeseries_root = outdir / "timeseries"
    cmor_root = outdir / "cmor"

    plan = Plan(timeseries_root=timeseries_root, cmor_root=cmor_root)

    for realm in wanted_realms:
        if realm not in REALM_COMPONENT:
            raise ValueError(
                f"Realm {realm!r} has no component directory; "
                f"known realms: {sorted(REALM_COMPONENT)}"
            )
        inputdir = history_dir(case_dir, realm)
        if not inputdir.is_dir():
            plan.skipped.append(f"{realm}: no history directory at {inputdir}")
            continue

        declared = _realm_frequencies(model, realm, frequencies)
        if not declared:
            plan.skipped.append(
                f"{realm}: none of the requested frequencies are declared"
            )
            continue

        sheet_for_patterns = (
            ice_sheet or DEFAULT_ICE_SHEET if realm == "landIce" else None
        )
        realm_frequencies = written_frequencies(
            case_dir, model, realm, declared, sheet_for_patterns
        )
        absent = [f for f in declared if f not in realm_frequencies]
        if absent:
            plan.skipped.append(
                f"{realm}: no history files for {absent}; this case did not "
                "write those streams"
            )
        if not realm_frequencies:
            continue

        sheet = ice_sheet or DEFAULT_ICE_SHEET if realm == "landIce" else None
        ts_dir = timeseries_root / realm

        if "timeseries" in stages:
            plan.steps.append(
                _timeseries_step(
                    scripts_dir=scripts_dir,
                    realm=realm,
                    inputdir=inputdir,
                    ts_dir=ts_dir,
                    model=model,
                    frequencies=realm_frequencies,
                    years=years,
                    workers=workers,
                    sheet=sheet,
                    overwrite=overwrite_timeseries,
                )
            )

        for frequency in realm_frequencies:
            if "cmor" in stages:
                plan.steps.append(
                    _cmor_step(
                        scripts_dir=scripts_dir,
                        realm=realm,
                        frequency=frequency,
                        ts_dir=ts_dir,
                        cmor_root=cmor_root,
                        model=model,
                        atmos_res=(atmos_res if needs_atmos_res(realm) else None),
                        experiment=experiment,
                        sheet=sheet,
                        variant_label=variant_label,
                        tables_root=tables_root,
                    )
                )
            if "validate" in stages:
                plan.steps.append(
                    _validate_step(
                        scripts_dir=scripts_dir,
                        realm=realm,
                        frequency=frequency,
                        cmor_root=cmor_root,
                        model=model,
                        experiment=experiment,
                        variant_label=variant_label,
                    )
                )

    return plan


def _realm_frequencies(
    model: str, realm: str, requested: Sequence[str] | None
) -> list[str]:
    """Return the declared frequencies of a realm, narrowed to those requested."""
    declared = frequencies_for(model, realm)
    if not requested:
        return declared
    return [frequency for frequency in declared if frequency in requested]


def _timeseries_step(
    *,
    scripts_dir,
    realm,
    inputdir,
    ts_dir,
    model,
    frequencies,
    years,
    workers,
    sheet,
    overwrite=False,
) -> Step:
    """Return the gen_timeseries.py step for one realm."""
    command = [
        PYTHON,
        str(Path(scripts_dir) / "gen_timeseries.py"),
        "--inputdir",
        str(inputdir),
        "--outputdir",
        str(ts_dir),
        "--realm",
        realm,
        "--model",
        model,
        "--workers",
        str(workers),
        "--frequency",
        *frequencies,
    ]
    if years:
        command += ["--years-spec", years]
    if sheet:
        command += ["--ice-sheet", sheet]
    if overwrite:
        command.append("--overwrite_timeseries")
    return Step(
        key=f"timeseries-{realm}", stage="timeseries", realm=realm, command=command
    )


def _cmor_step(
    *,
    scripts_dir,
    realm,
    frequency,
    ts_dir,
    cmor_root,
    model,
    atmos_res,
    experiment,
    sheet,
    variant_label=None,
    tables_root=None,
) -> Step:
    """Return the cmor_driver.py step for one realm and frequency."""
    command = [
        PYTHON,
        str(Path(scripts_dir) / "cmor_driver.py"),
        "--realm",
        realm,
        "--frequency",
        frequency,
        "--tsdir",
        str(ts_dir),
        "--outdir",
        str(cmor_root),
        "--model",
        model,
        "--experiment",
        experiment,
        "--workers",
        str(CMOR_WORKERS),
    ]
    if atmos_res:
        command += ["--atmos-res", atmos_res]
    if sheet:
        command += ["--ice-sheet", sheet]
    if variant_label:
        command += ["--variant-label", variant_label]
    if tables_root:
        command += ["--tables-root", str(tables_root)]
    return Step(
        key=f"cmor-{realm}-{frequency}",
        stage="cmor",
        realm=realm,
        frequency=frequency,
        command=command,
    )


def _validate_step(
    *,
    scripts_dir,
    realm,
    frequency,
    cmor_root,
    model,
    experiment,
    variant_label=None,
) -> Step:
    """Return the validate_cmor_output.py step for one realm and frequency."""
    command = [
        PYTHON,
        str(Path(scripts_dir) / "validate_cmor_output.py"),
        "--model",
        model,
        "--realm",
        realm,
        "--frequency",
        frequency,
        "--experiment",
        experiment,
        "--root-output-path",
        str(cmor_root),
        "--strict",
    ]
    if variant_label:
        command += ["--ensemble-member", variant_label]
    return Step(
        key=f"validate-{realm}-{frequency}",
        stage="validate",
        realm=realm,
        frequency=frequency,
        command=command,
    )
