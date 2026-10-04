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

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .include_patterns import load_include_patterns

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

# Grids the atmosphere and land may be run on.  'regular' means the output
# already carries lat/lon and is not regridded.
ATM_RESOLUTIONS = ("ne30", "ne16", "regular")

# Realms whose input grid is the atmosphere/land grid the case was run on.
ATM_GRID_REALMS = frozenset({"atmos", "atmosChem", "aerosol", "land"})

# Realms written on their native grid, with no ESMF regridding.  CISM land-ice
# output carries projected x/y coordinates which cmor_writer georeferences with
# the ice sheet's own map projection (see cism_grid.project_xy_to_latlon), so no
# weight files are involved and the resolution would be ignored.  The value
# below is a pass-through: it names the entry that builds a regridder and
# discards it, which is what the unregridded path expects.
NATIVE_GRID_REALMS = frozenset({"landIce"})
NATIVE_GRID = "regular"

# The ocean/sea-ice grid each model writes on.  These realms never share the
# atmosphere's grid, so passing one an atmosphere resolution would regrid
# through the wrong weights and produce plausible but wrong output.
OCEAN_GRID = {"noresm": "tnx1v4", "cesm": "tx2_3v2"}

OCEAN_GRID_REALMS = frozenset({"ocean", "ocnBgchem", "seaIce"})

STAGES = ("timeseries", "cmor", "validate")


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
    """Return the frequencies declared for one model and realm."""
    return sorted(load_include_patterns(model)[realm])


def resolution_for(model: str, realm: str, atm_resolution: str) -> str:
    """Return the input grid name for one realm.

    Every realm's grid is derived, so no caller has to know the model's grid
    layout.  ``atm_resolution`` is the grid the atmosphere and land were run on
    and applies only to those realms; ocean and sea ice are always on the
    model's tripolar grid; land ice is written on its native projected grid and
    is not regridded at all.
    """
    if atm_resolution not in ATM_RESOLUTIONS:
        raise ValueError(
            f"Unknown atmosphere resolution {atm_resolution!r}; "
            f"choose from {list(ATM_RESOLUTIONS)}"
        )
    if realm in ATM_GRID_REALMS:
        return atm_resolution
    if realm in OCEAN_GRID_REALMS:
        try:
            return OCEAN_GRID[model]
        except KeyError:
            raise ValueError(
                f"No ocean grid known for model={model!r}; "
                f"known models: {sorted(OCEAN_GRID)}"
            ) from None
    if realm in NATIVE_GRID_REALMS:
        return NATIVE_GRID
    raise ValueError(f"No input grid known for realm {realm!r}")


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
    atmos_resolution: str,
    experiment: str,
    workers: int = 4,
    ice_sheet: str | None = None,
    scripts_dir: os.PathLike | str = "scripts",
) -> Plan:
    """Return the full plan for one reference-case run.

    ``realms`` of None takes every realm the model declares; a realm whose
    history directory is absent is recorded in ``Plan.skipped`` rather than
    failing the run, since an archived case need not hold every component.

    ``atmos_resolution`` is the grid the atmosphere and land were run on; every
    other realm's grid is derived from it or from the model.

    ``years`` is passed through to gen_timeseries.py as ``--years-spec`` and so
    uses its format, ``first:last:increment``.
    """
    unknown = [stage for stage in stages if stage not in STAGES]
    if unknown:
        raise ValueError(f"Unknown stage(s) {unknown}; choose from {list(STAGES)}")

    case_dir = Path(case_dir)
    outdir = Path(outdir)
    scripts_dir = Path(scripts_dir)
    timeseries_root = outdir / "timeseries"
    cmor_root = outdir / "cmor"

    plan = Plan(timeseries_root=timeseries_root, cmor_root=cmor_root)
    wanted_realms = list(realms) if realms else realms_for(model)

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

        realm_frequencies = _realm_frequencies(model, realm, frequencies)
        if not realm_frequencies:
            plan.skipped.append(
                f"{realm}: none of the requested frequencies are declared"
            )
            continue

        sheet = ice_sheet or DEFAULT_ICE_SHEET if realm == "landIce" else None
        ts_dir = timeseries_root / realm

        realm_resolution = resolution_for(model, realm, atmos_resolution)

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
                        resolution=realm_resolution,
                        experiment=experiment,
                        workers=workers,
                        sheet=sheet,
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
    *, scripts_dir, realm, inputdir, ts_dir, model, frequencies, years, workers, sheet
) -> Step:
    """Return the gen_timeseries.py step for one realm."""
    command = [
        "python",
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
    resolution,
    experiment,
    workers,
    sheet,
) -> Step:
    """Return the cmor_driver.py step for one realm and frequency."""
    command = [
        "python",
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
        "--resolution",
        resolution,
        "--experiment",
        experiment,
        "--workers",
        str(workers),
    ]
    if sheet:
        command += ["--ice-sheet", sheet]
    return Step(
        key=f"cmor-{realm}-{frequency}",
        stage="cmor",
        realm=realm,
        frequency=frequency,
        command=command,
    )


def _validate_step(
    *, scripts_dir, realm, frequency, cmor_root, model, experiment
) -> Step:
    """Return the validate_cmor_output.py step for one realm and frequency."""
    command = [
        "python",
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
    return Step(
        key=f"validate-{realm}-{frequency}",
        stage="validate",
        realm=realm,
        frequency=frequency,
        command=command,
    )
