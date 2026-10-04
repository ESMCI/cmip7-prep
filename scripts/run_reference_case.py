#!/usr/bin/env python3

"""Run the whole chain over one archived case, for every realm, in one command.

    python scripts/run_reference_case.py --case-dir <archive> --outdir <where>

For each realm the model declares, this runs gen_timeseries.py, then
cmor_driver.py and validate_cmor_output.py for each of that realm's
frequencies, and prints one table at the end saying what succeeded, what
failed, and how much was produced.

It is meant to be run on the machine holding the data, interactively -- there
is no batch submission here.  A full run takes a while, so start it under
``tmux`` or ``screen`` if the connection might drop.  Use ``--dry-run`` first to
see exactly which commands would be issued.

Validation is what decides whether a run was good: validate_cmor_output.py is
invoked with --strict, so a missing variable or a CMOR log error fails the run.
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

_LOCAL_PATH = Path(__file__).resolve().parent
sys.path.insert(0, str(_LOCAL_PATH.parent / "src"))

# pylint: disable=wrong-import-position
from cmip7_prep.cv_lookup import validate as validate_cv
from cmip7_prep.reference_run import (
    ATM_RESOLUTIONS,
    STAGES,
    Plan,
    Step,
    build_plan,
)

logger = logging.getLogger("run_reference_case")


def parse_arguments():
    """Parse and return the command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run time series generation, CMORization and validation over one "
            "archived case, for every realm, and report the result."
        )
    )

    # Grouped so --help states plainly which arguments must be given.  All five
    # describe the case itself: a wrong value produces output that looks valid,
    # so none of them may be defaulted.
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "--case-dir",
        required=True,
        help="Archived case root, holding atm/, lnd/, ice/, glc/ and ocn/",
    )
    required.add_argument(
        "--outdir",
        required=True,
        help="Directory to write timeseries/ and cmor/ output into",
    )
    required.add_argument(
        "--model",
        choices=["noresm", "cesm"],
        required=True,
        help=(
            "Model whose include patterns and variable mappings to use. "
            "The wrong one produces plausible but wrong output."
        ),
    )
    required.add_argument(
        "--atmos-resolution",
        choices=list(ATM_RESOLUTIONS),
        required=True,
        help=(
            "Grid the atmosphere and land were run on. Ocean and sea ice are "
            "always on the model's own tripolar grid, and land ice is written "
            "on its native projected grid, so those are derived from this."
        ),
    )
    required.add_argument(
        "--experiment",
        required=True,
        help=(
            "CMIP7 experiment_id of the case, e.g. piControl. Checked against "
            "the controlled vocabulary before any work starts."
        ),
    )

    selection = parser.add_argument_group("selecting what to run")
    selection.add_argument(
        "--realms",
        nargs="+",
        default=None,
        metavar="REALM",
        help="Realms to process (default: every realm the model declares)",
    )
    selection.add_argument(
        "--frequencies",
        nargs="+",
        default=None,
        metavar="FREQ",
        help="Frequencies to process (default: every frequency each realm declares)",
    )
    selection.add_argument(
        "--years",
        default=None,
        help=(
            "Years to process, as gen_timeseries.py's --years-spec: "
            "first:last:increment, e.g. 1526:1527:2 for two years in one file"
        ),
    )
    selection.add_argument(
        "--stages",
        nargs="+",
        choices=list(STAGES),
        default=list(STAGES),
        help="Stages to run (default: all three)",
    )
    selection.add_argument(
        "--ice-sheet",
        choices=["gris", "ais"],
        default=None,
        help="Ice sheet for the landIce realm (default: gris)",
    )

    behaviour = parser.add_argument_group("how to run")
    behaviour.add_argument(
        "--workers",
        type=int,
        default=4,
        help=(
            "Worker processes per stage (default: 4, chosen to be polite on a "
            "shared interactive node rather than to be fast)"
        ),
    )
    behaviour.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would run, and exit",
    )
    behaviour.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue after a failing step instead of stopping",
    )
    behaviour.add_argument(
        "--tables-root",
        default=None,
        help=(
            "cmip7-cmor-tables checkout, used to check --experiment against the "
            "controlled vocabulary (default: the one beside this repo)"
        ),
    )
    behaviour.add_argument("--log-level", default="INFO", help="Default: INFO")
    return parser.parse_args()


def run_step(step: Step, log_dir: Path) -> dict:
    """Run one step, tee its output to a log file, and return the outcome."""
    log_path = log_dir / step.log_name
    logger.info("[%s] %s", step.key, " ".join(step.command))
    started = time.time()
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(" ".join(step.command) + "\n\n")
        log_file.flush()
        completed = subprocess.run(
            step.command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.time() - started
    outcome = {
        "key": step.key,
        "stage": step.stage,
        "realm": step.realm,
        "frequency": step.frequency,
        "returncode": completed.returncode,
        "seconds": elapsed,
        "log": str(log_path),
    }
    if completed.returncode == 0:
        logger.info("[%s] ok in %.1fs", step.key, elapsed)
    else:
        logger.error(
            "[%s] FAILED (exit %d) after %.1fs -- see %s",
            step.key,
            completed.returncode,
            elapsed,
            log_path,
        )
    return outcome


def print_plan(plan: Plan) -> None:
    """Print the commands a run would issue, without running them."""
    for stage in STAGES:
        steps = plan.for_stage(stage)
        if not steps:
            continue
        print(f"\n=== {stage} ({len(steps)} step(s)) ===")
        for step in steps:
            print(f"  {step.key}")
            print(f"    {' '.join(step.command)}")
    if plan.skipped:
        print("\n=== skipped ===")
        for reason in plan.skipped:
            print(f"  {reason}")


def print_report(outcomes: list[dict], plan: Plan) -> None:
    """Print the end-of-run summary table."""
    print("\n" + "=" * 78)
    print("REFERENCE CASE RUN")
    print("=" * 78)

    width = max((len(o["key"]) for o in outcomes), default=10)
    for outcome in outcomes:
        status = (
            "ok" if outcome["returncode"] == 0 else f"FAILED ({outcome['returncode']})"
        )
        print(f"  {outcome['key']:<{width}}  {outcome['seconds']:7.1f}s  {status}")

    failed = [o for o in outcomes if o["returncode"] != 0]
    print("-" * 78)
    print(f"  {len(outcomes) - len(failed)} ok, {len(failed)} failed")

    if plan.skipped:
        print("\n  skipped:")
        for reason in plan.skipped:
            print(f"    {reason}")

    if failed:
        print("\n  logs for the failures:")
        for outcome in failed:
            print(f"    {outcome['key']}: {outcome['log']}")


def main():
    """Build the plan, run it, and report."""
    args = parse_arguments()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    case_dir = Path(args.case_dir)
    if not case_dir.is_dir():
        logger.error("No such case directory: %s", case_dir)
        sys.exit(2)

    # Fail on a bad experiment now rather than after the time series stage: CMOR
    # would reject it at write time, by which point the slow part has run.
    tables_root = Path(args.tables_root or _LOCAL_PATH.parent / "cmip7-cmor-tables")
    try:
        validate_cv(tables_root, "experiment_id", args.experiment)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        logger.error("%s", exc)
        sys.exit(2)

    plan = build_plan(
        case_dir=case_dir,
        outdir=args.outdir,
        model=args.model,
        realms=args.realms,
        frequencies=args.frequencies,
        years=args.years,
        stages=args.stages,
        atmos_resolution=args.atmos_resolution,
        experiment=args.experiment,
        workers=args.workers,
        ice_sheet=args.ice_sheet,
        scripts_dir=_LOCAL_PATH,
    )

    if not plan.steps:
        logger.error("Nothing to do. Skipped: %s", plan.skipped or "(nothing)")
        sys.exit(2)

    logger.info(
        "%d step(s) over realms %s",
        len(plan.steps),
        sorted({step.realm for step in plan.steps}),
    )

    if args.dry_run:
        print_plan(plan)
        return

    outdir = Path(args.outdir)
    log_dir = outdir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    outcomes = []
    for step in plan.steps:
        outcome = run_step(step, log_dir)
        outcomes.append(outcome)
        if outcome["returncode"] != 0 and not args.keep_going:
            logger.error("Stopping at the first failure; pass --keep-going to continue")
            break

    print_report(outcomes, plan)

    failed = any(outcome["returncode"] != 0 for outcome in outcomes)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
