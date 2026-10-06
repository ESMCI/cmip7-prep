#!/usr/bin/env python3

"""Run the whole chain over one archived case, for every realm, in one command.

    python scripts/run_reference_case.py --case-dir <archive> --outdir <where>

For each realm the model declares, this runs gen_timeseries.py, then
cmor_driver.py and validate_cmor_output.py for each of that realm's
frequencies, and prints one table at the end saying what succeeded, what
failed, and how much was produced.

Steps that succeed are recorded in the output directory, so repeating the
command after an interruption continues rather than starting over; --force
runs everything again.

It is meant to be run on the machine holding the data, interactively -- there
is no batch submission here.  A full run takes a while, so start it under
``tmux`` or ``screen`` if the connection might drop.  Use ``--dry-run`` first to
see exactly which commands would be issued.

Validation is what decides whether a run was good: validate_cmor_output.py is
invoked with --strict, so a missing variable or a CMOR log error fails the run.
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

_LOCAL_PATH = Path(__file__).resolve().parent
sys.path.insert(0, str(_LOCAL_PATH.parent / "src"))

# pylint: disable=wrong-import-position
from cmip7_prep.cv_lookup import validate as validate_cv
from cmip7_prep.grids import ATM_RESOLUTIONS
from cmip7_prep.reference_run import STAGES, Plan, Step, build_plan

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
        "--experiment",
        required=True,
        help=(
            "CMIP7 experiment_id of the case, e.g. piControl. Checked against "
            "the controlled vocabulary before any work starts."
        ),
    )

    selection = parser.add_argument_group("selecting what to run")
    selection.add_argument(
        "--atmos-res",
        choices=list(ATM_RESOLUTIONS),
        default=None,
        help=(
            "Grid the atmosphere and land were run on. Required unless the "
            "realms being processed are all ocean, sea ice or land ice, which "
            "derive their own grid."
        ),
    )
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
        "--variant-label",
        default=None,
        metavar="rXiYpZfW",
        help=(
            "Ensemble member to label the output with, e.g. r1i1p1f1 "
            "(cmor_driver.py's --realization-initialization-physics-forcing). "
            "Needed when producing output for submission rather than just "
            "exercising the chain, since it distinguishes members."
        ),
    )
    selection.add_argument(
        "--ice-sheet",
        choices=["gris", "ais"],
        default=None,
        help="Ice sheet for the landIce realm (default: gris)",
    )

    reporting = parser.add_argument_group("what to report")
    reporting.add_argument(
        "--plots",
        action="store_true",
        help=(
            "Have validation plot each variable: a mean time series, and a "
            "time-mean map where the data are on a plottable grid"
        ),
    )
    reporting.add_argument(
        "--html",
        action="store_true",
        help=(
            "Have validation rebuild its static HTML index, so the reports "
            "and plots for every realm can be browsed in one place"
        ),
    )
    reporting.add_argument(
        "--max-plots",
        type=int,
        default=None,
        metavar="N",
        help="Plot at most N variables per plot type (default: no limit)",
    )

    behaviour = parser.add_argument_group("how to run")
    behaviour.add_argument(
        "--workers",
        type=int,
        default=4,
        help=(
            "Worker processes for time series generation (default: 4, chosen "
            "to be polite on a shared interactive node). CMORization always "
            "runs with one worker."
        ),
    )
    behaviour.add_argument(
        "--force",
        action="store_true",
        help=(
            "Run every step again, including those a previous run in this "
            "output directory already completed, and rewrite time series "
            "that already exist"
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
            "cmip7-cmor-tables checkout. Used to check --experiment against "
            "the controlled vocabulary before starting, and passed on to "
            "CMORization (default: the one beside this repo)"
        ),
    )
    behaviour.add_argument("--log-level", default="INFO", help="Default: INFO")
    return parser.parse_args()


# Steps that finished are recorded here so a repeated run continues instead of
# starting over.  One file per output directory, next to the logs.
COMPLETED_NAME = "completed_steps.json"


def read_completed(outdir: Path) -> set[str]:
    """Return the keys of steps a previous run completed in this directory."""
    path = outdir / COMPLETED_NAME
    if not path.is_file():
        return set()
    try:
        return set(json.loads(path.read_text(encoding="utf-8")))
    except (ValueError, OSError) as exc:
        logger.warning("Ignoring %s: %s", path, exc)
        return set()


def record_completed(outdir: Path, done: set[str]) -> None:
    """Write the set of completed step keys, sorted so the file diffs cleanly."""
    path = outdir / COMPLETED_NAME
    try:
        path.write_text(json.dumps(sorted(done), indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not record progress in %s: %s", path, exc)


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
    """Print the commands a run would issue, in the order it would issue them.

    Grouped by realm, because that is how a run proceeds: one realm's time
    series, CMORization and validation complete before the next realm starts.
    """
    realm = None
    for step in plan.steps:
        if step.realm != realm:
            realm = step.realm
            count = sum(1 for s in plan.steps if s.realm == realm)
            print(f"\n=== {realm} ({count} step(s)) ===")
        print(f"  {step.key}")
        print(f"    {' '.join(step.command)}")
    print(f"\n{len(plan.steps)} step(s) in total")
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

    try:
        plan = build_plan(
            case_dir=case_dir,
            outdir=args.outdir,
            model=args.model,
            realms=args.realms,
            frequencies=args.frequencies,
            years=args.years,
            stages=args.stages,
            atmos_res=args.atmos_res,
            experiment=args.experiment,
            workers=args.workers,
            ice_sheet=args.ice_sheet,
            overwrite_timeseries=args.force,
            plots=args.plots,
            html=args.html,
            max_plots=args.max_plots,
            variant_label=args.variant_label,
            tables_root=args.tables_root,
            scripts_dir=_LOCAL_PATH,
        )
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(2)

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

    completed = set() if args.force else read_completed(outdir)
    if completed:
        logger.info(
            "%d step(s) already completed in %s; pass --force to run them again",
            len(completed),
            outdir,
        )

    outcomes = []
    for step in plan.steps:
        if step.key in completed:
            logger.info("[%s] already done, skipping", step.key)
            continue
        outcome = run_step(step, log_dir)
        outcomes.append(outcome)
        if outcome["returncode"] == 0:
            completed.add(step.key)
            # Written after every step, so an interrupted run still knows what
            # it finished.
            record_completed(outdir, completed)
        elif not args.keep_going:
            logger.error("Stopping at the first failure; pass --keep-going to continue")
            break

    if not outcomes:
        logger.info("Nothing left to do: every planned step was already completed")
        return
    print_report(outcomes, plan)

    failed = any(outcome["returncode"] != 0 for outcome in outcomes)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
