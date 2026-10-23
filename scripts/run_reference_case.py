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
import datetime
import subprocess
import sys
import time
from pathlib import Path

_LOCAL_PATH = Path(__file__).resolve().parent
sys.path.insert(0, str(_LOCAL_PATH.parent / "src"))

# pylint: disable=wrong-import-position
from cmip7_prep.time_windows import year_windows
from cmip7_prep.validation_exit import RAN_AND_FOUND_PROBLEMS, RAN_AND_PASSED
from cmip7_prep.cv_lookup import validate as validate_cv
from cmip7_prep.grids import MODEL_RESOLUTIONS
from cmip7_prep.reference_run import (
    CMOR_WORKERS,
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
        "--model-res",
        choices=list(MODEL_RESOLUTIONS),
        required=True,
        help=(
            "Model resolution, as named in data/<model>_grids.yaml. It selects "
            "the grid each realm's input is on and the grid label its output "
            "carries, so every realm needs it."
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
        "--ts-workers",
        type=int,
        default=4,
        help=(
            "Worker processes for time series generation (default: 4, chosen "
            "to be polite on a shared interactive node). See --cmor-workers "
            "for the other stage."
        ),
    )
    behaviour.add_argument(
        "--cmor-workers",
        type=int,
        default=CMOR_WORKERS,
        help=(
            f"Worker processes for CMORization (default: {CMOR_WORKERS}). "
            "Counted separately from --workers because a CMOR worker holds a "
            "variable's regridded arrays -- measured at 3.4 to 6.7 GB each on "
            "a two-year ne16 atmosphere run -- so memory, not cores, is "
            "usually what limits how many are useful."
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
        help=(
            "Continue after a failing time series or CMORization step instead "
            "of stopping. Validation failures never stop a run whether this is "
            "given or not: validation reports rather than produces, and under "
            "--strict it fails while any requested variable is missing."
        ),
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


def period_tag(years: str | None) -> str | None:
    """Name the period a run covers, from its --years, or None without one.

    ``"1:50:10"`` gives ``"1-50"``.  Used to keep the things that belong to one
    run -- the step logs and the record of what finished -- apart from another
    run's, while the output tree and the time series stay shared: years 1-50
    and 51-100 are one dataset, but they are two runs.

    Without it, a second period would find every step recorded as complete and
    do nothing at all, and would overwrite the first period's step logs.
    """
    if not years:
        return None
    windows = year_windows(years)
    return f"{windows[0][0]}-{windows[-1][1]}"


def completed_path(outdir: Path, tag: str | None) -> Path:
    """Where a run's record of finished steps lives."""
    if tag is None:
        return outdir / COMPLETED_NAME
    return outdir / f"completed_steps_{tag}.json"


def step_log_dir(outdir: Path, tag: str | None) -> Path:
    """Where a run's per-step logs live.  run.log stays above this, shared."""
    if tag is None:
        return outdir / "logs"
    return outdir / "logs" / tag


def outcome_label(stage: str, returncode: int) -> str:
    """How a finished step's result reads in the summary.

    Validation reporting problems is not a failure of the run -- it is the
    report, and under --strict it says so whenever a requested variable is
    missing.  Calling that FAILED, in the same words as a crash, made twelve
    working reports look like twelve broken steps.
    """
    if returncode == 0:
        return "ok"
    if stage == "validate" and returncode == RAN_AND_FOUND_PROBLEMS:
        return "reported problems"
    return f"FAILED ({returncode})"


def counts_as_done(stage: str, returncode: int) -> bool:
    """Whether a resume can skip this step because it already ran.

    Time series and CMOR steps count only if they succeeded.  Validation counts
    unless it crashed, because reporting problems is its job rather than a
    failure.

    --force skips nothing.
    """
    if returncode == RAN_AND_PASSED:
        return True
    if stage == "validate" and returncode == RAN_AND_FOUND_PROBLEMS:
        return True
    return False


def should_stop(stage: str, returncode: int, keep_going: bool) -> bool:
    """Whether a finished step's result should stop the rest of the run.

    Never for validation.  It reports rather than produces, nothing after it
    depends on it, and under --strict it fails while any requested variable is
    missing -- so stopping for it would end every run at the third step.

    Otherwise a failure stops the run unless --keep-going: a missing time
    series or CMOR file means everything after it is working from input that
    is not there, and finding that out three hours later is worse than
    stopping.
    """
    if returncode == 0 or stage == "validate":
        return False
    return not keep_going


def read_completed(path: Path) -> set[str]:
    """Return the keys of steps a previous run of this period completed."""
    if not path.is_file():
        return set()
    try:
        return set(json.loads(path.read_text(encoding="utf-8")))
    except (ValueError, OSError) as exc:
        logger.warning("Ignoring %s: %s", path, exc)
        return set()


def record_completed(path: Path, done: set[str]) -> None:
    """Write the set of completed step keys, sorted so the file diffs cleanly."""
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
    label = outcome_label(step.stage, completed.returncode)
    if label == "ok":
        logger.info("[%s] ok in %.1fs", step.key, elapsed)
    elif label == "reported problems":
        # A warning, not an error: the report is the step's job, and under
        # --strict it has problems to report whenever a variable is missing.
        logger.warning(
            "[%s] reported problems after %.1fs -- see %s",
            step.key,
            elapsed,
            log_path,
        )
    else:
        logger.error(
            "[%s] FAILED (exit %d) after %.1fs -- see %s",
            step.key,
            completed.returncode,
            elapsed,
            log_path,
        )
    return outcome


# Opened by _start_run_log once the output directory is known.  The plan and
# the summary table are printed rather than logged, so they need routing to the
# file explicitly; everything that goes through a logger is handled by the
# file handler instead.
_RUN_LOG = None
RUN_LOG_NAME = "run.log"


def _start_run_log(log_dir: Path):
    """Send the logger and everything printed to <log_dir>/run.log as well.

    A run is hours long and its own account of itself -- which steps ran, how
    long each took, which failed -- existed only in the terminal, so it was
    lost to a closed window or a dropped connection.  The per-step logs were
    kept, but the timings and the summary were not.

    Appended to, never truncated.  Re-running one realm into a finished run's
    directory is the ordinary way to repair it, and that must not erase the
    timings of the run being repaired.  Each run writes a banner first so the
    file reads as a sequence of runs.
    """
    global _RUN_LOG  # pylint: disable=global-statement
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / RUN_LOG_NAME
    handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logging.getLogger().addHandler(handler)
    _RUN_LOG = open(path, "a", encoding="utf-8")  # pylint: disable=consider-using-with
    started = datetime.datetime.now().isoformat(timespec="seconds")
    _RUN_LOG.write(f"\n{'=' * 78}\nRUN STARTED {started}\n{'=' * 78}\n")
    _RUN_LOG.flush()
    return path


def say(text: str = "") -> None:
    """Print to the terminal, and to the run log if one is open."""
    print(text)
    if _RUN_LOG is not None:
        _RUN_LOG.write(text + "\n")
        _RUN_LOG.flush()


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
            say(f"\n=== {realm} ({count} step(s)) ===")
        say(f"  {step.key}")
        say(f"    {' '.join(step.command)}")
    say(f"\n{len(plan.steps)} step(s) in total")
    if plan.skipped:
        say("\n=== skipped ===")
        for reason in plan.skipped:
            say(f"  {reason}")


def print_report(outcomes: list[dict], plan: Plan) -> None:
    """Print the end-of-run summary table."""
    say("\n" + "=" * 78)
    say("REFERENCE CASE RUN")
    say("=" * 78)

    width = max((len(o["key"]) for o in outcomes), default=10)
    for outcome in outcomes:
        status = outcome_label(outcome["stage"], outcome["returncode"])
        say(f"  {outcome['key']:<{width}}  {outcome['seconds']:7.1f}s  {status}")

    # Counted the way they are labelled: a validation report is not a failure.
    broken = [
        o
        for o in outcomes
        if outcome_label(o["stage"], o["returncode"]).startswith("FAILED")
    ]
    reported = [
        o
        for o in outcomes
        if outcome_label(o["stage"], o["returncode"]) == "reported problems"
    ]
    say("-" * 78)
    tally = f"  {len(outcomes) - len(broken) - len(reported)} ok"
    if reported:
        tally += f", {len(reported)} reported problems"
    tally += f", {len(broken)} failed"
    total = sum(o["seconds"] for o in outcomes)
    say(f"{tally}, {total / 60:.1f} min")

    # Where the time went, which is the question asked after every long run.
    by_stage: dict[str, float] = {}
    for outcome in outcomes:
        by_stage[outcome["stage"]] = (
            by_stage.get(outcome["stage"], 0.0) + outcome["seconds"]
        )
    if len(by_stage) > 1:
        spent = ", ".join(
            f"{stage} {seconds / 60:.1f}"
            for stage, seconds in sorted(by_stage.items(), key=lambda kv: -kv[1])
        )
        say(f"  minutes by stage: {spent}")

    if plan.skipped:
        say("\n  skipped:")
        for reason in plan.skipped:
            say(f"    {reason}")

    if broken:
        say("\n  logs for the failures:")
        for outcome in broken:
            say(f"    {outcome['key']}: {outcome['log']}")
    if reported:
        say("\n  validation reports:")
        for outcome in reported:
            say(f"    {outcome['key']}: {outcome['log']}")


def main():
    """Build the plan, run it, and report."""
    args = parse_arguments()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    run_log = _start_run_log(Path(args.outdir) / "logs")

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
            model_res=args.model_res,
            experiment=args.experiment,
            ts_workers=args.ts_workers,
            cmor_workers=args.cmor_workers,
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

    if args.dry_run:
        print_plan(plan)
        return

    outdir = Path(args.outdir)
    # Per-period, so finishing years 51-100 cannot overwrite the step logs or
    # the progress record of years 1-50.  run.log stays shared and appended to,
    # so one file still tells the whole story.
    tag = period_tag(args.years)
    log_dir = step_log_dir(outdir, tag)
    log_dir.mkdir(parents=True, exist_ok=True)
    completed_file = completed_path(outdir, tag)
    completed = set() if args.force else read_completed(completed_file)

    # One block saying what this run will do and where it will put things,
    # rather than four lines each saying part of it.
    to_run = [step for step in plan.steps if step.key not in completed]
    done_already = len(plan.steps) - len(to_run)

    # Said as sentences rather than a list of fields, because the first thing
    # anyone wants from a run that takes hours is to know it is about to do
    # what they meant.
    logger.info(
        "Running %s for %s of %s, over %s.",
        " and ".join(args.stages),
        f"years {tag}" if tag else "every year present",
        args.experiment,
        ", ".join(sorted({step.realm for step in plan.steps})),
    )
    # Named, not just counted: "12 steps" says nothing about which realm or
    # frequency is about to take an hour.  One line per stage keeps it short.
    for stage in args.stages:
        in_stage = [step for step in to_run if step.stage == stage]
        if in_stage:
            logger.info(
                "  %d %s step(s): %s",
                len(in_stage),
                stage,
                ", ".join(f"{s.realm} {s.frequency}" for s in in_stage),
            )
    if done_already:
        logger.info(
            "  %d step(s) were done by an earlier run and will be skipped. Add "
            "--force to do them again.",
            done_already,
        )

    logger.info("Each step's output goes to %s.", log_dir)
    logger.info("A record of every run of this period goes to %s.", run_log)

    outcomes = []
    for step in plan.steps:
        if step.key in completed:
            logger.info("[%s] already done, skipping", step.key)
            continue
        outcome = run_step(step, log_dir)
        outcomes.append(outcome)

        if counts_as_done(step.stage, outcome["returncode"]):
            completed.add(step.key)
            # Written after every step, so an interrupted run still knows what
            # it finished.
            record_completed(completed_file, completed)

        # Decided separately from recording.
        if should_stop(step.stage, outcome["returncode"], args.keep_going):
            logger.error(
                "Stopping here; pass --keep-going to carry on past a failed %s step",
                step.stage,
            )
            break

    if not outcomes:
        logger.info("Nothing left to do: every planned step was already completed")
        return
    print_report(outcomes, plan)

    # A validation report is not a failure of the run, so it does not decide
    # the exit code either -- otherwise every run exits non-zero for ever,
    # which makes the code useless for telling whether anything is wrong.
    broken = [
        outcome
        for outcome in outcomes
        if outcome_label(outcome["stage"], outcome["returncode"]).startswith("FAILED")
    ]
    sys.exit(1 if broken else 0)


if __name__ == "__main__":
    main()
