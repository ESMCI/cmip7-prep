#!/usr/bin/env python3

"""
script: generate time series for all input files in a direcory
"""

# ++++++++++++++++++++++++++++++
# Import python modules
# ++++++++++++++++++++++++++++++

import os
import logging
import sys
import glob
import argparse
import logging
from concurrent.futures import ProcessPoolExecutor

# Determine local directory path:
_LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))

from pathlib import Path

# Time series generation imports
import xarray as xr
from gents.hfcollection import HFCollection
from gents.timeseries import TSCollection

from cmip7_prep.include_patterns import all_include_patterns, load_include_patterns

MODELS = ["cesm", "noresm"]

# Realms come from the include-pattern tables rather than a list kept here, so
# a realm added to a table is immediately runnable.
DECLARED_REALMS = sorted(
    {realm for model in MODELS for realm in load_include_patterns(model)}
)

# ++++++++++++++++++++++++++++++
# Unusable history file detection
# ++++++++++++++++++++++++++++++


def time_record_count(path):
    """
    Returns the number of time records in a history file: 0 if it holds none,
    None if it has no time dimension, or -1 if it will not open.

    No decoding: only the length of the time dimension is needed, and decoding
    a broken time coordinate would fail on the very files this is looking for.
    engine="netcdf4": naming it skips xarray's backend search, which imports
    every installed plugin.
    """
    try:
        with xr.open_dataset(
            path, engine="netcdf4", decode_times=False, decode_cf=False
        ) as ds:
            return ds.sizes.get("time")
    except (OSError, ValueError):
        return -1


def _time_record_counts(paths, workers):
    """Return each path's time record count, in order.

    A pool is only started when there is more than one worker and more than one
    file to look at: asking for one worker should not cost a pool, and some
    environments refuse to create one at all.
    """
    if workers > 1 and len(paths) > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(paths))) as executor:
            return list(executor.map(time_record_count, paths))
    return [time_record_count(path) for path in paths]


def _overwrite_or_resume(ts_collection, overwrite, logger):
    """Return the collection set either to overwrite output or to resume.

    Asked to overwrite, GenTS rewrites every time series.  Otherwise the
    collection is narrowed to the timesteps no output covers yet, so a rerun
    after an interrupted run continues instead of repeating work.
    """
    if overwrite:
        logger.info("Overwriting any existing time series")
        return ts_collection.apply_overwrite("*")
    resumed = ts_collection.skip_existing()
    logger.info(
        "Skipping time series that already exist; pass --overwrite_timeseries "
        "to rewrite them"
    )
    return resumed


def _execute_or_report_complete(ts_collection, logger):
    """Generate the time series, or report that there is nothing left to do.

    Narrowing a finished run to its missing timesteps leaves nothing, and GenTS
    treats an empty collection as an error.  Having already produced everything
    asked for is a success, so it is reported as one.
    """
    if len(ts_collection) == 0:
        logger.info("Every requested time series already exists; nothing to generate")
        return
    ts_collection.execute()


def find_unusable_files(inputdir, include_patterns, workers, logger):
    """
    Returns the history files GenTS cannot derive time bounds from: those with
    zero time records, and those that will not open.

    GenTS takes a minimum over each file's time coordinate to order the files
    into a series, so a file with no records raises a zero-size reduction in
    gents.meta.  Files like that are written when a run segment opens a history
    stream but ends before any sample reaches it.  They hold no data, so they
    are dropped here instead of being left for GenTS to warn about one logged
    traceback at a time.
    """
    candidates = sorted(
        {
            path
            for pattern in include_patterns
            for path in glob.glob(os.path.join(inputdir, pattern))
        }
    )
    if not candidates:
        return []

    unusable = []
    for path, count in zip(candidates, _time_record_counts(candidates, workers)):
        if count == 0:
            logger.warning("Skipping %s: zero time records", path)
            unusable.append(path)
        elif count == -1:
            logger.warning("Skipping %s: cannot be opened", path)
            unusable.append(path)

    if unusable:
        logger.warning(
            "Excluding %d of %d history file(s) from time series generation",
            len(unusable),
            len(candidates),
        )
    return unusable


# ++++++++++++++++++++++++++++++
# Input argument parser function
# ++++++++++++++++++++++++++++++


def parse_arguments():
    """
    Parses command-line input arguments using the argparse
    python module and outputs the final argument object.
    """

    parser = argparse.ArgumentParser(
        description="Utility to create time series for all time slice files in a directory"
    )

    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "--inputdir",
        type=str,
        required=True,
        help="Full pathname of the directory containing the input history files",
    )
    required.add_argument(
        "--model",
        choices=MODELS,
        required=True,
        help=(
            "Model whose include patterns to use. The wrong one selects the "
            "wrong history streams, so there is no default."
        ),
    )
    required.add_argument(
        "--realm",
        choices=DECLARED_REALMS,
        required=True,
        help="Realm to process; sets the include patterns for the time series",
    )

    selection = parser.add_argument_group("selecting what to process")
    selection.add_argument(
        "--frequency",
        nargs="+",
        default=None,
        metavar="FREQ",
        help=(
            "Only generate time series for these frequencies, e.g. '--frequency "
            "6hr' or '--frequency mon day'. Frequencies are those defined for the "
            "realm in <model>_include_patterns.yaml. "
            "(Default: every frequency the realm defines.)"
        ),
    )
    selection.add_argument(
        "--sampling",
        choices=["tavg", "tpt"],
        default=None,
        help=(
            "Restrict to time-averaged ('tavg') or instantaneous ('tpt') history "
            "files. Default: collect both, since which is needed depends on the "
            "CMIP7 variable being produced later."
        ),
    )
    selection.add_argument(
        "--varlist",
        type=str,
        default=None,
        help="Comma separated list of variables to process (default: all of them)",
    )
    selection.add_argument(
        "--ice-sheet",
        choices=["gris", "ais"],
        default=None,
        help=(
            "Ice sheet for the landIce realm: 'gris' (Greenland) or 'ais' "
            "(Antarctica). Required when --realm landIce; ignored otherwise."
        ),
    )
    selection.add_argument(
        "--years-spec",
        help="colon separated specification of years to process \n"
        " in format of year-first,year-last,year-increments \n "
        " where year-increments specifies how many years to user for each time series file \n"
        " (default: all files in inputdir are placed in one time series file)",
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "--outputdir",
        type=str,
        help="Full path to directory where output time series data will be placed "
        "(default: inputdir/../time_series)",
    )
    output.add_argument(
        "--overwrite_timeseries",
        action="store_true",
        help=(
            "Rewrite time series that already exist. Without this, existing "
            "output is kept and only the missing timesteps are produced, so "
            "an interrupted run can simply be repeated."
        ),
    )

    behaviour = parser.add_argument_group("how to run")
    behaviour.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Number of workers (default: 32)",
    )
    behaviour.add_argument(
        "--debug", action="store_true", help="Turn on debug output (False by default)."
    )

    return parser.parse_args()


# ++++++++++++++++++++++++++++++
# main time series script
# ++++++++++++++++++++++++++++++


def main():

    # Parse command-line arguments
    args = parse_arguments()

    # Set up logging
    if args.debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
        )
    logger = logging.getLogger("gen_timseries")

    # For each file in list of files - regrid data
    debug = args.debug

    # Determine include patterns.  Patterns may contain a '{ice_sheet}'
    # placeholder (landIce), filled in from --ice-sheet at run time.
    try:
        patterns = all_include_patterns(
            args.model, args.realm, args.ice_sheet, args.sampling, args.frequency
        )
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    include_patterns = [f"*{pattern}*" for pattern in patterns]
    if args.frequency:
        logger.info("Restricting to frequencies: %s", ", ".join(args.frequency))

    # Determine input directory
    inputdir = Path(args.inputdir)

    # Determine output directories
    if args.outputdir:
        outputdir = Path(args.outputdir)
    else:
        outputdir = inputdir / ".." / "time_series"

    # Determine parallelization
    workers = args.workers
    logger.info(f"Number of workers is {workers}")

    # Create time series by default
    logger.info(f"Timeseries generation starting for files in {inputdir}...")
    logger.info(f"  output will be placed in {outputdir}...")

    # Determine number of files used in time series creation
    cnt = 0
    filtered = []
    all_patterns = list(include_patterns)
    for include_pattern in include_patterns:
        num = len(glob.glob(os.path.join(inputdir, include_pattern)))
        if num == 0:
            logger.info(
                "No files match %s; this stream was not written, skipping it",
                include_pattern,
            )
        else:
            cnt += num
            logger.info("Processing %d file(s) matching %s", num, include_pattern)
            filtered.append(include_pattern)
    include_patterns = filtered
    if cnt == 0:
        # Being asked to process a realm and finding nothing is a failure, not
        # a no-op: the caller named this realm, so an empty result means the
        # input directory, the realm or the frequencies were wrong.
        logger.error(
            "No input files in %s for realm %s with patterns %s",
            inputdir,
            args.realm,
            all_patterns,
        )
        sys.exit(1)
    logger.info(f"include patterns are {include_patterns}")

    varlist = (
        [variable.strip() for variable in args.varlist.split(",") if variable.strip()]
        if args.varlist
        else None
    )

    # Drop files GenTS cannot read time bounds from before they reach it
    unusable_files = find_unusable_files(inputdir, include_patterns, workers, logger)

    # Determine how time series will be created
    if not args.years_spec:

        # Create base HFCollection
        logger.info("Starting hf_collection")
        hf_collection = HFCollection(inputdir, num_processes=workers)
        hf_collection = hf_collection.include(include_patterns)
        if unusable_files:
            hf_collection = hf_collection.exclude(unusable_files)
        logger.info("Finished hf_collection")

        # Create base TSCollection
        logger.info("Starting ts_collection")
        ts_collection = TSCollection(hf_collection, outputdir, num_processes=workers)
        if varlist:
            ts_collection = ts_collection.include("*", var_glob=varlist)
        # Empty here means nothing matched at all, which is a mistake worth
        # stopping for.  Empty only after narrowing below means the work was
        # already done, which is not.
        if len(ts_collection) == 0:
            raise RuntimeError(
                "No matching variables/files found for time series generation"
            )
        ts_collection = _overwrite_or_resume(
            ts_collection, args.overwrite_timeseries, logger
        )
        _execute_or_report_complete(ts_collection, logger)
        logger.info("Finished ts_collection")

    else:

        years = args.years_spec.split(":")
        year_first = int(years[0])
        year_last = int(years[1])
        nyears = int(years[2])
        logger.info("First year to use is %s", year_first)
        logger.info("Last year to use is %s", year_last)
        logger.info("Year increment for time series generation is %s", nyears)

        hf_collection = HFCollection(inputdir, num_processes=workers)
        if unusable_files:
            hf_collection = hf_collection.exclude(unusable_files)
        for include_pattern in include_patterns:
            logger.info("Processing files with pattern: %s", include_pattern)

            for year in range(year_first, year_last + 1, nyears):
                logger.info(f"Processing from year {year} to year {year+nyears-1}")
                hfp_collection = hf_collection.include([include_pattern])
                hfp_collection = hfp_collection.include_years(year, year + nyears - 1)

                logger.info(f"files to process for year {year} are")
                for item in list(hfp_collection):
                    logger.info(f"{item}")

                # Reads metadata from all files matching this pattern
                # Gets variable names, dimensions, time information, etc.
                hfp_collection.pull_metadata()

                # Set up the time series generation for this pattern's files
                logger.info("Calling ts_collection")
                ts_collection = TSCollection(
                    hfp_collection, outputdir, ts_orders=None, num_processes=workers
                )
                logger.info("Finished ts_collection")

                # Perform the time series generation for this pattern
                if varlist:
                    ts_collection = ts_collection.include("*", var_glob=varlist)
                if len(ts_collection) == 0:
                    raise RuntimeError(
                        "No matching variables/files found for time series generation"
                    )

                ts_collection = _overwrite_or_resume(
                    ts_collection, args.overwrite_timeseries, logger
                )
                if len(ts_collection):
                    logger.info(
                        "Variables scheduled: %s",
                        [order["primary_var"] for order in ts_collection],
                    )

                _execute_or_report_complete(ts_collection, logger)
                logger.info("Timeseries processing complete")


if __name__ == "__main__":
    main()
