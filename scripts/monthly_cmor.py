#!/usr/bin/env python3
"""
monthly_cmor.py: Combined CMIP7 monthly processing script for ATM and LND realms.

Usage:
  python monthly_cmor.py --realm atm [--tsdir]
  python monthly_cmor.py --realm lnd [--tsdir]

Preserves all comments and error handling from both atm_monthly.py and lnd_monthly.py.
"""
from __future__ import annotations
import argparse
from concurrent.futures import as_completed

import os
from pathlib import Path
import logging
import re
from typing import Optional, Tuple
import sys
from datetime import datetime, UTC
import glob
import numpy as np
import xarray as xr

from cmip7_prep.cmor_utils import bounds_from_centers_1d, roll_for_monotonic_with_bounds
from cmip7_prep.mapping_compat import Mapping
from cmip7_prep.pipeline import realize_regrid_prepare, open_native_for_cmip_vars
from cmip7_prep.cmor_writer import CmorSession

from data_request_api.query import data_request as dr
from data_request_api.content import dump_transformation as dt

from dask.distributed import LocalCluster
from dask.distributed import wait, as_completed
from dask import delayed

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("cmip7_prep.monthly_cmor")
# Regex for date extraction from filenames

_DATE_RE = re.compile(
    r"[\.\-](?P<year>\d{4})"  # year
    r"(?P<sep>-?)"  # optional hyphen
    r"(?P<month>0[1-9]|1[0-2])"  # month 01–12
    r"\.nc(?!\S)"  # literal .nc and then end (or whitespace)
)

#TABLES = "/glade/derecho/scratch/jedwards/cmip7-prep/cmip7-cmor-tables/tables"
TABLES = "/projects/NS9560K/mvertens/cmip7-prep/cmip7-cmor-tables/tables/"
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(
        description="CMIP7 monthly processing for atm/lnd realms"
    )
    parser.add_argument(
        "--cmip-vars",
        nargs="*",
        help="List of CMIP variable names to process directly (bypasses variable search)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=128,
        help="Number of Dask workers (default: 128, set to 1 for serial execution)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing timeseries outputs (default: False)",
    )
    parser.add_argument(
        "--realm",
        choices=[
            "atmos",
            "land",
            "aerosol",
            "atmosChem",
        ],
        required=True,
        help="Realm to process",
    )
    parser.add_argument (
        "--resolution",
        type=str,
        choices=[
            "ne16",
            "ne30"
        ],
        required=True,
        help="input_grid name (required)",
    )
    parser.add_argument(
        "--tsdir",
        type=str,
        help="Time series directory (optional)." 
        "If not specified, will use a preset time series test directory"
    )    
    parser.add_argument( # Move to a wrapper script
        "--caseroot",
        type=str,
        help="Case root directory"
    )
    parser.add_argument( # Move to a wrapper script
        "--cimeroot",
        type=str,
        help="CIME root directory"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode with default paths"
    )
    parser.add_argument(
        "--run-freq",
        type=str,
        default="10y",
        help="Request run frequency (e.g. '10y' or '120m'), default 10y",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Output directory for CMORized files",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="piControl",
        help="Experiment name for data request",
    )

    args = parser.parse_args()

    # Parse run_freq argument
    run_years = 10
    run_months = 120
    run_freq = args.run_freq.strip().lower()
    if run_freq.endswith("y"):
        try:
            run_years = int(run_freq[:-1])
            run_months = run_years * 12
        except Exception:
            logger.error(f"Invalid --run-freq value: {run_freq}")
            sys.exit(1)
    elif run_freq.endswith("m"):
        try:
            run_months = int(run_freq[:-1])
            run_years = run_months // 12
        except Exception:
            logger.error(f"Invalid --run-freq value: {run_freq}")
            sys.exit(1)
    else:
        logger.error(f"Invalid --run-freq value: {run_freq}")
        sys.exit(1)
    args.run_years = run_years
    args.run_months = run_months

    return args


def process_one_var(
    cmip_var,
    mapping,
    inputfile,
    tables_path,
    outdir,
    resolution,
    realm="atmos",
) -> list[tuple[str, str]]:
    """Compute+write one CMIP variable. Returns a list of (varname, 'ok' or error message) tuples."""
    varname = cmip_var.branded_variable_name.name

    # At this point you have a cmip_var (metadata from database query for the target variable)
    #   queried a cmor database from the cloud

    logger.info(f"Starting processing for variable: {varname}")
    results = [(str(varname), "started")]
    try:
        # This is what maps the cesm variable(s) to the cmor variable
        # This is yaml file - cesm_to_cmip7.yaml in this repo
        # the variable name in variables: is the output
        # Jim started with the cmip6 template and then converted it
        # (cmip6 file used for e3sm -> cmip)
        cfg = mapping.get_cfg(varname)
    except Exception as e:
        logger.error(f"Error retrieving config for {varname}: {e}")
        results.append((varname, f"ERROR: {e}"))
        return results

    # These are the dims on the destination
    # (interpolated dims if you WILL do interpolation - have not done it yet)
    dims_list = cfg.get("dims")
    # If dims is a single list (atm/lnd), wrap in a list for uniformity
    if dims_list and len(dims_list) > 0 and isinstance(dims_list[0], str):
        dims_list = [dims_list]

    # Loop over dims - for land and atmos realms there is only one entry 
    for dims in dims_list:
        logger.info(f"Processing {varname} with dims {dims}")
        try:
            open_kwargs = None
            logger.info("Opening native data for variable %s", varname)

            # This is where
            # ds_native is where you read the CESM time series file
            # ds_native is an xarray dataset
            # open_native_for_cmip_vars is in pipeline.py
            ds_native, var = open_native_for_cmip_vars(
                varname,
                inputfile,
                mapping,
                use_cftime=True,
                parallel=False,
                open_kwargs=open_kwargs,
            )
            logger.info("realm is %s", realm)
            # fx - grid definition like topography, fraction
            # ds_native.merge is an xarray command
            logger.info(
                "ds_native keys: %s for var %s with dims %s",
                list(ds_native.keys()),
                varname,
                dims,
            )
            if var is None:
                logger.warning(f"Source variable(s) not found for {varname}")
                results.append((varname, "ERROR: Source variable(s) not found."))
                continue
            # For lnd/atm or any other dims, use existing logic
            logger.debug(
                "Processing %s for dims %s (atm/lnd)", varname, dims
            )
            # Below is where you do the mapping from SE to lat/lon
            ds_cmor = realize_regrid_prepare(
                resolution,
                mapping,
                ds_native,
                varname,
                tables_path=tables_path,
                regrid_kwargs={
                    "dtype": "float32",
                },
                open_kwargs={"decode_timedelta": True},
            )
            logger.info("ds_cmor is not None")

        except AttributeError:
            results.append((varname, f"ERROR cesm input variable not found"))
            continue
        except Exception as e:
            logger.warning(
                "Exception during regridding of %s with dims %s: %r",
                varname,
                dims,
                e,
            )
            results.append((varname, f"ERROR during regridding: {e!r}"))
            continue

        # CMORize
        try:
            log_dir = outdir + "/logs"

            # Initialize CMOR class
            with CmorSession(
                tables_path=tables_path,
                log_dir=log_dir,
                log_name=f"cmor_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_{varname}.log",
                dataset_attrs={"institution_id": "NCAR", "GLOBAL_IS_CMIP7": True},
                outdir=outdir,
            ) as cm:
                cmip7name = cmip_var.attributes["branded_variable_name"]

                logger.info(f"Writing CMOR variable {cmip7name.name}")
                shortname = str(getattr(cmip_var, "physical_parameter").name)
                vdef = type(
                    "VDef",
                    (),
                    {
                        "name": shortname,
                        "table": cfg.get("table", "atmos"),
                        "units": cfg.get("units", ""),
                        "dims": dims,
                        "positive": cfg.get("positive", None),
                        "cell_methods": cfg.get("cell_methods", None),
                        "long_name": cfg.get("long_name", None),
                        "standard_name": cfg.get("standard_name", None),
                        "levels": cfg.get("levels", None),
                        "branded_variable_name": cmip7name,
                    },
                )()
                logger.info(f"Writing variable {varname} with dims {dims} ")

                # Now use CMOR utility to write out netcdf varialbe
                cm.write_variable(ds_cmor, cmip_var, vdef)

            logger.info(f"Finished processing for {varname} with dims {dims}")
            results.append((shortname, "ok"))
        except Exception as e:
            logger.error(
                f"Exception while processing {varname} with dims {dims}: {e!r}"
            )
            results.append((str(varname), f"ERROR: {e!r}"))
    logger.info(f"Completed all processing for variable: {varname}, results {results}")
    return results


process_one_var_delayed = delayed(process_one_var)


def latest_monthly_file(
    directory: Path, *, require_consistent_style: bool = True
) -> Optional[Tuple[Path, int, int]]:
    """
    Find the file in `directory` with the most recent YYYYMM.nc or YYYY-MM.nc date in its name.
    Returns (path, year, month) or None if no matching files are found.
    If `require_consistent_style` is True, raises ValueError if both styles are present.
    """
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    found = []
    seps = set()
    logger.debug(f"Looking for files in {str(directory)}")
    for p in directory.iterdir():
        if not p.is_file():
            continue
        m = _DATE_RE.search(p.name)
        if not m:
            continue
        year = int(m.group("year"))
        month = int(m.group("month"))
        sep = m.group("sep")
        seps.add(sep)
        found.append((year, month, p))
    if not found:
        return None
    if require_consistent_style and len(seps) > 1:
        raise ValueError("Mixed date styles detected (YYYYMM.nc and YYYY-MM.nc).")
    logger.debug(f"Found {len(found)} files in {str(directory)}")
    found.sort(key=lambda t: (t[0], t[1], t[2].name))
    year, month, path = found[-1]
    return path, year, month


def main():
    args = parse_args()
    scratch = os.getenv("SCRATCH")
    OUTDIR = args.outdir
    resolution = args.resolution

    if args.realm == "atmos":
        include_patterns = ["*cam.h0a*"]
        frequency = "mon"
        subdir = "atm"
    elif args.realm == "land":
        include_patterns = ["*clm2.h0*"]
        frequency = "mon"
        subdir = "lnd"

    # Setup input/output directories
    if args.tsdir:
        TSDIR = Path(args.tsdir)
        if not os.path.exists(str(TSDIR)):
            logger.info(f"Time series directory {str(TSDIR)} must exist")
            sys.exit(0)
        timeseries = latest_monthly_file(TSDIR)
        logger.info(f"latest monthly time series file is {timeseries}")
    else:
        # testing path
        if args.realm == "atmos":
            TSDIR = "/datalake/NS9560K/mvertens/test_regridder/atm/timeseries"
        elif args.realm == "land":
            TSDIR = "/datalake/NS9560K/mvertens/test_regridder/lnd/timeseries"

    # Make output directory if it does not exist
    OUTDIR = Path(args.outdir)
    if not os.path.exists(str(OUTDIR)):
        os.makedirs(str(OUTDIR))

    # Load all possible cmip vars for this realm and this experiment - create a data request
    logger.info("Loading data request content %s", args.realm)
    content_dic = dt.get_transformed_content()
    logger.info("Content dic obtained")
    DR = dr.DataRequest.from_separated_inputs(**content_dic)
    cmip_vars = []
    cmip_vars = DR.find_variables(
        skip_if_missing=False,
        operation="all",
        cmip7_frequency=frequency,
        modelling_realm=args.realm,
        experiment=args.experiment,
    )
    
    logger.debug(f"list of all cmip variables for realm {args.realm} and experiment {args.experiment}")
    for var in cmip_vars: 
        logger.debug(f"cmip_var is {var}")

    if args.cmip_vars:
        tmp_cmip_vars = cmip_vars
        cmip_vars = []
        for var in tmp_cmip_vars:
            logger.info(
                "Checking variable %s in %s",
                var.branded_variable_name.name,
                args.cmip_vars,
            )
            if var.branded_variable_name.name in args.cmip_vars:
                logger.info("Adding variable %s", var.branded_variable_name.name)
                if var.branded_variable_name.name not in [
                    v.branded_variable_name.name for v in cmip_vars
                ]:
                    cmip_vars.append(var)

    logger.info(f"CMORIZING {len(cmip_vars)} variables")
    if args.workers == 1:
        client = None
        cluster = None
    else:
        ncpus_env = os.getenv("NCPUS")
        if ncpus_env is not None:
            ml = 1.0 - float(int(ncpus_env) - 1) / 128.0
        else:
            ml = "auto"  # Default memory limit if NCPUS is not set
        cluster = LocalCluster(
            n_workers=min(args.workers, len(cmip_vars)),
            threads_per_worker=1,
            memory_limit=ml,
        )
        client = cluster.get_client()

    # Load requested variables
    if len(cmip_vars) > 0:
        if len(include_patterns) == 1:
            input_path = Path(str(TSDIR) + f"/*{include_patterns[0]}*")
        else:
            input_path = Path(str(TSDIR) + f"/*")

        # Load mapping
        mapping = Mapping.from_packaged_default()

        if args.workers == 1:
            results = [
                item
                for v in cmip_vars
                for item in process_one_var(
                    v,
                    mapping,
                    input_path,
                    TABLES,
                    OUTDIR,
                    resolution,
                    realm=args.realm,
                )
            ]

        else:
            futs = [
                process_one_var_delayed(
                    var,
                    mapping,
                    input_path,
                    TABLES,
                    OUTDIR,
                    resolution,
                    realm=args.realm,
                )
                for var in cmip_vars
            ]
            logger.info(f"launching {len(futs)} futures")
            futures = client.compute(futs)
            wait(
                futures, timeout="1200s"
            )  # optional soft check; won’t raise, just returns done/pending

            # iterate results; if anything stalls you can call dump_all_stacks(client)
            results = []
            for _, result in as_completed(futures, with_results=True):
                try:
                    # Handle result types: list of tuples, tuple, or other
                    if isinstance(result, list):
                        # If it's a list, check if it's a list of tuples
                        if all(isinstance(x, tuple) and len(x) == 2 for x in result):
                            results.extend(result)
                        else:
                            # Not a list of tuples, wrap as unknown
                            results.append((str(result), "unknown"))
                    elif isinstance(result, tuple) and len(result) == 2:
                        results.append(result)
                    else:
                        # Not a tuple/list, wrap as unknown
                        results.append((str(result), "unknown"))
                except Exception as e:
                    logger.error("Task error:", e)
                    raise

        for v, status in set(results):
            logger.info(f"Variable {v} processed with status: {status}")

    else:
        logger.info("No results to process.")
    if client:
        client.close()
    if cluster:
        cluster.close()


if __name__ == "__main__":
    main()
