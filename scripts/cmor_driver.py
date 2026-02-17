#!/usr/bin/env python3
"""
cmor_driver.py: Combined CMIP7 monthly processing script for ATM and LND realms.

Usage:
  python cmor_driver.py --realm atmos --tsdir
  python cmor_driver.py --realm land --tsdir

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
from cmor import set_cur_dataset_attribute

from cmip7_prep.cmor_utils import bounds_from_centers_1d, roll_for_monotonic_with_bounds
from cmip7_prep.mapping_compat import Mapping
from cmip7_prep.pipeline import realize_regrid_prepare, open_native_for_cmip_vars
from cmip7_prep.cmor_writer import CmorSession
from cmip7_prep.mom6_static import ocean_fx_fields

from data_request_api.query import data_request as dr
from data_request_api.content import dump_transformation as dt

from dask.distributed import LocalCluster
from dask.distributed import wait, as_completed
from dask import delayed

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

logger = logging.getLogger("cmip7_prep.cmor_driver")

# Regex for date extraction from filenames
_DATE_RE = re.compile(
    r"[\.\-](?P<year>\d{4})"  # year
    r"(?P<sep>-?)"  # optional hyphen
    r"(?P<month>0[1-9]|1[0-2])"  # month 01–12
    r"\.nc(?!\S)"  # literal .nc and then end (or whitespace)
)

# Path for cmor tables
TABLES_cesm = "/glade/derecho/scratch/jedwards/cmip7-prep/cmip7-cmor-tables/"
TABLES_noresm = "/nird/home/mvertens/packages/cmip7-prep/cmip7-cmor-tables/"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

INCLUDE_PATTERN_MAP = {
    "cesm": {
        "atmos": {
            "mon": ["cam.h0"],
            "day": ["cam.h1"],
        },
        "land": {
            "mon": ["clm2.h0a"],
        },
        "ocean": {
            "mon": ["mom6.h.z", "mom6.h.native."],
            "day": ["mom6.h.sfc"],
        },
    },
    "noresm": {
        "atmos": {
            "mon": ["cam.h0"],
            "day": ["cam.h1"],
        },
        "land": {
            "mon": ["clm2.h0a"],
        },
    },
}


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
            "aerosol",
            "atmosChem",
            "land",
            "ocean",
            "ocnBgchem",
            "seaIce",
            "landIce",
        ],
        required=True,
        help="Realm to process",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        choices=[
            "ne16",
            "ne30",
            "tx2_3v2",
        ],
        required=True,
        help="input_grid name (required)",
    )
    parser.add_argument(
        "--ocn-grid-file",
        type=str,
        default="/glade/campaign/cesm/cesmdata/inputdata/ocn/mom/tx2_3v2/ocean_hgrid_221123.nc",
        help="Path to ocean grid description file for CESM/MOM (optional)",
    )
    parser.add_argument(
        "--ocn-static-file",
        type=str,
        default=None,
        help="Path to static file for CESM/MOM variables (optional)",
    )
    parser.add_argument(
        "--tsdir",
        type=str,
        help="Time series directory (optional)."
        "If not specified, will use a preset time series test directory",
    )
    parser.add_argument(  # Move to a wrapper script
        "--caseroot", type=str, help="Case root directory"
    )
    parser.add_argument(  # Move to a wrapper script
        "--cimeroot", type=str, help="CIME root directory"
    )
    parser.add_argument(
        "--test", action="store_true", help="Run in test mode with default paths"
    )
    parser.add_argument(
        "--frequency",
        type=str,
        default="mon",
        choices=["mon", "day", "6hr"],
        help="Frequency of data to be translated (mon, day, 6hr,)",
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
    parser.add_argument(
        "--model",
        choices=["cesm", "noresm"],
        required=True,
        help="Model to use",
    )

    args = parser.parse_args()

    return args


def process_one_var(
    cmip_var,
    mapping,
    inputfile,
    tables_root,
    outdir,
    resolution,
    model,
    realm="atmos",
    frequency="mon",
    ocn_fx_fields=None,
) -> list[tuple[str, str]]:
    """Compute+write one CMIP variable. Returns a list of (varname, 'ok' or error message) tuples."""
    varname = cmip_var.branded_variable_name.name

    # At this point you have a cmip_var (metadata from database query for the target variable)
    # queried a cmor database from the cloud
    logger.info(f"Starting processing for variable: {varname}")
    results = [(str(varname), "started")]

    try:
        # This is what maps the CESM/NorESM history variable(s) to the cmor variable
        # This is obtained from reading cesm_to_cmip7.yaml
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

    # Loop over dims - in most cases there will only be one entry -
    # but for some variables (like ocean sos) there needs to be an
    # entry both the native and the interpolated grid - so there will
    # be two entries - hence the loop below

    for dims in dims_list:
        logger.info(f"Processing {varname} with dims {dims}")

        # ---------------------------------------------
        # Read in time series and regrid if necessary
        # ---------------------------------------------
        try:
            open_kwargs = None
            if realm == "ocean":
                open_kwargs = {"decode_timedelta": False}
            logger.info("Opening native data for variable %s", varname)

            # ds_native is an xarray dataset for the CESM/NorESM time series files
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

            if model == "cesm":
                # Append ocn_fx_fields to ds_native if available
                # fx - grid definition like topography, fraction
                if realm == "ocean" and ocn_fx_fields is not None:
                    logger.info("adding ocn_fx_fields to ds_native")
                    ds_native = ds_native.merge(ocn_fx_fields)

            # Output ds_native keys
            logger.info(
                "ds_native keys: %s for var %s with dims %s",
                list(ds_native.keys()),
                varname,
                dims,
            )
            # TODO: why does this not abort the program?
            if var is None:
                logger.warning(f"Source variable(s) not found for {varname}")
                results.append((varname, "ERROR: Source variable(s) not found."))
                continue

            # For ocean realm: distinguish native vs regridded by dims
            if model == "cesm" and "latitude" in dims and "longitude" in dims:
                # output ocn on the native grid
                logger.info(f"Preparing native grid output for mom6 variable {varname}")
                ds_cmor = ds_native
                results.append((str(varname), "analyzed native mom6 grid"))
            else:
                # For lnd/atm or any other dims, use existing logic
                logger.debug(
                    "Processing %s for dims %s (atm/lnd or other)", varname, dims
                )
                # Below is where you do the mapping from SE to lat/lon
                mom6_grid = None

                ds_cmor = realize_regrid_prepare(
                    resolution,
                    model,
                    mapping,
                    ds_native,
                    varname,
                    tables_path=tables_root / "tables",
                    mom6_grid=mom6_grid,
                    regrid_kwargs={
                        "dtype": "float32",
                    },
                    open_kwargs={"decode_timedelta": True},
                )
                logger.info("ds_cmor is not None")

                # Attach ocn_fx_fields to regridded output for writing
                if ocn_fx_fields is not None:
                    ds_cmor = ds_cmor.merge(ocn_fx_fields)

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

        # ---------------------------------------------
        # CMORize
        # ---------------------------------------------
        try:
            log_dir = outdir / "logs"

            # TODO: add NorESM institution_id below
            # Initialize CMOR class
            with CmorSession(
                tables_root=tables_root,
                log_dir=log_dir,
                log_name=f"cmor_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_{varname}.log",
                dataset_attrs={"institution_id": "NCAR", "GLOBAL_IS_CMIP7": True},
                outdir=outdir,
            ) as cm:
                cmip7name = cmip_var.attributes["branded_variable_name"]
                set_cur_dataset_attribute("frequency", frequency)

                logger.info(
                    f"Writing CMOR variable {cmip7name.name} with frequency {frequency}"
                )
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

                # Now use CMOR utility to write out netcdf varialbe
                logger.info(f"Writing CMOR variable {varname} with dims {dims} ")
                cm.write_variable(ds_cmor, cmip_var, vdef)

            logger.info(f"Finished processing for {varname} with dims {dims}")
            results.append((str(cmip7name), "ok"))
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


def get_include_patterns(model: str, realm: str, frequency: str) -> list[str]:
    try:
        logger.info(
            "Looking for pattern: %s", INCLUDE_PATTERN_MAP[model][realm][frequency]
        )
        return INCLUDE_PATTERN_MAP[model][realm][frequency]
    except KeyError:
        raise ValueError(
            f"No include_patterns defined for model={model}, "
            f"realm={realm}, frequency={frequency}"
        )


def main():
    args = parse_args()
    scratch = os.getenv("SCRATCH")
    OUTDIR = args.outdir
    resolution = args.resolution
    model = args.model
    frequency = args.frequency
    realm = args.realm

    mom6_grid = None
    ocn_grid = None
    ocn_fx_fields = None

    # Determine include patterns and frequency
    include_patterns = get_include_patterns(model, realm, frequency)

    # Setup input directory for noresm
    if model == "noresm":
        if args.tsdir:
            TSDIR = Path(args.tsdir)
            if not TSDIR.exists():
                logger.info(f"Time series directory {str(TSDIR)} must exist")
                sys.exit(0)
            timeseries = latest_monthly_file(TSDIR)
            logger.info(f"latest monthly time series file is {timeseries}")
        else:
            if realm == "atmos":
                TSDIR = "/datalake/NS9560K/mvertens/test_regridder/atm/timeseries"
            elif realm == "land":
                TSDIR = "/datalake/NS9560K/mvertens/test_regridder/lnd/timeseries"

    # Setup input directory for cesm
    if model == "cesm":
        if realm == "atmos":
            subdir = "atm"
        elif realm == "land":
            subdir = "lnd"
        elif realm == "ocean":
            subdir = "ocn"
            if args.ocn_grid_file:
                ocn_grid = args.ocn_grid_file
            if args.ocn_static_file:
                ocn_fx_fields = ocean_fx_fields(args.ocn_static_file)
                logger.info(
                    f"Loaded ocean fx fields from {args.ocn_static_file}: {list(ocn_fx_fields.keys())}"
                )
        if args.caseroot and args.cimeroot:
            caseroot = args.caseroot
            cimeroot = args.cimeroot
            sys.path.append(cimeroot)
            _LIBDIR = os.path.join(cimeroot, "CIME", "Tools")
            sys.path.append(_LIBDIR)
            try:
                from CIME.case import Case
            except ImportError as e:
                logger.warning(f"Error importing CIME modules: {e}")
                sys.exit(1)
            with Case(caseroot, read_only=True) as case:
                inputroot = case.get_value("DOUT_S_ROOT")
                casename = case.get_value("CASE")
            TSDIR = Path(inputroot).parent / "timeseries" / casename / subdir / "hist"
            INPUTDIR = os.path.join(inputroot, subdir, "hist")
            native = latest_monthly_file(Path(INPUTDIR))
            if native is None:
                print(f"No output files found in {INPUTDIR}")
                sys.exit(0)
        else:
            # testing path
            scratch = os.getenv("SCRATCH")
            if realm == "atmos":
                TSDIR = (
                    Path(scratch)
                    / "archive"
                    / "timeseries"
                    / "b.e30_beta06.B1850C_LTso.ne30_t232_wgx3.192.wrkflw.1"
                    / "atm"
                    / "hist"
                )
            elif realm == "land":
                TSDIR = (
                    Path(scratch)
                    / "archive"
                    / "timeseries"
                    / "b.e30_beta06.B1850C_LTso.ne30_t232_wgx3.192.wrkflw.1"
                    / "lnd"
                    / "hist"
                )
            elif realm == "ocean":
                TSDIR = (
                    Path(scratch)
                    / "archive"
                    / "timeseries"
                    / "b.e30_beta06.B1850C_LTso.ne30_t232_wgx3.192.wrkflw.1"
                    / "ocn"
                    / "hist"
                )

    # Make output directory if it does not exist
    OUTDIR = Path(args.outdir)
    if not os.path.exists(str(OUTDIR)):
        os.makedirs(str(OUTDIR))

    # Load all possible cmip vars for this realm and this experiment - create a data request
    logger.info("Loading data request content %s", realm)
    content_dic = dt.get_transformed_content()
    logger.info("Content dic obtained")
    DR = dr.DataRequest.from_separated_inputs(**content_dic)
    cmip_vars = []
    cmip_vars = DR.find_variables(
        skip_if_missing=False,
        operation="all",
        cmip7_frequency=frequency,
        modelling_realm=realm,
        experiment=args.experiment,
    )

    # Determine cmip variables that will process
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

    # Set up dask if appropriate
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
            input_path = TSDIR / f"*{include_patterns[0]}*"
        else:
            input_path = TSDIR / "*"

        # Load and evaluate the CMIP mapping YAML file (cesm_to_cmip7.yaml)
        mapping = Mapping.from_packaged_default()

        # Determine TABLES directory
        if model == "cesm":
            tables_root = Path(TABLES_cesm)
        else:
            tables_root = Path(TABLES_noresm)

        # Now process the variables
        if args.workers == 1:
            results = [
                item
                for v in cmip_vars
                for item in process_one_var(
                    v,
                    mapping,
                    input_path,
                    tables_root,
                    OUTDIR,
                    resolution,
                    model,
                    realm=realm,
                    frequency=frequency,
                    ocn_fx_fields=ocn_fx_fields,
                )
            ]
        else:
            futs = [
                process_one_var_delayed(
                    var,
                    mapping,
                    input_path,
                    tables_root,
                    OUTDIR,
                    resolution,
                    model,
                    realm=realm,
                    frequency=frequency,
                    ocn_fx_fields=ocn_fx_fields,
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
