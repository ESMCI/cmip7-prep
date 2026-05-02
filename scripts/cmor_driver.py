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

from cmip7_prep.cmor_utils import (
    bounds_from_centers_1d,
    roll_for_monotonic_with_bounds,
    packaged_dataset_json,
)
from cmip7_prep.mapping_compat import Mapping
from cmip7_prep.pipeline import (
    realize_regrid_prepare,
    open_native_for_cmip_vars,
    _collect_required_model_vars,
    _filename_contains_var,
)
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
TABLES_noresm = "/nird/datalake/NS9560K/mvertens/packages/cmip7-prep/cmip7-cmor-tables/"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

INCLUDE_PATTERN_MAP = {
    "cesm": {
        "aerosol": {
            "mon": ["cam.h0a"],
            "day": ["cam.h1a"],
            "6hr": ["cam.h2a"],
            "3hr": ["cam.h3a"],
        },
        "atmosChem": {
            "mon": ["cam.h0a"],
            "day": ["cam.h1a"],
            "6hr": ["cam.h2a"],
            "3hr": ["cam.h3a"],
        },
        "atmos": {
            "mon": ["cam.h0a"],
            "day": ["cam.h1a"],
            "6hr": ["cam.h2a"],
            "3hr": ["cam.h3a"],
        },
        "land": {
            "mon": ["clm2.h0a"],
        },
        "ocnBgchem": {
            "mon": ["mom6.h.z", "mom6.h.native."],
            "day": ["mom6.h.sfc"],
        },
        "ocean": {
            "mon": ["mom6.h.z", "mom6.h.native."],
            "day": ["mom6.h.sfc"],
        },
        "seaIce": {
            "mon": ["cice.h."],
            "day": ["cice.h1."],
        },
    },
    "noresm": {
        "atmos": {
            "mon": ["cam.h0a"],
            "day": ["cam.h1a"],
            "6hr": ["cam.h2a"],
            "3hr": ["cam.h4a"],
        },
        "land": {
            "mon": ["clm2.h0a"],
            "day": ["clm2.h1a"],
            "3hr": ["clm2.h2a"],
            "yr": ["clm2.h3a"],
        },
        "seaIce": {
            "mon": ["cice.h."],
            "day": ["cice.h1."],
        },
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="CMIP7 monthly processing for atm/lnd realms"
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show program version and exit",
    )
    parser.add_argument(
        "--cmip-vars",
        nargs="*",
        help="List of CMIP variable names to process directly (bypasses variable search)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of Dask workers (default: set to 1 for serial execution)",
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
        default="atmos",
        help="Realm to process. (Default: atmos)",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        choices=[
            "ne16",
            "ne30",
            "tx2_3v2",
        ],
        default="ne30",
        help="input_grid name (Default: ne30)",
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
        choices=["mon", "day", "6hr", "3hr"],
        help="Frequency of data to be translated (mon, day, 6hr, 3hr), (Default: mon)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=".",
        help="Output directory for CMORized files. (Default .)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="piControl",
        help="Experiment name for data request. (Default piControl)",
    )
    parser.add_argument(
        "--model",
        choices=["cesm", "noresm"],
        default="cesm",
        help="Model to use, default: cesm",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging output",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="log output level",
    )
    parser.add_argument(
        "--custom-yaml",
        default=False,
        help="Path to custom YAML mapping file (optional, overrides default packaged YAML)",
    )

    args = parser.parse_args()
    return args


def get_version():
    # Use dynamic version from cmip7_prep
    from cmip7_prep import __version__

    return __version__


def process_one_var(
    cmip_var,
    mapping,
    inputfiles,
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
        # This is obtained from reading cesm_to_cmip7.yaml or noresm_to_cmip7.yaml
        cfg = mapping.get_cfg(varname)
    except KeyError:
        logger.warning(f"Skipping '{varname}': no entry found in mapping YAML")
        results.append((varname, "WARNING: no mapping in YAML"))
        return results
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
        # Read in time series, do the mapping and then regrid if necessary
        # ---------------------------------------------
        # cmor_items is a list of (ds_cmor, write_cfg) — one per variant
        cmor_items = []
        try:
            open_kwargs = None
            if realm in ("ocean", "seaIce"):
                open_kwargs = {"decode_timedelta": False}
            logger.info("Opening native data for variable %s", varname)

            # ds_native is an xarray dataset for the CESM/NorESM time series files
            # open_native_for_cmip_vars is in pipeline.py
            ds_native, var = open_native_for_cmip_vars(
                varname,
                inputfiles,
                mapping,
                use_cftime=True,
                parallel=False,
                open_kwargs=open_kwargs,
            )
            if ds_native is None:
                logger.warning(f"Source variable(s) not found for {varname}, skipping")
                results.append((varname, "WARNING: Source variable(s) not found."))
                continue
            if "TLAT" in ds_native:
                logger.info("TLAT is present")
            if model == "cesm":
                # Append ocn_fx_fields to ds_native if available
                # fx - grid definition like topography, fraction
                if realm == "ocean" and ocn_fx_fields is not None:
                    logger.info("adding ocn_fx_fields to ds_native")
                    ds_native = ds_native.merge(ocn_fx_fields)

            # Output ds_native keys
            logger.debug(
                "ds_native keys: %s for var %s with dims %s",
                list(ds_native.variables.keys()),
                varname,
                dims,
            )
            # TODO: why does this not abort the program?
            # JPE: because I don't want to abort the whole program
            # if one variable is missing - I want to log the error and
            # move on to the next variable
            if var is None:
                logger.warning(f"Source variable(s) not found for {varname}")
                results.append((varname, "WARNING: Source variable(s) not found."))
                continue

            # For ocean realm: distinguish native vs regridded by dims
            if model == "cesm" and "latitude" in dims and "longitude" in dims:
                # output ocn on the native grid, but apply realize for formulas/mapping
                logger.info(
                    f"Preparing native grid output for mom6 variable {varname}, applying realize"
                )
                realized = mapping.realize(ds_native, varname)
                ds_c = (
                    realized
                    if isinstance(realized, xr.Dataset)
                    else xr.Dataset({varname: realized})
                )
                # Ensure time_bounds is included if present
                if "time_bounds" in ds_native and "time_bounds" not in ds_c:
                    ds_c = ds_c.assign(time_bounds=ds_native["time_bounds"])
                cmor_items = [(ds_c, cfg)]
                results.append(
                    (str(varname), "analyzed native mom6 grid (realize applied)")
                )
            elif realm == "seaIce" and len(dims) == 1:
                logger.info(
                    f"Preparing seaIce field variants via realize_all for {varname}"
                )
                for da, variant_cfg in mapping.realize_all(
                    ds_native, varname, freq=frequency
                ):
                    ds_v = (
                        da if isinstance(da, xr.Dataset) else xr.Dataset({varname: da})
                    )
                    if "time_bounds" in ds_native and "time_bounds" not in ds_v:
                        ds_v = ds_v.assign(time_bounds=ds_native["time_bounds"])
                    cmor_items.append((ds_v, variant_cfg))
                results.append(
                    (
                        str(varname),
                        f"seaIce field ({len(cmor_items)} variant(s), realize_all applied)",
                    )
                )
            else:
                # For lnd/atm or any other dims, use existing logic
                logger.debug(
                    "Processing %s for dims %s (atm/lnd or other)", varname, dims
                )
                # Obtain an xr.Dataset (ds_cmor) with the requested CMIP variable ready for CMOR.
                # (this will include mapping from SE to lat/lon)
                ds_cmor = realize_regrid_prepare(
                    resolution,
                    model,
                    mapping,
                    ds_native,
                    varname,
                    tables_path=tables_root / "tables",
                    regrid_kwargs={
                        "dtype": "float32",
                    },
                    open_kwargs={"decode_timedelta": True},
                )
                logger.debug("ds_cmor is not None")

                # Attach ocn_fx_fields to regridded output for writing
                if ocn_fx_fields is not None:
                    ds_cmor = ds_cmor.merge(ocn_fx_fields)
                cmor_items = [(ds_cmor, cfg)]

        except AttributeError:
            results.append((varname, f"ERROR {model} input variable not found."))
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
        # CMORize — loop over variants (usually just one)
        # ---------------------------------------------
        shortname = str(getattr(cmip_var, "physical_parameter").name)
        cmip7name = cmip_var.attributes["branded_variable_name"]
        for ds_cmor_write, write_cfg in cmor_items:
            try:
                log_dir = outdir / "logs"

                # TODO: add NorESM institution_id below
                # Initialize CMOR class
                metadata_json = None
                if model == "noresm":
                    metadata_json = packaged_dataset_json("cmor_dataset_noresm.json")

                with CmorSession(
                    tables_root=tables_root,
                    log_dir=log_dir,
                    log_name=f"cmor_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_{varname}.log",
                    dataset_json=metadata_json,
                    dataset_attrs={"institution_id": "NCC", "GLOBAL_IS_CMIP7": True},
                    outdir=outdir,
                ) as cm:
                    set_cur_dataset_attribute("frequency", frequency)
                    region = write_cfg.get("region", "glb")
                    logger.info("Setting region: %s", region)
                    set_cur_dataset_attribute("region", region)

                    logger.info(
                        f"Writing CMOR variable {cmip7name.name} with frequency {frequency}"
                    )
                    vdef = type(
                        "VDef",
                        (),
                        {
                            "name": shortname,
                            "table": write_cfg.get("table", "atmos"),
                            "units": write_cfg.get("units", ""),
                            "dims": dims,
                            "positive": write_cfg.get("positive", None),
                            "cell_methods": write_cfg.get("cell_methods", None),
                            "long_name": write_cfg.get("long_name", None),
                            "standard_name": write_cfg.get("standard_name", None),
                            "levels": write_cfg.get("levels", None),
                            "branded_variable_name": cmip7name,
                        },
                    )()

                    # Now use CMOR utility to write out netcdf variable
                    logger.info(f"Writing CMOR variable {varname} with dims {dims} ")
                    cm.write_variable(ds_cmor_write, cmip_var, vdef)

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

    # Set logging level
    logger.setLevel(getattr(logging, args.log_level))
    logger.debug(f"Parsed arguments: {args}")

    scratch = os.getenv("SCRATCH")
    OUTDIR = args.outdir
    resolution = args.resolution
    model = args.model
    frequency = args.frequency
    realm = args.realm
    logger.debug("Realm is %s", realm)

    mom6_grid = None
    ocn_grid = None
    ocn_fx_fields = None

    # Determine include patterns and frequency
    include_patterns = get_include_patterns(model, realm, frequency)
    TSDIR = None
    # Setup input directory for noresm
    if args.tsdir:
        TSDIR = Path(args.tsdir)
        if not TSDIR.exists():
            logger.info(f"Time series directory {str(TSDIR)} does not exist")
            sys.exit(0)
        timeseries = latest_monthly_file(TSDIR)
        logger.info(f"latest monthly time series file is {timeseries}")
    elif model == "noresm":
        if realm == "atmos":
            TSDIR = "/datalake/NS9560K/mvertens/test_regridder/atm/timeseries"
        elif realm == "land":
            TSDIR = "/datalake/NS9560K/mvertens/test_regridder/lnd/timeseries"

    # Setup input directory for cesm
    if model == "cesm":
        if realm in ["ocean", "seaIce"]:
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
            if realm in ("atmos", "aerosol", "atmosChem"):
                TSDIR = Path(inputroot) / "atm" / "proc" / "tseries"
            elif realm == "land":
                TSDIR = Path(inputroot) / "lnd" / "proc" / "tseries"
            elif realm in ("ocean", "ocnBgchem"):
                TSDIR = Path(inputroot) / "ocn" / "proc" / "tseries"
            elif realm == "seaIce":
                TSDIR = Path(inputroot) / "ice" / "proc" / "tseries"
            elif realm == "landIce":
                TSDIR = Path(inputroot) / "glc" / "proc" / "tseries"
            TSDIR = TSDIR / args.frequency
        elif not TSDIR or not os.path.exists(TSDIR):
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

    # Load all possible cmip vars for this realm and this experiment
    # The data_request_api is a CMIP7-specific Python package that is
    # separate from CMOR itself but closely related to it.
    # It produce lists of variables requested for each CMIP7 experiment
    logger.info("Loading data request content %s", realm)
    content_dic = dt.get_transformed_content()
    logger.info("Content dictionary obtained")
    DR = dr.DataRequest.from_separated_inputs(**content_dic)
    cmip_vars = []
    cmip_vars = DR.find_variables(
        skip_if_missing=False,
        operation="all",
        cmip7_frequency=frequency,
        modelling_realm=realm,
        experiment=args.experiment,
    )
    # cmip_vars = [var for var in cmip_vars if getattr(var, "region", "") == "glb"]

    # Determine cmip variables that will process
    if args.cmip_vars:
        # Make a copy of the cmip_vars from the data request
        tmp_cmip_vars = cmip_vars
        # Now process the cmip_vars from the input argument, if any
        cmip_vars = []
        for var in tmp_cmip_vars:
            logger.debug(
                "Checking variable %s in %s relative to the data request",
                var.branded_variable_name.name,
                args.cmip_vars,
            )
            if var.branded_variable_name.name in args.cmip_vars:
                logger.info(
                    "Adding variable %s with priority %s",
                    var.branded_variable_name.name,
                    DR.find_priority_per_variable(variable=var),
                )
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
            glob_pattern = f"*{include_patterns[0]}*.nc"
        else:
            glob_pattern = "*.nc"

        # Load and evaluate the CMIP mapping YAML file (cesm_to_cmip7.yaml or noresm_to_cmip7.yaml)
        if custom_yaml := args.custom_yaml:
            logger.info(f"Using custom YAML mapping file: {custom_yaml}")
            mapping = Mapping.from_yaml(custom_yaml)
        elif model == "noresm":
            mapping = Mapping.from_packaged_default(filename="noresm_to_cmip7.yaml")
        else:
            mapping = Mapping.from_packaged_default()
        mapping.default_freq = frequency

        # Determine TABLES directory
        if model == "cesm":
            tables_root = Path(TABLES_cesm)
        else:
            tables_root = Path(TABLES_noresm)

        all_ts_files = sorted(Path(TSDIR).glob(glob_pattern))
        logger.info(
            f"Found {len(all_ts_files)} candidate timeseries files matching '{glob_pattern}'"
        )
        if not all_ts_files:
            logger.error(
                f"No timeseries files found in {TSDIR} matching '{glob_pattern}'"
            )
            sys.exit(1)

        results = []
        for v in cmip_vars:
            varname = v.branded_variable_name.name
            # Filter to only files containing the native model vars needed for this variable
            try:
                model_vars = _collect_required_model_vars(mapping, [varname])
            except Exception:
                model_vars = []
            ts_files = sorted(
                {
                    p
                    for p in all_ts_files
                    if any(_filename_contains_var(p, mv) for mv in model_vars)
                }
            )
            logger.info("=" * 60)
            if not ts_files:
                logger.warning(f"No timeseries files found for variable {varname}")
                continue
            else:
                logger.info(
                    f"Found {len(ts_files)} timeseries files for variable {varname} "
                    f"(model vars: {model_vars})"
                )
            if args.workers == 1:
                res = process_one_var(
                    v,
                    mapping,
                    ts_files,
                    tables_root,
                    OUTDIR,
                    resolution,
                    model,
                    realm=realm,
                    frequency=frequency,
                    ocn_fx_fields=ocn_fx_fields,
                )
                results.extend(res)
            else:
                fut = process_one_var_delayed(
                    v,
                    mapping,
                    ts_files,
                    tables_root,
                    OUTDIR,
                    resolution,
                    model,
                    realm=realm,
                    frequency=frequency,
                    ocn_fx_fields=ocn_fx_fields,
                )
                futures = client.compute([fut])
                wait(futures, timeout="1200s")
                for _, result in as_completed(futures, with_results=True):
                    if isinstance(result, list):
                        results.extend(result)
                    elif isinstance(result, tuple) and len(result) == 2:
                        results.append(result)
                    else:
                        results.append((str(result), "unknown"))
        for v, status in set(results):
            logger.info(f"Variable {v} processed with status: {status}")
    else:
        logger.info("No results to process.")
    if client:
        client.close()
    if cluster:
        cluster.close()


if __name__ == "__main__":
    args = parse_args()
    if getattr(args, "version", False):
        print(f"cmor_driver.py version: {get_version()}")
        sys.exit(0)
    main()
