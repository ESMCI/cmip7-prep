#!/usr/bin/env python3
"""
monthly_cmor.py: Combined CMIP7 monthly processing script for ATM and LND realms.

Usage:
  python monthly_cmor.py --realm atm [--caseroot ... --cimeroot ...]
  python monthly_cmor.py --realm lnd [--caseroot ... --cimeroot ...]
  python monthly_cmor.py --realm atm --test
  python monthly_cmor.py --realm lnd --test

Preserves all comments and error handling from both atm_monthly.py and lnd_monthly.py.
"""
from __future__ import annotations
import argparse
from concurrent.futures import as_completed
from email import parser
from http import client
import os
from pathlib import Path
import logging
import re
from typing import Optional, Tuple
import sys
from datetime import datetime, UTC
import glob

import xarray as xr
from cmip7_prep.mapping_compat import Mapping
from cmip7_prep.pipeline import realize_regrid_prepare, open_native_for_cmip_vars
from cmip7_prep.cmor_writer import CmorSession
from cmip7_prep.dreq_search import find_variables_by_prefix
from cmip7_prep.mom6_static import load_mom6_static
from gents.hfcollection import HFCollection
from gents.timeseries import TSCollection
from dask.distributed import LocalCluster
from dask.distributed import wait, as_completed
from dask import delayed
import dask.array as da
import dask

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

TABLES = "/glade/work/cmip7/e3sm_to_cmip/cmip6-cmor-tables/Tables"
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(
        description="CMIP7 monthly processing for atm/lnd realms"
    )

    parser.add_argument(
        "--static-grid",
        type=str,
        default=None,
        help="Path to static grid file for MOM variables (optional)",
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
        "--realm", choices=["atm", "lnd", "ocn"], required=True, help="Realm to process"
    )
    parser.add_argument("--caseroot", type=str, help="Case root directory")
    parser.add_argument("--cimeroot", type=str, help="CIME root directory")
    parser.add_argument(
        "--test", action="store_true", help="Run in test mode with default paths"
    )
    scratch = os.getenv("SCRATCH")
    default_outdir = scratch + "/CMIP7" if scratch else "./CMIP7"
    parser.add_argument(
        "--run-freq",
        type=str,
        default="10y",
        help="Request run frequency (e.g. '10y' or '120m'), default 10y",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=default_outdir,
        help="Output directory for CMORized files (default: $SCRATCH/CMIP7)",
    )
    parser.add_argument(
        "--skip-timeseries",
        action="store_true",
        help="Skip timeseries processing and go directly to CMORization.",
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
    varname: str, mapping, inputfile, tables_path, outdir, realm="atm", mom6_grid=None
) -> list[tuple[str, str]]:
    """Compute+write one CMIP variable. Returns list of (varname, 'ok' or error message)."""
    logger.info(f"Starting processing for variable: {varname}")
    results = [(varname, "started")]
    try:
        cfg = mapping.get_cfg(varname)
    except Exception as e:
        logger.error(f"Error retrieving config for {varname}: {e}")
        results.append((varname, f"ERROR: {e}"))
        return results

    dims_list = cfg.get("dims")
    # If dims is a single list (atm/lnd), wrap in a list for uniformity
    if dims_list and isinstance(dims_list[0], str):
        dims_list = [dims_list]
    for dims in dims_list:
        logger.info(f"Processing {varname} with dims {dims}")
        try:
            open_kwargs = None
            if realm == "ocn":
                open_kwargs = {"decode_timedelta": False}

            ds_native, var = open_native_for_cmip_vars(
                varname,
                inputfile,
                mapping,
                use_cftime=True,
                parallel=True,
                open_kwargs=open_kwargs,
            )
            logger.info(
                f"ds_native keys: {list(ds_native.keys())} for var {varname} with dims {dims}"
            )
            if var is None:
                logger.warning(f"Source variable(s) not found for {varname}")
                results.append((varname, "ERROR: Source variable(s) not found."))
                continue

            # --- OCN: distinguish native vs regridded by dims ---
            if "xh" in dims and "yh" in dims:
                logger.info(f"Preparing native grid output for mom6 variable {varname}")
                ds_cmor = ds_native
                results.append((varname, "analyzed native mom6 grid"))
                if mom6_grid is not None:
                    logger.info(f"Add geolat to ds_cmor")

                    def _extract_array(val):
                        # If val is a tuple, return the first element, else return as is
                        return val[0] if isinstance(val, tuple) else val

                    ds_cmor["geolat"] = xr.DataArray(mom6_grid[0], dims=("yh", "xh"))
                    ds_cmor["geolon"] = xr.DataArray(mom6_grid[1], dims=("yh", "xh"))
                    ds_cmor["geolat_c"] = xr.DataArray(
                        mom6_grid[2], dims=("yhp", "xhp")
                    )
                    ds_cmor["geolon_c"] = xr.DataArray(
                        mom6_grid[3], dims=("yhp", "xhp")
                    )
            else:
                # For lnd/atm or any other dims, use existing logic
                logger.info(f"Processing {varname} for dims {dims} (atm/lnd or other)")
                sftlf_path = next(Path(outdir).rglob("sftlf_fx_*.nc"), None)
                ds_cmor = realize_regrid_prepare(
                    mapping,
                    ds_native,
                    varname,
                    tables_path=tables_path,
                    time_chunk=12,
                    mom6_grid=mom6_grid,
                    regrid_kwargs={
                        "output_time_chunk": 12,
                        "dtype": "float32",
                        "sftlf_path": sftlf_path,
                    },
                    open_kwargs={"decode_timedelta": True},
                )
        except Exception as e:
            logger.error(
                f"Exception during regridding of {varname} with dims {dims}: {e!r}"
            )
            continue
        try:
            # CMORize
            logger.info(f"CMOR writing for {varname} with dims {dims}")
            log_dir = outdir + "/logs"
            with CmorSession(
                tables_path=tables_path,
                log_dir=log_dir,
                log_name=f"cmor_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_{varname}.log",
                dataset_attrs={"institution_id": "NCAR"},
                outdir=outdir,
            ) as cm:

                vdef = type(
                    "VDef",
                    (),
                    {
                        "name": varname,
                        "table": cfg.get("table", "Amon"),
                        "units": cfg.get("units", ""),
                        "dims": dims,
                        "positive": cfg.get("positive", None),
                        "cell_methods": cfg.get("cell_methods", None),
                        "long_name": cfg.get("long_name", None),
                        "standard_name": cfg.get("standard_name", None),
                        "levels": cfg.get("levels", None),
                    },
                )()
                logger.info(f"Writing variable {varname} with dims {dims}")
                cm.write_variable(ds_cmor, varname, vdef)
            logger.info(f"Finished processing for {varname} with dims {dims}")
            results.append((varname, "ok"))
        except Exception as e:
            logger.error(
                f"Exception while processing {varname} with dims {dims}: {e!r}"
            )
            results.append((varname, f"ERROR: {e!r}"))
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

    mom6_grid = None
    if args.realm == "atm":
        include_patterns = ["*cam.h0a*"]
        var_prefix = "Amon."
        subdir = "atm"
    elif args.realm == "lnd":
        include_patterns = ["*clm2.h0*"]
        var_prefix = "Lmon."
        subdir = "lnd"
    elif args.realm == "ocn":
        # we do not want to match static files
        include_patterns = ["*mom6.h.rho2.*", "*mom6.h.native.*", "*mom6.h.sfc.*"]
        var_prefix = "Omon."
        subdir = "ocn"
        if not args.static_grid:
            ocn_static_grid = "/glade/campaign/cesm/cesmdata/inputdata/ocn/mom/tx2_3v2/ocean_hgrid_221123.nc"
    # Setup input/output directories
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
        INPUTDIR = os.path.join(inputroot, subdir, "hist")
        TSDIR = Path(inputroot).parent / "timeseries" / casename / subdir / "hist"
        native = latest_monthly_file(Path(INPUTDIR))
        if native is None:
            print(f"No output files found in {INPUTDIR}")
            sys.exit(0)
        if TSDIR.exists():
            timeseries = latest_monthly_file(TSDIR)
            if timeseries is not None:
                _, tsyr, _ = timeseries
                _, nyr, _ = native
                # Calculate span in months
                span_months = (nyr - tsyr) * 12
                if span_months < args.run_months:
                    logger.info(
                        f"Less than required run frequency ready ({span_months} months, need {args.run_months}), not processing {nyr}, {tsyr}"
                    )
                    sys.exit(0)
    else:
        # testing path
        if args.realm == "atm":
            INPUTDIR = "/glade/derecho/scratch/cmip7/archive/b.e30_beta06.B1850C_LTso.ne30_t232_wgx3.192.wrkflw.1/atm/hist"
            TSDIR = (
                scratch
                + "/archive/timeseries/b.e30_beta06.B1850C_LTso.ne30_t232_wgx3.192.wrkflw.1/atm/hist"
            )
        elif args.realm == "lnd":
            INPUTDIR = "/glade/derecho/scratch/cmip7/archive/b.e30_beta06.B1850C_LTso.ne30_t232_wgx3.192.wrkflw.1/lnd/hist"
            TSDIR = (
                scratch
                + "/archive/timeseries/b.e30_beta06.B1850C_LTso.ne30_t232_wgx3.192.wrkflw.1/lnd/hist"
            )
        elif args.realm == "ocn":
            INPUTDIR = "/glade/derecho/scratch/cmip7/archive/b.e30_beta06.B1850C_LTso.ne30_t232_wgx3.192.wrkflw.1/ocn/hist"
            TSDIR = (
                scratch
                + "/archive/timeseries/b.e30_beta06.B1850C_LTso.ne30_t232_wgx3.192.wrkflw.1/ocn/hist"
            )

    if not os.path.exists(str(OUTDIR)):
        os.makedirs(str(OUTDIR))
    if not os.path.exists(str(TSDIR)):
        os.makedirs(str(TSDIR))
    # Load MOM6 static grid if needed (ocn realm)
    if args.realm == "ocn" and ocn_static_grid:
        mom6_grid = load_mom6_static(ocn_static_grid)
        logger.info(f"Using MOM static grid file: {ocn_static_grid}")
    # Dask cluster setup
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
            n_workers=args.workers, threads_per_worker=1, memory_limit=ml
        )
        client = cluster.get_client()
    input_head_dir = INPUTDIR
    output_head_dir = TSDIR
    if args.skip_timeseries:
        logger.info("Skipping timeseries processing as per --skip-timeseries flag.")
    else:
        cnt = 0
        for include_pattern in include_patterns:
            cnt = cnt + len(glob.glob(os.path.join(input_head_dir, include_pattern)))
        if cnt == 0:
            logger.warning(
                f"No input files to process in {input_head_dir} with {include_patterns}"
            )
            sys.exit(0)
        hf_collection = HFCollection(input_head_dir, dask_client=client)
        for include_pattern in include_patterns:
            logger.info(f"Processing files with pattern: {include_pattern}")
            hfp_collection = hf_collection.include_patterns([include_pattern])
            hfp_collection.pull_metadata()
            ts_collection = TSCollection(
                hfp_collection, output_head_dir, ts_orders=None, dask_client=client
            )
            if args.overwrite:
                ts_collection = ts_collection.apply_overwrite("*")
            ts_collection.execute()
            logger.info("Timeseries processing complete, starting CMORization...")
    mapping = Mapping.from_packaged_default()
    logger.info(f"Finding variables with prefix {var_prefix}")
    if args.cmip_vars and len(args.cmip_vars) > 0:
        cmip_vars = args.cmip_vars
    else:
        cmip_vars = find_variables_by_prefix(
            None, var_prefix, where={"List of Experiments": "piControl"}
        )
    logger.info(f"CMORIZING {len(cmip_vars)} variables")
    # Load requested variables
    if len(cmip_vars) > 0:
        if len(include_patterns) == 1:
            input_path = Path(str(TSDIR) + f"/*{include_patterns[0]}*")
        else:
            input_path = Path(str(TSDIR) + f"/*")

        if args.workers == 1:
            results = [
                process_one_var(
                    v,
                    mapping,
                    input_path,
                    TABLES,
                    OUTDIR,
                    realm=args.realm,
                    mom6_grid=mom6_grid,
                )
                for v in cmip_vars
            ]
        else:
            futs = [
                process_one_var_delayed(
                    var,
                    mapping,
                    input_path,
                    TABLES,
                    OUTDIR,
                    realm=args.realm,
                    mom6_grid=mom6_grid,
                )
                for var in cmip_vars
            ]
            futures = client.compute(futs)
            wait(
                futures, timeout="600s"
            )  # optional soft check; won’t raise, just returns done/pending

            # iterate results; if anything stalls you can call dump_all_stacks(client)
            results = []
            for _, result in as_completed(futures, with_results=True):
                try:
                    results.append(result)  # (v, status)
                except Exception as e:
                    logger.error("Task error:", e)
                    raise

        for v, status in results:
            logger.info(f"Variable {v} processed with status: {status}")
    else:
        logger.info("No results to process.")
    if client:
        client.close()
    if cluster:
        cluster.close()


if __name__ == "__main__":
    main()
