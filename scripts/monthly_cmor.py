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
import os
from pathlib import Path
import logging
import re
from typing import Optional, Tuple
import sys
from datetime import datetime, UTC

import xarray as xr
from cmip7_prep.mapping_compat import Mapping
from cmip7_prep.pipeline import realize_regrid_prepare, open_native_for_cmip_vars
from cmip7_prep.cmor_writer import CmorSession
from cmip7_prep.dreq_search import find_variables_by_prefix
from gents.hfcollection import HFCollection
from gents.timeseries import TSCollection
from dask.distributed import LocalCluster
from dask.distributed import Client
from dask import delayed
import dask

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
        "--realm", choices=["atm", "lnd"], required=True, help="Realm to process"
    )
    parser.add_argument("--caseroot", type=str, help="Case root directory")
    parser.add_argument("--cimeroot", type=str, help="CIME root directory")
    parser.add_argument(
        "--test", action="store_true", help="Run in test mode with default paths"
    )
    parser.add_argument(
        "--run-freq",
        type=str,
        default="10y",
        help="Minimum run frequency required (e.g. '10y' or '120m'), default 10y",
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
            print(f"Invalid --run-freq value: {run_freq}")
            sys.exit(1)
    elif run_freq.endswith("m"):
        try:
            run_months = int(run_freq[:-1])
            run_years = run_months // 12
        except Exception:
            print(f"Invalid --run-freq value: {run_freq}")
            sys.exit(1)
    else:
        print(f"Invalid --run-freq value: {run_freq}")
        sys.exit(1)
    args.run_years = run_years
    args.run_months = run_months
    return args


@delayed
def process_one_var(varname: str, mapping, ds_native, OUTDIR) -> tuple[str, str]:
    """Compute+write one CMIP variable. Returns (varname, 'ok' or error message)."""
    try:
        # Realize → verticalize (if needed) → regrid for a single variable
        ds_cmor = realize_regrid_prepare(
            mapping,
            ds_native,
            varname,
            tables_path=TABLES,
            time_chunk=12,
            regrid_kwargs={
                "output_time_chunk": 12,
                "dtype": "float32",
                "bilinear_map": Path(
                    "/glade/campaign/cesm/cesmdata/inputdata/cpl/gridmaps/ne30pg3/map_ne30pg3_to_1x1d_bilin.nc"
                ),
                "conservative_map": Path(
                    "/glade/campaign/cesm/cesmdata/inputdata/cpl/gridmaps/ne30pg3/map_ne30pg3_to_1x1d_aave.nc"
                ),
            },
        )
        # Unique log per *run* is in your CmorSession; still fine to reuse here.
        log_dir = Path(OUTDIR) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        with CmorSession(
            tables_path=TABLES,
            log_dir=log_dir,
            log_name=f"cmor_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_{varname}.log",
            dataset_attrs={"institution_id": "NCAR"},
            outdir=OUTDIR,
        ) as cm:
            cfg = mapping.get_cfg(varname)
            vdef = type(
                "VDef",
                (),
                {
                    "name": varname,
                    "table": cfg.get("table", "Amon"),
                    "units": cfg.get("units", ""),
                    "dims": cfg.get("dims", []),
                    "positive": cfg.get("positive", None),
                    "cell_methods": cfg.get("cell_methods", None),
                    "long_name": cfg.get("long_name", None),
                    "standard_name": cfg.get("standard_name", None),
                    "levels": cfg.get("levels", None),
                },
            )()
            cm.write_variable(ds_cmor, varname, vdef)
        return (varname, "ok")
    except Exception as e:
        return (varname, f"ERROR: {e!r}")


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
    print(f"Looking for files in {str(directory)}")
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
    print(f"Found {len(found)} files in {str(directory)}")
    found.sort(key=lambda t: (t[0], t[1], t[2].name))
    year, month, path = found[-1]
    return path, year, month


def main():
    parser = argparse.ArgumentParser(
        description="CMIP7 monthly processing for atm/lnd realms"
    )
    parser.add_argument(
        "--realm", choices=["atm", "lnd"], required=True, help="Realm to process"
    )
    parser.add_argument("--caseroot", type=str, help="Case root directory")
    parser.add_argument("--cimeroot", type=str, help="CIME root directory")
    parser.add_argument(
        "--test", action="store_true", help="Run in test mode with default paths"
    )
    parser.add_argument(
        "--run-freq",
        type=str,
        default="10y",
        help="Requested run frequency (e.g. '10y' or '120m'), default 10y",
    )
    args = parser.parse_args()

    args = parse_args()
    scratch = os.getenv("SCRATCH")
    OUTDIR = scratch + "/CMIP7"
    # Set realm-specific parameters
    if args.realm == "atm":
        include_pattern = "*cam.h0a*"
        var_prefix = "Amon."
        subdir = "atm"
    else:
        include_pattern = "*clm2.h0*"
        var_prefix = "Lmon."
        subdir = "lnd"
    # Setup input/output directories
    if args.caseroot and args.cimeroot:
        caseroot = args.caseroot
        cimeroot = args.cimeroot
        _LIBDIR = os.path.join(cimeroot, "CIME", "Tools")
        sys.path.append(_LIBDIR)
        try:
            from CIME.case import Case
        except ImportError as e:
            print(f"Error importing CIME modules: {e}")
            sys.exit(1)
        with Case(caseroot, read_only=True) as case:
            inputroot = case.get_value("DOUT_S_ROOT")
            casename = case.get_value("CASE")
        INPUTDIR = os.path.join(inputroot, subdir, "hist")
        TSDIR = Path(inputroot).parent / "timeseries" / casename / subdir / "hist"
        native = latest_monthly_file(Path(INPUTDIR))
        if TSDIR.exists():
            timeseries = latest_monthly_file(TSDIR)
            if timeseries is not None:
                _, tsyr, _ = timeseries
                _, nyr, _ = native
                # Calculate span in months
                span_months = (nyr - tsyr) * 12
                if span_months < args.run_months:
                    print(
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
        else:
            INPUTDIR = "/glade/derecho/scratch/cmip7/archive/b.e30_beta06.B1850C_LTso.ne30_t232_wgx3.192.wrkflw.1/lnd/hist"
            TSDIR = (
                scratch
                + "/archive/timeseries/b.e30_beta06.B1850C_LTso.ne30_t232_wgx3.192.wrkflw.1/lnd/hist"
            )
    # Ensure output directories exist
    if not os.path.exists(str(OUTDIR)):
        os.makedirs(str(OUTDIR))
    if not os.path.exists(str(TSDIR)):
        os.makedirs(str(TSDIR))
    # Dask cluster setup
    cluster = LocalCluster(n_workers=128, threads_per_worker=1, memory_limit="235GB")
    client = cluster.get_client()
    input_head_dir = INPUTDIR
    output_head_dir = TSDIR
    hf_collection = HFCollection(input_head_dir, dask_client=client)
    hf_collection = hf_collection.include_patterns([include_pattern])
    hf_collection.pull_metadata()
    ts_collection = TSCollection(
        hf_collection, output_head_dir, ts_orders=None, dask_client=client
    )
    ts_collection = ts_collection.apply_overwrite("*")
    ts_collection.execute()
    # Load mapping
    mapping = Mapping.from_packaged_default()
    cmip_vars = find_variables_by_prefix(
        None, var_prefix, include_groups={"baseline_monthly"}
    )
    print(f"CMORIZING {len(cmip_vars)} variables")
    # Load requested variables
    ds_native, cmip_vars = open_native_for_cmip_vars(
        cmip_vars,
        Path(str(TSDIR) + f"/*{include_pattern}*"),
        mapping,
        use_cftime=True,
        parallel=True,
    )
    futs = [process_one_var(v, mapping, ds_native, TABLES, OUTDIR) for v in cmip_vars]
    results = dask.compute(*futs)
    for v, status in results:
        print(v, "→", status)
    client.close()


if __name__ == "__main__":
    main()
