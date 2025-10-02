from __future__ import annotations
import os
from pathlib import Path
import logging
import re
from typing import Optional, Tuple
import sys

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
from datetime import datetime, UTC

import dask
from dask import delayed

_DATE_RE = re.compile(
    r"[\.\-](?P<year>\d{4})"  # year
    r"(?P<sep>-?)"  # optional hyphen
    r"(?P<month>0[1-9]|1[0-2])"  # month 01–12
    r"\.nc(?!\S)"  # literal .nc and then end (or whitespace)
)

TABLES = "/glade/work/cmip7/e3sm_to_cmip/cmip6-cmor-tables/Tables"

scratch = os.getenv("SCRATCH")
OUTDIR = scratch + "/CMIP7"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


@delayed
def process_one_var(varname: str) -> tuple[str, str]:
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
            # one log file per worker/run (timestamp + var suffix helps debugging)
            log_dir=log_dir,
            log_name=f"cmor_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_{varname}.log",
            dataset_attrs={"institution_id": "NCAR"},  # plus your other attrs if needed
            outdir=OUTDIR,
        ) as cm:
            # vdef from mapping cfg
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

            # Your writer expects a dataset with varname present:
            cm.write_variable(ds_cmor, varname, vdef)

        return (varname, "ok")
    except Exception as e:  # keep task alive; report failure
        return (varname, f"ERROR: {e!r}")


def link_files(
    src: Path, dest: Path, pattern: str, *, relative: bool, overwrite: bool
) -> int:
    if not src.is_dir():
        raise SystemExit(f"Error: source directory not found: {src}")
    dest.mkdir(parents=True, exist_ok=True)

    # Non-recursive: Path.glob does not descend into subdirectories
    matches = [p for p in src.glob(pattern) if p.is_file()]
    if not matches:
        print(f"No matches for pattern {pattern!r} in {src}")
        return 0

    made = 0
    for p in matches:
        link_path = dest / p.name

        if link_path.exists() or link_path.is_symlink():
            if overwrite:
                link_path.unlink()
            else:
                print(f"Skipping existing: {link_path}")
                continue

        if relative:
            # make target path relative to the link's directory
            target = os.path.relpath(p.resolve(), start=link_path.parent)
            link_path.symlink_to(target)
        else:
            link_path.symlink_to(p.resolve())

        print(f"Linked: {link_path} -> {p}")
        made += 1

    return made


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
        sep = m.group("sep")  # "" or "-"
        seps.add(sep)
        found.append((year, month, p))

    if not found:
        return None

    if require_consistent_style and len(seps) > 1:
        raise ValueError("Mixed date styles detected (YYYYMM.nc and YYYY-MM.nc).")
    print(f"Found {len(found)} files in {str(directory)}")
    # Pick the max by (year, month). If tie, fall back to lexicographic filename to be deterministic.
    found.sort(key=lambda t: (t[0], t[1], t[2].name))
    year, month, path = found[-1]
    return path, year, month


if __name__ == "__main__":
    # JPE: THIS MECHANISM in hf_collection include_pattern is currently broken
    # Only atm monthly 32 bit
    # include_pattern = "*cam.h0a.*"
    # Only atm monthly 64 bit
    include_pattern = "*cam.h0a*"

    if len(sys.argv) > 2:
        caseroot = sys.argv[1]
        cimeroot = sys.argv[2]
        _LIBDIR = os.path.join(cimeroot, "CIME", "Tools")
        sys.path.append(_LIBDIR)

        from standard_script_setup import *
        from CIME.case import Case

        with Case(caseroot, read_only=True) as case:
            inputroot = case.get_value("DOUT_S_ROOT")
            casename = case.get_value("CASE")
        # Currently due to a problem in GenTS we need to create another directory and link only files we need
        INPUTDIR = os.path.join(inputroot, "atm", "hist_amon")
        link_files(
            Path(os.path.join(inputroot, "atm", "hist")),
            Path(INPUTDIR),
            include_pattern,
            relative=True,
            overwrite=False,
        )
        TSDIR = Path(inputroot).parent / "timeseries" / casename / "atm" / "hist"

        native = latest_monthly_file(Path(INPUTDIR))
        if TSDIR.exists():
            timeseries = latest_monthly_file(TSDIR)
            if timeseries is not None:
                _, tsyr, _ = timeseries
                _, nyr, _ = native
                if nyr < tsyr + 10:
                    print(f"Less than 10 years ready, not processing {nyr}, {tsyr}")
                    sys.exit(0)
    else:
        # testing path
        INPUTDIR = "/glade/derecho/scratch/cmip7/archive/b.e30_beta06.B1850C_LTso.ne30_t232_wgx3.192.wrkflw.1/atm/hist_amon64"
        TSDIR = (
            scratch
            + "/archive/timeseries/b.e30_beta06.B1850C_LTso.ne30_t232_wgx3.192.wrkflw.1/atm/hist"
        )

    if not os.path.exists(str(OUTDIR)):
        os.makedirs(str(OUTDIR))
    if not os.path.exists(str(TSDIR)):
        os.makedirs(str(TSDIR))

    cluster = LocalCluster(n_workers=128, threads_per_worker=1, memory_limit="235GB")
    client = cluster.get_client()
    input_head_dir = INPUTDIR
    output_head_dir = TSDIR
    hf_collection = HFCollection(input_head_dir, dask_client=client)
    # hf_collection = hf_collection.include_patterns([include_pattern])

    hf_collection.pull_metadata()
    ts_collection = TSCollection(
        hf_collection, output_head_dir, ts_orders=None, dask_client=client
    )
    ts_collection = ts_collection.apply_overwrite("*")
    ts_collection.execute()

    # 0) Load mapping (uses packaged data/cesm_to_cmip7.yaml by default)
    mapping = Mapping.from_packaged_default()

    cmip_vars = find_variables_by_prefix(
        None, "Amon.", include_groups={"baseline_monthly"}
    )
    print(f"CMORIZING {len(cmip_vars)} variables")
    # 1) Load requested variables
    ds_native, cmip_vars = open_native_for_cmip_vars(
        cmip_vars,
        Path(TSDIR / "*cam.h0a.*"),
        mapping,
        use_cftime=True,
        parallel=True,
    )

    futs = [process_one_var(v) for v in cmip_vars]
    results = dask.compute(*futs)  # blocks until all finish

    for v, status in results:
        print(v, "→", status)
