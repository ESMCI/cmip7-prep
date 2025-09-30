import os
from pathlib import Path
import logging

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

scratch = os.getenv("SCRATCH")
TABLES = "/glade/work/cmip7/e3sm_to_cmip/cmip6-cmor-tables/Tables"
INPUTDIR = "/glade/derecho/scratch/cmip7/archive/b.e30_beta06.B1850C_LTso.ne30_t232_wgx3.192.wrkflw.1/atm/hist_amon64"
TSDIR = (
    scratch
    + "/archive/timeseries/b.e30_beta06.B1850C_LTso.ne30_t232_wgx3.192.wrkflw.1/atm/hist"
)
OUTDIR = scratch + "/CMIP7"

if not os.path.exists(str(OUTDIR)):
    os.makedirs(str(OUTDIR))
if not os.path.exists(str(TSDIR)):
    os.makedirs(str(TSDIR))

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
            cm.write_variable(ds_cmor, varname, vdef, outdir=OUTDIR)

        return (varname, "ok")
    except Exception as e:  # keep task alive; report failure
        return (varname, f"ERROR: {e!r}")


if __name__ == "__main__":
    # JPE: THIS MECHANISM is currently broken
    # Only atm monthly 32 bit
    # include_pattern = "*cam.h0a.*"
    # Only atm monthly 64 bit
    #    include_pattern = "*cam.h0a*"

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
        Path(TSDIR + "/*"),
        mapping,
        use_cftime=True,
        parallel=True,
    )

    futs = [process_one_var(v) for v in cmip_vars]
    results = dask.compute(*futs)  # blocks until all finish

    for v, status in results:
        print(v, "→", status)
