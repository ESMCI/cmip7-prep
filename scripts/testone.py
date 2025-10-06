import os
from pathlib import Path
import xarray as xr
from cmip7_prep.mapping_compat import Mapping
from cmip7_prep.pipeline import realize_regrid_prepare, open_native_for_cmip_vars
from cmip7_prep.cmor_writer import CmorSession
from datetime import datetime, UTC

scratch = os.environ.get("SCRATCH")
basedir = Path(
    scratch
    + "/archive/timeseries/b.e30_beta06.B1850C_LTso.ne30_t232_wgx3.192.wrkflw.1/lnd/hist/"
)
TABLES = Path("/glade/work/cmip7/e3sm_to_cmip/cmip6-cmor-tables/Tables")

# 0) Load mapping (uses packaged data/cesm_to_cmip7.yaml by default)
mapping = Mapping.from_packaged_default()
OUTDIR = scratch + "/CMIP7"

cmip_vars = ["lai"]

ds_native, cmip_vars = open_native_for_cmip_vars(
    cmip_vars,
    basedir / "*",
    mapping,
    use_cftime=True,
    parallel=True,
)
varname = cmip_vars[0]

# 2) One call: realize → chunk → regrid → carry time+bounds
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

# 3) CMOR write
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

print("ok")
