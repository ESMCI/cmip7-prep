import os
from pathlib import Path
import xarray as xr
from cmip7_prep.mapping_compat import Mapping
from cmip7_prep.pipeline import realize_regrid_prepare_many, open_native_for_cmip_vars
from cmip7_prep.cmor_writer import CmorSession

basedir = Path(
    "/glade/derecho/scratch/cmip7/archive/timeseries/b.e30_beta06.B1850C_LTso.ne30_t232_wgx3.192.wrkflw.1_32/atm/hist_monthly/"
)

# 0) Load mapping (uses packaged data/cesm_to_cmip7.yaml by default)
mapping = Mapping.from_packaged_default()

cmip_vars = ["tas"]

ds_native = open_native_for_cmip_vars(
    cmip_vars,
    os.path.join(basedir, "*cam.h0*"),
    mapping,
    use_cftime=True,
    parallel=True,
)

# 2) One call: realize → chunk → regrid → carry time+bounds
ds_cmor = realize_regrid_prepare_many(
    mapping,
    ds_native,
    cmip_vars,
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
with CmorSession(
    tables_path="/glade/work/cmip7/e3sm_to_cmip/cmip6-cmor-tables/Tables",
) as cm:
    cm.write_variables(
        ds_cmor, cmip_vars, mapping, outdir=Path("/glade/derecho/scratch/cmip7/CMIP7")
    )

print("ok")
