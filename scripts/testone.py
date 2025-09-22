from pathlib import Path
import xarray as xr
from cmip7_prep.mapping_compat import Mapping
from cmip7_prep.pipeline import realize_regrid_prepare
from cmip7_prep.cmor_writer import CmorSession

# 0) Load mapping (uses packaged data/cesm_to_cmip7.yaml by default)
mapping = Mapping.from_packaged_default()
cmip_var = "tas"

# 1) Only open the files required for this CMIP var
cfg = mapping.get_cfg(cmip_var)
cesm_vars = sorted(
    {*(cfg.get("raw_variables") or []), *([cfg["source"]] if cfg.get("source") else [])}
)

basedir = Path(
    "/glade/derecho/scratch/cmip7/archive/timeseries/b.e30_beta06.B1850C_LTso.ne30_t232_wgx3.192.wrkflw.1_32/atm/hist_monthly/"
)
files = sorted({str(p) for v in cesm_vars for p in basedir.glob(f"*{v}*.nc")})
if not files:
    raise FileNotFoundError(f"No files for {cesm_vars}")

ds_native = xr.open_mfdataset(
    files, combine="by_coords", use_cftime=True, parallel=True
)

# 2) One call: realize → chunk → regrid → carry time+bounds
ds_cmor = realize_regrid_prepare(
    mapping,
    ds_native,
    cmip_var,
    time_chunk=12,
    regrid_kwargs={"output_time_chunk": 12, "dtype": "float32"},
)

# 3) CMOR write
with CmorSession(
    tables_path="/glade/work/cmip7/e3sm_to_cmip/cmip6-cmor-tables/Tables",
) as cm:
    vdef = type(
        "VDef", (), {"name": cmip_var, "realm": "Amon", "units": cfg.get("units", "")}
    )()
    cm.write_variable(
        ds_cmor, cmip_var, vdef, outdir=Path("/glade/derecho/scratch/cmip7/CMIP7")
    )

print("ok")
