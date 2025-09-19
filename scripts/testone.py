from pathlib import Path
import xarray as xr
from cmip7_prep.mapping_compat import Mapping
from cmip7_prep.regrid import regrid_to_1deg
from cmip7_prep.cmor_writer import CmorSession

mapping = Mapping.from_packaged_default()  # uses cmip7_prep/mapping/cesm_to_cmip7.yaml
var = "tas"  # pick a mapped Amon var
cfg = mapping.get_cfg(var)
cesm_vars = []
if cfg.get("source"):  # direct 1:1 mapping
    cesm_vars.append(cfg["source"])
if cfg.get("raw_variables"):  # identity or formula inputs
    cesm_vars.extend(cfg["raw_variables"])
cesm_vars = sorted(set(cesm_vars))  # unique, ordered

# 2) Build the file list by variable name match
basedir = Path(
    "/glade/derecho/scratch/cmip7/archive/timeseries/b.e30_beta06.B1850C_LTso.ne30_t232_wgx3.192.wrkflw.1_32/atm/hist_monthly/"
)
files = sorted({str(p) for v in cesm_vars for p in basedir.glob(f"*{v}*.nc")})

if not files:
    raise FileNotFoundError(f"No files matched for {cesm_vars} in {basedir}")

ds = xr.open_mfdataset(files, combine="by_coords", use_cftime=True, parallel=True)

ds = ds.isel(time=slice(0, 12))

da = mapping.realize(ds, var)
# 3) Choose a time chunk that is a MULTIPLE of the stored chunk size
t_stored = None
chunksizes = da.encoding.get("chunksizes")
if chunksizes and "time" in da.dims:
    t_axis = da.dims.index("time")
    try:
        t_stored = int(chunksizes[t_axis])
    except Exception:
        t_stored = None

# If the file is chunked 1-month along time (common), this picks 12 months per task.
# If stored time-chunk is 24, this picks 24 or 48, etc.—always a multiple → no warning.
if t_stored:
    time_chunk = t_stored * 12
else:
    time_chunk = 12  # fallback; may warn on some datasets
da = da.chunk({"time": time_chunk})

da_rg = regrid_to_1deg(
    xr.Dataset({var: da}), var, output_time_chunk=time_chunk, dtype="float32"
)

with CmorSession(tables_path="Tables", dataset_json="cmor_dataset.json") as cm:
    vdef = type(
        "VDef", (), {"name": var, "realm": "Amon", "units": cfg.get("units", "")}
    )()
    cm.write_variable(xr.Dataset({var: da_rg}), var, vdef, outdir=Path("out_smoke"))
print("ok")
