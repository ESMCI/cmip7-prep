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
tbname = ds_cmor["time"].attrs.get("bounds")
tb = ds_cmor[tbname]
print(
    "CHECK:",
    "time dtype",
    ds_cmor["time"].dtype,
    "| tb dims",
    tb.dims,
    "dtype",
    tb.dtype,
    "| equal lengths",
    tb.sizes["time"] == ds_cmor.sizes["time"],
)
print(
    "CHECK lat/lon: ",
    "lat",
    ds_cmor.lat.size,
    float(ds_cmor.lat.min()),
    float(ds_cmor.lat.max()),
    "| lon",
    ds_cmor.lon.size,
    float(ds_cmor.lon.min()),
    float(ds_cmor.lon.max()),
)
attrs = {
    "mip_era": "CMIP6",
    "activity_id": "CMIP",
    "institution_id": "NCAR",
    "institution": "National Center for Atmospheric Research, Climate and Global Dynamics Laboratory, 1850 Table Mesa Drive, Boulder, CO 80305, USA",
    "product": "model-output",
    "source_id": "CESM2",
    "source_type": "AOGCM",
    "experiment_id": "piControl",
    "experiment": "Pre-Industrial Control",
    "sub_experiment_id": "none",
    "sub_experiment": "none",
    "realization_index": 1,
    "initialization_index": 1,
    "physics_index": 1,
    "forcing_index": 1,
    "member_id": "r1i1p1f1",
    "grid_label": "gr",
    "calendar": "NO_LEAP",
    "frequency": "mon",
    "title": "CESM2 piControl on 1x1 degree grid (regridded)",
    "outpath": "/glade/derecho/scratch/cmip7/CMIP7",
    "_controlled_vocabulary_file": "CMIP6_CV.json",
    "_AXIS_ENTRY_FILE": "CMIP6_coordinate.json",
    "_FORMULA_VAR_FILE": "CMIP6_formula_terms.json",
}

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
