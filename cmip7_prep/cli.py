# cmip7_prep/cli.py
import typer
from pathlib import Path
from cmip7_prep.dreq import DReq
from cmip7_prep.regrid import RegridderPool
from cmip7_prep.cmor_writer import CmorSession
from cmip7_prep.vertical import to_plev19

app = typer.Typer(help="Prepare CESM output for CMIP7 with CMOR.")

@app.command()
def make_target(out: Path = Path("grids/target_1deg.nc")):
    """Build a canonical 1° grid template with lat/lon, bounds, cell_area."""
    import xarray as xr, numpy as np
    lat = np.arange(-89.5, 90.5, 1.0)
    lon = np.arange(0.5, 360.5, 1.0)
    ds = xr.Dataset(
        dict(),
        coords=dict(lat=("lat", lat), lon=("lon", lon))
    )
    # add bounds + area (simple spherical approx or from ESMF Mesh)
    # ... (fill) ...
    ds.to_netcdf(out)

@app.command()
def build_weights(cam_grid: Path, target: Path="grids/target_1deg.nc", outdir: Path="grids/weights"):
    """(Optional) Document how you produced weights with TempestRemap; store here."""
    # We keep the CLI hook mainly for provenance notes.
    # Actual weights are produced offline with TempestRemap (see README).

@app.command()
def prepare(
    var: str,
    in_files: list[Path],
    realm: str,              # 'Amon', 'Lmon', 'Omon', etc.
    dreq_export: Path,       # Airtable export CSV/JSON
    mapping_yaml: Path="mapping/cesm_to_cmip7.yaml",
    cmor_tables: Path="Tables",    # CMOR JSON (CMIP7-or-6 as applicable)
    outdir: Path="out"
):
    """Main entry: regrid (if needed) and CMOR-write one variable."""
    import xarray as xr
    dreq = DReq(dreq_export, mapping_yaml)
    vdef = dreq.lookup(realm, var)          # all required metadata

    ds = xr.open_mfdataset([str(p) for p in in_files], combine="by_coords", use_cftime=True)

    # Regrid for ATM/LND only
    if realm[0] in ("A","L"):
        from cmip7_prep.regrid import regrid_to_1deg
        ds[var] = regrid_to_1deg(var, ds, method=vdef.regrid_method)

    # Vertical levels if needed (e.g., plev19)
    if vdef.requires_plev19:
        ds = to_plev19(ds, var, vdef)

    # CMOR write
    with CmorSession(tables_path=cmor_tables, dataset_attrs=dreq.dataset_attrs()) as cm:
        cm.write_variable(ds, var, vdef, outdir=Path(outdir))

if __name__ == "__main__":
    app()

