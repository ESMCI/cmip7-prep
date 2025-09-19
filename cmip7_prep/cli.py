
"""Simple CLI entry points for cmip7_prep (argparse-based)."""
from __future__ import annotations
import argparse
from pathlib import Path
import xarray as xr
import numpy as np

from cmip7_prep.regrid import regrid_to_1deg
from cmip7_prep.cmor_writer import CmorSession

def make_target(out: Path) -> None:
    """Create a canonical 1° grid template with lat/lon and (placeholder) bounds."""
    lat = np.arange(-89.5, 90.5, 1.0)
    lon = np.arange(0.5, 360.5, 1.0)
    ds = xr.Dataset(coords={"lat": ("lat", lat), "lon": ("lon", lon)})
    ds.to_netcdf(out)

def prepare(
    *,
    var: str,
    in_files: list[Path],
    realm: str,
    cmor_tables: Path,
    dataset_json: Path,
    outdir: Path,
) -> None:
    """Regrid (if needed) and CMOR-write a single variable (thin wrapper demo)."""
    ds = xr.open_mfdataset([str(p) for p in in_files], combine="by_coords", use_cftime=True)
    da = ds[var]
    ds_tmp = xr.Dataset({var: da})
    da1 = regrid_to_1deg(ds_tmp, var)
    ds_out = xr.Dataset({var: da1})
    with CmorSession(tables_path=str(cmor_tables), dataset_json=str(dataset_json)) as cm:
        vdef = type("VDef", (), {"name": var, "realm": realm, "units": da.attrs.get("units", "")})()
        cm.write_variable(ds_out, var, vdef, outdir=outdir)

def main(argv: list[str] | None = None) -> int:
    """Entry point for the cmip7_prep command-line interface."""
    p = argparse.ArgumentParser(prog="cmip7-prep")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_make = sub.add_parser("make-target", help="Write a simple 1° lat/lon grid file.")
    p_make.add_argument("out", type=Path, help="Output NetCDF path for the target grid.")
    p_make.set_defaults(func=lambda a: make_target(a.out))

    p_prep = sub.add_parser("prepare", help="Regrid one variable and write CMOR output.")
    p_prep.add_argument("--var", required=True, help="Variable name in input files.")
    p_prep.add_argument("--in-file", "-i", action="append", required=True, type=Path, help="Input files (repeatable).")
    p_prep.add_argument("--realm", required=True, help="CMOR realm/table, e.g., Amon.")
    p_prep.add_argument("--cmor-tables", required=True, type=Path, help="Path to CMOR Tables directory.")
    p_prep.add_argument("--dataset-json", required=True, type=Path, help="Path to cmor_dataset.json.")
    p_prep.add_argument("--outdir", default=Path("out"), type=Path, help="Output directory for CMORized files.")
    p_prep.set_defaults(func=lambda a: prepare(
        var=a.var,
        in_files=a.in_file,
        realm=a.realm,
        cmor_tables=a.cmor_tables,
        dataset_json=a.dataset_json,
        outdir=a.outdir,
    ))

    args = p.parse_args(argv)
    args.func(args)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
