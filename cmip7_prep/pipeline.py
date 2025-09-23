# cmip7_prep/pipeline.py
"""Open native CESM timeseries, realize mappings, optional vertical transforms, and regrid."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union, Dict, List
import re
import glob

import xarray as xr

from .mapping_compat import Mapping
from .regrid import regrid_to_1deg_ds
from .vertical import to_plev19


# --------------------------- file discovery ---------------------------

_VAR_TOKEN = re.compile(r"(?<![A-Za-z0-9_])([A-Za-z0-9_]+)(?![A-Za-z0-9_])")


def _filename_contains_var(path: Union[str, Path], var: str) -> bool:
    """
    True if filename contains the variable as a dot-delimited token: '.var.'.

    Example matches: '...cam.h0.TS.0001-01.nc' contains '.TS.' -> True
    """
    name = Path(path).name
    needle = f".{var}."
    return needle in name


def _collect_required_cesm_vars(
    mapping: Mapping, cmip_vars: Sequence[str]
) -> List[str]:
    """Gather all native CESM vars needed to realize the requested CMIP vars."""
    needed: set[str] = set()
    for v in cmip_vars:
        cfg = mapping.get_cfg(v) or {}
        src = cfg.get("source")
        raws = cfg.get("raw_variables") or cfg.get("sources") or []
        if src:
            needed.add(src)
        for r in raws:
            # 'sources' items may be dicts with 'cesm_var'
            if isinstance(r, dict) and "cesm_var" in r:
                needed.add(r["cesm_var"])
            elif isinstance(r, str):
                needed.add(r)
        # vertical dependencies if plev19
        levels = cfg.get("levels") or {}
        if (levels.get("name") or "").lower() == "plev19":
            needed.update({"PS", "hyam", "hybm", "P0"})
    return sorted(needed)


def open_native_for_cmip_vars(
    cmip_vars: Sequence[str],
    files_glob: Union[str, Path],
    mapping: Mapping,
    *,
    use_cftime: bool = True,
    parallel: bool = False,
    open_kwargs: Optional[Dict] = None,
) -> xr.Dataset:
    """
    Open CESM timeseries files needed for the requested CMIP variables.

    Parameters
    ----------
    cmip_vars : list of CMIP variable names (e.g., ["tas", "ta"])
    files_glob : glob pattern pointing at native timeseries files
                 (e.g., "/path/atm/hist_monthly/*cam.h0*")
    mapping : Mapping object that knows how to realize CMIP vars
    use_cftime, parallel : forwarded to xarray.open_mfdataset
    open_kwargs : extra kwargs for open_mfdataset

    Returns
    -------
    xr.Dataset containing only the required CESM variables.
    """
    open_kwargs = dict(open_kwargs or {})
    required = _collect_required_cesm_vars(mapping, cmip_vars)

    candidates = glob.glob(str(files_glob))
    if not candidates:
        raise FileNotFoundError(f"No files matched glob: {files_glob}")

    # keep any file that contains ANY of the required CESM vars as '.var.' in the name
    selected = sorted(
        {p for p in candidates if any(_filename_contains_var(p, v) for v in required)}
    )
    if not selected:
        raise FileNotFoundError(
            f"No files under glob matched required variables {required} with '.VAR.' token."
        )

    ds = xr.open_mfdataset(
        selected,
        combine="by_coords",
        use_cftime=use_cftime,
        parallel=parallel,
        **open_kwargs,
    )
    return ds


# ----------------------- realization / vertical -----------------------


def _apply_vertical_if_needed(
    ds_var: xr.Dataset,
    ds_native: xr.Dataset,
    cmip_var: str,
    mapping: Mapping,
    tables_path: Optional[Union[str, Path]],
) -> xr.Dataset:
    """Apply vertical transforms (e.g., plev19) to a single CMIP variable if required."""
    cfg = mapping.get_cfg(cmip_var) or {}
    levels = cfg.get("levels") or {}
    if (levels.get("name") or "").lower() != "plev19":
        return ds_var
    if not tables_path:
        raise ValueError("tables_path is required for plev19 vertical interpolation.")

    need = ["PS", "hyam", "hybm", "P0"]
    base = xr.Dataset({k: ds_native[k] for k in need if k in ds_native})
    base[cmip_var] = ds_var[cmip_var]
    v_plev = to_plev19(
        base, cmip_var, tables_path
    )  # returns dataset with var on 'plev'
    return xr.Dataset({cmip_var: v_plev[cmip_var]})


def _apply_vertical_if_needed_many(
    ds_vars: xr.Dataset,
    ds_native: xr.Dataset,
    cmip_vars: Sequence[str],
    mapping: Mapping,
    tables_path: Optional[Union[str, Path]],
) -> xr.Dataset:
    """Vectorized wrapper to apply vertical transforms per variable."""
    out = ds_vars.copy()
    for v in cmip_vars:
        out = out.update(
            _apply_vertical_if_needed(out[[v]], ds_native, v, mapping, tables_path)
        )
    return out


# ----------------------- single / multi var pipeline -----------------------


def realize_regrid_prepare(
    mapping: Mapping,
    ds_or_glob: Union[str, Path, xr.Dataset],
    cmip_var: str,
    *,
    tables_path: Optional[Union[str, Path]] = None,
    time_chunk: Optional[int] = 12,
    regrid_kwargs: Optional[dict] = None,
    open_kwargs: Optional[dict] = None,
) -> xr.Dataset:
    """
    Open native (if needed) → realize one CMIP variable → verticalize if needed → regrid to 1°.
    Returns an xr.Dataset with that single CMIP variable ready for CMOR.
    """
    regrid_kwargs = dict(regrid_kwargs or {})
    open_kwargs = dict(open_kwargs or {})

    if isinstance(ds_or_glob, xr.Dataset):
        ds_native = ds_or_glob
    else:
        ds_native = open_native_for_cmip_vars(
            [cmip_var], ds_or_glob, mapping, **open_kwargs
        )

    # realize -> ensure name is cmip_var
    ds_tmp = mapping.realize(ds_native, cmip_var)
    if isinstance(ds_tmp, xr.DataArray):
        ds_tmp = xr.Dataset({cmip_var: ds_tmp})
    if cmip_var not in ds_tmp:
        raise KeyError(f"Mapping produced no '{cmip_var}' variable.")

    if time_chunk and "time" in ds_tmp[cmip_var].dims:
        ds_tmp[cmip_var] = ds_tmp[cmip_var].chunk({"time": int(time_chunk)})

    # vertical transform if needed
    ds_vert = _apply_vertical_if_needed(
        ds_tmp, ds_native, cmip_var, mapping, tables_path
    )

    # regrid
    ds_regr = regrid_to_1deg_ds(ds_vert, cmip_var, time_from=ds_native, **regrid_kwargs)
    return ds_regr


def realize_regrid_prepare_many(
    mapping: Mapping,
    ds_or_glob: Union[str, Path, xr.Dataset],
    cmip_vars: Sequence[str],
    *,
    tables_path: Optional[Union[str, Path]] = None,
    time_chunk: Optional[int] = 12,
    regrid_kwargs: Optional[dict] = None,
    open_kwargs: Optional[dict] = None,
) -> xr.Dataset:
    """
    Open native (if needed) → realize all CMIP variables → verticalize if needed → regrid all to 1°.
    Returns an xr.Dataset with all requested CMIP variables ready for CMOR.
    """
    regrid_kwargs = dict(regrid_kwargs or {})
    open_kwargs = dict(open_kwargs or {})

    if isinstance(ds_or_glob, xr.Dataset):
        ds_native = ds_or_glob
    else:
        ds_native = open_native_for_cmip_vars(
            list(cmip_vars), ds_or_glob, mapping, **open_kwargs
        )

    # realize all
    realized = {}
    for v in cmip_vars:
        ds_v = mapping.realize(ds_native, v)
        if isinstance(ds_v, xr.DataArray):
            da = ds_v
        else:
            if v not in ds_v:
                raise KeyError(f"Mapping produced no '{v}' variable.")
            da = ds_v[v]
        if time_chunk and "time" in da.dims:
            da = da.chunk({"time": int(time_chunk)})
        realized[v] = da

    ds_vars = xr.Dataset(realized)

    # verticals
    ds_vert = _apply_vertical_if_needed_many(
        ds_vars, ds_native, cmip_vars, mapping, tables_path
    )

    # regrid together
    ds_regr = regrid_to_1deg_ds(
        ds_vert, list(cmip_vars), time_from=ds_native, **regrid_kwargs
    )
    return ds_regr
