# cmip7_prep/pipeline.py
"""High-level pipeline utilities to realize → regrid → prepare CMOR-ready datasets."""
from __future__ import annotations
from typing import Optional, Dict, Any, Set, List, Sequence
import re
from pathlib import Path
from glob import glob as _glob
import xarray as xr

from .mapping_compat import Mapping
from .regrid import regrid_to_1deg_ds

_VAR_TOKEN_RE = re.compile(r"[A-Za-z_]\w*")


def _cesm_vars_from_cfg(cfg: dict) -> Set[str]:
    """Extract CESM native variable names needed by a single CMIP var config."""
    out: Set[str] = set()

    # 1:1 source
    if isinstance(cfg.get("source"), str):
        out.add(cfg["source"])

    # explicit sources list
    for src in cfg.get("sources") or []:
        for key in ("cesm_var", "variable", "name"):
            v = src.get(key)
            if isinstance(v, str):
                out.add(v)
                break

    # raw/formula inputs
    for v in cfg.get("raw_variables") or []:
        if isinstance(v, str):
            out.add(v)

    # parse formula tokens (best-effort)
    formula = cfg.get("formula")
    if isinstance(formula, str):
        for tok in _VAR_TOKEN_RE.findall(formula):
            if tok not in {"np", "xr"}:
                out.add(tok)

    return out


def _filename_contains_var(name: str, var: str) -> bool:
    """Return True only if the filename contains the token `.var.` (dot on both sides)."""
    base = Path(name).name
    pat = re.compile(rf"\.{re.escape(var)}\.")
    return bool(pat.search(base))


def open_native_for_cmip_vars(
    cmip_vars: Sequence[str],
    files_glob: str | Path,
    mapping: Optional[Mapping] = None,
    use_cftime: bool = True,
    parallel: bool = True,
) -> xr.Dataset:
    """
    Open a native CESM timeseries Dataset containing only the variables needed
    to realize the requested CMIP variables.

    Parameters
    ----------
    cmip_vars : list[str]
        CMIP variable names (keys in cesm_to_cmip7.yaml), e.g. ["tas", "pr"].
    files_glob : str | Path
        Glob matching your native timeseries files (e.g., '/path/atm/hist_monthly/*.nc').
        We will filter this candidate set to files whose names include any required CESM vars.
    mapping : Mapping, optional
        If omitted, uses Mapping.from_packaged_default().
    use_cftime : bool
        Passed to xarray.open_mfdataset.
    parallel : bool
        Passed to xarray.open_mfdataset.

    Returns
    -------
    xr.Dataset
        Dataset with only the required native CESM variables (plus coords/bounds if present).
    """
    mapping = mapping or Mapping.from_packaged_default()

    # Collect required CESM variables across all requested CMIP variables
    required: Set[str] = set()
    missing: List[str] = []
    for cmv in cmip_vars:
        try:
            cfg = mapping.get_cfg(cmv)
        except KeyError:
            missing.append(cmv)
            continue
        required |= _cesm_vars_from_cfg(cfg)

    if missing:
        raise KeyError(f"No mapping found for: {missing}")

    if not required:
        raise ValueError(f"Mapping yielded no CESM variables for {cmip_vars}")

    # Expand the candidate list from the glob, then filter by variable token
    candidates = [Path(p) for p in _glob(str(files_glob))]
    selected = [
        p
        for p in candidates
        if any(_filename_contains_var(p.name, v) for v in required)
    ]

    if not selected:
        raise FileNotFoundError(
            f"No files under glob {files_glob!s} contained any of: {sorted(required)}"
        )

    # Open only the selected files
    ds = xr.open_mfdataset(
        [str(p) for p in sorted(set(selected))],
        combine="by_coords",
        use_cftime=use_cftime,
        parallel=parallel,
    )

    # Keep just the needed vars + helpful coords/bounds if present
    keep_vars = [v for v in sorted(required) if v in ds.data_vars]
    coord_like = [
        c
        for c in (
            "time_bounds",
            "time_bnds",
            "lat",
            "lon",
            "lev",
            "ilev",
            "hyam",
            "hybm",
            "P0",
            "PS",
            "lat_bnds",
            "lon_bnds",
        )
        if c in ds
    ]
    keep = sorted(set(keep_vars + coord_like + list(ds.coords)))
    return ds[keep]


def realize_regrid_prepare(
    mapping: Mapping,
    ds_native: xr.Dataset,
    cmip_var: str,
    *,
    time_chunk: Optional[int] = 12,
    regrid_kwargs: Optional[Dict[str, Any]] = None,
) -> xr.Dataset:
    """Realize a CMIP var from native CESM, chunk it, regrid to 1°, and attach time+bounds.

    Parameters
    ----------
    mapping : Mapping
        Loaded mapping (e.g., Mapping.from_packaged_default()).
    ds_native : xr.Dataset
        Native dataset containing the raw CESM variables.
    cmip_var : str
        CMIP short name to build (e.g., 'tas').
    time_chunk : int or None
        If provided and 'time' present, chunk the realized DataArray along time.
    regrid_kwargs : dict
        Passed through to regrid_to_1deg_ds (e.g., {'output_time_chunk': 12, 'dtype': 'float32'}).

    Returns
    -------
    xr.Dataset with the regridded variable and propagated time + time bounds.
    """
    regrid_kwargs = regrid_kwargs or {}
    da_native = mapping.realize(ds_native, cmip_var)

    # Optional chunking along time (keeps memory bounded)
    if time_chunk and "time" in da_native.dims:
        da_native = da_native.chunk({"time": time_chunk})

    ds_tmp = xr.Dataset({cmip_var: da_native})
    ds_regr = regrid_to_1deg_ds(
        ds_tmp,
        cmip_var,
        time_from=ds_native,
        **regrid_kwargs,
    )
    return ds_regr
