# cmip7_prep/pipeline.py
"""Open native MODEL timeseries, realize mappings, optional vertical transforms, and regrid."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union, Dict, List
import re
import logging
import xarray as xr

from .mapping_compat import Mapping
from .regrid import regrid_to_latlon_ds
from .vertical import to_plev

logger = logging.getLogger(__name__)
# --------------------------- file discovery ---------------------------

_VAR_TOKEN = re.compile(r"(?<![A-Za-z0-9_])([A-Za-z0-9_]+)(?![A-Za-z0-9_])")


def _filename_contains_var(
    path: Union[str, Path], var: str, fname_pattern: Optional[str] = None
) -> bool:
    """
    True if filename contains the variable as a dot-delimited token: '.var.'
    or matches the optional fname_pattern substring.

    Example matches: '...cam.h0.TS.0001-01.nc' contains '.TS.' -> True
    >>> _filename_contains_var("b.e30.fredscomp.cam.h1.TS.ne30pg3.001.nc", "TS")
    True
    >>> _filename_contains_var("b.e30.fredscomp.cam.h1.PS.ne30pg3.001.nc", "TS")
    False
    >>> _filename_contains_var(Path("b.e30.fredscomp.cam.h1.TS.ne30pg3.001.nc"), "TS")
    True
    >>> _filename_contains_var("file.TS.001.nc", "TS", fname_pattern=".001.")
    True
    """
    name = Path(path).name
    needle = f".{var}."
    if fname_pattern is not None:
        return fname_pattern in name and needle in name
    logger.debug("Checking if filename '%s' contains variable token '%s'", name, needle)
    return needle in name


def _collect_required_model_vars(
    mapping: Mapping, cmip_vars: Sequence[str]
) -> List[str]:
    """Gather all native model vars needed to realize the requested CMIP vars."""
    needed: set[str] = set()
    for var in cmip_vars:
        try:
            cfg = mapping.get_cfg(var) or {}
        except KeyError:
            logger.warning("Skipping '%s': no mapping found in %s", var, mapping.path)
            continue
        src = cfg.get("source")
        raws = cfg.get("raw_variables") or cfg.get("sources") or []
        if src:
            needed.add(src)
        for raw in raws:
            # 'sources' items may be dicts with 'model_var'
            if isinstance(raw, dict) and "model_var" in raw:
                needed.add(raw["model_var"])
            elif isinstance(raw, str):
                needed.add(raw)
        # vertical dependencies if plev19 or plev39
        levels = cfg.get("levels") or {}
        if "plev" in (levels.get("name") or "").lower():
            needed.update({"PS", "hyam", "hybm", "P0"})
        elif (levels.get("name") or "").lower() == "standard_hybrid_sigma":
            needed.update({"PS", "hyam", "hybm", "hyai", "hybi", "P0", "ilev"})
        for var in ("area", "landmask", "landfrac"):
            logger.info("Adding auxiliary variable '%s' ", var)
            needed.add(var)
    return sorted(needed)


def open_native_for_cmip_vars(
    cmip_vars: Sequence[str],
    files: Union[str, Path],
    mapping: Mapping,
    *,
    use_cftime: bool = True,
    parallel: bool = False,
    open_kwargs: Optional[Dict] = None,
) -> xr.Dataset:
    """
    Open timeseries files needed for the requested CMIP variables.

    Parameters
    ----------
    cmip_vars : list of CMIP variable names (e.g., ["tas", "ta"])
    files : list of native timeseries files
                 (e.g., "/path/atm/hist_monthly/*cam.h0*")
    mapping : Mapping object that knows how to realize CMIP vars
    use_cftime, parallel : forwarded to xarray.open_mfdataset
    open_kwargs : extra kwargs for open_mfdataset

    Returns
    -------
    xr.Dataset containing only the required variables.
    """
    open_kwargs = dict(open_kwargs or {})
    # Allow cmip_vars to be a single variable (str) or a list
    if not isinstance(cmip_vars, list):
        cmip_vars = [cmip_vars]
    new_cmip_vars = []

    for var in cmip_vars:
        logger.info("Processing CMIP var collecting model vars'%s'", var)
        rvar = _collect_required_model_vars(mapping, [var])
        logger.info(
            "Looking for native files for CMIP var '%s' needing model vars: %s",
            var,
            rvar,
        )
        selected = sorted(
            {p for p in files if any(_filename_contains_var(p, v) for v in rvar)}
        )
        if selected:
            new_cmip_vars.append(var)
    required = _collect_required_model_vars(mapping, new_cmip_vars)

    # keep any file that contains ANY of the required model vars as '.var.' in the name
    selected = sorted(
        {str(p) for p in files if any(_filename_contains_var(p, v) for v in required)}
    )

    if not selected:
        logger.warning(
            "no native inputs found for requested CMIP variables: %s", cmip_vars
        )
        return None, None

    ds = xr.open_mfdataset(
        selected,
        combine="by_coords",
        use_cftime=use_cftime,
        parallel=parallel,
        data_vars="minimal",
        compat="equals",
        **open_kwargs,
    )
    # Convert "lev" and "ilev" units from mb to Pa for downstream operations.
    if "lev" in ds:
        ds["lev"] = ds["lev"] / 1000
    if "ilev" in ds:
        ds["ilev"] = ds["ilev"] / 1000

    logger.info("Returning from open_native_for_cmip_vars")
    return ds, new_cmip_vars


# ----------------------- realization / vertical -----------------------
#    ds_vert = _apply_vertical_if_needed(ds_vars, cmip_var, cfg, mapping, tables_path=tables_path)


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
    logger.info("levels for %s: %s", cmip_var, levels)
    if "plev" not in (levels.get("name") or "").lower():
        return ds_var
    if not tables_path:
        raise ValueError("tables_path is required for plev19 vertical interpolation.")

    need = ["PS", "hyam", "hybm", "P0"]
    base = xr.Dataset({k: ds_native[k] for k in need if k in ds_native})

    base[str(cmip_var)] = ds_var[str(cmip_var)]
    logger.info("Applying plev19 vertical transform for variable: %s", cmip_var)

    v_plev = to_plev(
        ds=base, var=cmip_var, tables_path=tables_path, target=levels.get("name")
    )  # returns dataset with var on 'plev'
    logger.info("After vertical transform : %s", cmip_var)
    return xr.Dataset({str(cmip_var): v_plev[str(cmip_var)]})


# ----------------------- single / multi var pipeline -----------------------
# pylint: disable=unused-argument


def realize_regrid_prepare(
    resolution: str,
    model: str,
    mapping: Mapping,
    ds_or_glob: Union[str, Path, xr.Dataset],
    cmip_var: str,
    *,
    tables_path: Optional[Union[str, Path]] = None,
    time_chunk: Optional[int] = 12,
    mom6_grid: Optional[Dict[str, xr.DataArray]] = None,
    regrid_kwargs: Optional[dict] = None,
    open_kwargs: Optional[dict] = None,
) -> xr.Dataset:
    """Open native (if needed) → realize one CMIP variable → verticalize if needed → regrid to 1°.

    Returns an xr.Dataset with the requested CMIP variable ready for CMOR.
    Ensures hybrid-σ auxiliaries (hyam, hybm, P0, PS, lev/ilev) are present when needed.
    """

    regrid_kwargs = dict(regrid_kwargs or {})
    open_kwargs = dict(open_kwargs or {})
    aux = []

    # 1) Get native dataset
    if isinstance(ds_or_glob, xr.Dataset):
        ds_native = ds_or_glob
    else:
        # Use your existing native opener (don’t add PS here to avoid mapping lookups)
        ds_native = open_native_for_cmip_vars(
            [cmip_var], ds_or_glob, mapping, **open_kwargs
        )
    logger.info("Opened native dataset with dims: %s", ds_native.dims)

    # 2) Realize the target variable
    ds_v = mapping.realize(ds_native, cmip_var)
    da = ds_v if isinstance(ds_v, xr.DataArray) else ds_v[cmip_var]
    if time_chunk and "time" in da.dims:
        da = da.chunk({"time": int(time_chunk)})

    ds_vars = xr.Dataset({cmip_var: da})
    for var in ("landfrac", "area", "landmask", "wet"):
        if var in ds_native and var not in ds_vars:
            ds_vars = ds_vars.assign(**{var: ds_native[var]})

    # 3) Check whether hybrid-σ is required
    cfg = mapping.get_cfg(cmip_var) or {}
    logger.info("Mapping cfg for %s: %s", cmip_var, cfg)
    levels = cfg.get("levels", {}) or {}
    lev_kind = (levels.get("name") or "").lower()
    is_hybrid = lev_kind in {"standard_hybrid_sigma", "alev", "alevel"}

    # 4) If hybrid: carry PS in the working dataset (so we can regrid it)
    # and make sure 1-D coefficients are available
    if is_hybrid:
        # PS is on (time,ncol) natively; add it so regridding
        # can produce PS(time,lat,lon)
        if "PS" in ds_native and "PS" not in ds_vars:
            ds_vars = ds_vars.assign(PS=ds_native["PS"])
        aux = [
            nm
            for nm in ("hyai", "hybi", "hyam", "hybm", "P0", "ilev", "lev")
            if nm in ds_native
        ]

    # 5) Apply vertical transform if needed (plev19, etc.).
    #    Single-var helper already takes cfg + tables_path
    logger.info("Applying vertical handling if needed for variable: %s", cmip_var)
    ds_vert = _apply_vertical_if_needed(
        ds_vars, ds_native, cmip_var, mapping, tables_path=tables_path
    )
    logger.info("After vertical handling, dataset dims: %s", ds_vert.dims)

    # 6) Regrid (include PS if present)
    names_to_regrid = [str(cmip_var)]
    if is_hybrid and "PS" in ds_vert:
        names_to_regrid.append("PS")

    # 7) Rename levgrnd if present to sdepth
    logger.info(
        "Checking for 'levgrnd' dimension in variables to regrid. %s", names_to_regrid
    )
    for lev in ("levgrnd", "levsoi"):
        if lev in ds_vert.dims:
            logger.info("Renaming '%s' dimension to 'sdepth'", lev)
            ds_vert = ds_vert.rename_dims({lev: "sdepth"})
            # Ensure the coordinate variable is also copied
            ds_vert = ds_vert.assign_coords(sdepth=ds_native[lev].values)

    # 8) Regrid to lat/lon
    logger.info("Calling regrid_to_latlon_ds")
    ds_regr = regrid_to_latlon_ds(
        ds_vert,
        names_to_regrid,
        resolution,
        model,
        time_from=ds_native,
        **regrid_kwargs,
    )
    logger.info("Regridded dataset dims: %s", ds_regr.dims)
    if aux:
        ds_regr = ds_regr.merge(ds_native[aux], compat="override")

    return ds_regr
