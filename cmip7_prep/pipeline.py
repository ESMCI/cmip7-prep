# cmip7_prep/pipeline.py
"""Open native CESM timeseries, realize mappings, optional vertical transforms, and regrid."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union, Dict, List
import re
import warnings
import glob
import sys
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
    >>> _filename_contains_var("b.e30.fredscomp.cam.h1.TS.ne30pg3.001.nc", "TS")
    True
    >>> _filename_contains_var("b.e30.fredscomp.cam.h1.PS.ne30pg3.001.nc", "TS")
    False
    >>> _filename_contains_var(Path("b.e30.fredscomp.cam.h1.TS.ne30pg3.001.nc"), "TS")
    True
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
        try:
            cfg = mapping.get_cfg(v) or {}
            print(f"v is {v} cfg is {cfg}")
        except KeyError:
            print(
                f"WARNING: skipping '{v}': no mapping found in {mapping.path}",
                file=sys.stderr,
            )
            continue
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
        elif (levels.get("name") or "").lower() == "standard_hybrid_sigma":
            needed.update({"PS", "hyam", "hybm", "hyai", "hybi", "P0", "ilev"})
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
    new_cmip_vars = []
    for var in cmip_vars:
        rvar = _collect_required_cesm_vars(mapping, [var])
        print(f"var {var} rvar {rvar}")
        candidates = glob.glob(str(files_glob))
        selected = sorted(
            {p for p in candidates if any(_filename_contains_var(p, v) for v in rvar)}
        )
        if var == "evspsblsoi":
            print(f"rvar = {rvar}, selected={selected} candidates = {candidates}")
        if selected:
            new_cmip_vars.append(var)
        else:
            warnings.warn(
                f"[mapping] missing native inputs for {var} - skipping",
                RuntimeWarning,
            )
    required = _collect_required_cesm_vars(mapping, new_cmip_vars)

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
    # Convert "lev" and "ilev" units from mb to Pa for downstream operations.
    if "lev" in ds:
        ds["lev"] = ds["lev"] / 1000
    if "ilev" in ds:
        ds["ilev"] = ds["ilev"] / 1000

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
    """Open native (if needed) → realize one CMIP variable → verticalize if needed → regrid to 1°.

    Returns an xr.Dataset with the requested CMIP variable ready for CMOR.
    Ensures hybrid-σ auxiliaries (hyam, hybm, P0, PS, lev/ilev) are present when needed.
    """
    regrid_kwargs = dict(regrid_kwargs or {})
    open_kwargs = dict(open_kwargs or {})

    # 1) Get native dataset
    if isinstance(ds_or_glob, xr.Dataset):
        ds_native = ds_or_glob
    else:
        # Use your existing native opener (don’t add PS here to avoid mapping lookups)
        ds_native = open_native_for_cmip_vars(
            [cmip_var], ds_or_glob, mapping, **open_kwargs
        )

    # 2) Realize the target variable
    ds_v = mapping.realize(ds_native, cmip_var)
    da = ds_v if isinstance(ds_v, xr.DataArray) else ds_v[cmip_var]
    if time_chunk and "time" in da.dims:
        da = da.chunk({"time": int(time_chunk)})

    ds_vars = xr.Dataset({cmip_var: da})

    if "area" in ds_native and "area" not in ds_vars and "ncol" in ds_native.dims:
        ds_vars = ds_vars.assign(area=ds_native["area"])
    if "landfrac" in ds_native and "landfrac" not in ds_vars:
        ds_vars = ds_vars.assign(landfrac=ds_native["landfrac"])
    print(f"ds_vars {ds_vars}")
    # 3) Check whether hybrid-σ is required
    cfg = mapping.get_cfg(cmip_var) or {}
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

    # 5) Apply vertical transform if needed (plev19, etc.).
    # Single-var helper already takes cfg + tables_path
    ds_vert = _apply_vertical_if_needed(
        ds_vars, ds_native, cmip_var, mapping, tables_path=tables_path
    )

    # 6) Regrid (include PS if present)
    names_to_regrid = [cmip_var]
    if is_hybrid and "PS" in ds_vert:
        names_to_regrid.append("PS")

    ds_regr = regrid_to_1deg_ds(
        ds_vert, names_to_regrid, time_from=ds_native, **regrid_kwargs
    )

    print(f"Here 1 {'areacella' in ds_regr}")
    # 7) If hybrid: merge in 1-D hybrid coefficients directly from native (no regridding needed)
    if is_hybrid:
        aux = [
            nm
            for nm in ("hyai", "hybi", "hyam", "hybm", "P0", "ilev", "lev")
            if nm in ds_native
        ]
        if aux:
            ds_regr = ds_regr.merge(ds_native[aux], compat="override")

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
    Open native (if needed) → realize requested CMIP variables →
    ensure hybrid-σ support fields are present (PS & coeffs) →
    apply vertical handling for CMIP vars only → regrid CMIP vars + PS →
    return 1° dataset that still includes hybrid coeffs (unmodified).
    """
    regrid_kwargs = dict(regrid_kwargs or {})
    open_kwargs = dict(open_kwargs or {})

    # 1) Open native once
    if isinstance(ds_or_glob, xr.Dataset):
        ds_native = ds_or_glob
    else:
        ds_native = open_native_for_cmip_vars(
            list(cmip_vars), ds_or_glob, mapping, **open_kwargs
        )

    # 2) Realize each CMIP var and overlay onto native so we keep auxiliaries
    realized: dict[str, xr.DataArray] = {}
    for v in cmip_vars:
        ds_v = mapping.realize(ds_native, v)
        da = ds_v if isinstance(ds_v, xr.DataArray) else ds_v[v]
        if time_chunk and "time" in da.dims:
            da = da.chunk({"time": int(time_chunk)})
        realized[v] = da

    ds_tmp = ds_native.assign(**realized)

    # Promote vertical axes to coords if present (helps CF/CMOR)
    for name in ("lev", "ilev"):
        if name in ds_tmp:
            ds_tmp = ds_tmp.set_coords(name)

    # 3) Detect if any requested CMIP var uses hybrid-σ and, if so, prepare support
    regrid_vars = set(cmip_vars)  # variables to actually regrid (lat/lon)
    carry_along: dict[str, xr.DataArray] = {}  # coeffs kept unchanged (1D)

    for v in cmip_vars:
        cfg = mapping.get_cfg(v) or {}
        levels = cfg.get("levels") or {}
        kind = str(levels.get("name", "")).lower()
        if kind in {"standard_hybrid_sigma", "alev", "alevel"}:
            # names as specified in mapping (with robust defaults)
            ps_name = levels.get("ps", "PS")
            hyam_name = levels.get("hyam", "hyam")
            hybm_name = levels.get("hybm", "hybm")
            p0_name = levels.get("P0", "P0")
            ilev_name = levels.get("src_axis_bnds", "ilev")
            lev_name = levels.get("src_axis_name", "lev")

            # PS must be regridded (time,lat,lon); include only if present
            if ps_name in ds_tmp:
                regrid_vars.add(ps_name)

            # Hybrid coeffs are 1D; just carry them through unchanged if present
            for aux in (hyam_name, hybm_name, p0_name, ilev_name, lev_name):
                if aux in ds_tmp and aux not in carry_along:
                    carry_along[aux] = ds_tmp[aux]

    # 4) Vertical handling for CMIP vars only (do NOT include PS here)
    ds_vert = _apply_vertical_if_needed_many(
        ds_tmp, ds_native, list(cmip_vars), mapping, tables_path=tables_path
    )

    # Make sure hybrid coeffs (1D) are still present for CMOR writer
    if carry_along:
        ds_vert = ds_vert.assign(
            **{k: v for k, v in carry_along.items() if k not in ds_vert}
        )

    # 5) Regrid CMIP vars + PS (PS will be present if required & found above)
    ds_regr = regrid_to_1deg_ds(
        ds_vert, list(sorted(regrid_vars)), time_from=ds_native, **regrid_kwargs
    )

    # Reattach coeffs after regridding (they're 1D; no regridding needed)
    if carry_along:
        missing = {k: v for k, v in carry_along.items() if k not in ds_regr}
        if missing:
            ds_regr = ds_regr.assign(**missing)

    return ds_regr
