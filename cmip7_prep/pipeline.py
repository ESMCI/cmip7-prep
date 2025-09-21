# cmip7_prep/pipeline.py
"""High-level pipeline utilities to realize → regrid → prepare CMOR-ready datasets."""
from __future__ import annotations
from typing import Optional, Dict, Any
import xarray as xr

from .mapping_compat import Mapping
from .regrid import regrid_to_1deg_ds


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
