"""Thin CMOR wrapper used by cmip7_prep.

This module centralizes CMOR session setup and writing so that the rest of the
pipeline can stay xarray-first. It supports either a dataset JSON file (preferred)
or directly injected global attributes, and creates axes based on the coordinates
present in the provided dataset. It also supports a packaged default
`cmor_dataset.json` living under `cmip7_prep/data/`.
"""

from pathlib import Path
import collections.abc
import json
import tempfile
import types
import warnings

from contextlib import AbstractContextManager
from typing import Any, Sequence, Optional, Union
import datetime as dt

import logging
import cmor

import numpy as np
import xarray as xr
from .cmor_utils import (
    packaged_dataset_json,
    get_cmor_attr,
    set_cmor_attr,
    encode_time_to_num,
    sigma_mid_and_bounds,
    bounds_from_centers_1d,
    resolve_table_filename,
    filled_for_cmor,
    open_existing_fx,
)

# from .mom6_static import compute_cell_bounds_from_corners

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DatasetJsonLike = Union[str, Path, AbstractContextManager]


# ---------------------------------------------------------------------
# CMOR session
# ---------------------------------------------------------------------
class CmorSession(
    AbstractContextManager
):  # pylint: disable=too-many-instance-attributes
    """Context manager for CMOR sessions.

    Parameters
    ----------
    tables_path : str or Path
        Directory containing CMOR table JSONs (e.g., CMIP6_*.json or CMIP7_*.json).
    dataset_json : str or Path, optional
        Path to a cmor_dataset.json with the experiment/source metadata.
    dataset_attrs : dict, optional
        Alternative to `dataset_json`, allowing direct attribute injection.
    """

    def __init__(
        self,
        *,
        tables_path: Path | str,
        dataset_attrs: dict[str, str] | None = None,
        dataset_json: Optional[DatasetJsonLike] = None,
        tracking_prefix: str | None = None,
        # NEW: one log per run (session)
        log_dir: Path | str | None = None,
        log_name: str | None = None,
        outdir: Path | str | None = None,
    ) -> None:
        self.tables_path = Path(tables_path)
        self.dataset_attrs = dict(dataset_attrs or {})
        self.dataset_json = dataset_json
        self._dataset_json_cm = None
        self.tracking_prefix = tracking_prefix
        # logging config
        self._log_dir = Path(log_dir) if log_dir is not None else None
        self._log_name = log_name
        self._log_path: Path | None = None
        self._pending_ps = None

        self._outdir = Path(outdir or "./CMIP7").resolve()
        self._outdir.mkdir(parents=True, exist_ok=True)
        self._fx_written: set[str] = (
            set()
        )  # remembers which fx vars were written this run
        self._fx_cache: dict[str, xr.DataArray] = (
            {}
        )  # regridded fx fields cached in-memory

        # Attach grid mapping attribute to variable if ocean

    def __enter__(self) -> "CmorSession":
        # Resolve logfile path if requested
        if self._log_dir is not None:
            ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            fname = self._log_name or f"cmor_{ts}.log"
            self._log_dir.mkdir(parents=True, exist_ok=True)
            self._log_path = (self._log_dir / fname).resolve()

        # Setup CMOR; pass logfile if CMOR supports it, else fall back
        try:
            if self._log_path is not None:
                logger.info("CMOR logfile: %s", self._log_path)
                cmor.setup(
                    inpath=str(self.tables_path),
                    netcdf_file_action=getattr(
                        cmor, "CMOR_REPLACE_3", getattr(cmor, "CMOR_REPLACE", 3)
                    ),
                    set_verbosity=cmor.CMOR_NORMAL,
                    logfile=str(self._log_path),  # supported by newer CMOR builds
                )
            else:
                cmor.setup(
                    inpath=str(self.tables_path),
                    netcdf_file_action=getattr(
                        cmor, "CMOR_REPLACE_3", getattr(cmor, "CMOR_REPLACE", 3)
                    ),
                    set_verbosity=cmor.CMOR_NORMAL,
                )
        except TypeError:
            # Older CMOR with no 'logfile' kw
            cmor.setup(
                inpath=str(self.tables_path),
                netcdf_file_action=getattr(
                    cmor, "CMOR_REPLACE_3", getattr(cmor, "CMOR_REPLACE", 3)
                ),
                set_verbosity=cmor.CMOR_NORMAL,
            )
            # best-effort fallback setter
            for name in ("set_logfile", "setLogFile", "set_log_file", "setLogfile"):
                if self._log_path is not None and hasattr(cmor, name):
                    getattr(cmor, name)(str(self._log_path))
                    break

        # Resolve dataset_json to a real filesystem path
        dj = self.dataset_json
        if dj is None:
            # packaged file → returns a context manager
            cm = packaged_dataset_json("cmor_dataset.json")
            self._dataset_json_cm = cm
            p = cm.__enter__()  # ← ENTER the CM, get a Path
        elif isinstance(dj, (str, Path)):
            p = Path(dj)
        else:
            # caller passed a context manager directly
            self._dataset_json_cm = dj
            p = dj.__enter__()  # ← ENTER the CM, get a Path

        with open(p, encoding="utf-8") as f:
            cfg = json.load(f)
        cfg["outpath"] = str(self._outdir)
        tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(cfg, tmp)
        tmp.close()
        cmor.dataset_json(str(tmp.name))

        try:
            prod = cmor.get_cur_dataset_attribute("product")  # type: ignore[attr-defined]
        except Exception:  # pylint: disable=broad-except
            prod = None
        if prod != "model-output":
            cmor.set_cur_dataset_attribute("product", "model-output")

        # long paragraph; split to keep lines < 100
        inst = get_cmor_attr("institution_id") or "NCAR"
        license_text = (
            f"CMIP6 model data produced by {inst} is licensed under a Creative Commons "
            "Attribution 4.0 International License "
            "(https://creativecommons.org/licenses/by/4.0/). "
            "Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use "
            "governing CMIP6 output, including citation requirements and proper "
            "acknowledgment. Further information about this data, including some "
            "limitations, can be found via the further_info_url (recorded as a "
            "global attribute in this file). The data producers and data providers "
            "make no warranty, either express or implied, including, but not "
            "limited to, warranties of merchantability and fitness for a "
            "particular purpose. All liabilities arising from the supply of the "
            "information (including any liability arising in negligence) are "
            "excluded to the fullest extent permitted by law."
        )

        cmor.set_cur_dataset_attribute("license", license_text)

        # Tell CMOR how to build tracking_id; let CMOR generate it
        prefix = get_cmor_attr("tracking_prefix")
        if not (isinstance(prefix, str) and prefix.startswith("hdl:")):
            set_cmor_attr("tracking_prefix", "hdl:21.14100/")

        # If a non-handle tracking_id snuck in (e.g., bare UUID from JSON), clear it
        tid = get_cmor_attr("tracking_id")
        if isinstance(tid, str) and not tid.startswith("hdl:21.14100/"):
            set_cmor_attr(
                "tracking_id", ""
            )  # empty lets CMOR regenerate from tracking_prefix

        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            cmor.close()  # or your per-var close logic
        finally:
            if self._dataset_json_cm is not None:
                # Close the context manager we entered in __enter__
                self._dataset_json_cm.__exit__(exc_type, exc, tb)
                self._dataset_json_cm = None

    # -------------------------
    # internal helpers
    # -------------------------
    def _define_axes(self, ds: xr.Dataset, vdef: any) -> list[int]:
        """Define CMOR axis IDs for the variable in ds according to the CMOR tables.

        Rules:
        - time: numeric coord + optional bounds
        - horizontal: table_entry='latitude'/'longitude', with bounds (synth if missing)
        - hybrid sigma: table_entry='alev' defined by length only (+ z-factors a, b, p0, ps)
        - pressure levels: table_entry='plev' with coord_vals (+ optional bounds)
        - axis_ids order MUST match the order of ds[vdef.name].dims
        """

        # ---- helpers ----
        def _get_time_and_bounds(dsi: xr.Dataset):
            time_da = (
                dsi.coords["time"]
                if "time" in dsi.coords
                else (dsi["time"] if "time" in dsi else None)
            )
            if time_da is None:
                return None, None, None
            units = time_da.attrs.get("units", "days since 1850-01-01")
            cal = time_da.attrs.get(
                "calendar", time_da.encoding.get("calendar", "noleap")
            )
            tvals = encode_time_to_num(time_da, units, cal)

            bname = time_da.attrs.get("bounds")
            tb = (
                dsi[bname]
                if isinstance(bname, str) and bname in dsi
                else (
                    dsi["time_bounds"]
                    if "time_bounds" in dsi
                    else (dsi["time_bnds"] if "time_bnds" in dsi else None)
                )
            )
            tbnum = encode_time_to_num(tb, units, cal) if tb is not None else None
            return tvals, tbnum, str(units)

        def _get_1d_with_bounds(dsi: xr.Dataset, name: str, units_default: str):
            da = dsi.coords[name] if name in dsi.coords else dsi[name]
            vals = np.asarray(da.values, dtype="f8")
            units = da.attrs.get("units", units_default)

            # try common names; else synthesize conservative bounds
            cand = (f"{name}_bnds", f"{name}_bounds")
            b = None
            for cn in cand:
                if cn in dsi:
                    b = np.asarray(dsi[cn].values, dtype="f8")
                    break
            if b is None:
                dv = np.diff(vals)
                if vals.size >= 2:
                    lower = np.concatenate(
                        ([vals[0] - dv[0] / 2.0], vals[:-1] + dv / 2.0)
                    )
                    upper = np.concatenate(
                        (vals[1:] - dv / 2.0, [vals[-1] + dv[-1] / 2.0])
                    )
                    b = np.stack([lower, upper], axis=-1)
                else:
                    b = np.array([[vals[0] - 0.5, vals[0] + 0.5]], dtype="f8")

            return vals, b, units

        axes_ids = []
        var_name = getattr(vdef, "name", None)
        if var_name is None or var_name not in ds:
            raise KeyError(f"Variable to write not found in dataset: {var_name!r}")
        var_da = ds[var_name]
        var_dims = list(var_da.dims)
        alev_id = None
        plev_id = None
        lat_id = None
        lon_id = None
        sdepth_id = None

        logger.debug("[CMOR axis debug] var_dims: %s", var_dims)
        if "xh" in var_dims and "yh" in var_dims:
            # MOM6/curvilinear grid: register xh/yh as generic axes (i/j), not as lat/lon
            # Define the native grid using the coordinate arrays
            logger.debug(
                "[CMOR axis debug] Defining unstructured grid for variable %s.",
                var_name,
            )

            i_id = cmor.axis(
                table_entry="i",
                units="1",
                length=ds["xh"].size,
            )
            j_id = cmor.axis(
                table_entry="j",
                units="1",
                length=ds["yh"].size,
            )
            logger.debug("[CMOR axis debug] Defining unstructured grid_id.")
            grid_id = cmor.grid(
                axis_ids=[j_id, i_id],  # note CMOR wants fastest varying last
                longitude=ds["geolon"].values,
                latitude=ds["geolat"].values,
                longitude_vertices=ds["geolon_c"].values,
                latitude_vertices=ds["geolat_c"].values,
            )

            # No lat/lon axis registration for curvilinear grid
            # 2D geolat/geolon should be written as auxiliary variables elsewhere
            # Map axes for dimension order
            dim_to_axis = {"time": None, "i": i_id, "j": j_id, "xh": i_id, "yh": j_id}
            axes_ids = []
            for d in var_dims:
                axis_id = dim_to_axis.get(d)
                if axis_id is None and d != "time":
                    raise KeyError(
                        f"No axis ID found for dimension '{d}'"
                        f" in variable '{var_name}' (curvilinear)"
                    )
                axes_ids.append(axis_id)
            logger.debug("[CMOR axis debug] Appending grid_id: %s", grid_id)
            axes_ids.append(grid_id)
            logger.debug("[CMOR axis debug] axes_ids: %s", axes_ids)
            return axes_ids

        # --- horizontal axes (use CMOR names) ----
        if "lat" not in ds or "lon" not in ds:
            raise KeyError(
                "Expected 'lat' and 'lon' in dataset for CMOR horizontal axes."
            )
        lat_vals, lat_bnds, _ = _get_1d_with_bounds(ds, "lat", "degrees_north")
        lon_vals, lon_bnds, _ = _get_1d_with_bounds(ds, "lon", "degrees_east")

        logger.info("write lat axis")
        lat_id = cmor.axis(
            table_entry="latitude",
            units="degrees_north",
            coord_vals=lat_vals,
            cell_bounds=lat_bnds,
        )
        logger.info("write lon axis")
        lon_id = cmor.axis(
            table_entry="longitude",
            units="degrees_east",
            coord_vals=lon_vals,
            cell_bounds=lon_bnds,
        )

        # ---- time axis ----
        time_id = None
        tvals, tbnds, t_units = _get_time_and_bounds(ds)
        if tvals is not None:
            logger.info("write time axis")
            time_id = cmor.axis(
                table_entry="time",
                units=t_units,
                coord_vals=tvals,
                cell_bounds=tbnds if tbnds is not None else None,
            )
        # --- vertical: standard_hybrid_sigma ---
        levels = getattr(vdef, "levels", {}) or {}

        if (levels.get("name") or "").lower() in {
            "standard_hybrid_sigma",
            "alevel",
            "alev",
        } or "lev" in var_dims:
            # names in the native ds
            hyam_name = levels.get("hyam", "hyam")  # A mid (dimensionless)
            hybm_name = levels.get("hybm", "hybm")  # B mid (dimensionless)
            hyai_name = levels.get("hyai", "hyai")  # A interfaces (optional)
            hybi_name = levels.get("hybi", "hybi")  # B interfaces (preferred)
            ps_name = levels.get("ps", "PS")

            # 0) sigma mid and bounds (0..1)
            sigma_mid, sigma_bnds = sigma_mid_and_bounds(ds, levels)

            # 1) define axis using sigma
            alev_id = cmor.axis(
                table_entry="standard_hybrid_sigma",
                units="1",
                coord_vals=sigma_mid,
                cell_bounds=sigma_bnds,
            )

            # 2) z-factors: a(lev), b(lev), p0(scalar), ps(time,lat,lon)

            cmor.zfactor(
                zaxis_id=alev_id,
                zfactor_name="a",
                units="1",
                axis_ids=[alev_id],
                zfactor_values=np.asarray(ds[hyam_name].values),
                zfactor_bounds=np.asarray(ds[hyai_name].values),
            )
            cmor.zfactor(
                zaxis_id=alev_id,
                zfactor_name="b",
                units="1",
                axis_ids=[alev_id],
                zfactor_values=np.asarray(ds[hybm_name].values),
                zfactor_bounds=np.asarray(ds[hybi_name].values),
            )

            # p0 scalar
            cmor.zfactor(
                zaxis_id=alev_id, zfactor_name="p0", units="Pa", zfactor_values=1.0e5
            )
            # ps(time,lat,lon) zfactor must be DEFINED before the main variable
            ps_da = ds[ps_name]  # ensure units are Pa
            ps_zvar_id = cmor.zfactor(
                zaxis_id=alev_id,
                zfactor_name="ps",
                units="Pa",
                axis_ids=[
                    time_id,
                    lat_id,
                    lon_id,
                ],  # order must match var’s non-vertical axes
            )
            # stash to write before main variable
            self._pending_ps = (ps_zvar_id, ps_da)

        elif "plev" in var_dims:
            # Pressure levels expected in data
            plev = ds["plev"]
            pvals = np.asarray(plev.values, dtype="f8")
            punits = str(plev.attrs.get("units", "")).strip().lower()

            # Optional bounds
            pb = None
            for nm in ("plev_bnds", "plev_bounds"):
                if nm in ds:
                    pb = np.asarray(ds[nm].values, dtype="f8")
                    break

            # --- NEW: normalize to Pa ---
            # Convert hPa/mb/etc → Pa;
            # if units missing but magnitudes look like hPa (<= ~2000), assume hPa
            if punits in {"hpa", "mb", "mbar", "millibar"} or (
                punits == "" and (np.nanmax(pvals) if pvals.size else 0.0) <= 2000.0
            ):
                pvals = pvals * 100.0
                if pb is not None:
                    pb = pb * 100.0
            # (if punits already "pa" or values are already in Pa, do nothing)

            # Debug (optional)
            if pvals.size == 19:
                table_entry = "plev19"
            else:
                table_entry = "plev"

            levels = getattr(vdef, "levels", None) or {}
            name = levels.get("name")
            if isinstance(name, str):
                key = name.strip()
                # Accept entries like plev19, plev7h, plev27, etc.
                if key.lower().startswith("plev"):
                    table_entry = key

            plev_id = cmor.axis(
                table_entry=table_entry,
                units="Pa",
                coord_vals=pvals,
                cell_bounds=pb if pb is not None else None,
            )
        elif "sdepth" in var_dims:
            values = ds["sdepth"].values
            logger.info("write sdepth axis")
            bnds = bounds_from_centers_1d(values, "sdepth")
            if bnds[0, 0] < 0:
                bnds[0, 0] = 0.0  # no negative soil depth bounds
            sdepth_id = cmor.axis(
                table_entry="sdepth",
                units="m",
                coord_vals=np.asarray(values),
                cell_bounds=bnds,
            )
        # Map dimension names to axis IDs
        dim_to_axis = {
            "time": time_id,
            "alev": alev_id,  # hybrid sigma
            "lev": alev_id,  # sometimes used for hybrid
            "sdepth": sdepth_id,
            "plev": plev_id,
            "lat": lat_id,
            "latitude": lat_id,
            "lon": lon_id,
            "longitude": lon_id,
            "xh": lon_id,  # MOM6
            "yh": lat_id,  # MOM6
        }
        axes_ids = []
        for d in var_dims:
            axis_id = dim_to_axis.get(d)
            if axis_id is None:
                raise KeyError(
                    f"No axis ID found for dimension '{d}' in variable '{var_name}'"
                )
            axes_ids.append(axis_id)
        return axes_ids

    def _write_fx_2d(self, ds: xr.Dataset, name: str, units: str) -> None:
        if name not in ds:
            return
        table_filename = resolve_table_filename(self.tables_path, "fx")
        cmor.load_table(table_filename)

        lat = ds["lat"].values
        lon = ds["lon"].values
        lat_b = ds.get("lat_bnds")
        lon_b = ds.get("lon_bnds")
        lat_b = (
            lat_b.values
            if isinstance(lat_b, xr.DataArray)
            else bounds_from_centers_1d(lat, "lat")
        )
        lon_b = (
            lon_b.values
            if isinstance(lon_b, xr.DataArray)
            else bounds_from_centers_1d(lon, "lon")
        )

        lat_id = cmor.axis(
            "latitude", "degrees_north", coord_vals=lat, cell_bounds=lat_b
        )
        lon_id = cmor.axis(
            "longitude", "degrees_east", coord_vals=lon, cell_bounds=lon_b
        )
        data_filled, fillv = filled_for_cmor(ds[name])

        var_id = cmor.variable(name, units, [lat_id, lon_id], missing_value=fillv)

        cmor.write(
            var_id,
            np.asarray(data_filled),
        )
        cmor.close(var_id)

    def ensure_fx_written_and_cached(self, ds_regr: xr.Dataset) -> xr.Dataset:
        """Ensure sftlf and areacella exist in ds_regr and are written once as fx.
        If not present in ds_regr, try to read from existing CMOR fx files in outdir.
        If present in ds_regr but not yet written this run, write and cache them.
        Returns ds_regr augmented with any missing fx fields.
        """
        need = [
            ("sftlf", "%"),
            ("areacella", "m2"),
            ("sftof", "%"),
            ("areacello", "m3"),
            ("mrsofc", "m3 s-1"),
            ("orog", "m"),
            ("thkcello", "m"),
            ("slthick", "m"),
            ("basin", "m2"),
            ("deptho", "m"),
            ("hfgeou", "m"),
            ("masscello", "m3"),
            ("thkcello", "m"),
            ("rootd", "m"),
            ("sftgif", "%"),
            ("sftif", "%"),
        ]  # land fraction, ocean cell area, soil moisture fraction
        out = ds_regr

        for name, units in need:
            # 1) Already cached this run?
            if name in self._fx_cache:
                if name not in out:
                    out = out.assign({name: self._fx_cache[name]})
                continue

            # 2) Present in regridded dataset? (best case)
            if name in out:
                self._fx_cache[name] = out[name]
                if name not in self._fx_written:
                    # Convert landfrac to % if needed
                    if name == "sftlf":
                        v = out[name]
                        if (np.nanmax(v.values) <= 1.0) and v.attrs.get(
                            "units", ""
                        ) not in ("%", "percent"):
                            out = out.assign(
                                {
                                    name: (v * 100.0).assign_attrs(
                                        v.attrs | {"units": "%"}
                                    )
                                }
                            )
                            self._fx_cache[name] = out[name]
                    self._write_fx_2d(out, name, units)
                    self._fx_written.add(name)
                    continue

            # 3) Not present in ds_regr → try reading existing CMOR fx output
            if self._outdir:
                fx_da = open_existing_fx(self._outdir, name)
                if fx_da is not None:
                    # Verify grid match (simple equality on lat/lon values)
                    if (
                        "lat" in out
                        and "lon" in out
                        and np.array_equal(out["lat"].values, fx_da["lat"].values)
                        and np.array_equal(out["lon"].values, fx_da["lon"].values)
                    ):
                        out = out.assign({name: fx_da})
                        self._fx_cache[name] = out[name]
                        self._fx_written.add(name)  # already exists on disk
                        continue
                # If grid mismatch, you could regrid fx_da here; for now, skip.
                # 4) Last resort: leave missing; caller may compute it later
        return out

    # public API
    # -------------------------
    def write_variable(
        self,
        ds: xr.Dataset,
        varname: str,
        vdef: Any,
    ) -> None:
        """Write one variable from ds to a CMOR-compliant NetCDF file."""
        # Pick CMOR table: prefer vdef.table, else vdef.realm (default Amon)
        table_key = (
            getattr(vdef, "table", None) or getattr(vdef, "realm", None) or "Amon"
        )

        table_filename = resolve_table_filename(self.tables_path, table_key)
        cmor.load_table(table_filename)

        data = ds[vdef.name]
        logger.info("Prepare data for CMOR %s", data.dtype)  # debug
        data_filled, fillv = filled_for_cmor(data)
        logger.info("Define axes data_filled dtype: %s", data_filled.dtype)  # debug
        axes_ids = self._define_axes(ds, vdef)
        units = getattr(vdef, "units", "") or ""
        # Debug logging for axis mapping

        # Try to get axis table entries for each axis_id
        try:
            for i, aid in enumerate(axes_ids):
                entry = cmor.axis_entry(aid) if hasattr(cmor, "axis_entry") else None
                logger.debug(
                    "[CMOR DEBUG] axis %d: id=%s, table_entry=%s", i, aid, entry
                )
        # pylint: disable=broad-exception-caught
        except Exception as e:
            logger.warning("[CMOR DEBUG] Could not retrieve axis table entries: %s", e)
        var_id = cmor.variable(
            getattr(vdef, "name", varname),
            units,
            axes_ids,
            positive=getattr(vdef, "positive", None),
            missing_value=fillv,
        )

        # ---- Prepare time info for this write (local, not cached) ----
        time_da = ds.coords.get("time")
        if time_da is None:
            time_da = ds.get("time")
        nt = 0

        self.ensure_fx_written_and_cached(ds)

        # ---- Main variable write ----

        cmor.write(
            var_id,
            np.asarray(data_filled),
            ntimes_passed=nt,
        )
        # ---- Hybrid ps streaming (if present) ----
        if self._pending_ps is not None:
            ps_id, ps_da = self._pending_ps
            if "time" in ps_da.dims:
                ps_da = ps_da.transpose("time", "lat", "lon")
                nt_ps = int(ps_da.sizes["time"])
            else:
                nt_ps = 0
            ps_filled, _ = filled_for_cmor(ps_da)
            if nt_ps > 0:
                cmor.write(
                    ps_id,
                    np.asarray(ps_filled),
                    ntimes_passed=nt_ps,
                    store_with=var_id,
                )
            else:
                cmor.write(ps_id, np.asarray(ps_filled), store_with=var_id)
            self._pending_ps = None

        cmor.close(var_id)

    def write_variables(
        self,
        ds: xr.Dataset,
        cmip_vars: Sequence[str],
        mapping: collections.abc.Mapping,
    ) -> None:
        """Write multiple CMIP variables from one dataset."""
        for v in cmip_vars:
            cfg = mapping.get_cfg(v) or {}
            table = cfg.get("table", "Amon")
            units = cfg.get("units", "")
            positive = cfg.get("positive") or None
            vdef = types.SimpleNamespace(
                name=v,
                table=table,
                realm=table,
                units=units,
                positive=positive,
            )
            # pylint: disable=broad-exception-caught
            try:
                self.write_variable(ds, v, vdef)
            except Exception as e:
                warnings.warn(f"[cmor] skipping {v} due to error: {e}", RuntimeWarning)
                # continue to next variable
