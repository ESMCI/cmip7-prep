"""Thin CMOR wrapper used by cmip7_prep.

This module centralizes CMOR session setup and writing so that the rest of the
pipeline can stay xarray-first. It supports either a dataset JSON file (preferred)
or directly injected global attributes, and creates axes based on the coordinates
present in the provided dataset. It also supports a packaged default
`cmor_dataset.json` living under `cmip7_prep/data/`.
"""

from pathlib import Path
import json
import tempfile

from contextlib import AbstractContextManager
from typing import Any, Optional, Union
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
    roll_for_monotonic_with_bounds,
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
        self.primarytable: Optional[str] = None
        self.currenttable: Optional[str] = None
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
        logger.info("Using dataset JSON: %s", p)
        with open(p, encoding="utf-8") as f:
            cfg = json.load(f)
        cfg["outpath"] = str(self._outdir)
        tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(cfg, tmp)
        tmp.close()
        if not Path(tmp.name).exists():
            raise FileNotFoundError(f"Temporary dataset_json not found: {tmp.name}")
        cmor.dataset_json(str(tmp.name))
        logger.info("CMOR dataset_json loaded from: %s", tmp.name)
        try:
            prod = cmor.get_cur_dataset_attribute("product")  # type: ignore[attr-defined]
        except Exception:  # pylint: disable=broad-except
            prod = None
        if prod != "model-output":
            cmor.set_cur_dataset_attribute("product", "model-output")

        # long paragraph; split to keep lines < 100
        inst = get_cmor_attr("institution_id") or "NCAR"
        cmor.set_cur_dataset_attribute("institution_id", inst)
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
        logger.info("set _controlled_vocabulary_file:")
        cmor.set_cur_dataset_attribute("_controlled_vocabulary_file", "CMIP7_CV.json")
        logger.info("set _axis_entry_file:")
        cmor.set_cur_dataset_attribute("_AXIS_ENTRY_FILE", "CMIP7_coordinate.json")
        logger.info("set _formula_var_file:")
        cmor.set_cur_dataset_attribute("_FORMULA_VAR_FILE", "CMIP7_formula_terms.json")
        logger.info("CMOR session initialized")
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
    def load_table(self, tables_path: Path, key: str) -> dict:
        """Load CMOR table JSON for a given key by searching common patterns."""
        if key == self.currenttable:
            return {}  # already loaded
        table_filename = resolve_table_filename(tables_path, key)
        self.currenttable = key
        return cmor.load_table(table_filename)

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

            units = time_da.attrs.get(
                "units", time_da.encoding.get("units", "days since 1850-01-01")
            )
            # spinup run calendar adjustment
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
            # Compute bounds
            bnds = bounds_from_centers_1d(vals, name)
            # For longitude, ensure monotonicity and alignment
            if name == "lon":
                vals, bnds, _ = roll_for_monotonic_with_bounds(vals, bnds)
            return vals, bnds, units

        axes_ids = []
        var_name = getattr(vdef, "name", None)
        if var_name is None or var_name not in ds:
            raise KeyError(f"Variable to write not found in dataset: {var_name!r}")
        var_da = ds[str(var_name)]
        var_dims = list(var_da.dims)
        alev_id = None
        plev_id = None
        lat_id = None
        lon_id = None
        sdepth_id = None
        lev_id = None

        logger.debug("[CMOR axis debug] var_dims: %s", var_dims)
        if ("xh" in var_dims or "xq" in var_dims) and (
            "yh" in var_dims or "yq" in var_dims
        ):
            # MOM6/curvilinear grid: register xh/yh as generic axes (i/j), not as lat/lon
            # Define the native grid using the coordinate arrays
            logger.debug(
                "[CMOR axis debug] Defining unstructured grid for variable %s.",
                var_name,
            )
            logger.info("[CMOR axis debug] write geolon axes")

            geo_path = Path(__file__).parent / "data" / "ocean_geometry.nc"
            if not geo_path.exists():
                raise FileNotFoundError(f"Expected geometry file not found: {geo_path}")
            ds_geo = xr.open_dataset(geo_path)
            if "xh" in var_dims:
                lon_raw = np.mod(ds_geo["lonh"].values, 360.0)
            else:
                lon_raw = np.mod(ds_geo["lonq"].values, 360.0)
            lon_bnds_raw = bounds_from_centers_1d(lon_raw, "lon")
            lon_vals_1d, lon_bnds, shift = roll_for_monotonic_with_bounds(
                lon_raw, lon_bnds_raw
            )
            if "yh" in var_dims:
                lat_vals_1d = ds_geo["lath"].values
            elif "yq" in var_dims:
                lat_vals_1d = ds_geo["latq"].values
            else:
                raise KeyError(
                    "Expected 'yh' or 'yq' dimension for latitude not found."
                )
            # Fix first and last bounds to wrap correctly
            if lon_bnds.shape[0] > 1:
                # Ensure bounds are strictly increasing and wrap at dateline
                if lon_bnds[0, 0] > 0.0:
                    # Set first lower bound to 0, last upper bound to 360
                    lon_bnds[0, 0] = 0.0
                    lon_bnds[-1, 1] = 360.0
                # Also correct first upper bound to match the first cell
                lon_bnds[0, 1] = lon_bnds[1, 0]
            logger.info("[CMOR axis debug] corrected lon_bnds: %s", lon_bnds)
            # Print lon_bnds for a range (debug)
            i_id = cmor.axis(
                table_entry="latitude",
                units="degrees_north",
                coord_vals=lat_vals_1d,
                cell_bounds=bounds_from_centers_1d(lat_vals_1d, "lat"),
            )
            logger.info("[CMOR axis debug] write geolat axes")
            j_id = cmor.axis(
                table_entry="longitude",
                units="degrees_east",
                coord_vals=lon_vals_1d,
                cell_bounds=lon_bnds,
            )
            axes_ids.extend([j_id, i_id])
            for dim in ("xh", "xq", "yh", "yq"):
                if dim in var_dims:
                    # rename dims xh and yh to longitude and latitude
                    logger.info("[CMOR axis debug] renaming dim %s", dim)
                    if dim in ["xh", "xq"]:
                        if dim == "xh":
                            ds[var_name] = var_da.roll(xh=-shift, roll_coords=True)
                        elif dim == "xq":
                            ds[var_name] = var_da.roll(xq=-shift, roll_coords=True)
                        ds[var_name] = ds[var_name].rename({dim: "longitude"})
                    if dim in ["yh", "yq"]:
                        ds[var_name] = ds[var_name].rename({dim: "latitude"})

            var_da = ds[var_name]
            var_dims = list(var_da.dims)

        # --- horizontal axes (use CMOR names) ----
        elif "lat" in var_dims and "lon" in var_dims:
            logger.info("*** Define horizontal axes")
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
        self.load_table(self.tables_path, self.primarytable)
        tvals, tbnds, t_units = _get_time_and_bounds(ds)
        if tvals is not None:
            time_id = cmor.axis(
                table_entry="time",
                units=t_units,
                coord_vals=tvals,
                cell_bounds=tbnds if tbnds is not None else None,
            )
            logger.info("time axis id: %s", time_id)
        # --- vertical: standard_hybrid_sigma ---
        levels = getattr(vdef, "levels", {}) or {}

        if (levels.get("name") or "").lower() in {
            "standard_hybrid_sigma",
            "alevel",
            "alev",
        } or "lev" in var_dims:
            # names in the native ds
            logger.info("*** Define hybrid sigma axis")
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
            cmor.set_cur_dataset_attribute(
                "vertical_label", "alevel"
            )  # or another valid value
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
            if pb is None:
                pb = bounds_from_centers_1d(pvals, "plev")

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
        elif "z_l" in var_dims:
            values = ds["z_l"].values
            logger.info("write z_l axis")
            bnds = bounds_from_centers_1d(values, "z_l")
            lev_id = cmor.axis(
                table_entry="depth_coord",
                units="m",
                coord_vals=np.asarray(values),
                cell_bounds=bnds,
            )
        elif "zl" in var_dims:
            ds[var_name] = ds[var_name].rename({"zl": "olevel"})
            var_dims = list(ds[var_name].dims)
            cmor.set_cur_dataset_attribute("vertical_label", "olevel")
            logger.info("*** Define olevel axis")
            values = ds["olevel"].values
            logger.info("write olevel axis")
            zi = ds["zi"].values
            bnds = np.column_stack((zi[:-1], zi[1:]))
            lev_id = cmor.axis(
                table_entry="depth_coord",
                units="m",
                coord_vals=np.asarray(values),
                cell_bounds=bnds,
            )
        elif "zi" in var_dims:
            ds[var_name] = ds[var_name].rename({"zi": "olevel"})
            var_dims = list(ds[var_name].dims)
            cmor.set_cur_dataset_attribute("vertical_label", "olevel")
            logger.info("*** Define olevel axis")
            values = ds["olevel"].values
            logger.info("write olevel axis")
            zl = ds["zl"].values
            bnds = np.column_stack((zl[:-1], zl[1:]))
            lev_id = cmor.axis(
                table_entry="depth_coord",
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
            "z_l": lev_id,
            "olevel": lev_id,
            "plev": plev_id,
            "lat": lat_id,
            "latitude": lat_id if lat_id is not None else i_id,
            "lon": lon_id,
            "longitude": lon_id if lon_id is not None else j_id,
            "xh": lon_id,  # MOM6
            "yh": lat_id,  # MOM6
            "xq": lon_id,  # MOM6
            "yq": lat_id,  # MOM6
        }
        axes_ids = []
        for d in var_dims:
            axis_id = dim_to_axis.get(d)
            logger.info("[CMOR axis debug] dim '%s' → axis_id: %s", d, axis_id)
            if axis_id is None:
                raise KeyError(
                    f"No axis ID found for dimension '{d}' in variable '{var_name}' {var_dims}"
                )
            axes_ids.append(axis_id)
        self.load_table(self.tables_path, self.primarytable)

        return axes_ids

    def _write_fx_2d(self, ds: xr.Dataset, name: str, units: str) -> None:
        if name not in ds:
            return

        # Handle curvilinear grid (yh, xh) for deptho
        da = ds[name]
        logger.info("FX variable %s dims: %s", name, da.dims)
        if set(da.dims) == {"xh", "yh"}:
            self.load_table(self.tables_path, "ocean")
            geo_path = Path(__file__).parent / "data" / "ocean_geometry.nc"
            ds_geo = xr.open_dataset(geo_path)
            lat = ds_geo["lath"].values
            lon_raw = np.mod(ds_geo["lonh"].values, 360.0)
            lon_bnds_raw = bounds_from_centers_1d(lon_raw, "lon")
            lon_vals_1d, lon_bnds, shift = roll_for_monotonic_with_bounds(
                lon_raw, lon_bnds_raw
            )
            if lon_bnds.shape[0] > 1:
                # Ensure bounds are strictly increasing and wrap at dateline
                if lon_bnds[0, 0] > 0.0:
                    # Set first lower bound to 0, last upper bound to 360
                    lon_bnds[0, 0] = 0.0
                    lon_bnds[-1, 1] = 360.0
                # Also correct first upper bound to match the first cell
                lon_bnds[0, 1] = lon_bnds[1, 0]
            # Print lon_bnds for a range (debug)
            data = np.roll(da.values, -shift, axis=1)
            # Assign new dims for writing
            data_filled, fillv = filled_for_cmor(
                xr.DataArray(data, dims=["latitude", "longitude"])
            )
            lat_b = bounds_from_centers_1d(lat, "lat")

            lat_id = cmor.axis(
                "latitude", "degrees_north", coord_vals=lat, cell_bounds=lat_b
            )
            logger.info("FX variable %s define lat_id %s", name, lat_id)

            lon_id = cmor.axis(
                "longitude",
                "degrees_east",
                coord_vals=lon_vals_1d,
                cell_bounds=lon_bnds,
            )
            logger.info("FX variable %s define lon_id %s", name, lon_id)
            logger.info("Writing fx variable %s on curvilinear grid", name)
            cmor.set_cur_dataset_attribute("grid", "curvilinear")
            cmor.set_cur_dataset_attribute("grid_label", "gn")
            if name == "deptho":
                name = "deptho_ti-u-hxy-sea"
        else:
            cmor.set_cur_dataset_attribute("grid", "1x1 degree")
            cmor.set_cur_dataset_attribute("grid_label", "gr")
            if name in ("areacella_ti-u-hxy-u", "sftlf_ti-u-hxy-u"):
                self.load_table(self.tables_path, "land")
            elif name in ("sftof_ti-u-hxy-u", "deptho", "areacello"):
                self.load_table(self.tables_path, "ocean")
                if name == "deptho":
                    name = "deptho_ti-u-hxy-sea"
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
            data_filled, fillv = filled_for_cmor(da)
        logger.info("Defining fx variable %s", name)

        var_id = cmor.variable(name, units, [lat_id, lon_id], missing_value=fillv)
        if cmor.has_variable_attribute(var_id, "outputpath"):
            logger.info(
                "CMOR variable object has outputpath attribute: %s",
                cmor.get_variable_attribute(var_id, "outputpath"),
            )
        logger.info("Now writing FX variable %s id: %s", name, var_id)
        cmor.write(var_id, np.asarray(data_filled))
        cmor.close(var_id)
        logger.info("Finished writing fx variable %s", name)

    def ensure_fx_written_and_cached(self, ds_regr: xr.Dataset) -> xr.Dataset:
        """Ensure <fx variables> exist in ds_regr and are written once as fx.
        If not present in ds_regr, try to read from existing CMOR fx files in outdir.
        If present in ds_regr but not yet written this run, write and cache them.
        Returns ds_regr augmented with any missing fx fields.
        """
        need = [
            ("sftlf_ti-u-hxy-u", "%"),
            ("areacella_ti-u-hxy-u", "m2"),
            ("sftof_ti-u-hxy-u", "%"),
            ("wet", "%"),
            ("areacello_ti-u-hxy-u", "m2"),
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

        # Write sftof_ti-u-hxy-u if present (native grid sea fraction: wet)
        if "sftof_ti-u-hxy-u" in ds_regr and "sftof_ti-u-hxy-u" not in self._fx_written:
            logger.info("Writing fx variable sftof_ti-u-hxy-u")
            self.load_table(self.tables_path, "ocean")
            self._write_fx_2d(ds_regr, "sftof_ti-u-hxy-u", "%")
            self._fx_written.add("sftof_ti-u-hxy-u")

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
                    if name == "sftlf_ti-u-hxy-u":
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
                try:
                    fx_da = open_existing_fx(self._outdir, name)
                except FileNotFoundError:
                    fx_da = None
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
        cmip_var: object,
        vdef: Any,
    ) -> None:
        """Write one variable from ds to a CMOR-compliant NetCDF file."""
        # Pick CMOR table: prefer vdef.table, else vdef.realm (default Amon)
        self.primarytable = (
            getattr(vdef, "table", None) or getattr(vdef, "realm", None) or "atmos"
        )

        logger.info(
            "Using CMOR table key: %s %s", self.tables_path, self.primarytable
        )  # debug
        self.load_table(self.tables_path, self.primarytable)
        varname = getattr(cmip_var, "physical_parameter").name
        logger.info("Preparing to write variable: %s", varname)  # debug
        data = ds[str(varname)]

        logger.info("Ensure fx variables are written and cached")  # debug
        self.ensure_fx_written_and_cached(ds)

        units = getattr(vdef, "units", "") or ""
        self.load_table(self.tables_path, self.primarytable)
        logger.info("Define CMOR axes for variable %s", vdef.name)  # debug
        axes_ids = self._define_axes(ds, vdef)
        logger.info("Prepare data for CMOR %s", data.dtype)  # debug
        data_filled, fillv = filled_for_cmor(data)
        if "zl" in data_filled.dims:
            data_filled = data_filled.rename({"zl": "olevel"})
        elif "zi" in data_filled.dims:
            data_filled = data_filled.rename({"zi": "olevel"})
        self.load_table(self.tables_path, self.primarytable)

        var_entry = getattr(cmip_var, "branded_variable_name", varname)
        if hasattr(var_entry, "name"):
            var_entry = var_entry.name
        elif hasattr(var_entry, "value"):
            var_entry = var_entry.value
        else:
            var_entry = str(var_entry)

        logger.info("Define CMOR variable %s", var_entry)  # debug
        var_id = cmor.variable(
            var_entry,
            units,
            axes_ids,
            positive=getattr(vdef, "positive", None),
            missing_value=fillv,
        )
        logger.info("Now define time dimension and write data")  # debug
        if "lat" in data.dims and "lon" in data.dims:
            cmor.set_cur_dataset_attribute("grid", "1x1 degree")
            cmor.set_cur_dataset_attribute("grid_label", "gr")
        else:
            cmor.set_cur_dataset_attribute("grid", "curvilinear")
            cmor.set_cur_dataset_attribute("grid_label", "gn")
        # ---- Prepare time info for this write (local, not cached) ----
        time_da = ds.coords.get("time")
        if time_da is None:
            time_da = ds.get("time")
        nt = 0

        # ---- Main variable write ----
        logger.info("Writing CMOR variable %s", var_id)  # debug
        cmor.write(
            var_id,
            np.asarray(data_filled),
            ntimes_passed=nt,
        )
        logger.info("Finished writing CMOR variable %s", var_id)  # debug
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
