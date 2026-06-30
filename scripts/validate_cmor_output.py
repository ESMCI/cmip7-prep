#!/usr/bin/env python3

"""Quick-check validation for CMORized CMIP7 output trees.

This script validates a selected CMIP7 output subset after ``cmor_driver.py``
has run. It reports:

* variables with CMOR log errors
* variables expected for the selected subset that were not produced
* dimension inventory for variables that were produced

It can also optionally create:

* composite mean time-series plots for produced variables
* per-variable time-mean maps where the data are on plottable lat/lon grids
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
import yaml

from cmip7_prep.mapping_compat import packaged_mapping_resource

from cmor_driver import REALM_YAML_MAP

logger = logging.getLogger("cmip7_prep.validate_cmor_output")

MODEL_NAMING_MAPS = {
    "noresm": ["NCC", "NorESM3"],
    "cesm": ["NCAR", "CESM3"],
}

CANONICAL_REALM_MAP = {
    "aerosol": "atmos",
    "atmosChem": "atmos",
    "ocnBgchem": "ocean",
}

LOG_NAME_RE = re.compile(r"^cmor_\d{8}T\d{6}Z_(?P<variable>.+)\.log$")
LOG_ERROR_RE = re.compile(r"Error: ", re.IGNORECASE)
LOG_SUCCESS_RE = re.compile(r"\b(success|complete(?:d)?|finished)\b", re.IGNORECASE)


@dataclass
class LogRecord:
    """A parsed record for one CMOR log file."""

    variable: str
    path: str
    has_error: bool
    has_success_marker: bool
    error_lines: list[str]


@dataclass
class ProducedFileRecord:
    """Inventory record for one produced CMOR file."""

    variable: str
    consortium: str
    model: str
    experiment: str
    ensemble_member: str
    region: str
    frequency: str
    dimension_folder: str
    grid_type: str
    file_path: str
    data_var: str
    dims: list[str]
    sizes: dict[str, int]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Quick-check validation for CMIP7 CMOR output trees"
    )
    parser.add_argument(
        "--model",
        choices=sorted(REALM_YAML_MAP),
        required=True,
        help="Model whose CMIP7 output should be validated",
    )
    parser.add_argument(
        "--realm",
        choices=[
            "atmos",
            "aerosol",
            "atmosChem",
            "land",
            "landIce",
            "ocean",
            "ocnBgchem",
            "seaIce",
        ],
        required=True,
        help="Realm to validate; aerosol/atmosChem map to atmos YAML, ocnBgchem maps to ocean YAML",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="piControl",
        help="Experiment name to validate (Default piControl)",
    )
    parser.add_argument(
        "--frequency",
        choices=["mon", "day", "6hr", "3hr", "yr", "fx"],
        required=True,
        help="CMIP7 output frequency to validate",
    )
    parser.add_argument(
        "--resolution",
        default=None,
        help="Optional best-effort filter applied to dimension folder, grid type and file name",
    )
    parser.add_argument(
        "--root-output-path",
        required=True,
        help="Root output directory containing CMIP7/ and logs/",
    )
    parser.add_argument(
        "--ensemble-member",
        default=None,
        help="Optional ensemble member filter",
    )
    parser.add_argument(
        "--custom-yaml",
        default=None,
        help="Optional custom YAML mapping file",
    )
    parser.add_argument(
        "--variables",
        nargs="*",
        default=None,
        help="Optional explicit list of branded variable names to validate",
    )
    parser.add_argument(
        "--report-dir",
        default=None,
        help="Directory for validation reports; defaults to <root-output-path>/validation_reports/<subset>",
    )
    parser.add_argument(
        "--plot-timeseries",
        action="store_true",
        help="Create composite mean time-series plots for produced variables",
    )
    parser.add_argument(
        "--plot-maps",
        action="store_true",
        help="Create per-variable time-mean maps where possible",
    )
    parser.add_argument(
        "--plot-dir",
        default=None,
        help="Output directory for plot files; defaults to <report-dir>/plots",
    )
    parser.add_argument(
        "--max-plots",
        type=int,
        default=36,
        help="Maximum number of variables to plot per plot mode",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if missing variables or log errors are found",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser.parse_args()


def canonical_realm(realm: str) -> str:
    """Return the YAML realm corresponding to the requested realm."""
    return CANONICAL_REALM_MAP.get(realm, realm)


def resolve_cmip_root(root_output_path: str | Path) -> Path:
    """Resolve the CMIP7 root directory from either a parent or direct path."""
    root = Path(root_output_path).expanduser().resolve()
    if (root / "CMIP").is_dir():
        return root
    cmip_root = root / "CMIP7"
    if (cmip_root / "CMIP").is_dir():
        return cmip_root
    raise FileNotFoundError(
        f"Could not locate CMIP output under {root}. Expected either {root / 'CMIP7' / 'CMIP'} or {root / 'CMIP'}."
    )


def resolve_logs_dir(root_output_path: str | Path, cmip_root: Path) -> Path:
    """Resolve the logs directory adjacent to the CMIP7 output tree."""
    root = Path(root_output_path).expanduser().resolve()
    if (root / "logs").is_dir():
        return root / "logs"
    if (cmip_root.parent / "logs").is_dir():
        return cmip_root.parent / "logs"
    return root / "logs"


def default_report_dir(args: argparse.Namespace) -> Path:
    """Compute the default report directory for a validation subset."""
    root = Path(args.root_output_path).expanduser().resolve()
    subset = f"{args.model}_{args.realm}_{args.experiment}_{args.frequency}"
    return root / "validation_reports" / subset


def get_yaml_path(model: str, realm: str, custom_yaml: str | None) -> Path:
    """Resolve the YAML mapping path for the selected model/realm."""
    if custom_yaml:
        path = Path(custom_yaml).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    yaml_realm = canonical_realm(realm)
    yaml_name = REALM_YAML_MAP.get(model, {}).get(yaml_realm)
    if yaml_name is None:
        raise ValueError(f"No YAML mapping defined for model={model}, realm={realm}")
    with packaged_mapping_resource(yaml_name) as resource_path:
        return Path(resource_path)


def load_yaml_variables(yaml_path: Path) -> dict[str, dict[str, Any]]:
    """Load the variables block from a CMIP7 mapping YAML file."""
    with open(yaml_path, encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    variables = payload.get("variables")
    if not isinstance(variables, dict):
        raise ValueError(
            f"Unsupported YAML structure in {yaml_path}: expected top-level variables dict"
        )
    return variables


def get_requested_variables(
    realm: str,
    experiment: str,
    frequency: str,
) -> set[str] | None:
    """Query the CMIP7 data request, if the package is available."""
    try:
        from data_request_api.content import dump_transformation as dt
        from data_request_api.query import data_request as dr
    except ImportError:
        logger.warning(
            "data_request_api is not available; expected-variable filtering will use YAML only"
        )
        return None

    logger.info(
        "Querying data request for realm=%s experiment=%s frequency=%s",
        realm,
        experiment,
        frequency,
    )
    content_dic = dt.get_transformed_content()
    logger.debug(content_dic)
    data_request = dr.DataRequest.from_separated_inputs(**content_dic)
    cmip_vars = data_request.find_variables(
        skip_if_missing=False,
        operation="all",
        cmip7_frequency=frequency,
        modelling_realm=realm,
        experiment=experiment.lower(),
    )
    return {var.branded_variable_name.name for var in cmip_vars}


def filter_expected_variables(
    yaml_variables: dict[str, dict[str, Any]],
    requested_variables: set[str] | None,
    selected_variables: list[str] | None,
) -> list[str]:
    """Resolve the expected variable list for this validation subset."""
    yaml_names = set(yaml_variables)
    if requested_variables is not None:
        yaml_names &= requested_variables
    if selected_variables:
        yaml_names &= set(selected_variables)
    return sorted(yaml_names)


def parse_log_variable(log_path: Path) -> str | None:
    """Extract the CMIP variable name from a CMOR log filename."""
    match = LOG_NAME_RE.match(log_path.name)
    if match is None:
        return None
    return match.group("variable")


def parse_log_file(log_path: Path) -> LogRecord | None:
    """Parse one CMOR log file for basic error markers."""
    variable = parse_log_variable(log_path)
    if variable is None:
        return None
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    error_lines = []
    for line in text.splitlines():
        if LOG_ERROR_RE.search(line):
            error_lines.append(line.strip())
    return LogRecord(
        variable=variable,
        path=str(log_path),
        has_error=bool(error_lines),
        has_success_marker=bool(LOG_SUCCESS_RE.search(text)),
        error_lines=error_lines[:10],
    )


def collect_log_records(
    log_dir: Path, expected_variables: set[str]
) -> dict[str, list[LogRecord]]:
    """Collect log records for variables that belong to the selected subset."""
    records: dict[str, list[LogRecord]] = defaultdict(list)
    if not log_dir.is_dir():
        logger.warning("Log directory not found: %s", log_dir)
        return records
    for log_path in sorted(log_dir.glob("*.log")):
        record = parse_log_file(log_path)
        if record is None:
            continue
        if expected_variables and record.variable not in expected_variables:
            continue

        records[record.variable].append(record)
    return records


def path_matches_resolution(
    resolution: str | None,
    dimension_folder: str,
    grid_type: str,
    file_name: str,
) -> bool:
    """Apply a best-effort resolution filter to discovered output paths."""
    if not resolution:
        return True
    target = resolution.lower()
    candidates = (dimension_folder.lower(), grid_type.lower(), file_name.lower())
    return any(target in candidate for candidate in candidates)


def open_dataset_inventory(
    file_path: Path, variable: str
) -> tuple[str, list[str], dict[str, int]]:
    """Open a CMOR file and inspect the target data variable dimensions."""
    with xr.open_dataset(file_path, decode_times=False) as dataset:
        if variable in dataset.data_vars:
            data_var = variable
        else:
            candidates = [
                name
                for name in dataset.data_vars
                if not name.endswith("_bnds")
                and name not in {"time_bnds", "lat_bnds", "lon_bnds"}
            ]
            if not candidates:
                raise ValueError(f"No data variables found in {file_path}")
            data_var = candidates[0]
        array = dataset[data_var]
        dims = list(array.dims)
        sizes = {dim: int(array.sizes[dim]) for dim in array.dims}
    return data_var, dims, sizes


def scan_output_tree(
    cmip_root: Path,
    *,
    model: str,
    experiment: str,
    frequency: str,
    expected_variables: set[str],
    ensemble_member: str | None = None,
    resolution: str | None = None,
) -> tuple[dict[str, list[Path]], list[ProducedFileRecord], list[dict[str, str]]]:
    """Scan the CMIP7 tree and inventory produced files for the selected subset."""
    produced: dict[str, list[Path]] = defaultdict(list)
    inventory: list[ProducedFileRecord] = []
    inspection_errors: list[dict[str, str]] = []
    institution_id = MODEL_NAMING_MAPS[model][0]
    pattern = cmip_root.glob(
        f"CMIP/{institution_id}/{MODEL_NAMING_MAPS[model][1]}/{experiment}/*/glb/{frequency}/*/*/*/*.nc"
    )
    for file_path in sorted(pattern):
        relative = file_path.relative_to(cmip_root)
        # sys.exit(4)
        parts = relative.parts
        if len(parts) < 11:
            logger.debug("Skipping unexpected output path layout: %s", file_path)
            continue

        (
            _,
            consortium,
            path_model,
            path_experiment,
            path_ensemble,
            region,
            path_frequency,
            variable,
            dimension_folder,
            grid_type,
            _,
        ) = parts[:11]

        if institution_id and consortium != institution_id:
            logger.debug("Skipping due to institution_id filter: %s", institution_id)
            continue
        if ensemble_member and path_ensemble != ensemble_member:
            logger.debug("Skipping due to ensemble filter: %s", ensemble_member)
            continue
        if (
            expected_variables
            and f"{variable}_{dimension_folder}" not in expected_variables
        ):
            logger.debug("Skipping due to expected_variables filter: %s", variable)
            continue
        # if not path_matches_resolution(
        #     resolution, dimension_folder, grid_type, file_path.name
        # ):
        #     print("Skipping due to resolution filter: %s", resolution)
        #     logger.debug("Skipping due to resolution filter: %s", resolution)
        #     continue
        produced[f"{variable}_{dimension_folder}"].append(file_path)
        try:
            data_var, dims, sizes = open_dataset_inventory(file_path, variable)
        except Exception as exc:  # pylint: disable=broad-except
            inspection_errors.append(
                {
                    "variable": variable,
                    "path": str(file_path),
                    "error": repr(exc),
                }
            )
            continue

        inventory.append(
            ProducedFileRecord(
                variable=variable,
                consortium=consortium,
                model=path_model,
                experiment=path_experiment,
                ensemble_member=path_ensemble,
                region=region,
                frequency=path_frequency,
                dimension_folder=dimension_folder,
                grid_type=grid_type,
                file_path=str(file_path),
                data_var=data_var,
                dims=dims,
                sizes=sizes,
            )
        )
    return produced, inventory, inspection_errors


def summarize_dimension_inventory(
    inventory: list[ProducedFileRecord],
) -> list[dict[str, Any]]:
    """Collapse file-level inventory into one summary row per variable."""
    grouped: dict[str, dict[str, set[str] | list[dict[str, Any]]]] = {}
    for record in inventory:
        bucket = grouped.setdefault(
            record.variable,
            {
                "dims": set(),
                "dimension_folders": set(),
                "grid_types": set(),
                "data_vars": set(),
                "sample_sizes": [],
                "sample_path": record.file_path,
            },
        )
        bucket["dims"].add(", ".join(record.dims))
        bucket["dimension_folders"].add(record.dimension_folder)
        bucket["grid_types"].add(record.grid_type)
        bucket["data_vars"].add(record.data_var)
        if len(bucket["sample_sizes"]) < 3:
            bucket["sample_sizes"].append(record.sizes)

    summary = []
    for variable in sorted(grouped):
        item = grouped[variable]
        summary.append(
            {
                "variable": variable,
                "dims": sorted(item["dims"]),
                "dimension_folders": sorted(item["dimension_folders"]),
                "grid_types": sorted(item["grid_types"]),
                "data_vars": sorted(item["data_vars"]),
                "sample_sizes": item["sample_sizes"],
                "sample_path": item["sample_path"],
            }
        )
    return summary


def write_json_report(report: dict[str, Any], output_path: Path) -> None:
    """Write the main validation report as JSON."""
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )


def write_csv_rows(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write a homogeneous list of dict rows to CSV."""
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({field for row in rows for field in row})
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_summary(report: dict[str, Any], output_path: Path) -> None:
    """Write a concise markdown summary."""
    lines = [
        "# CMOR Validation Summary",
        "",
        f"- Model: {report['scope']['model']}",
        f"- Realm: {report['scope']['realm']}",
        f"- Experiment: {report['scope']['experiment']}",
        f"- Frequency: {report['scope']['frequency']}",
        f"- Expected variables: {report['counts']['expected_variables']}",
        f"- Produced variables: {report['counts']['produced_variables']}",
        f"- Variables with log errors: {report['counts']['variables_with_log_errors']}",
        f"- Missing variables: {report['counts']['missing_variables']}",
        "",
        "## Variables With Log Errors",
        "",
    ]
    for variable in report["variables_with_log_errors"]:
        lines.append(f"- {variable}")
    lines.extend(["", "## Missing Variables", ""])
    for variable in report["expected_but_not_produced"]:
        lines.append(f"- {variable}")
    lines.extend(["", "## Produced Variable Inventory", ""])
    for item in report["dimension_inventory"]:
        dims = "; ".join(item["dims"])
        grids = ", ".join(item["grid_types"])
        lines.append(f"- {item['variable']}: dims={dims}; grids={grids}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _open_variable_timeseries(
    file_paths: list[Path], variable: str
) -> xr.DataArray | None:
    """Open a variable across files and reduce it to a 1D time series if possible."""
    if not file_paths:
        return None
    with xr.open_mfdataset(
        file_paths, combine="by_coords", decode_times=True
    ) as dataset:
        data_var = (
            variable if variable in dataset.data_vars else list(dataset.data_vars)[0]
        )
        array = dataset[data_var]
        if "time" not in array.dims:
            return None
        reduce_dims = [
            dim for dim in array.dims if dim != "time" and not dim.endswith("bnds")
        ]
        if reduce_dims:
            array = array.mean(dim=reduce_dims, skipna=True)
        return array.load()


def _open_variable_map(file_paths: list[Path], variable: str) -> xr.DataArray | None:
    """Open a variable across files and reduce it to a 2D spatial field if possible."""
    if not file_paths:
        return None
    with xr.open_mfdataset(
        file_paths, combine="by_coords", decode_times=True
    ) as dataset:
        data_var = (
            variable if variable in dataset.data_vars else list(dataset.data_vars)[0]
        )
        array = dataset[data_var]
        if "time" in array.dims:
            array = array.mean(dim="time", skipna=True)
        spatial_dims = [
            dim
            for dim in array.dims
            if dim.lower()
            in {"lat", "lon", "latitude", "longitude", "xh", "yh", "i", "j"}
        ]
        if len(spatial_dims) != 2:
            return None
        extra_dims = [dim for dim in array.dims if dim not in spatial_dims]
        if extra_dims:
            array = array.isel({dim: 0 for dim in extra_dims})
        return array.load()


def create_timeseries_plots(
    produced_files: dict[str, list[Path]],
    plot_dir: Path,
    max_plots: int,
) -> list[str]:
    """Create paginated composite mean time-series plots."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib is not available; skipping time-series plots")
        return []

    plotted = []
    variables = sorted(produced_files)[:max_plots]
    if not variables:
        return plotted

    page_size = 9
    for page_index in range(0, len(variables), page_size):
        page_variables = variables[page_index : page_index + page_size]
        fig, axes = plt.subplots(3, 3, figsize=(15, 11), squeeze=False)
        for axis, variable in zip(axes.flat, page_variables):
            series = _open_variable_timeseries(
                produced_files[variable], variable.split("_")[0]
            )
            if series is None:
                axis.set_title(variable)
                axis.text(
                    0.5, 0.5, "No plottable time series", ha="center", va="center"
                )
                axis.set_axis_off()
                continue
            axis.plot(get_plottble_times(series), series.values, linewidth=1.0)
            axis.set_title(variable)
            axis.tick_params(axis="x", rotation=30)
            axis.set_xlabel("Time (years)")
            axis.set_ylabel(
                f"{variable.split('_')[0]} ({series.attrs.get('units', 'unknown')})"
            )
        for axis in axes.flat[len(page_variables) :]:
            axis.set_axis_off()

        fig.tight_layout()
        output_path = (
            plot_dir / f"timeseries_composite_{page_index // page_size + 1:02d}.png"
        )
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plotted.append(str(output_path))
    return plotted


def get_plottble_times(tseries: xr.DataArray) -> np.ndarray:
    monlength = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=int)
    strings = [date.strftime("%Y-%m-%d") for date in tseries["time"].values]
    numbers_lists = [strings.split("-") for strings in strings]
    numbers = np.array(
        [
            (365 * int(val[0]) + monlength[: int(val[1])].sum() + int(val[2])) / 365.0
            for val in numbers_lists
        ],
        dtype=float,
    )
    return numbers


def create_map_plots(
    produced_files: dict[str, list[Path]],
    plot_dir: Path,
    max_plots: int,
) -> list[str]:
    """Create per-variable time-mean map plots where the data shape allows it."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib is not available; skipping map plots")
        return []

    plotted = []
    for variable in sorted(produced_files)[:max_plots]:
        field = _open_variable_map(produced_files[variable], variable.split("_")[0])
        if field is None:
            continue
        fig, axis = plt.subplots(figsize=(8, 4.5))
        field.plot(ax=axis)
        axis.set_title(f"{variable} time-mean")
        output_path = plot_dir / f"map_{variable}.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plotted.append(str(output_path))
    return plotted


def build_report(
    args: argparse.Namespace,
    yaml_path: Path,
    expected_variables: list[str],
    log_records: dict[str, list[LogRecord]],
    produced_files: dict[str, list[Path]],
    dimension_inventory: list[dict[str, Any]],
    inspection_errors: list[dict[str, str]],
    plot_outputs: dict[str, list[str]],
) -> dict[str, Any]:
    """Build the structured validation report payload."""
    variables_with_log_errors = sorted(
        variable
        for variable, records in log_records.items()
        if any(record.has_error for record in records)
    )
    produced_variables = sorted(produced_files)
    expected_but_not_produced = sorted(
        set(expected_variables) - set(produced_variables)
    )
    flattened_log_records = [
        asdict(record) for records in log_records.values() for record in records
    ]

    return {
        "scope": {
            "model": args.model,
            "realm": args.realm,
            "experiment": args.experiment,
            "frequency": args.frequency,
            "resolution": args.resolution,
            "ensemble_member": args.ensemble_member,
            "institution_id": MODEL_NAMING_MAPS.get(args.model, [None, None])[0],
            "root_output_path": str(Path(args.root_output_path).expanduser().resolve()),
            "yaml_path": str(yaml_path),
        },
        "counts": {
            "expected_variables": len(expected_variables),
            "produced_variables": len(produced_variables),
            "variables_with_log_errors": len(variables_with_log_errors),
            "missing_variables": len(expected_but_not_produced),
            "inspection_errors": len(inspection_errors),
        },
        "expected_variables": expected_variables,
        "variables_with_log_errors": variables_with_log_errors,
        "expected_but_not_produced": expected_but_not_produced,
        "produced_variables": produced_variables,
        "dimension_inventory": dimension_inventory,
        "log_records": flattened_log_records,
        "inspection_errors": inspection_errors,
        "plots": plot_outputs,
    }


def print_summary(report: dict[str, Any]) -> None:
    """Print a concise terminal summary."""
    print(
        f"Validated {report['counts']['expected_variables']} expected variables; "
        f"found {report['counts']['produced_variables']} produced, "
        f"{report['counts']['variables_with_log_errors']} with log errors, "
        f"{report['counts']['missing_variables']} missing."
    )
    if report["variables_with_log_errors"]:
        print("Variables with log errors:")
        for variable in report["variables_with_log_errors"]:
            print(f"  - {variable}")
    if report["expected_but_not_produced"]:
        print("Expected but not produced:")
        for variable in report["expected_but_not_produced"]:
            print(f"  - {variable}")


def main() -> int:
    """Run the CMOR output validation workflow."""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cmip_root = resolve_cmip_root(args.root_output_path)
    logs_dir = resolve_logs_dir(args.root_output_path, cmip_root)
    report_dir = (
        Path(args.report_dir).expanduser().resolve()
        if args.report_dir
        else default_report_dir(args)
    )
    report_dir.mkdir(parents=True, exist_ok=True)

    yaml_path = get_yaml_path(args.model, args.realm, args.custom_yaml)
    yaml_variables = load_yaml_variables(yaml_path)
    requested_variables = get_requested_variables(
        args.realm, args.experiment, args.frequency
    )
    expected_variables = filter_expected_variables(
        yaml_variables, requested_variables, args.variables
    )
    expected_variable_set = set(expected_variables)

    log_records = collect_log_records(logs_dir, expected_variable_set)
    produced_files, inventory_records, inspection_errors = scan_output_tree(
        cmip_root,
        model=args.model,
        experiment=args.experiment,
        frequency=args.frequency,
        expected_variables=expected_variable_set,
        ensemble_member=args.ensemble_member,
        resolution=args.resolution,
    )
    dimension_inventory = summarize_dimension_inventory(inventory_records)

    plot_outputs = {"timeseries": [], "maps": []}
    if args.plot_timeseries or args.plot_maps:
        plot_dir = (
            Path(args.plot_dir).expanduser().resolve()
            if args.plot_dir
            else report_dir / "plots"
        )
        plot_dir.mkdir(parents=True, exist_ok=True)
        if args.plot_timeseries:
            plot_outputs["timeseries"] = create_timeseries_plots(
                produced_files, plot_dir, args.max_plots
            )
        if args.plot_maps:
            plot_outputs["maps"] = create_map_plots(
                produced_files, plot_dir, args.max_plots
            )

    report = build_report(
        args,
        yaml_path,
        expected_variables,
        log_records,
        produced_files,
        dimension_inventory,
        inspection_errors,
        plot_outputs,
    )

    write_json_report(report, report_dir / "validation_summary.json")
    write_csv_rows(report["log_records"], report_dir / "log_records.csv")
    write_csv_rows(
        [{"variable": variable} for variable in report["expected_but_not_produced"]],
        report_dir / "missing_variables.csv",
    )
    write_csv_rows(
        report["dimension_inventory"], report_dir / "dimension_inventory.csv"
    )
    write_csv_rows(report["inspection_errors"], report_dir / "inspection_errors.csv")
    write_markdown_summary(report, report_dir / "validation_summary.md")

    print_summary(report)
    print(f"Validation report written to {report_dir / 'validation_summary.json'}")
    if plot_outputs["timeseries"]:
        print(f"Time-series plots written to {plot_dir}")
    if plot_outputs["maps"]:
        print(f"Map plots written to {plot_dir}")

    if args.strict and (
        report["variables_with_log_errors"] or report["expected_but_not_produced"]
    ):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
