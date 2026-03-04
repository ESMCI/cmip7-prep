# Copilot Instructions for cmip7-prep

## Project Overview
- **Purpose:** Prepares climate model output for CMIP7 submission, including variable mapping, regridding, and CMORization.
- **Main Components:**
  - `cmip7_prep/`: Core logic (CLI, data mapping, regridding, vertical interpolation, CMOR writing)
  - `scripts/`: Example workflows and test scripts
  - `tests/`: Pytest-based unit tests for core modules
  - `data/`: Reference files (mapping YAML, CMOR tables, data request CSV)

## Developer Workflows
**Install dependencies:** `poetry install`
**Run CLI:** `poetry run cmip7-prep prepare ...` (see README for full example)
**Validate output:** For CMIP7, output validation tools are under development. Refer to project documentation or CMIP7 guidance for recommended validation methods.
**Run tests:** `pytest` in the `tests/` directory
**Build:** No explicit build step; Python package managed by Poetry

## Key Patterns & Conventions
- **Configuration:**
  - Uses YAML (`cesm_to_cmip7.yaml`) for variable mapping
  - Uses JSON (`cmor_dataset.json`) for dataset metadata
  - Data request CSV must match expected format
- **Regridding:**
  - Requires ESMF weights file (`map_ne30pg3_to_1x1d_aave.nc`)
  - Regridding logic in `regrid.py`
- **CMORization:**
  - Table and attribute handling in `cmor_writer.py`
  - CMOR tables expected in `Tables/` directory
**CLI Entrypoint:**
  - Main CLI is provided by the `cmip7-prep` Python package (installed via Poetry)
  - Usage examples and argument patterns are documented in the README
  - There is no `cli.py` file and no `prepare` command; use the documented subcommands and options
  - Pytest tests in `tests/`, use realistic data and mocks
  - No test data generation; tests expect files in `data/`

## Integration Points
- **External tools:**
  - ESMF for regridding (weights file)
  - Output validation tools for CMIP7 are under development; do not use PrePARE.
- **Data flow:**
  - Input: CESM NetCDF files, mapping YAML, data request CSV
  - Output: CMORized NetCDF files in `out/`

## Examples
See the README for up-to-date usage examples. The CLI does not use a `prepare` command; use the available subcommands and options as documented.

## Tips for AI Agents
 Always check for required config/data files before running workflows
 Follow CLI argument patterns from README
 Use Poetry for all Python environment management
 Reference `tests/` for usage patterns and edge cases
 When writing documentation, target scientists, researchers, and system integrators. Organize information hierarchically, balance depth with accessibility, and include runnable, tested examples. Document edge cases and limitations. Ask clarifying questions if requirements are ambiguous.
 When writing or reviewing tests, use pytest best practices: ensure tests are isolated, deterministic, and well-documented. Focus on test files and avoid modifying production code unless specifically requested.

---
 **Config files:** YAML for variable mapping ([data/cesm_to_cmip7.yaml](data/cesm_to_cmip7.yaml)), JSON for dataset metadata ([data/cmor_dataset.json](data/cmor_dataset.json))
 **Testing:** Pytest, realistic data in [tests/](tests/), no synthetic test data. Write unit, integration, and end-to-end tests with clear descriptions and appropriate patterns. Review test quality for maintainability and isolation. Use pytest and avoid modifying production code unless specifically requested.
 **Error handling:** Use exceptions for critical errors, log warnings for recoverable issues (see [src/cmip7_prep/cache_tools.py](src/cmip7_prep/cache_tools.py))
 **Logging:** Use Python logging module, logs in [logs/](logs/)
 **No test data generation:** Tests expect files in [data/](data/)
- **Naming:** Use snake_case for functions/variables, PascalCase for classes
- **Imports:** Absolute imports preferred; see [src/cmip7_prep/cache_tools.py](src/cmip7_prep/cache_tools.py) for examples
- **Type hints:** Used in new code, see [src/cmip7_prep/pipeline.py](src/cmip7_prep/pipeline.py)
- **Docstrings:** Google-style, see [src/cmip7_prep/regrid.py](src/cmip7_prep/regrid.py)

## Architecture
- **Core modules:** Located in [src/cmip7_prep/](src/cmip7_prep)
  - `pipeline.py`: Orchestrates mapping, regridding, vertical interpolation, and CMOR writing
  - `cmor_writer.py`: Handles CMOR table logic and NetCDF output
  - `regrid.py`: ESMF-based regridding
  - `vertical.py`: Vertical interpolation
- **Data flow:**
  1. Input CESM NetCDF → mapping via YAML/CSV → regridding (ESMF weights) → vertical interpolation → CMORization → output NetCDF
  2. See [src/cmip7_prep/pipeline.py](src/cmip7_prep/pipeline.py) for orchestration
- **Service boundaries:** Each module is self-contained; CLI orchestrates workflow
- **Why:** Modular design enables easy extension for new models, variables, or workflows

## Project Conventions
- **Config files:** YAML for variable mapping ([data/cesm_to_cmip7.yaml](data/cesm_to_cmip7.yaml)), JSON for dataset metadata ([data/cmor_dataset.json](data/cmor_dataset.json))
- **Testing:** Pytest, realistic data in [tests/](tests/), no synthetic test data
- **Error handling:** Use exceptions for critical errors, log warnings for recoverable issues (see [src/cmip7_prep/cache_tools.py](src/cmip7_prep/cache_tools.py))
- **Logging:** Use Python logging module, logs in [logs/](logs/)
- **No test data generation:** Tests expect files in [data/](data/)

## Security
- **Sensitive data:** No authentication or secrets in codebase
- **Data files:** Only reference files in [data/](data/) and [cmip7-cmor-tables/reference/](cmip7-cmor-tables/reference/)
- **External access:** All external tools (ESMF, PrePARE) run locally; no remote API calls
