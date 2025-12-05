# Copilot Instructions for cmip7-prep

## Project Overview
- **Purpose:** Prepares climate model output for CMIP7 submission, including variable mapping, regridding, and CMORization.
- **Main Components:**
  - `cmip7_prep/`: Core logic (CLI, data mapping, regridding, vertical interpolation, CMOR writing)
  - `scripts/`: Example workflows and test scripts
  - `tests/`: Pytest-based unit tests for core modules
  - `data/`: Reference files (mapping YAML, CMOR tables, data request CSV)

## Developer Workflows
- **Install dependencies:** `poetry install`
- **Run CLI:** `poetry run cmip7-prep prepare ...` (see README for full example)
- **Validate output:** Use `PrePARE` tool with generated NetCDF files
- **Run tests:** `pytest` in the `tests/` directory
- **Build:** No explicit build step; Python package managed by Poetry

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
  - PrePARE for output validation
- **Data flow:**
  - Input: CESM NetCDF files, mapping YAML, data request CSV
  - Output: CMORized NetCDF files in `out/`

## Examples
See the README for up-to-date usage examples. The CLI does not use a `prepare` command; use the available subcommands and options as documented.

## Tips for AI Agents
- Always check for required config/data files before running workflows
- Follow CLI argument patterns from `cli.py` and README
- Use Poetry for all Python environment management
- Reference `tests/` for usage patterns and edge cases

---
*Please review and suggest edits for any unclear or missing sections.*
