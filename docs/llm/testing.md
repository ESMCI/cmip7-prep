# Testing

Verified against commit `b90eb2a`. Last run on 2026-09-28 in the Docker image
(CMOR 3.15.3, Python 3.12, cesm-dev tables): 295 passed, 1 harmless numpy
binary-compatibility warning.

## Run

Tests import `cmor`, `xesmf` and `geocat`, so run them in Docker. Mount only
the directories you edit. Mounting the whole repo hides the image's tables.

```bash
docker run --rm \
  -v "$(pwd)/src":/opt/cmip7-prep/src -v "$(pwd)/tests":/opt/cmip7-prep/tests \
  -v "$(pwd)/scripts":/opt/cmip7-prep/scripts -v "$(pwd)/data":/opt/cmip7-prep/data \
  -e PYTHONPATH=/opt/cmip7-prep/src:/opt/cmip7-prep \
  cmip7-prep python -m pytest tests/ -q
```

Configuration:
- `pytest.ini` sets `--doctest-modules` and `pythonpath = src .`, and skips `scripts/` during collection.
- Doctests in `src/` run. Doctests in `scripts/` do not.
- Script tests add `scripts/` to `sys.path` themselves.

## Test map

| Test file | Covers | Needs real CMOR or tables |
|---|---|---|
| `test_mapping_compat.py`, `test_mapping_compat_extra.py` | YAML loading, freq filter, variants, formulas | no |
| `test_pipeline.py` | `_filename_contains_var`, `_collect_required_model_vars` | no |
| `test_regrid_latlon.py` | lat/lon output, time bounds, CICE dims, `_pick_maps`, map tables vs driver choices | no (weights monkeypatched) |
| `test_include_patterns.py` | Pattern tables, sampling, ice-sheet substitution | no |
| `test_vertical.py` | `to_plev` helpers | geocat import |
| `test_cmor_writer.py` | `CmorSession` end to end | yes: real CMOR plus `<repo>/cmip7-cmor-tables` (ENV-5) |
| `test_cmor_writer_helpers.py`, `test_cmor_utils_extra.py` | cmor_utils helpers | cmor import |
| `test_cache_tools.py` | `FXCache`, `RegridderCache` | xesmf import |
| `test_cism_grid.py` | CISM projection | pyproj |
| `test_mom6_static.py` | MOM6 helpers | no |
| `test_variable_selection.py` | Synthetic variables | no |
| `test_unused_functions.py` | `_apply_unit_conversion` | no |
| `test_convert_csv_to_yaml.py` | Converter rules (93 tests) | no |
| `test_yaml_to_csv.py` | Reverse converter | no |
| `test_validate_cmor_output.py` | Validator helpers | cmor (imports `cmor_driver`) |

`tests/conftest.py` provides a `fake_cmor` fixture (a `FakeCMOR` monkeypatched
into `cmor_writer`) for writer tests that must not need real CMOR.

Not covered by tests:
- `cmor_driver.main` and `process_one_var`
- `realize_regrid_prepare`
- `gen_timeseries.py`
- `build_validation_html.py`
- any end-to-end run with real weights

## Lint

`pre-commit run --all-files` runs black, pylint (`.pylintrc`, max line length
100, `scripts/` ignored) and the whitespace hooks. CI runs it in `pylint.yml`.

## Writing tests

- Use small synthetic `xr.Dataset`s. Existing tests build them inline.
- Monkeypatch `RegridderCache.get` and `get_map_paths` instead of needing weight files. `test_regrid_latlon.py` shows the pattern.
- Use `fake_cmor` for writer logic. Keep the real-CMOR tests few.
- Put doctests on small pure functions (the repo convention).
