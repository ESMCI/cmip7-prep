# CLAUDE.md

Instructions for coding agents working in this repository. Any agent can use
them; `AGENTS.md` points here. Human docs are separate (`README.md`,
`INSTALL.md`).

Detailed docs are in `docs/llm/`. Load only the doc your task needs (index
below). If a doc disagrees with the code, trust the code, fix the doc, and say
so in your summary.

## Ground rules

1. Never commit or push unless the user says to in the current request. Finish the change, summarize it, and ask for review.
2. **Local context:** if `SESSION.md` exists at the repo root, read it. It is git-ignored and describes this checkout only: local data, uncommitted local files, sandbox and tool quirks. Put anything specific to one machine, sandbox or session there, never in `docs/llm/` or this file. Never commit `SESSION.md`, or any file it lists as local-only.
3. Do not change behaviour that alters published values without the user's sign-off. That includes:
   - `data/intensive_vars.yaml` (regrid method)
   - `data/*_positive.yaml` (flux signs)
   - `REALM_GRIDS` and `GRIDS_OVERRIDES` (native vs regridded output)
   - weight files or target grids
   - `lev/1000` (Q-1)
   - any item marked `Q-n` in `docs/llm/issues.md`
4. Run anything that imports `cmor`, `xesmf` or `geocat` (including the tests) in Docker. See `docs/llm/testing.md`.
5. **Keep the LLM docs current.** When you change code, config, CLI flags, `data/` files or behaviour, update every stale reference in `docs/llm/` and in this file in the same change. Use the table below to find which doc to update. When you fix an issue, set its Status in `docs/llm/issues.md`; never delete or renumber it. When you find a new problem, add it to `issues.md` with the next free ID and cite that ID elsewhere. Update the `Verified against commit` line of every doc you re-check.
6. Repo-wide AI-written docs live only in `docs/llm/`; local context goes in `SESSION.md`. Keep the docs terse: tables, `file::symbol` references (not line numbers), and issue IDs instead of repeated explanations.
7. When a request touches something marked `Q-n`, or the scientific intent is unclear (units, signs, grids, vertical coordinates), ask the user before you change it.

| If you changed | Update |
|---|---|
| A module's public symbols, or moved code | `layout.md` |
| The driver flow, dispatch or writer axes | `pipeline.md` |
| The YAML schema, the converter, or `data/` files | `mapping.md`, and `features.md` if coverage changed |
| Realm, resolution or feature support | `features.md` |
| Dependencies, Docker, CI or environment | `stack.md`, `testing.md` |
| Tests or the test setup | `testing.md` |
| A fix or a new bug, gap or question | `issues.md`, plus the "Top gotchas" table below |
| A fact about this machine, sandbox or local data only | `SESSION.md` (not `docs/llm/`) |

## Project in brief

`cmip7-prep` converts CESM3 and NorESM3 output into CMOR 3.15 NetCDF for CMIP7.
It must keep working for variables and experiments outside CMIP7 (GAP-6).

1. Scientists maintain a spreadsheet. `scripts/convert_csv_to_yaml.py` turns it into `data/<model>_to_cmip7_<realm>.yaml`.
2. `scripts/gen_timeseries.py` turns history files into per-variable time series.
3. `scripts/cmor_driver.py` queries the CMIP7 Data Request, then for each variable: realize (rename or formula), vertical interpolation, regrid with ESMF weights, write with CMOR.

Library: `src/cmip7_prep/`. Scripts: `scripts/`. Config: `data/`.

Grid policy: output is regridded to lat/lon (`gr`) unless the mapping says
otherwise. Today seaIce, landIce and most ocean output stays native (`gn`)
(GAP-4).

## Quick commands

```bash
docker build -t cmip7-prep .   # or: git archive HEAD | docker build -t cmip7-prep -  (ENV-1)
docker run --rm -v "$(pwd)/src":/opt/cmip7-prep/src -v "$(pwd)/tests":/opt/cmip7-prep/tests \
  -v "$(pwd)/scripts":/opt/cmip7-prep/scripts -v "$(pwd)/data":/opt/cmip7-prep/data \
  -e PYTHONPATH=/opt/cmip7-prep/src:/opt/cmip7-prep cmip7-prep python -m pytest tests/ -q
```

For real runs:
- Always pass `--tables-root` (BUG-1).
- Set `PYTHONPATH=/opt/cmip7-prep/src` in the image (BUG-12).
- Use the table fork that matches the model (ENV-2).

## Top gotchas

| ID | Gotcha |
|---|---|
| BUG-1 | CESM driver run without `--tables-root` raises `NameError` |
| BUG-3 | `scale:` and `unit_conversion:` are ignored; convert units in `formula` |
| BUG-4 | 123 CESM land formulas call undefined functions |
| BUG-6 | The output `grid_label` comes from the data dims, not the YAML `grids` |
| BUG-7 | Converter output file names do not match the names the driver reads |
| BUG-12 | The Docker image imports a copy of the package that has no `data/` |
| BUG-13 | `data/ocean_geometry.nc` is empty, so MOM6 native output fails |
| ENV-2 | cesm-dev and noresm-dev tables have incompatible CVs |
| GAP-1 | CESM aerosol, atmosChem, ocnBgchem and landIce, and NorESM ocean, have no mapping |

## Docs index (`docs/llm/`)

| Doc | Read it when | Contents |
|---|---|---|
| [issues.md](docs/llm/issues.md) | Before changing behaviour; when something looks wrong | Register of BUG, GAP, DEBT, ENV, DOC and Q items, with location, workaround and status |
| [features.md](docs/llm/features.md) | "Is X supported for model or realm Y?" | Model x realm matrix, resolutions, capability table |
| [pipeline.md](docs/llm/pipeline.md) | You need the end-to-end flow or a dispatch rule | Stage commands, `cmor_driver.main` steps, `process_one_var` branches, regrid dispatch by dims, CMOR axis rules |
| [layout.md](docs/llm/layout.md) | "Where do I change X?" | Repo tree, module to symbol table, import graph, task to file table |
| [mapping.md](docs/llm/mapping.md) | Editing the mapping YAML, the converter or `data/` | YAML schema, formula namespace, spreadsheet rules, `data/` inventory, add-a-variable checklist |
| [stack.md](docs/llm/stack.md) | Setting up, building or running | Dependencies and versions, external inputs, Docker, conda, CI |
| [testing.md](docs/llm/testing.md) | Running or writing tests | Docker test command, test map, coverage gaps, lint |
