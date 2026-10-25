# cmip7-prep

A Python library and set of scripts for preparing NorESM and CESM native model
output for submission to CMIP7, through CMOR (the Climate Model Output
Rewriter).

[![Documentation](https://readthedocs.org/projects/cmip7prep/badge/?version=latest)](https://cmip7prep.readthedocs.io/en/latest/)

**Documentation: <https://cmip7prep.readthedocs.io>**

The guide covers what the three stages do, how to run them over a case, how to
read what they produced, what to do when something goes wrong, and how the
mapping files and tables work. Its source is `docs/` in this repository, and it
rebuilds on every push to `main`.

To read it locally instead:

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
open docs/_build/html/index.html
```

## Installing

On NIRD, from inside a clone of this repository:

```bash
cd scripts
. setup_nird.sh
```

Anywhere else:

```bash
conda env create -f environment.yml
conda activate cmip7-prep
pip install -e .
```

See the guide's installation page for Derecho, Docker and Apptainer.

## Running

```bash
python scripts/run_reference_case.py \
  --case-dir <the archived case directory> \
  --outdir <where the output should go> \
  --model noresm --model-res NorESM3-LM \
  --experiment piControl \
  --years 1:50:10 \
  --dry-run
```

`--dry-run` prints the steps and exits. See the guide's tutorial page for what
each argument does, and its recipes page for running one realm or one stage.

## Tests

```bash
pytest
```
