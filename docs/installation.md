# Installation

## NIRD

From inside a clone of this repository:

```bash
cd scripts
. setup_nird.sh
```

That activates the shared environment and puts the package on your path. To do
it by hand instead:

```bash
conda activate /projects/NS9560K/diagnostics/cmordev_env_312/
export PYTHONPATH=/path/to/cmip7-prep/src:$PYTHONPATH
```

:::{warning}
Give `PYTHONPATH` an absolute path. A relative one such as `../src` resolves
only when your working directory happens to be `scripts/`, and the import then
fails from anywhere else with no obvious reason.
:::

`run_reference_case.py` needs no `PYTHONPATH` at all: it adds `src` to the path
itself, by absolute path, so it works from any directory. The other three
scripts rely on the environment.

## Derecho

```bash
module load conda
conda activate /glade/work/jedwards/conda-envs/CMORDEV
pip install -e .
```

## Your own machine

```bash
conda env create -f environment.yml
conda activate cmip7-prep
pip install -e .
```

## Docker

The image carries CMOR, ESMF, the CMOR tables and the package itself, so the
host needs no conda environment.

```bash
docker build -t cmip7-prep .
```

Run it with the input and output directories mounted:

```bash
docker run --rm -v /path/to/timeseries:/data -v /path/to/output:/out cmip7-prep \
    python scripts/cmor_driver.py --realm atmos --tsdir /data --outdir /out
```

The tables are cloned at build time from the fork CI uses. To build against a
different checkout:

```bash
docker build -t cmip7-prep \
    --build-arg TABLES_REPO=https://github.com/NorESMhub/cmip7-cmor-tables.git \
    --build-arg TABLES_REF=noresm-dev .
```

On a system that runs Apptainer rather than Docker, convert the image:

```bash
apptainer build cmip7-prep.sif docker-daemon://cmip7-prep:latest
```

## Checking it worked

```bash
python -c "import cmip7_prep; print(cmip7_prep.__file__)"
```

If that prints a path, the package is importable and no `PYTHONPATH` is needed.
If it raises `ModuleNotFoundError`, set `PYTHONPATH` as above.

## The CMOR tables

The tables are a git submodule, so a plain `git clone` leaves the directory
empty. Populate it with:

```bash
git submodule update --init --recursive
```

The driver reads `cmip7-cmor-tables/` at the root of this repository by
default, and `--tables-root` points it at a different checkout. The tables
decide which variables the data request asks for, so an out-of-date submodule
changes what a run produces.
