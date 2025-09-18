
# Makefile for cmip7_prep
PY ?= python
POETRY ?= poetry

# Map files (adjust paths if needed)
CONS_MAP = map_ne30pg3_to_1x1d_aave.nc
BILIN_MAP = map_ne30pg3_to_1x1d_bilin.nc

OUTDIR = out

.PHONY: all env install target weights tas pr fx prepare validate clean

all: install

env:
	$(POETRY) env use $(shell which python)

install: pyproject.toml
	$(POETRY) install

target:
	$(POETRY) run cmip7-prep make-target grids/target_1deg.nc

# Example: prepare tas from CESM TS
tas:
	$(POETRY) run cmip7-prep prepare \	  --var TS \	  --realm Amon \	  --dreq-export data_request_v1.2.2.csv \	  --mapping-yaml cesm_to_cmip7.yaml \	  --cmor-tables Tables \	  --outdir $(OUTDIR) \	  --in-files cam.h1.TS.ne30pg3.000101-000112.nc

# Example: precipitation from PRECT
pr:
	$(POETRY) run cmip7-prep prepare \	  --var PRECT \	  --realm Amon \	  --dreq-export data_request_v1.2.2.csv \	  --mapping-yaml cesm_to_cmip7.yaml \	  --cmor-tables Tables \	  --outdir $(OUTDIR) \	  --in-files cam.h1.PRECT.ne30pg3.000101-000112.nc

fx:
	@echo "Regridding fx (areacella, sftlf) alongside variables is handled in cmor_writer.py."

validate:
	@which PrePARE >/dev/null || (echo "PrePARE not found in PATH"; exit 1)
	PrePARE --table-path Tables $(OUTDIR)/*.nc

clean:
	rm -rf $(OUTDIR)
