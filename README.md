
# cmip7-prep (skeleton)

Quickstart:
1) Ensure you have a precomputed E3SM weights file: `map_ne30pg3_to_1x1d_aave.nc`.
2) Fill in `cmor_dataset.json` placeholders (experiment_id, etc.).
3) Export the CMIP7 Data Request v1.2.2 (CSV) as `data_request_v1.2.2.csv`.
4) `poetry install`
5) Prepare a variable, e.g., tas from TS:
   `poetry run cmip7-prep prepare --var TS --realm Amon --dreq-export data_request_v1.2.2.csv --mapping-yaml cesm_to_cmip7.yaml --cmor-tables Tables --outdir out --in-files cam.h1.TS.ne30pg3.*.nc`
6) Validate with PrePARE:
   `PrePARE --table-path Tables out/*.nc`
