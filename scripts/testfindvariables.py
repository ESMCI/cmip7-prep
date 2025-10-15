import os
from pathlib import Path
from cmip7_prep.mapping_compat import Mapping
from cmip7_prep.dreq_search import find_variables_by_prefix

basedir = Path(os.getenv("SCRATCH")) / Path(
    "archive/timeseries/b.e30_beta06.B1850C_LTso.ne30_t232_wgx3.192.wrkflw.1/atm/hist/"
)

# 0) Load mapping (uses packaged data/cesm_to_cmip7.yaml by default)
mapping = Mapping.from_packaged_default()

cmip_vars = find_variables_by_prefix(
    None, "Amon.", where={"List of Experiments": "picontrol"}
)
print(f"Found {len(cmip_vars)} variables: {cmip_vars}")
