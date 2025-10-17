import os
from cmip7_prep.dreq_search import find_variables_by_prefix


cmip_vars = find_variables_by_prefix(
    None, "Omon.", where={"List of Experiments": "picontrol"}
)
print(f"Found {len(cmip_vars)} variables: {cmip_vars}")
