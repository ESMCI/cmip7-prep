"""Tests for unique realm ownership of CMIP variable mappings."""

from cmip7_prep.mapping_compat import Mapping
from cmip7_prep.mapping_ownership import (
    classify_mapping_ownership,
    mapping_owners,
)


def test_mapping_owners_classify_variables_for_selected_realm():
    """Selected owners keep shared variables; unrelated conflicts stay ambiguous."""
    mappings = {
        "atmosChem": Mapping.from_packaged_default("noresm_to_cmip7_atmosChem.yaml"),
        "aerosol": Mapping.from_packaged_default("noresm_to_cmip7_aerosol.yaml"),
    }

    owners = mapping_owners(mappings, "mon")
    classification = classify_mapping_ownership(
        {
            "ch4_tavg-p19-hxy-air",
            "abs550bc_tavg-u-hxy-u",
            "unknown_tavg-u-hxy-u",
            "shared_tavg-u-hxy-u",
            "shared_elsewhere_tavg-u-hxy-u",
        },
        owners
        | {
            "shared_tavg-u-hxy-u": ("aerosol", "atmosChem"),
            "shared_elsewhere_tavg-u-hxy-u": ("aerosol", "landIce"),
        },
        "atmosChem",
    )

    assert classification.owned_here == {
        "ch4_tavg-p19-hxy-air",
        "shared_tavg-u-hxy-u",
    }
    assert classification.mapped_elsewhere == {"abs550bc_tavg-u-hxy-u": ("aerosol",)}
    assert classification.unmapped == {"unknown_tavg-u-hxy-u"}
    assert classification.ambiguous == {
        "shared_elsewhere_tavg-u-hxy-u": ("aerosol", "landIce")
    }
