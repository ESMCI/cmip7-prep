"""Tests for unique realm ownership of CMIP variable mappings."""

from cmip7_prep.mapping_compat import Mapping
from cmip7_prep.mapping_ownership import (
    classify_mapping_ownership,
    mapping_owners,
)


def test_mapping_owners_classify_shared_variables_for_every_owner():
    """Each owner keeps a shared variable; other realms record all owners."""
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
    land_classification = classify_mapping_ownership(
        {"shared_tavg-u-hxy-u"},
        {"shared_tavg-u-hxy-u": ("aerosol", "atmosChem")},
        "land",
    )

    assert classification.owned_here == {
        "ch4_tavg-p19-hxy-air",
        "shared_tavg-u-hxy-u",
    }
    assert classification.mapped_elsewhere == {
        "abs550bc_tavg-u-hxy-u": ("aerosol",),
        "shared_elsewhere_tavg-u-hxy-u": ("aerosol", "landIce"),
    }
    assert classification.unmapped == {"unknown_tavg-u-hxy-u"}
    assert land_classification.owned_here == set()
    assert land_classification.mapped_elsewhere == {
        "shared_tavg-u-hxy-u": ("aerosol", "atmosChem")
    }
    assert land_classification.unmapped == set()


def test_cesm_shared_atmos_landice_variables_are_mapped_elsewhere_for_land():
    """The nine shared CESM brands are informational in the land realm."""
    variables = {
        f"{name}_tavg-u-hxy-is"
        for name in (
            "hfls",
            "hfss",
            "prra",
            "prsn",
            "rlds",
            "rlus",
            "rsds",
            "rsus",
            "ts",
        )
    }
    mappings = {
        "atmos": Mapping.from_packaged_default("cesm_to_cmip7_atmos.yaml"),
        "landIce": Mapping.from_packaged_default("cesm_to_cmip7_landIce.yaml"),
    }
    owners = mapping_owners(mappings, "mon")

    land = classify_mapping_ownership(variables, owners, "land")
    land_ice = classify_mapping_ownership(variables, owners, "landIce")

    assert land.owned_here == set()
    assert land.unmapped == set()
    assert land.mapped_elsewhere == {
        variable: ("atmos", "landIce") for variable in variables
    }
    assert land_ice.owned_here == variables
