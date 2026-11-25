"""Tests for the parent attributes build_dataset_cfg gives a dataset.

A run that starts from another run names it and the date it started from.  A
run with no parent carries none of those attributes.
"""

from pathlib import Path

from cmip7_prep.cmor_utils import build_dataset_cfg

TABLES_ROOT = Path(__file__).parent.parent / "cmip7-cmor-tables"

BRANCH = {
    "branch_method": "standard",
    "branch_time_in_parent": 65700.0,
    "branch_time_in_child": 0.0,
    "parent_time_units": "days since 0001-01-01 00:00:00",
}


def _parent_attributes(experiment, **kwargs):
    """Return the parent and branch attributes of one experiment's dataset."""
    cfg = build_dataset_cfg(
        model="noresm",
        resolution="NorESM3-LM",
        experiment=experiment,
        frequency="mon",
        ripf="r1i1p1f1",
        tables_root=TABLES_ROOT,
        data_version="v20200101",
        **kwargs,
    )
    return {
        name: value
        for name, value in cfg.items()
        if name.startswith(("parent_", "branch_"))
    }


def test_a_run_with_a_parent_carries_its_branch_attributes():
    """The values read from the case reach the dataset unchanged."""
    parent = _parent_attributes("historical", branch=BRANCH)

    assert parent["parent_experiment_id"] == "piControl"
    assert parent["parent_time_units"] == "days since 0001-01-01 00:00:00"
    assert parent["branch_time_in_parent"] == 65700.0


def test_a_startup_run_names_no_parent():
    """CMOR requires parent_experiment_id of piControl, so it says no parent."""
    parent = _parent_attributes("piControl", branch={})

    assert parent == {"parent_experiment_id": "no parent"}


def test_an_experiment_without_a_parent_has_no_parent_attributes():
    """The CV gives amip no parent, whatever the run started from."""
    assert not _parent_attributes("amip", branch={})
    assert not _parent_attributes("amip", branch=BRANCH)


def test_a_run_that_was_not_looked_at_keeps_the_placeholders():
    """Without the case there is nothing to say the run has no parent."""
    parent = _parent_attributes("historical")

    assert parent["parent_time_units"] == "days since 1850-01-01"
    assert parent["branch_time_in_parent"] == 0.0
