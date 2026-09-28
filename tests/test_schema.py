"""Tests for the mapping schema (data/schemas/mapping.schema.yaml)."""

import pytest

from cmip7_prep.schema import mapping_errors


@pytest.mark.parametrize(
    "entry, expected",
    [
        ("", "should be non-empty"),
        (1.0, "1.0 is not of type 'object'"),
        ({"formula": "T", "grids": ["gm"]}, "'gm' is not one of ['gn', 'gr']"),
        ({"formula": "T", "units": 1}, "1 is not of type 'string'"),
        ({"formula": "T", "positive": "upward"}, "'upward' is not one of"),
        ({"formula": {"month": "T"}}, "'month' is not one of"),
        (
            {"formula": "T", "sources": [{"model_var": "T"}]},
            "('sources' was unexpected)",
        ),
        ({"units": "K"}, "'formula' is a required property"),
        (
            {"formula": "T", "variants": [{"region": "nh", "formula": "T"}]},
            "should not be valid under",
        ),
        ({"variants": [{"formula": "T"}]}, "'region' is a required property"),
    ],
)
def test_invalid_entries_are_reported(entry, expected):
    """Common mistakes, including keys of the old format, are caught."""
    errors = mapping_errors({"variables": {"tas": entry}})
    assert any(expected in message for message in errors), errors


def test_file_grids_default():
    """The file-level grids default is checked like an entry's."""
    assert mapping_errors({"grids": ["gm"], "variables": {}}) == [
        "grids/0: 'gm' is not one of ['gn', 'gr']"
    ]
