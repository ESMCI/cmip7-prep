"""Tests for reading the controlled vocabulary to reject bad values early."""

import json
from pathlib import Path

import pytest

from cmip7_prep import cv_lookup

TABLES_ROOT = "cmip7-cmor-tables"


@pytest.fixture(name="fake_tables")
def fake_tables_fixture(tmp_path):
    """Return a tables root holding a small stand-in vocabulary."""
    cv_dir = tmp_path / "tables-cvs"
    cv_dir.mkdir()
    (cv_dir / "cmor-cvs.json").write_text(
        json.dumps(
            {
                "CV": {
                    "experiment_id": {
                        "piControl": {"description": "control"},
                        "historical": {"description": "historical"},
                        "abrupt-4xCO2": {"description": "idealised"},
                    },
                    "license": ["CC-BY-4.0"],
                    "required_global_attributes": ["source_id"],
                }
            }
        ),
        encoding="utf-8",
    )
    cv_lookup._load.cache_clear()  # pylint: disable=protected-access
    return tmp_path


# ---------------------------------------------------------------- the real CV


class TestAgainstTheRepoVocabulary:
    """Checks against the vocabulary actually checked out in this repo."""

    def test_experiment_ids_are_readable(self):
        """The repo's vocabulary yields a non-trivial list of experiments."""
        experiments = cv_lookup.allowed_values(TABLES_ROOT, "experiment_id")
        assert len(experiments) > 20
        assert "piControl" in experiments

    def test_a_valid_experiment_passes(self):
        """A registered experiment validates silently."""
        cv_lookup.validate(TABLES_ROOT, "experiment_id", "piControl")

    @pytest.mark.parametrize(
        "dataset_file", ["cmor_dataset.json", "cmor_dataset_noresm.json"]
    )
    def test_the_license_we_request_is_registered(self, dataset_file):
        """Whatever a dataset file asks for must be in the vocabulary.

        CMOR refuses to write anything at all when it is not, and the message
        it gives names only the value, not the file it came from.  The spelling
        itself is not asserted: it has differed between tables versions
        ("CC-BY-4.0" against "CC-BY-4-0"), so pinning one here would fail on a
        tables update rather than on a real mismatch.
        """
        path = Path(__file__).parent.parent / "data" / dataset_file
        if not path.is_file():
            pytest.skip(f"{dataset_file} is not part of this branch")
        requested = json.loads(path.read_text(encoding="utf-8"))["license_id"]
        registered = cv_lookup.allowed_values(TABLES_ROOT, "license.license_id")
        assert requested in registered, (
            f"{dataset_file} asks for license_id {requested!r}, which the "
            f"controlled vocabulary does not register; it allows {registered}"
        )


# --------------------------------------------------------- the stand-in tables


class TestLookup:
    """Tests for allowed_values and validate."""

    def test_values_are_sorted(self, fake_tables):
        """Registered values come back sorted, for stable messages."""
        assert cv_lookup.allowed_values(fake_tables, "experiment_id") == [
            "abrupt-4xCO2",
            "historical",
            "piControl",
        ]

    def test_a_list_valued_key_works(self, fake_tables):
        """A key whose value is a list is read the same way as a mapping."""
        assert cv_lookup.allowed_values(fake_tables, "license") == ["CC-BY-4.0"]

    def test_a_dotted_path_walks_into_nested_keys(self):
        """The licences nest under license.license_id in the real vocabulary.

        What they are called is not this test's business -- only that the path
        reaches them and returns something.
        """
        licenses = cv_lookup.allowed_values(TABLES_ROOT, "license.license_id")
        assert licenses
        assert all(isinstance(name, str) and name for name in licenses)

    def test_a_dotted_path_that_goes_nowhere_raises(self):
        """A wrong path says what is controlled at the level it reached."""
        with pytest.raises(KeyError, match="not a controlled attribute"):
            cv_lookup.allowed_values(TABLES_ROOT, "license.no_such_level")

    def test_unknown_key_lists_what_is_controlled(self, fake_tables):
        """A typo in the attribute name is as loud as one in the value."""
        with pytest.raises(KeyError, match="not a controlled attribute"):
            cv_lookup.allowed_values(fake_tables, "experimnet_id")

    def test_missing_vocabulary_says_where_it_looked(self, tmp_path):
        """A tables root without a vocabulary names the path it wanted."""
        cv_lookup._load.cache_clear()  # pylint: disable=protected-access
        with pytest.raises(FileNotFoundError, match="cmor-cvs.json"):
            cv_lookup.allowed_values(tmp_path, "experiment_id")


class TestValidate:
    """Tests for the rejection messages."""

    def test_valid_value_is_accepted(self, fake_tables):
        """A registered value raises nothing."""
        cv_lookup.validate(fake_tables, "experiment_id", "historical")

    def test_near_miss_suggests_the_real_thing(self, fake_tables):
        """A misspelling is answered with the nearest registered values."""
        with pytest.raises(ValueError, match="Did you mean: piControl"):
            cv_lookup.validate(fake_tables, "experiment_id", "piControll")

    def test_wild_value_points_at_the_file(self, fake_tables):
        """With nothing close, the message says where the list lives."""
        with pytest.raises(ValueError, match="cmor-cvs.json"):
            cv_lookup.validate(fake_tables, "experiment_id", "banana")

    def test_case_matters(self, fake_tables):
        """Vocabulary values are case sensitive, as CMOR treats them."""
        with pytest.raises(ValueError, match="not in the controlled vocabulary"):
            cv_lookup.validate(fake_tables, "experiment_id", "picontrol")
