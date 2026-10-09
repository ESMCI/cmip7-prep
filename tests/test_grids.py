"""Tests for reading each realm's input grid from the model tables.

Nothing here names a grid or a resolution.  Both are data, stated per model in
``data/<model>_regrid_maps.yaml``, and restating them in a test would only
assert that this file and that one were edited together.  So the tests check
the properties the table must have, and that the lookup returns what the table
says, whatever the table says.
"""

import pytest

from cmip7_prep.grids import MODEL_RESOLUTIONS, CUSTOM, resolution_for
from cmip7_prep.regrid_maps import load_regrid_maps

MODELS = ("noresm", "cesm")


def _rows(model):
    """Yield (resolution, realm, row) for every row in a model's table."""
    for resolution, realms in load_regrid_maps(model)["grid_names_per_realm"].items():
        for realm, row in realms.items():
            yield resolution, realm, row


class TestTheLookupReturnsWhatTheTableSays:
    """resolution_for is a reader of the table, and decides nothing itself."""

    @pytest.mark.parametrize("model", MODELS)
    def test_every_row_is_returned_verbatim(self, model):
        """Each realm gets its own row's input_grid, for each resolution."""
        for resolution, realm, row in _rows(model):
            assert (
                resolution_for(model, realm, resolution) == row["input_grid"]
            ), f"{model}/{resolution}/{realm}"

    @pytest.mark.parametrize("model", MODELS)
    def test_the_resolution_selects_the_row(self, model):
        """Two resolutions that differ for a realm give different answers.

        This is what the original bug got wrong: a realm was given another
        realm's grid, which regrids through the wrong weights and produces
        output that looks fine.
        """
        table = load_regrid_maps(model)["grid_names_per_realm"]
        resolutions = sorted(table)
        differing = [
            (realm, [table[res][realm]["input_grid"] for res in resolutions])
            for realm in table[resolutions[0]]
            if len({table[res][realm]["input_grid"] for res in resolutions}) > 1
        ]
        if not differing:
            pytest.skip(f"{model} has only one resolution in its table")
        for realm, expected in differing:
            got = [resolution_for(model, realm, res) for res in resolutions]
            assert got == expected, realm


class TestEveryRowIsUsable:
    """Each row must name things the rest of the table can act on."""

    @pytest.mark.parametrize("model", MODELS)
    def test_every_input_grid_has_weights(self, model):
        """A grid with no 'resolutions' entry would fail at run time."""
        defined = set(load_regrid_maps(model)["resolutions"])
        for resolution, realm, row in _rows(model):
            assert row["input_grid"] in defined, f"{model}/{resolution}/{realm}"

    @pytest.mark.parametrize("model", MODELS)
    def test_every_row_has_both_fields_and_nothing_else(self, model):
        """Neither field may be left out, and no third field is read."""
        for resolution, realm, row in _rows(model):
            assert set(row) == {
                "input_grid",
                "grid_label",
            }, f"{model}/{resolution}/{realm}"

    @pytest.mark.parametrize("model", MODELS)
    def test_every_resolution_defines_the_same_realms(self, model):
        """A realm present at one resolution and missing at another would fail
        only for the run that asked for it."""
        table = load_regrid_maps(model)["grid_names_per_realm"]
        realms = [set(rows) for rows in table.values()]
        assert all(group == realms[0] for group in realms)


class TestWhatIsRefused:
    """A missing or unknown value fails, rather than being guessed."""

    def test_a_missing_resolution_is_rejected(self):
        """The resolution is never guessed, for any realm."""
        with pytest.raises(ValueError, match="must be given"):
            resolution_for("noresm", next(iter(_rows("noresm")))[1])

    def test_an_unknown_resolution_is_rejected(self):
        """A resolution the command line does not offer fails up front."""
        unknown = "".join(MODEL_RESOLUTIONS) + "x"
        with pytest.raises(ValueError, match="Unknown atmosphere resolution"):
            resolution_for("noresm", "atmos", unknown)

    def test_a_resolution_the_model_lacks_is_rejected(self):
        """An offered resolution with no rows for this model fails.

        Not every model runs every resolution, and the message names the ones
        it has rather than falling back to another.
        """
        for model in MODELS:
            has_rows = set(load_regrid_maps(model)["grid_names_per_realm"])
            missing = [
                res
                for res in MODEL_RESOLUTIONS
                if res not in has_rows and res != CUSTOM
            ]
            for res in missing:
                with pytest.raises(ValueError, match="No grid names defined"):
                    resolution_for(model, "atmos", res)

    def test_an_unknown_realm_is_rejected(self):
        """A misspelled realm is answered with the table's own keys."""
        resolution = next(iter(_rows("noresm")))[0]
        with pytest.raises(ValueError, match="No grid names defined"):
            resolution_for("noresm", "atmosphere", resolution)


class TestACustomCase:
    """A case that is not one of the standard configurations."""

    @pytest.mark.parametrize("model", MODELS)
    def test_undeclared_grids_mean_nothing_is_regridded(self, model):
        """With no block of its own, every realm is left alone.

        Nothing can be assumed about the grids of a case the table says
        nothing about, so the one safe answer is not to regrid.
        """
        declared = load_regrid_maps(model)["grid_names_per_realm"]
        if CUSTOM in declared:
            pytest.skip(f"{model} declares a {CUSTOM} block")
        for _, realm, _ in _rows(model):
            assert resolution_for(model, realm, CUSTOM) == CUSTOM

    @pytest.mark.parametrize("model", MODELS)
    def test_declared_grids_are_used(self, model):
        """Filling in the block makes the case behave like any other.

        This is what the commented-out template at the end of the table is
        for, so a custom case on, say, regular lat/lon is not forced through
        the no-regrid path.
        """
        declared = load_regrid_maps(model)["grid_names_per_realm"]
        if CUSTOM not in declared:
            pytest.skip(f"{model} declares no {CUSTOM} block")
        for realm, row in declared[CUSTOM].items():
            assert resolution_for(model, realm, CUSTOM) == row["input_grid"]

    def test_it_is_offered_on_the_command_line(self):
        """It is a --model-res choice, so the lookup has to accept it."""
        assert CUSTOM in MODEL_RESOLUTIONS
