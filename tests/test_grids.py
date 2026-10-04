"""Tests for deriving each realm's input grid."""

import pytest

from cmip7_prep.grids import (
    ATM_RESOLUTIONS,
    needs_atmos_res,
    NATIVE_GRID,
    OCEAN_GRID,
    resolution_for,
)


class TestResolutionPerRealm:
    """Tests for choosing each realm's input grid.

    The grid is not a free choice: sea ice is on the model's tripolar grid
    whatever the atmosphere was run on, so passing one realm another's grid
    would regrid through the wrong weights and yield plausible wrong output.
    """

    @pytest.mark.parametrize("realm", ["atmos", "atmosChem", "aerosol", "land"])
    @pytest.mark.parametrize("atm", ATM_RESOLUTIONS)
    def test_atmosphere_realms_take_the_given_grid(self, realm, atm):
        """Atmosphere and land use the resolution the case was run at."""
        assert resolution_for("noresm", realm, atm) == atm

    @pytest.mark.parametrize("realm", ["seaIce", "ocean", "ocnBgchem"])
    @pytest.mark.parametrize("model", ["noresm", "cesm"])
    def test_ocean_realms_take_the_model_grid(self, realm, model):
        """Ocean and sea ice ignore the atmosphere resolution entirely."""
        assert resolution_for(model, realm, "ne16") == OCEAN_GRID[model]

    @pytest.mark.parametrize("realm", ["seaIce", "ocean", "ocnBgchem", "landIce"])
    def test_realms_with_their_own_grid_need_no_resolution(self, realm):
        """A sea-ice or land-ice run need not supply an atmosphere resolution."""
        assert resolution_for("noresm", realm)
        assert not needs_atmos_res(realm)

    @pytest.mark.parametrize("realm", ["atmos", "atmosChem", "aerosol", "land"])
    def test_atmosphere_realms_say_so_when_it_is_missing(self, realm):
        """A realm on the atmosphere grid refuses to guess its resolution."""
        assert needs_atmos_res(realm)
        with pytest.raises(ValueError, match="resolution must be given"):
            resolution_for("noresm", realm)

    def test_unknown_atmosphere_resolution_is_rejected(self):
        """A resolution with no weight files fails before any run starts."""
        with pytest.raises(ValueError, match="Unknown atmosphere resolution"):
            resolution_for("noresm", "atmos", "ne120")

    @pytest.mark.parametrize("model", ["noresm", "cesm"])
    @pytest.mark.parametrize("atm", ATM_RESOLUTIONS)
    def test_landice_is_never_regridded(self, model, atm):
        """land ice takes the pass-through grid whatever else was asked for.

        CISM output is written on its native projected grid, georeferenced from
        its x/y coordinates, so no weight files apply and the atmosphere's
        resolution is irrelevant to it.
        """
        assert resolution_for(model, "landIce", atm) == NATIVE_GRID
