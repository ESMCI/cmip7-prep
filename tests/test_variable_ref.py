"""Tests for the transportable stand-in for a data-request variable."""

import pickle

from cmip7_prep.variable_ref import VariableRef, _Named


class _Recursive:
    """A stand-in for the data-request objects, with their fatal habit.

    They answer any unknown attribute with their own ``value``.  Unpickling
    makes an empty shell and then fills it, so the shell is asked for ``value``
    before it has one, which asks for ``value``, until Python gives up.
    """

    def __init__(self, branded, parameter):
        self.branded_variable_name = _Named(branded)
        self.physical_parameter = _Named(parameter)
        self.attributes = {"branded_variable_name": branded}

    def __getattr__(self, item):  # pragma: no cover - only reached when empty
        return self.value


class TestItSurvivesBeingSentToAnotherProcess:
    """The point of the class: the original cannot be pickled, this can."""

    def test_the_original_cannot_be_pickled(self):
        """Confirms the problem is real, so the stand-in has a reason to be."""
        original = _Recursive("tas_tavg-h2m-hxy-u", "tas")
        try:
            pickle.loads(pickle.dumps(original))
        except RecursionError:
            return
        raise AssertionError("expected unpickling to recurse")

    def test_the_stand_in_round_trips(self):
        """What actually gets sent to a worker comes back unchanged."""
        ref = VariableRef.from_data_request(_Recursive("tas_tavg-h2m-hxy-u", "tas"))
        assert pickle.loads(pickle.dumps(ref)) == ref


class TestItReadsLikeTheOriginal:
    """The writer is given one of these and must not notice the difference."""

    def test_the_three_strings_are_carried(self):
        """Exactly what cmor_driver and cmor_writer read from the original."""
        ref = VariableRef.from_data_request(_Recursive("tas_tavg-h2m-hxy-u", "tas"))
        assert ref.branded_variable_name.name == "tas_tavg-h2m-hxy-u"
        assert ref.physical_parameter.name == "tas"
        assert ref.attributes["branded_variable_name"] == "tas_tavg-h2m-hxy-u"

    def test_the_branded_name_is_an_object_with_a_name(self):
        """cmor_writer does `bvn_attr.name`, so a bare string would fail."""
        ref = VariableRef.from_data_request(_Recursive("siu_tavg-u-hxy-si", "siu"))
        assert hasattr(ref.branded_variable_name, "name")

    def test_printing_it_names_the_variable(self):
        """Log lines use it directly, so it should not print as a dataclass."""
        ref = VariableRef.from_data_request(_Recursive("siu_tavg-u-hxy-si", "siu"))
        assert str(ref) == "siu_tavg-u-hxy-si"

    def test_a_missing_physical_parameter_is_tolerated(self):
        """Not every variable carries one; absence should not raise."""

        class _NoParameter:
            branded_variable_name = _Named("x_tavg-u-hxy-u")
            attributes: dict = {}

        ref = VariableRef.from_data_request(_NoParameter())
        assert ref.physical_parameter.name == ""
        assert ref.attributes["branded_variable_name"] == "x_tavg-u-hxy-u"
