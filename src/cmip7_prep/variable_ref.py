"""Three names copied out of a data-request variable, so they can be sent to
another process.

cmor_driver.py can work on several variables at once, each in its own process.
Separate processes share nothing, so a description of the variable has to be
copied to each one. The data-request library's variable objects break when
copied -- the copy comes back asking itself the same question forever:

    File "data_request_api/query/vocabulary_server.py", line 238, in __getattr__
        return self.value
    [Previous line repeated 978 more times]
    RecursionError: maximum recursion depth exceeded

But nothing here uses the whole object. Only three names are ever read from
it: the branded variable name, the short name, and the branded name again from
its attributes. So those three are copied onto something simple and that is
sent instead -- like writing the name and address off a parcel rather than
posting the parcel.

The names are offered under the names the original used, so the code reading
them is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class _Named:
    """Holds one name.

    The code reading these writes ``variable.branded_variable_name.name``, so
    the name has to sit inside something rather than being a plain string.
    """

    name: str


@dataclass(frozen=True)
class VariableRef:
    """The three pieces of text that CMORizing one variable needs."""

    branded_variable_name: _Named
    physical_parameter: _Named
    attributes: dict = field(default_factory=dict)

    @classmethod
    def from_data_request(cls, cmip_var) -> "VariableRef":
        """Copy the three names out of a data-request variable.

        Everything else about it is left behind, which is the point: what is
        left behind is what could not be copied to another process.
        """
        # Each of the three may arrive either as an object with a .name or as
        # a plain string, depending on where it came from, so take the .name
        # when there is one and use the thing itself when there is not.
        branded = cmip_var.branded_variable_name
        branded_name = str(getattr(branded, "name", branded))

        parameter = getattr(cmip_var, "physical_parameter", None) or ""
        parameter_name = str(getattr(parameter, "name", parameter))

        # The branded name a second time, from the attributes dict.  Some
        # variables have no attributes at all, and some have the key missing,
        # so fall back to the name read above.
        attributes = getattr(cmip_var, "attributes", None) or {}
        attribute_name = str(attributes.get("branded_variable_name", branded_name))

        return cls(
            branded_variable_name=_Named(branded_name),
            physical_parameter=_Named(parameter_name),
            attributes={"branded_variable_name": attribute_name},
        )

    def __str__(self) -> str:
        return self.branded_variable_name.name
