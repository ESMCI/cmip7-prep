"""Determine realm ownership from the model's variable-mapping YAML files.

Ownership here means that a branded variable name is present in a realm's
mapping for the requested frequency. It is not inferred from the CMIP tables.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .mapping_compat import Mapping


@dataclass
class MappingOwnership:
    """Classify requested branded variable names by their mapping realm.

    Attributes:
        owned_here: Variables mapped in exactly the realm being processed.
        mapped_elsewhere: Variables mapped in exactly one other realm, with
            that realm stored as a one-item tuple.
        unmapped: Variables absent from all realm mappings.
        ambiguous: Variables present in multiple realm mappings, with all
            matching realms stored as a tuple.
    """

    owned_here: set[str]
    mapped_elsewhere: dict[str, tuple[str, ...]]
    unmapped: set[str]
    ambiguous: dict[str, tuple[str, ...]]


def load_mapping_catalog(
    model: str,
    realm_yaml_map: dict[str, dict[str, str]],
    *,
    selected_realm: str,
    selected_mapping: Mapping | None = None,
) -> dict[str, Mapping]:
    """Load all configured realm mappings for a model.

    The selected realm's already-loaded mapping is reused when supplied,
    avoiding a second load and allowing a custom mapping to participate in the
    catalog. If that realm is not listed in the model configuration, the
    supplied mapping is still included.

    Args:
        model: Model key to look up in ``realm_yaml_map``.
        realm_yaml_map: Model-to-realm-to-packaged-YAML-filename configuration.
        selected_realm: Realm currently being processed.
        selected_mapping: Optional mapping already loaded for
            ``selected_realm``.

    Returns:
        A dictionary mapping each realm name to its loaded ``Mapping``.

    Raises:
        ValueError: If ``model`` has no entry in ``realm_yaml_map``.
    """
    model_mappings = realm_yaml_map.get(model)
    if model_mappings is None:
        raise ValueError(f"No realm mappings configured for model={model!r}")

    mappings = {}
    for realm, filename in model_mappings.items():
        if realm == selected_realm and selected_mapping is not None:
            mappings[realm] = selected_mapping
        else:
            mappings[realm] = Mapping.from_packaged_default(filename)
    if selected_realm not in mappings and selected_mapping is not None:
        mappings[selected_realm] = selected_mapping
    return mappings


def mapping_owners(
    mappings: dict[str, Mapping], frequency: str
) -> dict[str, tuple[str, ...]]:
    """Build a lookup from each mapped variable to its realm or realms.

    Each mapping contributes the variable names it supports at ``frequency``.
    A name appearing in multiple mappings is associated with every matching
    realm; this function does not resolve or prefer one realm over another.

    Args:
        mappings: Realm names and their loaded variable mappings.
        frequency: Frequency used to filter the variables in each mapping.

    Returns:
        A dictionary from branded variable names to sorted tuples of realm
        names. Variables with no matching mapping are not included.
    """
    owners: dict[str, list[str]] = defaultdict(list)
    for realm, mapping in mappings.items():
        for variable in mapping.iter_variable_names(freq=frequency):
            owners[variable].append(realm)
    return {variable: tuple(sorted(realms)) for variable, realms in owners.items()}


def classify_mapping_ownership(
    variables: set[str] | list[str],
    owners: dict[str, tuple[str, ...]],
    realm: str,
) -> MappingOwnership:
    """Classify variables relative to the realm currently being processed.

    A variable is classified as owned here if its sole mapped realm is
    ``realm``, mapped elsewhere if it has exactly one different owner, unmapped
    if it has no owners, or ambiguous if it has multiple owners. The ownership
    lookup is typically produced by :func:`mapping_owners` from the model's
    realm YAML mappings.

    Args:
        variables: Branded variable names to classify, such as names requested
            for processing or validation.
        owners: Variable-to-realm lookup, usually returned by
            :func:`mapping_owners`.
        realm: Realm whose processing run is being considered.

    Returns:
        A ``MappingOwnership`` containing the four disjoint classifications.
    """
    owned_here = set()
    mapped_elsewhere = {}
    unmapped = set()
    ambiguous = {}

    for variable in variables:
        variable_owners = owners.get(variable, ())
        if not variable_owners:
            unmapped.add(variable)
        elif len(variable_owners) > 1:
            ambiguous[variable] = variable_owners
        elif variable_owners[0] == realm:
            owned_here.add(variable)
        else:
            mapped_elsewhere[variable] = variable_owners

    return MappingOwnership(owned_here, mapped_elsewhere, unmapped, ambiguous)
