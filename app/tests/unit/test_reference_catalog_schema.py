from __future__ import annotations

import repositories.schemas.configuration as configuration
from repositories.schemas.configuration import ReferenceCatalogEntry, ReferenceCatalogManifest
from sqlalchemy import UniqueConstraint

###############################################################################
def test_reference_catalog_tables_exist_in_schema_models() -> None:
    assert ReferenceCatalogEntry.__tablename__ == "reference_catalog_entries"
    assert ReferenceCatalogManifest.__tablename__ == "reference_catalog_manifests"

###############################################################################
def test_reference_catalog_entry_unique_constraint_shape() -> None:
    constraints = [
        item
        for item in ReferenceCatalogEntry.__table_args__
        if isinstance(item, UniqueConstraint)
    ]
    assert constraints
    identity = next(
        constraint
        for constraint in constraints
        if constraint.name == "uq_reference_catalog_entries_identity"
    )
    assert tuple(identity.columns.keys()) == (
        "manifest",
        "domain",
        "category",
        "key",
        "locale",
        "normalized_value",
    )

###############################################################################
def test_reference_catalog_manifest_unique_constraint_shape() -> None:
    constraints = [
        item
        for item in ReferenceCatalogManifest.__table_args__
        if isinstance(item, UniqueConstraint)
    ]
    assert constraints
    identity = next(
        constraint
        for constraint in constraints
        if constraint.name == "uq_reference_catalog_manifests_manifest"
    )
    assert tuple(identity.columns.keys()) == ("manifest",)

###############################################################################
def test_legacy_text_normalization_model_removed() -> None:
    assert not hasattr(configuration, "TextNormalizationTerm")
